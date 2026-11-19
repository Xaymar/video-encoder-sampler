// The converter: our own Vulkan compute shaders (shaders/cv_*.comp) on the device the scorer and FFmpeg's decoders use (vk_compute.h). Reads frames in
// whatever sample layout the source has (planar, NV12/P010/NV24, packed RGB, ...) from memory or from FFmpeg Vulkan frames, scales them in linear light
// and converts them to planar YUV, into memory or into FFmpeg Vulkan frames (Picture::gpu) for the GPU scorer.
//
// The steps, each a pass over planes of 32-bit floats (nominal values: Y and RGB 0..1, Cb and Cr -0.5..0.5):
//   unpack     every component of the source into its own plane, exactly (any layout gives the same floats for the same samples)
//   resize, or a change of transfer curve (the "linear" path):
//     chroma up  subsampled chroma to the luma grid (upscaler), sited left
//     to linear  Y'CbCr -> R'G'B' (the source's matrix) -> linear light (the source's transfer curve)
//     scale      each of R, G, B to the output size (upscaler or downscaler; separable, or EWA for polar filters)
//     from lin.  linear light -> R'G'B' (the output's curve) -> Y'CbCr (the output's matrix)
//     chroma dn  chroma to the output's subsampling (downscaler), sited left
//   otherwise (the "direct" path: same size and curve): chroma resampled straight from the source's grid to the output's where they differ
//   pack       to the output's range and bit depth, rounded to the nearest code value (no dithering), clamped
// RGB sources always take the linear path (they need the matrix); with their own size and curve the curve is left out (not applied and undone).
#define _USE_MATH_DEFINES // M_PI on Windows
#include "convert.h"
#include "decode.h"
#include "gpu.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

#ifndef VES_NO_VULKAN
#include "ff.h"
#include "vk_compute.h"
extern "C" {
#include <libavutil/hwcontext.h>
#include <libavutil/hwcontext_vulkan.h>
}
#endif

namespace {

// ---- Filters (the names and shapes are libplacebo's, so that configurations keep working) ----

struct FilterDef {
	const char* name;
	int kernel; // 0 box, 1 triangle, 2 cubic (b, c), 3 spline16, 4 spline36, 5 spline64, 6 sinc, 7 jinc
	int window; // -1 none, 6 sinc, 7 jinc, 8 hann
	double radius;
	bool polar;
	double b = 0, c = 0, blur = 0;
};

constexpr double JINC_ZERO1 = 1.2196698912665045, JINC_ZERO3 = 3.2383154841662362076499, JINC_ZERO4 = 4.2410628637960698819573;
const double ROB_B = 12 / (19 + 9 * M_SQRT2), ROB_C = 113 / (58 + 216 * M_SQRT2), ROBS_B = 6 / (13 + 7 * M_SQRT2), ROBS_C = 7 / (2 + 12 * M_SQRT2);

const FilterDef kFilters[] = {
	{"nearest", 0, -1, 0.5, false},
	{"box", 0, -1, 0.5, false},
	{"bilinear", 1, -1, 1.0, false},
	{"linear", 1, -1, 1.0, false},
	{"triangle", 1, -1, 1.0, false},
	{"bicubic", 2, -1, 2.0, false, 1.0, 0.0},
	{"cubic", 2, -1, 2.0, false, 1.0, 0.0},
	{"hermite", 2, -1, 1.0, false, 0.0, 0.0},
	{"catmull_rom", 2, -1, 2.0, false, 0.0, 0.5},
	{"mitchell", 2, -1, 2.0, false, 1 / 3.0, 1 / 3.0},
	{"robidoux", 2, -1, 2.0, false, ROB_B, ROB_C},
	{"robidouxsharp", 2, -1, 2.0, false, ROBS_B, ROBS_C},
	{"spline16", 3, -1, 2.0, false},
	{"spline36", 4, -1, 3.0, false},
	{"spline64", 5, -1, 4.0, false},
	{"sinc", 6, -1, 2.0, false},
	{"lanczos", 6, 6, 3.0, false},
	{"ginseng", 6, 7, 3.0, false},
	{"ewa_jinc", 7, -1, JINC_ZERO3, true},
	{"ewa_lanczos", 7, 7, JINC_ZERO3, true},
	{"ewa_lanczossharp", 7, 7, JINC_ZERO3, true, 0, 0, 0.98125058372237073562493},
	{"ewa_ginseng", 7, 6, JINC_ZERO3, true},
	{"ewa_hann", 7, 8, JINC_ZERO3, true},
	{"ewa_hanning", 7, 8, JINC_ZERO3, true},
	{"ewa_robidoux", 2, -1, 2.0, true, ROB_B, ROB_C},
	{"ewa_robidouxsharp", 2, -1, 2.0, true, ROBS_B, ROBS_C},
};

const FilterDef* find_filter(const std::string& name) {
	for (auto& f : kFilters) if (name == f.name) return &f;
	return nullptr;
}

double bessel_j1(double x) { // J1 by its power series up to 12, by the asymptotic expansion beyond (|error| < 1e-12 for the radii used here)
	if (std::fabs(x) < 12) {
		double term = x / 2, sum = term;
		for (int k = 1; k < 60; k++) {
			term *= -(x * x / 4) / (k * (k + 1.0));
			sum += term;
			if (std::fabs(term) < 1e-17 * std::fabs(sum)) break;
		}
		return sum;
	}
	const double ax = std::fabs(x), t = ax - 3 * M_PI / 4;
	double p = 1, q = 0.375 / ax;
	{ // two more terms of P and Q (Hankel's expansion, mu = 4)
		const double mu = 4, z = 8 * ax;
		p = 1 - (mu - 1) * (mu - 9) / (2 * z * z) + (mu - 1) * (mu - 9) * (mu - 25) * (mu - 49) / (24 * z * z * z * z);
		q = (mu - 1) / z - (mu - 1) * (mu - 9) * (mu - 25) / (6 * z * z * z);
	}
	const double r = std::sqrt(2 / (M_PI * ax)) * (p * std::cos(t) - q * std::sin(t));
	return x < 0 ? -r : r;
}

double fn(int kind, double x, double radius, double b, double c) { // x >= 0
	switch (kind) {
	case 0: return 1.0;
	case 1: return 1.0 - x / radius;
	case 2: {
		const double p0 = 6 - 2 * b, p2 = -18 + 12 * b + 6 * c, p3 = 12 - 9 * b - 6 * c, q0 = 8 * b + 24 * c, q1 = -12 * b - 48 * c, q2 = 6 * b + 30 * c, q3 = -b - 6 * c;
		return x < 1 ? (p0 + x * x * (p2 + x * p3)) / p0 : x < 2 ? (q0 + x * (q1 + x * (q2 + x * q3))) / p0 : 0.0;
	}
	case 3: return x < 1 ? ((x - 9.0 / 5) * x - 1.0 / 5) * x + 1 : ((-1.0 / 3 * (x - 1) + 4.0 / 5) * (x - 1) - 7.0 / 15) * (x - 1);
	case 4:
		if (x < 1) return ((13.0 / 11 * x - 453.0 / 209) * x - 3.0 / 209) * x + 1;
		if (x < 2) return ((-6.0 / 11 * (x - 1) + 270.0 / 209) * (x - 1) - 156.0 / 209) * (x - 1);
		return ((1.0 / 11 * (x - 2) - 45.0 / 209) * (x - 2) + 26.0 / 209) * (x - 2);
	case 5:
		if (x < 1) return ((49.0 / 41 * x - 6387.0 / 2911) * x - 3.0 / 2911) * x + 1;
		if (x < 2) return ((-24.0 / 41 * (x - 1) + 4032.0 / 2911) * (x - 1) - 2328.0 / 2911) * (x - 1);
		if (x < 3) return ((6.0 / 41 * (x - 2) - 1008.0 / 2911) * (x - 2) + 582.0 / 2911) * (x - 2);
		return ((-1.0 / 41 * (x - 3) + 168.0 / 2911) * (x - 3) - 97.0 / 2911) * (x - 3);
	case 6: return x < 1e-8 ? 1.0 : std::sin(M_PI * x) / (M_PI * x);
	case 7: return x < 1e-8 ? 1.0 : 2 * bessel_j1(M_PI * x) / (M_PI * x);
	case 8: return 0.5 + 0.5 * std::cos(M_PI * x);
	}
	return 0.0;
}

// The radius of the window functions' own shape: the window is stretched so that it reaches its end (its first zero) at the filter's radius.
double window_radius(int kind) { return kind == 7 ? JINC_ZERO1 : 1.0; }

double weight(const FilterDef& f, double x) {
	x = std::fabs(x);
	if (x > f.radius) return 0.0;
	const double kx = f.blur > 0 ? x / f.blur : x;
	double k = fn(f.kernel, kx, f.radius, f.b, f.c);
	if (f.window >= 0) k *= fn(f.window, x / f.radius * window_radius(f.window), window_radius(f.window), 0, 0);
	return k;
}

constexpr int LUT_N = 1025;

// ---- Colour ----

// Kr, Kb of a matrix (H.273 MatrixCoefficients); unknown: by size, as libplacebo and most players guess (HD and above BT.709, else BT.601).
void matrix_of(int space, int w, int h, float& kr, float& kb) {
	switch (space) {
	case 1: kr = 0.2126f; kb = 0.0722f; return;
	case 4: kr = 0.30f; kb = 0.11f; return; // FCC
	case 5: case 6: kr = 0.299f; kb = 0.114f; return;
	case 7: kr = 0.212f; kb = 0.087f; return;
	case 9: case 10: kr = 0.2627f; kb = 0.0593f; return; // 10 (constant luminance) is taken as its non-constant twin
	default:
		if (w >= 1280 || h > 576) { kr = 0.2126f; kb = 0.0722f; }
		else { kr = 0.299f; kb = 0.114f; }
	}
}

std::once_flag g_warn_space;

// The curve a transfer tag is decoded/encoded with in the shaders (cv_common.glsl): unknown ones as BT.709 (YUV) or sRGB (RGB); the BT.709 family
// by the bt709 curve setting.
int curve_of(int trc, bool rgb, int bt709_curve) {
	const int b709 = bt709_curve == 1886 ? 1886 : bt709_curve == 13 ? 13 : 1;
	switch (trc) {
	case 1: case 6: case 11: case 12: case 14: case 15: return b709;
	case 4: case 5: case 7: case 8: case 9: case 10: case 13: case 16: case 17: case 18: return trc;
	default: return rgb ? 13 : b709;
	}
}

int plane_w(int w, int log2, bool chroma) { return chroma ? (w + (1 << log2) - 1) >> log2 : w; }

} // namespace

// ---- Settings ----

namespace {
std::atomic<bool> g_ffmpeg_vulkan{false};
}

int bt709_curve_of(const std::string& name) {
	if (name.empty() || name == "oetf" || name == "bt709") return 1;
	if (name == "bt1886" || name == "display") return 1886;
	if (name == "srgb") return 13; // as browsers decode it (Chrome treats BT.709 tags as sRGB)
	return 0;
}
std::string bt709_curve_name(int curve) { return curve == 1886 ? "bt1886" : curve == 13 ? "srgb" : "oetf"; }

int trc_of(const std::string& name) {
	if (name.empty() || name == "source") return 0;
	if (name == "srgb") return 13;
	if (name == "pq") return 16;
	if (name == "hlg") return 18;
	if (name.find_first_not_of("0123456789") == std::string::npos) {
		const int n = atoi(name.c_str());
		return n >= 1 && n <= 18 && n != 2 && n != 3 ? n : -1;
	}
#ifndef VES_NO_VULKAN
	const int t = av_color_transfer_from_name(name.c_str());
	return t >= 1 && t <= 18 && t != 2 && t != 3 ? t : -1;
#else
	return -1;
#endif
}

void set_converter_ffmpeg_vulkan(bool on) {
	const char* e = getenv("VES_SHARED_VULKAN");
	g_ffmpeg_vulkan = on && !(e && atoi(e) == 0);
}
bool converter_shares_ffmpeg_vulkan() { return g_ffmpeg_vulkan; }

// VES_CONVERT_UPSCALE=linear|sigmoid|stored: the light upscaling is done in (downscaling always is in linear light). By default the stored
// values, as most players scale (the NV12 or RGB texture as it is); a change of transfer curve still goes through linear light after it.
int upscale_light_setting() {
	static const int v = [] {
		const char* e = getenv("VES_CONVERT_UPSCALE");
		if (!e || !strcmp(e, "stored") || !strcmp(e, "gamma")) return 2;
		if (!strcmp(e, "linear")) return 0;
		if (!strcmp(e, "sigmoid")) return 1;
		fprintf(stderr, "converter: VES_CONVERT_UPSCALE=%s is not linear, sigmoid or stored; upscaling the stored values\n", e);
		return 2;
	}();
	return v;
}
// In the cache key: a change in how the converter works makes new scores.
std::string scaler_library_version() {
	static const char* light[] = {"linear", "sigmoid", "stored"};
	return std::string("ves converter 2 (downscale linear light, upscale ") + light[upscale_light_setting()] + ")";
}

bool scaler_known(const std::string& name) { return find_filter(name) != nullptr; }

#ifndef VES_NO_VULKAN
namespace {

// The Vulkan device of a name (empty: the home GPU, else the fastest usable one); -1 if there is none.
int device_named(const std::string& name) {
	std::string why;
	if (name.empty()) {
		if (home_vulkan_device() >= 0) return home_vulkan_device();
		for (auto& d : vulkan_devices_fastest_first()) if (d.usable) return d.index;
		return -1;
	}
	for (auto& d : Vk::devices(why)) if (d.name == name) return d.index;
	return -1;
}

constexpr int IN_FLIGHT = 3;


struct Planes { std::vector<uint8_t> mem[3]; }; // the memory of an output picture

// Reads a buffer frame (VkDecodedFrame::buffer, the converter's output) into memory, as picture `p` (whose size, format, times and colour stay),
// for a reader that cannot take it on the GPU. Through a Vk of its own per device (the converter's may be busy on another thread).
bool download_buffer_frame(int device, const VkDecodedFrame& f, const Fmt& fmt, Picture& p) {
	static std::mutex m;
	static std::vector<std::pair<int, std::unique_ptr<Vk>>> vks;
	std::lock_guard<std::mutex> g(m);
	try {
		Vk* vk = nullptr;
		for (auto& [d, v] : vks) if (d == device) vk = v.get();
		if (!vk) { vks.emplace_back(device, std::make_unique<Vk>(device, 1)); vk = vks.back().second.get(); }
		VkDeviceSize size = 0, off[3];
		for (int i = 0; i < 3; i++) { off[i] = size; size += ((VkDeviceSize)f.planes[i].w * f.planes[i].h * f.bytes + 3) / 4 * 4; }
		static std::vector<std::pair<Vk*, Buf>> hosts; // host buffers to read into (a Vk keeps every buffer it made, so reuse what fits)
		Buf* h = nullptr;
		for (auto& [v, b] : hosts) if (v == vk && b.size >= size) h = &b;
		if (!h) { hosts.emplace_back(vk, vk->buffer(size, 2)); h = &hosts.back().second; }
		vk->begin(0);
		vk->wait_timeline(0, f.buf_sem, f.buf_value);
		for (int i = 0; i < 3; i++) {
			const VkDeviceSize row = (VkDeviceSize)f.planes[i].w * f.bytes;
			vk->copy_rows(f.buffer, f.buf_offset[i], row, h->buf, off[i], row, f.planes[i].h);
		}
		vk->barrier();
		vk->submit(0);
		vk->wait(0);
		Picture o;
		o.width = p.width; o.height = p.height; o.fmt = fmt; o.pts = p.pts; o.duration = p.duration; o.color = p.color;
		auto planes = std::make_shared<Planes>();
		for (int i = 0; i < 3; i++) {
			const size_t row = (size_t)f.planes[i].w * f.bytes, n = row * f.planes[i].h;
			planes->mem[i].resize(n);
			memcpy(planes->mem[i].data(), (const uint8_t*)h->map + off[i], n);
			o.data[i] = planes->mem[i].data();
			o.stride[i] = (int)row;
		}
		o.owner = planes;
		o.layout = layout_of(av_planar_format(fmt.chroma, fmt.depth));
		p = std::move(o);
		return true;
	} catch (const std::exception& e) {
		fprintf(stderr, "converter: reading a converted frame back: %s\n", e.what());
		return false;
	}
}

// One step of resampling a plane: positions and kernel stretch along both axes, and how to run it (see cv_resample.comp).
struct Resample {
	int in_w = 0, in_h = 0, out_w = 0, out_h = 0;
	double ax = 1, bx = 0, ay = 1, by = 0;
	const FilterDef* f = nullptr;
	bool identity() const { return in_w == out_w && in_h == out_h && ax == 1 && bx == 0 && ay == 1 && by == 0; }
};

// Where output sample j of a plane sits on the input plane, along one axis: luma sizes `in`, `out`; the planes' subsampling (log2) and whether that
// axis sites chroma on the left sample (cosited, horizontal) or between (centered, vertical).
void axis(int in, int out, int log2_in, int log2_out, bool cosited, double& a, double& b) {
	const double si = 1 << log2_in, so = 1 << log2_out, r = (double)in / out;
	// output sample j's centre in output luma pixels (pixel p's centre at p + 0.5), then in input luma pixels, then in input plane samples
	// cosited: x = so * j + 0.5; centered: x = so * (j + 0.5); input sample i: cosited at si * i + 0.5, centered at si * (i + 0.5)
	const double x0 = cosited ? 0.5 : so * 0.5, dx = so; // x = dx * j + x0
	const double X0 = x0 * r, dX = dx * r;
	if (cosited) { a = dX / si; b = (X0 - 0.5) / si; }
	else { a = dX / si; b = X0 / si - 0.5; }
}

class VkConverter : public Converter {
	std::shared_ptr<Vk> vk; // shared with the buffer frames handed out (buf_out), which may outlive the converter
	// buf_out: converted frames for the GPU scorer go out as buffer frames (VkDecodedFrame::buffer), from a pool of buffers that each comes back when
	// its frame is let go; `tl` is signalled with each frame's number when its work is done
	struct BufPool { std::mutex m; std::vector<Buf> free; VkDeviceSize size = 0; };
	bool buf_out = false;
	std::shared_ptr<BufPool> pool;
	VkSemaphore tl = VK_NULL_HANDLE;
	uint64_t tl_value = 0;
	int device = -1;
	bool gpu_out_wanted = false, gpu_out = false;
	AVBufferRef* out_frames = nullptr;

	int in_w = 0, in_h = 0, out_w = 0, out_h = 0;
	Fmt out_fmt;
	int target_range = 0, target_trc = 0, bt709_curve = 1;
	const FilterDef *up = nullptr, *down = nullptr;

	// The shape of the frames of the current setup (made by setup() for the first frame, again when one comes that differs).
	struct Setup {
		Layout L;
		bool rgb = false, linear = false, scale = false;
		int in_depth = 8;
		int log2_cw = 0, log2_ch = 0; // the source's subsampling
		int out_cw = 0, out_ch = 0;   // the output's (log2)
		int curve_in = 0, curve_out = 0; // 0: none (same curve, no scaling)
		int upscale_light = 0; // an upscale (both sides): 0 in linear light, 1 sigmoidal light, 2 the stored values (VES_CONVERT_UPSCALE)
		float kr_in = 0, kb_in = 0, kr_out = 0, kb_out = 0;
		Color color_out;
		bool valid = false;
	} st;
	Color st_color; Layout st_layout; bool st_gpu = false; bool st_rgb = false; int st_depth = 0;

	// Buffers (device local unless said), shared by the frames in flight: their work runs in order on the queue (a barrier at the start of each).
	Buf raw;          // the source's planes as they are stored
	Buf A[3];         // unpacked: Y/R, Cb/G, Cr/B at their own sizes
	Buf B[2];         // chroma on the luma grid (linear path, subsampled sources)
	Buf T;            // between the passes of a separable scale
	Buf C[3];         // scaled (linear path)
	Buf D[2];         // chroma at the output's subsampling
	Buf out;          // the packed output planes
	Buf luts[3];      // kernel tables: upscaler, downscaler (by their FilterDef), host visible
	const FilterDef* lut_of[3] = {};
	size_t raw_size = 0;
	size_t plane_off[3] = {}; // in `raw`
	size_t out_off[3] = {}, out_stride[3] = {};

	struct Slot {
		Buf up, dl;   // host memory: the planes going in, the output coming out
		size_t up_size = 0;
		bool pending = false;
		int64_t pts = 0, duration = 1;
		Color color;
		std::shared_ptr<void> keep_in; // the source frame, until the GPU is done with it
		AVFrame* frame = nullptr; // gpu_out: the frame it renders into
		bool handed = false; // gpu_out: the frame went on at submit (its users wait for its semaphores), only the input is still held
	} slots[IN_FLIGHT];
	// VES_SCORE_TIMING: the converter's GPU time per frame (stamps 0..1 around its work), summed for the report when it closes
	bool timing = false;
	double gpu_sum = 0;
	long long gpu_frames = 0;
	// VES_SCORE_TIMING=2: GPU time per shader (the first one also holds the copy in), summed over the frames
	std::vector<std::pair<std::string, double>> pass_sum;
	std::vector<std::pair<const char*, double>> pass_now;
	std::deque<int> order;
	int next_slot = 0;
	std::deque<Picture> ready;
	// The pipelines (each with its descriptor set) of the current setup, in the order record_convert() asks for them: made for its first frame,
	// reused for the others (the buffers they bind stay while the setup does).
	std::vector<Pipe> pipes;
	size_t pipe_next = 0;
	int setups = 0; // setup() calls (a new shape)
	Pipe& pipe(const char* name, const std::vector<const Buf*>& b) {
		if (pipe_next < pipes.size()) return pipes[pipe_next++];
		std::vector<Buf*> bb;
		for (const Buf* x : b) bb.push_back(const_cast<Buf*>(x));
		pipes.push_back(vk->pipeline(name, bb));
		return pipes[pipe_next++];
	}

	size_t stride_of(int i) const {
		const bool c = i > 0;
		const int w = c ? plane_w(out_w, st.out_cw, true) : out_w;
		return ((size_t)w * (out_fmt.depth > 8 ? 2 : 1) + 63) & ~(size_t)63;
	}
	int out_plane_w(int i) const { return i ? plane_w(out_w, st.out_cw, true) : out_w; }
	int out_plane_h(int i) const { return i ? plane_w(out_h, st.out_ch, true) : out_h; }

	Buf float_buf(Buf& b, size_t samples) {
		const size_t size = samples * 4 + 16;
		if (!b.buf || b.size < size) b = vk->buffer(size, 1); // the old one is freed with the Vk (rare: only when the frame shape changes)
		return b;
	}
	Buf& lut_for(const FilterDef* f) {
		for (int i = 0; i < 3; i++) if (lut_of[i] == f) return luts[i];
		for (int i = 0; i < 3; i++) if (!lut_of[i]) {
			luts[i] = vk->buffer(LUT_N * 4, 0);
			float* p = (float*)luts[i].map;
			for (int k = 0; k < LUT_N; k++) p[k] = (float)weight(*f, f->radius * k / (LUT_N - 1));
			lut_of[i] = f;
			return luts[i];
		}
		throw std::runtime_error("converter: too many filters");
	}

	static uint32_t groups(int n, int per) { return (uint32_t)((n + per - 1) / per); }
	void dispatch_2d(int w, int h) { vk->dispatch(groups(w, 16), groups(h, 16)); }
	void dispatch_1d(int n) {
		const uint32_t g = groups(n, 256);
		vk->dispatch(std::min<uint32_t>(g, 65535u), (g + 65534u) / 65535u);
	}

	// Records one resample from `src` to `dst` (through T for separable filters).
	// EWA weights once per phase (cv_ewa3, cv_resample_ewa) where the sample positions repeat exactly: steps and offsets in 1/256ths (exact in
	// float), a period of up to 8 outputs, and the weights of every phase fit the shaders' table (1024). px = py = 0 otherwise.
	static void phases(const Resample& r, double sx, double sy, int32_t& px, int32_t& py) {
		px = py = 0;
		static const bool off = [] { const char* e = getenv("VES_CONVERT_PHASES"); return e && atoi(e) == 0; }();
		auto period = [](double a, double b) {
			if (std::fabs(a * 256 - std::round(a * 256)) > 1e-12 || std::fabs(b * 256 - std::round(b * 256)) > 1e-12 || a * 256 >= 4096) return 0;
			for (int q = 1; q <= 8; q++) if (std::fabs(a * q - std::round(a * q)) < 1e-12) return q;
			return 0;
		};
		const int qx = period(r.ax, r.bx), qy = period(r.ay, r.by);
		const int nx = (int)std::floor(2 * (float)sx) + 2, ny = (int)std::floor(2 * (float)sy) + 2;
		if (!off && qx && qy && qx * qy * nx * ny <= 1024) { px = qx; py = qy; }
	}

	void resample(const Buf& src, const Buf& dst, const Resample& r) {
		struct { int32_t in_w, in_h, out_w, out_h, mode, lut_n; float ax, bx, ay, by, fx, fy, radius; int32_t px = 0, py = 0; } pc;
		Buf& lut = lut_for(r.f);
		pc.lut_n = LUT_N;
		pc.radius = (float)r.f->radius;
		pc.ax = (float)r.ax; pc.bx = (float)r.bx; pc.ay = (float)r.ay; pc.by = (float)r.by;
		pc.fx = (float)std::max(1.0, r.ax); pc.fy = (float)std::max(1.0, r.ay);
		if (r.f->polar) {
			// in tiles through shared memory (cv_resample_ewa.comp) where a tile's input fits: its bounds as the shader works them out, plus a margin
			const double sx = r.f->radius * pc.fx, sy = r.f->radius * pc.fy;
			const double tw = std::ceil(r.ax * 15 + 2 * sx) + 3, th = std::ceil(r.ay * 15 + 2 * sy) + 3;
			const bool tiled = tw * th <= 7168 && !getenv("VES_CONVERT_EWA_UNTILED");
			Pipe& p = pipe(tiled ? "cv_resample_ewa" : "cv_resample", {&src, &dst, &lut});
			if (tiled) phases(r, (float)sx, (float)sy, pc.px, pc.py);
			pc.in_w = r.in_w; pc.in_h = r.in_h; pc.out_w = r.out_w; pc.out_h = r.out_h; pc.mode = 2;
			vk->bind(p); vk->push(&pc, sizeof pc); dispatch_2d(r.out_w, r.out_h);
			vk->barrier();
			return;
		}
		if ((size_t)r.out_w * r.in_h * 4 > T.size) throw std::runtime_error("converter: the scaler's temporary plane is too small");
		Pipe& ph = pipe("cv_resample", {&src, &T, &lut});
		Pipe& pv = pipe("cv_resample", {&T, &dst, &lut});
		pc.in_w = r.in_w; pc.in_h = r.in_h; pc.out_w = r.out_w; pc.out_h = r.in_h; pc.mode = 0;
		vk->bind(ph); vk->push(&pc, sizeof pc); dispatch_2d(pc.out_w, pc.out_h);
		vk->barrier();
		pc.in_w = r.out_w; pc.in_h = r.in_h; pc.out_w = r.out_w; pc.out_h = r.out_h; pc.mode = 1;
		vk->bind(pv); vk->push(&pc, sizeof pc); dispatch_2d(pc.out_w, pc.out_h);
		vk->barrier();
	}

	bool shape_changed(const Picture& p, bool gpu) const {
		return !st.valid || p.width != in_w || p.height != in_h || p.fmt.rgb != st_rgb || p.fmt.depth != st_depth || gpu != st_gpu || p.color.space != st_color.space ||
		       p.color.trc != st_color.trc || p.color.range != st_color.range || p.layout != st_layout;
	}

	// The setup for frames like `p` (its layout as it will be read: for frames from the GPU, the planes as copied out of their images).
	bool setup(const Picture& p, const Layout& L, bool gpu) {
		if (++setups == 3) { // a stream's frames all have one shape: a converter that keeps making new setups waits for its work every frame
			static std::once_flag told;
			std::call_once(told, [&] { fprintf(stderr, "converter: the frame shape keeps changing (%dx%d, %s); each change waits for the frames in flight\n", p.width, p.height, gpu ? "from the GPU" : "from memory"); });
		}
		for (auto& pp : pipes) vk->release(pp); // push() waited for the work in flight first; the pool would run out otherwise
		pipes.clear();
		pipe_next = 0;
		st = Setup();
		st.L = L;
		st.rgb = p.fmt.rgb;
		st.in_depth = p.fmt.depth;
		st.log2_cw = st.rgb ? 0 : L.log2_cw;
		st.log2_ch = st.rgb ? 0 : L.log2_ch;
		st.out_cw = out_fmt.chroma == 444 ? 0 : 1;
		st.out_ch = out_fmt.chroma == 420 ? 1 : 0;
		const int cin = curve_of(p.color.trc, st.rgb, bt709_curve);
		const int cout = target_trc ? curve_of(target_trc, false, bt709_curve) : cin;
		st.scale = p.width != out_w || p.height != out_h;
		st.linear = st.scale || cin != cout || st.rgb;
		if (st.linear && (st.scale || cin != cout)) { st.curve_in = cin; st.curve_out = cout; }
		if (st.scale && out_w >= p.width && out_h >= p.height) st.upscale_light = upscale_light_setting();
		matrix_of(st.rgb ? 1 : p.color.space, p.width, p.height, st.kr_in, st.kb_in);
		if (st.rgb) { st.kr_out = 0.2126f; st.kb_out = 0.0722f; }
		else { st.kr_out = st.kr_in; st.kb_out = st.kb_in; }
		if (!st.rgb && (p.color.space == 10 || p.color.space == 8 || p.color.space == 11 || p.color.space >= 12))
			std::call_once(g_warn_space, [&] { fprintf(stderr, "converter: matrix %d is taken as an ordinary one (no constant luminance / YCgCo / ICtCp)\n", p.color.space); });
		Color c = p.color;
		if (st.rgb) c.space = 1;
		if (target_trc) c.trc = target_trc;
		if (target_range) c.range = target_range;
		else if (st.rgb) c.range = 1;
		st.color_out = c;
		in_w = p.width; in_h = p.height;
		st_color = p.color; st_layout = p.layout; st_gpu = gpu; st_rgb = p.fmt.rgb; st_depth = p.fmt.depth;

		// raw planes: rows of the source's stride (frames from memory) or of their width (copied out of images); planes 16-byte aligned
		raw_size = 0;
		for (int pn = 0; pn < L.planes && pn < 3; pn++) {
			plane_off[pn] = raw_size;
			raw_size += ((size_t)plane_stride(p, L, pn, gpu) * plane_rows(p, L, pn) + 15) & ~(size_t)15;
		}
		if (!raw.buf || raw.size < raw_size + 16) raw = vk->buffer(raw_size + 16, 1);
		for (int c3 = 0; c3 < 3; c3++) float_buf(A[c3], (size_t)plane_w(in_w, st.log2_cw, c3 > 0) * plane_w(in_h, st.log2_ch, c3 > 0));
		if (st.linear) {
			if (st.log2_cw || st.log2_ch) for (auto& b : B) float_buf(b, (size_t)in_w * in_h);
			if (st.scale) for (auto& b : C) float_buf(b, (size_t)out_w * out_h);
		}
		if (st.linear ? (st.out_cw || st.out_ch) : (st.log2_cw != st.out_cw || st.log2_ch != st.out_ch))
			for (auto& b : D) float_buf(b, (size_t)out_plane_w(1) * out_plane_h(1));
		float_buf(T, (size_t)std::max(in_w, out_w) * std::max(in_h, out_h)); // big enough for any pass of a separable scale
		size_t o = 0;
		for (int i = 0; i < 3; i++) { out_off[i] = o; out_stride[i] = stride_of(i); o += out_stride[i] * out_plane_h(i); }
		if (!out.buf || out.size < o) out = vk->buffer(o, 1);
		for (auto& s : slots) {
			if (!s.dl.buf || s.dl.size < o) s.dl = vk->buffer(o, 2);
		}
		st.valid = true;
		return true;
	}

	static int plane_rows(const Picture& p, const Layout& L, int pn) {
		for (int c = 1; c < 3; c++) if (L.comp[c].plane == pn && L.comp[0].plane != pn && !p.fmt.rgb) return plane_w(p.height, L.log2_ch, true);
		return p.height;
	}
	static int plane_samples_w(const Picture& p, const Layout& L, int pn) {
		for (int c = 1; c < 3; c++) if (L.comp[c].plane == pn && L.comp[0].plane != pn && !p.fmt.rgb) return plane_w(p.width, L.log2_cw, true);
		return p.width;
	}
	static int plane_stride(const Picture& p, const Layout& L, int pn, bool gpu) {
		return gpu ? plane_samples_w(p, L, pn) * L.step[pn] : p.stride[pn];
	}

	// Records the work for one frame whose raw planes are (or will be, by commands recorded before) in `raw`.
	void record_convert() {
		const Setup& s = st;
		pipe_next = 0;
		struct { int32_t w, h, base, stride, step, offset, bytes, shift, depth, big_endian; float mul, add; } up_pc;
		// unpack: nominal values from code values; full range: Y = v / (2^d - 1), C = (v - 2^(d-1)) / (2^d - 1); limited: Y = (v - 16 << (d-8)) / (219 << (d-8)), C = (v - 128 << (d-8)) / (224 << (d-8))
		const bool full = st_color.range == 2 || (st_color.range == 0 && s.rgb);
		for (int c = 0; c < 3; c++) {
			const Layout::Comp& k = s.L.comp[c];
			const int depth = k.depth;
			const double sc = std::ldexp(1.0, depth - 8);
			double mul, add;
			const bool chroma = !s.rgb && c > 0;
			if (full) { mul = 1.0 / (std::ldexp(1.0, depth) - 1); add = chroma ? -std::ldexp(1.0, depth - 1) * mul : 0.0; }
			else if (chroma) { mul = 1.0 / (224 * sc); add = -128 * sc * mul; }
			else if (s.rgb) { mul = 1.0 / (219 * sc); add = -16 * sc * mul; } // limited-range RGB
			else { mul = 1.0 / (219 * sc); add = -16 * sc * mul; }
			up_pc.w = plane_w(in_w, s.log2_cw, c > 0); up_pc.h = plane_w(in_h, s.log2_ch, c > 0);
			up_pc.base = (int)plane_off[k.plane];
			up_pc.stride = cur_stride[k.plane];
			up_pc.step = s.L.step[k.plane];
			up_pc.offset = k.offset;
			up_pc.bytes = k.depth + k.shift > 8 ? 2 : 1;
			up_pc.shift = k.shift;
			up_pc.depth = k.depth;
			up_pc.big_endian = s.L.big_endian;
			up_pc.mul = (float)mul; up_pc.add = (float)add;
			Pipe& p = pipe("cv_unpack", {&raw, &A[c]});
			vk->bind(p); vk->push(&up_pc, sizeof up_pc); dispatch_2d(up_pc.w, up_pc.h);
		}
		vk->barrier();
		const Buf* Y = &A[0];
		const Buf* U = &A[1];
		const Buf* V = &A[2];
		const int ocw = out_plane_w(1), och = out_plane_h(1);
		if (s.linear) {
			if (s.log2_cw || s.log2_ch) {
				Resample r;
				r.in_w = plane_w(in_w, s.log2_cw, true); r.in_h = plane_w(in_h, s.log2_ch, true); r.out_w = in_w; r.out_h = in_h;
				axis(in_w, in_w, s.log2_cw, 0, true, r.ax, r.bx);
				axis(in_h, in_h, s.log2_ch, 0, false, r.ay, r.by);
				r.f = up;
				resample(A[1], B[0], r);
				resample(A[2], B[1], r);
				U = &B[0]; V = &B[1];
			}
			const bool sig = s.upscale_light == 1, stored = s.upscale_light == 2;
			const bool change = s.curve_in != s.curve_out;
			const int trc_in = stored ? 0 : s.curve_in, trc_out = stored && !change ? 0 : s.curve_out, pre = stored && change ? s.curve_in : 0;
			if (s.scale) { // a polar kernel: to linear light, scale and back in one pass (cv_ewa3.comp) where its tile fits
				Resample r;
				r.in_w = in_w; r.in_h = in_h; r.out_w = out_w; r.out_h = out_h;
				axis(in_w, out_w, 0, 0, true, r.ax, r.bx);
				axis(in_h, out_h, 0, 0, false, r.ay, r.by);
				r.f = (out_w < in_w || out_h < in_h) ? down : up;
				const double fx = std::max(1.0, r.ax), fy = std::max(1.0, r.ay), sx = r.f->radius * fx, sy = r.f->radius * fy;
				const double tw = std::ceil(r.ax * 15 + 2 * sx) + 3, th = std::ceil(r.ay * 7 + 2 * sy) + 3;
				static const bool off = [] { const char* e = getenv("VES_CONVERT_FUSED"); return e && atoi(e) == 0; }();
				if (r.f->polar && tw * th <= 3072 && !off) {
					struct {
						int32_t in_w, in_h, out_w, out_h, lut_n;
						float ax, bx, ay, by, fx, fy, radius;
						int32_t rgb, trc_in, sig, trc_out, pre;
						float kr_in, kb_in, kr_out, kb_out;
						int32_t px, py;
					} pc{in_w, in_h, out_w, out_h, LUT_N, (float)r.ax, (float)r.bx, (float)r.ay, (float)r.by, (float)fx, (float)fy, (float)r.f->radius,
					     s.rgb ? 1 : 0, trc_in, sig ? 1 : 0, trc_out, pre, s.kr_in, s.kb_in, s.kr_out, s.kb_out, 0, 0};
					// The weights once per phase where the sample positions repeat exactly: steps and offsets in 1/256ths (exact in float), a
					// period of up to 8 outputs, and the weights of every phase fit the shader's table.
					phases(r, sx, sy, pc.px, pc.py);
					Buf& lut = lut_for(r.f);
					Pipe& p = pipe("cv_ewa3", {Y, U, V, &C[0], &C[1], &C[2], &lut});
					vk->bind(p); vk->push(&pc, sizeof pc); vk->dispatch(groups(out_w, 16), groups(out_h, 8));
					vk->barrier();
					Y = &C[0]; U = &C[1]; V = &C[2];
					goto chroma_down;
				}
			}
			{
			struct { int32_t n, rgb, trc, sig; float kr, kb; } lp{in_w * in_h, s.rgb ? 1 : 0, trc_in, sig ? 1 : 0, s.kr_in, s.kb_in};
			Pipe& pl = pipe("cv_to_linear", {Y, U, V});
			vk->bind(pl); vk->push(&lp, sizeof lp); dispatch_1d(lp.n);
			vk->barrier();
			if (s.scale) {
				Resample r;
				r.in_w = in_w; r.in_h = in_h; r.out_w = out_w; r.out_h = out_h;
				axis(in_w, out_w, 0, 0, true, r.ax, r.bx);
				axis(in_h, out_h, 0, 0, false, r.ay, r.by);
				r.f = (out_w < in_w || out_h < in_h) ? down : up;
				resample(*Y, C[0], r);
				resample(*U, C[1], r);
				resample(*V, C[2], r);
				Y = &C[0]; U = &C[1]; V = &C[2];
			}
			struct { int32_t n, trc, sig, pre; float kr, kb; } fp{out_w * out_h, trc_out, sig ? 1 : 0, pre, s.kr_out, s.kb_out};
			Pipe& pf = pipe("cv_from_linear", {Y, U, V});
			vk->bind(pf); vk->push(&fp, sizeof fp); dispatch_1d(fp.n);
			vk->barrier();
			}
		chroma_down:
			if (s.out_cw || s.out_ch) {
				Resample r;
				r.in_w = out_w; r.in_h = out_h; r.out_w = ocw; r.out_h = och;
				axis(out_w, out_w, 0, s.out_cw, true, r.ax, r.bx);
				axis(out_h, out_h, 0, s.out_ch, false, r.ay, r.by);
				r.f = down;
				resample(*U, D[0], r);
				resample(*V, D[1], r);
				U = &D[0]; V = &D[1];
			}
		} else if (s.log2_cw != s.out_cw || s.log2_ch != s.out_ch) {
			Resample r;
			r.in_w = plane_w(in_w, s.log2_cw, true); r.in_h = plane_w(in_h, s.log2_ch, true); r.out_w = ocw; r.out_h = och;
			axis(in_w, out_w, s.log2_cw, s.out_cw, true, r.ax, r.bx);
			axis(in_h, out_h, s.log2_ch, s.out_ch, false, r.ay, r.by);
			r.f = (s.out_cw > s.log2_cw || s.out_ch > s.log2_ch) ? down : up;
			resample(A[1], D[0], r);
			resample(A[2], D[1], r);
			U = &D[0]; V = &D[1];
		}
		// pack
		const bool ofull = st.color_out.range == 2;
		const int d = out_fmt.depth;
		const double sc = std::ldexp(1.0, d - 8);
		const Buf* src[3] = {Y, U, V};
		for (int i = 0; i < 3; i++) {
			struct { int32_t w, h, base_words, stride_words, bytes; float mul, add, maxv; } pp;
			pp.w = out_plane_w(i); pp.h = out_plane_h(i);
			pp.bytes = d > 8 ? 2 : 1;
			pp.base_words = (int)(out_off[i] / 4);
			pp.stride_words = (int)(out_stride[i] / 4);
			const double maxv = std::ldexp(1.0, d) - 1;
			if (ofull) { pp.mul = (float)maxv; pp.add = i ? (float)std::ldexp(1.0, d - 1) : 0.0f; }
			else if (i) { pp.mul = (float)(224 * sc); pp.add = (float)(128 * sc); }
			else { pp.mul = (float)(219 * sc); pp.add = (float)(16 * sc); }
			pp.maxv = (float)maxv;
			Pipe& p = pipe("cv_pack", {src[i], &out});
			vk->bind(p); vk->push(&pp, sizeof pp); vk->dispatch(groups(groups(pp.w, 4 / pp.bytes), 16), groups(pp.h, 16));
		}
		vk->barrier();
	}
	int cur_stride[3] = {};

	// Waits for the oldest frame in flight (if `wait`; otherwise only takes it if it is done) and hands it out. 1: taken, 0: not done yet, -1: failed.
	int take_oldest(bool wait) {
		const int k = order.front();
		Slot& s = slots[k];
		try {
			if (wait) vk->wait(k);
			else if (!vk->done(k)) return 0;
		} catch (const std::exception& e) { fprintf(stderr, "converter: %s\n", e.what()); return -1; }
		if (timing) if (const double t = vk->gpu_seconds(k); t >= 0) {
			gpu_sum += t; gpu_frames++;
			if (vk->pass_timing()) {
				vk->pass_seconds(k, pass_now);
				for (auto& [n, sec] : pass_now) {
					auto it = std::find_if(pass_sum.begin(), pass_sum.end(), [&](auto& e) { return e.first == n; });
					if (it == pass_sum.end()) pass_sum.emplace_back(n, sec); else it->second += sec;
				}
			}
		}
		if (s.handed) { // the picture went on at submit
			s.keep_in.reset();
			s.handed = false;
			s.pending = false;
			order.pop_front();
			return 1;
		}
		Picture o;
		o.width = out_w; o.height = out_h;
		o.fmt = out_fmt;
		o.pts = s.pts;
		o.duration = s.duration;
		o.color = s.color;
		if (s.frame) {
			s.frame->pts = s.pts;
			s.frame->duration = s.duration;
			const bool ok = vulkan_picture(s.frame, device, false, o);
			av_frame_free(&s.frame);
			if (!ok) { fprintf(stderr, "converter: the converted Vulkan frame cannot be handed on\n"); return -1; }
			o.width = out_w; o.height = out_h; o.fmt = out_fmt; o.pts = s.pts; o.duration = s.duration; o.color = s.color;
		} else {
			auto planes = std::make_shared<Planes>();
			for (int i = 0; i < 3; i++) {
				const size_t n = out_stride[i] * out_plane_h(i);
				planes->mem[i].resize(n);
				memcpy(planes->mem[i].data(), (const uint8_t*)s.dl.map + out_off[i], n);
				o.data[i] = planes->mem[i].data();
				o.stride[i] = (int)out_stride[i];
			}
			o.owner = planes;
			o.layout = layout_of(av_planar_format(out_fmt.chroma, out_fmt.depth));
		}
		s.keep_in.reset();
		s.pending = false;
		ready.push_back(std::move(o));
		order.pop_front();
		return 1;
	}

public:
	// Takes the frames in flight that are done, oldest first, without waiting.
	bool take_done() {
		while (!order.empty()) {
			const int t = take_oldest(false);
			if (t < 0) return false;
			if (t == 0) break;
		}
		return true;
	}

	~VkConverter() override {
		if (vk) for (int i = 0; i < IN_FLIGHT; i++) if (slots[i].pending) { try { vk->wait(i); } catch (...) {} }
		if (timing && gpu_frames) {
			fprintf(stderr, "converter timing: %lld frames, GPU %.3f ms per frame (%dx%d -> %dx%d)\n", gpu_frames, gpu_sum * 1e3 / (double)gpu_frames, in_w, in_h, out_w, out_h);
			double in_passes = 0;
			for (auto& [n, sec] : pass_sum) { fprintf(stderr, "converter timing:   %s %.3f ms\n", n.c_str(), sec * 1e3 / (double)gpu_frames); in_passes += sec; }
			if (!pass_sum.empty()) fprintf(stderr, "converter timing:   after the last shader (copies out) %.3f ms\n", (gpu_sum - in_passes) * 1e3 / (double)gpu_frames);
		}
		for (auto& s : slots) av_frame_free(&s.frame);
		av_buffer_unref(&out_frames);
	}

	bool open(const Picture& first, Rational, int w, int h, Fmt outf, const ScaleOptions& opt, int range) override {
		target_range = range;
		target_trc = opt.trc;
		bt709_curve = opt.bt709_curve;
		out_w = w; out_h = h; out_fmt = outf;
		up = find_filter(opt.upscaler);
		down = find_filter(opt.downscaler);
		if (!up || !down) { fprintf(stderr, "converter: unknown scaler '%s' / '%s'\n", opt.upscaler.c_str(), opt.downscaler.c_str()); return false; }
		if (!outf.planar || outf.rgb || (outf.depth != 8 && outf.depth != 10 && outf.depth != 12 && outf.depth != 16)) { fprintf(stderr, "converter: cannot make %d-bit %s\n", outf.depth, outf.rgb ? "RGB" : "YUV"); return false; }
		device = device_named(opt.device);
		if (device < 0) { fprintf(stderr, "converter: no Vulkan GPU%s\n", opt.device.empty() ? "" : (" named '" + opt.device + "'").c_str()); return false; }
		try {
			vk = std::make_shared<Vk>(device, IN_FLIGHT);
		} catch (const std::exception& e) {
			fprintf(stderr, "converter: %s\n", e.what());
			return false;
		}
		if (const char* t = getenv("VES_SCORE_TIMING"); t && atoi(t) >= 1) timing = true;
		gpu_out = false;
		buf_out = false;
		const char* e = getenv("VES_CONVERT_TO_GPU");
		const char* eb = getenv("VES_CONVERT_TO_BUFFER");
		if (gpu_out_wanted && vk->external() && !(e && atoi(e) == 0) && !(eb && atoi(eb) == 0)) {
			try { tl = vk->timeline_semaphore(); } catch (const std::exception& x) { fprintf(stderr, "converter: %s\n", x.what()); return false; }
			pool = std::make_shared<BufPool>();
			buf_out = true;
		} else if (gpu_out_wanted && vk->external() && !(e && atoi(e) == 0)) {
			const AVPixelFormat sw = av_planar_format(outf.chroma, outf.depth);
			AVBufferRef* dev = nullptr;
			for (auto& d : vulkan_devices_fastest_first()) if (d.index == device) dev = ffmpeg_vulkan_device(d.index, d.vendor_id, d.device_id);
			AVBufferRef* fr = dev && sw != AV_PIX_FMT_NONE ? av_hwframe_ctx_alloc(dev) : nullptr;
			av_buffer_unref(&dev);
			if (fr) {
				AVHWFramesContext* fc = (AVHWFramesContext*)fr->data;
				fc->format = AV_PIX_FMT_VULKAN;
				fc->sw_format = sw;
				fc->width = w; fc->height = h;
				fc->initial_pool_size = 0;
				AVVulkanFramesContext* vfc = (AVVulkanFramesContext*)fc->hwctx;
				vfc->flags = (AVVkFrameFlags)(vfc->flags | AV_VK_FRAME_FLAG_DISABLE_MULTIPLANE); // one image per plane (R8/R16)
				vfc->usage = (VkImageUsageFlagBits)(VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT);
				if (av_hwframe_ctx_init(fr) < 0) av_buffer_unref(&fr);
			}
			if (fr) { out_frames = fr; gpu_out = true; }
			else fprintf(stderr, "converter: cannot make Vulkan frames on FFmpeg's device: the converted frames go through memory\n");
		}
		(void)first;
		return true;
	}

	void set_output_on_gpu(bool on) override { gpu_out_wanted = on; }

	bool push(const Picture* p) override {
		if (!vk) return false;
		if (!p) {
			while (!order.empty()) if (take_oldest(true) < 0) return false;
			return true;
		}
		const VkDecodedFrame* gf = nullptr;
		VkDecodedFrame here;
		if (p->gpu) {
			const AVFrame* af = gpu_avframe(*p);
			if (!af || !vulkan_frame(af, device, here)) { // on another device, or not readable in place: through memory
				Picture m = *p;
				if (!picture_to_memory(m)) return false;
				return push(&m);
			}
			gf = &here;
		}
		if (slots[next_slot].pending && take_oldest(true) < 0) return false;
		const int k = next_slot;
		Slot& s = slots[k];

		// the layout as it will be read
		Layout L = p->layout;
		if (gf) {
			const AVHWFramesContext* fc = (const AVHWFramesContext*)gpu_avframe(*p)->hw_frames_ctx->data;
			L = layout_of(fc->sw_format);
		}
		if (L.planes == 0 && p->fmt.planar) L = layout_of(av_planar_format(p->fmt.chroma, p->fmt.depth));
		if (L.planes == 0 || L.planes > 3) { fprintf(stderr, "converter: unsupported sample layout\n"); return false; }
		for (int c = 0; c < 3; c++) if (L.comp[c].plane < 0 || L.comp[c].depth < 8) { fprintf(stderr, "converter: unsupported sample layout\n"); return false; }
		Picture shape = *p;
		shape.layout = L;
		// A frame's lock is held from reading its semaphore value until the work that waits for it is submitted: the scorer and other converters
		// read the same frames, and work submitted to the one queue ahead of the work that signals what it waits for would wait forever.
		struct FrameLock {
			const VkDecodedFrame* f = nullptr;
			void take(const VkDecodedFrame* x) { f = x; if (f->lock) f->lock(f->ctx); }
			void release() { if (f && f->unlock) f->unlock(f->ctx); f = nullptr; }
			~FrameLock() { release(); }
		};
		std::shared_ptr<VkDecodedFrame> bf; // buf_out: the frame handed on
		VkDecodedFrame o; // gpu_out: the frame the converted picture goes into (declared before its lock, which outlives the try block)
		FrameLock in_lock, out_lock;
		bool recording = false; // begun and not yet submitted
		try {
			if (shape_changed(shape, gf != nullptr)) {
				while (!order.empty()) if (take_oldest(true) < 0) return false; // the shared buffers change
				if (!setup(shape, L, gf != nullptr)) return false;
			}
			for (int pn = 0; pn < L.planes; pn++) cur_stride[pn] = plane_stride(shape, L, pn, gf != nullptr);
			vk->begin(k);
			recording = true;
			if (timing) vk->stamp(k, 0);
			vk->barrier(); // after everything before on the queue: the shared buffers are free
			if (gf) {
				in_lock.take(gf);
				for (int i = 0; i < gf->nimages; i++) {
					vk->wait_timeline(k, gf->sem[i], *gf->sem_value[i]);
					vk->image_to_transfer(gf->images[i], *gf->layout[i], *gf->access[i]);
					*gf->sem_value[i] += 1;
					vk->signal_timeline(k, gf->sem[i], *gf->sem_value[i]);
				}
				for (int pn = 0; pn < gf->nplanes; pn++) {
					VkImagePlane pl = gf->planes[pn];
					pl.w = (uint32_t)plane_samples_w(shape, L, pn); // cut to the picture's size (the coded size can be larger)
					pl.h = (uint32_t)plane_rows(shape, L, pn);
					vk->copy_image(pl, raw, plane_off[pn]);
				}
				s.keep_in = gf->keep;
			} else {
				if (!s.up.buf || s.up.size < raw_size) s.up = vk->buffer(raw_size + 16, 3);
				for (int pn = 0; pn < L.planes; pn++) {
					const size_t n = (size_t)cur_stride[pn] * (plane_rows(shape, L, pn) - 1) + (size_t)plane_samples_w(shape, L, pn) * L.step[pn];
					memcpy((uint8_t*)s.up.map + plane_off[pn], p->data[pn], n);
				}
				vk->copy(s.up, raw, raw_size);
			}
			vk->barrier();
			record_convert();
			s.frame = nullptr;
			if (buf_out) {
				const int bytes = out_fmt.depth > 8 ? 2 : 1;
				VkDeviceSize off[3], size = 0;
				for (int i = 0; i < 3; i++) { off[i] = size; size += ((VkDeviceSize)out_plane_w(i) * out_plane_h(i) * bytes + 255) / 256 * 256; }
				Buf b;
				{
					std::lock_guard<std::mutex> g(pool->m);
					if (pool->size != size) { pool->free.clear(); pool->size = size; } // a new shape: the old ones stay with the Vk until it goes
					if (!pool->free.empty()) { b = pool->free.back(); pool->free.pop_back(); }
				}
				if (!b.buf) b = vk->buffer(size, 1);
				for (int i = 0; i < 3; i++)
					vk->copy_rows(out.buf, out_off[i], out_stride[i], b.buf, off[i], (VkDeviceSize)out_plane_w(i) * bytes, (uint32_t)out_plane_h(i));
				vk->signal_timeline(k, tl, ++tl_value);
				bf = std::make_shared<VkDecodedFrame>();
				bf->nplanes = 3;
				bf->bytes = bytes;
				for (int i = 0; i < 3; i++) { bf->planes[i].w = (uint32_t)out_plane_w(i); bf->planes[i].h = (uint32_t)out_plane_h(i); bf->buf_offset[i] = off[i]; }
				bf->buffer = b.buf;
				bf->buf_sem = tl;
				bf->buf_value = tl_value;
				bf->device = device;
				struct Hold {
					std::shared_ptr<Vk> vk; std::shared_ptr<BufPool> pool; Buf b; VkDeviceSize size;
					~Hold() { std::lock_guard<std::mutex> g(pool->m); if (pool->size == size) pool->free.push_back(b); }
				};
				std::shared_ptr<Hold> hold(new Hold{vk, pool, b, size}); // (not make_shared(Hold{...}): the temporary's destructor would give the buffer back)
				bf->keep = hold;
				const int dev = device;
				const Fmt f = out_fmt;
				std::weak_ptr<VkDecodedFrame> self = bf;
				bf->download = [dev, f, self](void* pic) {
					auto me = self.lock();
					return me && download_buffer_frame(dev, *me, f, *(Picture*)pic);
				};
				static std::once_flag told;
				std::call_once(told, [&] { fprintf(stderr, "converter: converted frames go to the GPU scorer in buffers (%s), without copies into images\n", av_get_pix_fmt_name(av_planar_format(out_fmt.chroma, out_fmt.depth))); });
			} else if (gpu_out) {
				AVFrame* of = av_frame_alloc();
				if (!of || av_hwframe_get_buffer(out_frames, of, 0) < 0 || !vulkan_frame(of, device, o) || o.nimages != 3) {
					av_frame_free(&of);
					fprintf(stderr, "converter: cannot get a Vulkan frame for the converted picture\n");
					vk->submit(k);
					vk->wait(k);
					return false;
				}
				out_lock.take(&o);
				for (int i = 0; i < 3; i++) {
					vk->wait_timeline(k, o.sem[i], *o.sem_value[i]);
					vk->image_to_transfer_dst(o.images[i], *o.layout[i], *o.access[i]);
					VkImagePlane pl = o.planes[i];
					vk->copy_to_image(out, out_off[i], (uint32_t)(out_stride[i] / (out_fmt.depth > 8 ? 2 : 1)), pl);
					*o.sem_value[i] += 1;
					vk->signal_timeline(k, o.sem[i], *o.sem_value[i]);
				}
				s.frame = of;
				static std::once_flag told;
				std::call_once(told, [&] { fprintf(stderr, "converter: converted frames stay on the GPU (%s), without reading them back\n", av_get_pix_fmt_name(av_planar_format(out_fmt.chroma, out_fmt.depth))); });
			} else {
				vk->copy(out, s.dl, out.size);
				vk->barrier();
			}
			if (timing) vk->stamp(k, 1);
			recording = false;
			vk->submit(k);
			out_lock.release();
			in_lock.release();
		} catch (const std::exception& e) {
			fprintf(stderr, "converter: %s\n", e.what()); // a lost device has been reported (vk_compute.cpp's chk())
			// The semaphore values of the frames were raised already: their other users (FFmpeg's decoder, freeing or reusing the frame, and the
			// scorer) wait for them, forever if the work that signals them never goes in. So it goes in as far as it was recorded.
			if (recording) {
				try { vk->submit(k); vk->wait(k); } catch (const std::exception& e2) { fprintf(stderr, "converter: %s\n", e2.what()); }
			}
			return false;
		}
		static std::once_flag told;
		std::call_once(told, [&] { fprintf(stderr, "converter: %s path, %s, %dx%d -> %dx%d\n", st.linear ? (st.curve_in ? "linear light" : "RGB matrix") : "direct", gf ? "from the GPU" : "from memory", in_w, in_h, out_w, out_h); });
		s.pending = true;
		s.pts = p->pts;
		s.duration = p->duration;
		s.color = st.color_out;
		order.push_back(k);
		next_slot = (next_slot + 1) % IN_FLIGHT;
		if (bf) { // in a buffer: hand it on now; whoever reads it waits for `tl`
			Picture o;
			o.width = out_w; o.height = out_h; o.fmt = out_fmt; o.pts = s.pts; o.duration = s.duration; o.color = s.color;
			o.layout = layout_of(av_planar_format(out_fmt.chroma, out_fmt.depth));
			o.gpu = bf;
			s.handed = true;
			ready.push_back(std::move(o));
		} else if (s.frame) { // on the GPU: hand it on now; whoever reads it waits for its semaphores, which the submitted work signals
			Picture o;
			s.frame->pts = s.pts;
			s.frame->duration = s.duration;
			const bool ok = vulkan_picture(s.frame, device, false, o);
			av_frame_free(&s.frame);
			if (!ok) { fprintf(stderr, "converter: the converted Vulkan frame cannot be handed on\n"); return false; }
			o.width = out_w; o.height = out_h; o.fmt = out_fmt; o.pts = s.pts; o.duration = s.duration; o.color = s.color;
			s.handed = true;
			ready.push_back(std::move(o));
		}
		return take_done();
	}

	int pop(Picture& o) override {
		if (ready.empty() && vk && !take_done()) return -1;
		if (ready.empty()) return 0;
		o = std::move(ready.front());
		ready.pop_front();
		return 1;
	}
};

} // namespace

std::unique_ptr<Converter> make_converter() { return std::make_unique<VkConverter>(); }
#else
namespace {
class NoConverter : public Converter {
public:
	bool open(const Picture&, Rational, int, int, Fmt, const ScaleOptions&, int) override { fprintf(stderr, "converter: built without Vulkan\n"); return false; }
	bool push(const Picture*) override { return false; }
	int pop(Picture&) override { return -1; }
};
}
std::unique_ptr<Converter> make_converter() { return std::make_unique<NoConverter>(); }
#endif

bool converter_works(const ScaleOptions& opt, std::string& why) {
	// 64x64 8-bit 4:4:4: luma rises from left to right, Cb from top to bottom, Cr is flat; it goes to 32x32 4:2:0, which takes the same path as a real frame (scale, subsample, read back).
	const int w = 64, h = 64;
	auto mem = std::make_shared<std::vector<uint8_t>>((size_t)w * h * 3);
	uint8_t* y = mem->data();
	uint8_t* u = y + w * h;
	uint8_t* v = u + w * h;
	for (int r = 0; r < h; r++)
		for (int c = 0; c < w; c++) { y[r * w + c] = (uint8_t)(32 + c * 3); u[r * w + c] = (uint8_t)(64 + r * 2); v[r * w + c] = 160; }
	Picture in;
	in.width = w; in.height = h;
	in.fmt = Fmt{444, 8, false, true};
	in.layout.planes = 3;
	for (int i = 0; i < 3; i++) { in.layout.step[i] = 1; in.layout.comp[i].plane = i; in.layout.comp[i].depth = 8; in.data[i] = y + (size_t)i * w * h; in.stride[i] = w; }
	in.color.range = 1; in.color.space = 1; in.color.primaries = 1; in.color.trc = 1;
	in.owner = mem;
	auto conv = make_converter();
	const Fmt out_fmt{420, 8, false, true};
	ScaleOptions o = opt;
	o.trc = 0;
	if (!conv->open(in, Rational{1, 60}, 32, 32, out_fmt, o, 1)) { why = "it cannot convert at all"; return false; }
	Picture out;
	bool got = false;
	for (int i = 0; i < 3 && !got; i++) {
		in.pts = i;
		if (!conv->push(&in)) { why = "converting failed"; return false; }
		got = conv->pop(out) == 1;
	}
	if (!got) { conv->push(nullptr); got = conv->pop(out) == 1; }
	if (!got) { why = "no converted picture came back"; return false; }
	auto mean = [&](int plane, int pw, int ph) {
		double sum = 0;
		for (int r = 0; r < ph; r++) for (int c = 0; c < pw; c++) sum += out.data[plane][(size_t)r * out.stride[plane] + c];
		return sum / (pw * ph);
	};
	const double my = mean(0, 32, 32), mu = mean(1, 16, 16), mv = mean(2, 16, 16);
	const double ey = 32 + 31.5 * 3, eu = 64 + 31.5 * 2, ev = 160; // the means of the picture that went in
	char b[200];
	snprintf(b, sizeof b, "a test picture came back with means Y %.0f, Cb %.0f, Cr %.0f instead of about %.0f, %.0f, %.0f", my, mu, mv, ey, eu, ev);
	if (std::fabs(my - ey) > 8 || std::fabs(mu - eu) > 8 || std::fabs(mv - ev) > 8) { why = b; return false; }
	const uint8_t* row = out.data[0];
	if (row[31] <= row[0] + 40) { snprintf(b, sizeof b, "a test picture came back without its left-to-right ramp (Y %d .. %d)", row[0], row[31]); why = b; return false; }
	return true;
}
