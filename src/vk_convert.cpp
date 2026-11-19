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
	switch (trc) {
	case 1: case 6: case 11: case 12: case 14: case 15: return bt709_curve == 1886 ? 1886 : 1;
	case 4: case 5: case 7: case 8: case 9: case 10: case 13: case 16: case 17: case 18: return trc;
	default: return rgb ? 13 : (bt709_curve == 1886 ? 1886 : 1);
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
	return 0;
}
std::string bt709_curve_name(int curve) { return curve == 1886 ? "bt1886" : "oetf"; }

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

std::string scaler_library_version() { return "ves converter 1 (linear light)"; }

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
	std::unique_ptr<Vk> vk;
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
	} slots[IN_FLIGHT];
	std::deque<int> order;
	int next_slot = 0;
	std::deque<Picture> ready;
	// The pipelines (each with its descriptor set) of the current setup, in the order record_convert() asks for them: made for its first frame,
	// reused for the others (the buffers they bind stay while the setup does).
	std::vector<Pipe> pipes;
	size_t pipe_next = 0;
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
	void resample(const Buf& src, const Buf& dst, const Resample& r) {
		struct { int32_t in_w, in_h, out_w, out_h, mode, lut_n; float ax, bx, ay, by, fx, fy, radius; } pc;
		Buf& lut = lut_for(r.f);
		pc.lut_n = LUT_N;
		pc.radius = (float)r.f->radius;
		pc.ax = (float)r.ax; pc.bx = (float)r.bx; pc.ay = (float)r.ay; pc.by = (float)r.by;
		pc.fx = (float)std::max(1.0, r.ax); pc.fy = (float)std::max(1.0, r.ay);
		if (r.f->polar) {
			Pipe& p = pipe("cv_resample", {&src, &dst, &lut});
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
		       p.color.trc != st_color.trc || p.color.range != st_color.range || memcmp(&p.layout, &st_layout, sizeof st_layout) != 0;
	}

	// The setup for frames like `p` (its layout as it will be read: for frames from the GPU, the planes as copied out of their images).
	bool setup(const Picture& p, const Layout& L, bool gpu) {
		pipes.clear(); // their descriptor sets stay allocated until the converter ends; a new shape is rare
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
			struct { int32_t n, rgb, trc; float kr, kb; } lp{in_w * in_h, s.rgb ? 1 : 0, s.curve_in, s.kr_in, s.kb_in};
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
			struct { int32_t n, trc; float kr, kb; } fp{out_w * out_h, s.curve_out, s.kr_out, s.kb_out};
			Pipe& pf = pipe("cv_from_linear", {Y, U, V});
			vk->bind(pf); vk->push(&fp, sizeof fp); dispatch_1d(fp.n);
			vk->barrier();
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
		Slot& s = slots[order.front()];
		if (!wait) return 0; // (no poll in Vk) the frames come out when their slots are needed again, or at the end
		try { vk->wait(order.front()); } catch (const std::exception& e) { fprintf(stderr, "converter: %s\n", e.what()); return -1; }
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
	~VkConverter() override {
		if (vk) for (int i = 0; i < IN_FLIGHT; i++) if (slots[i].pending) { try { vk->wait(i); } catch (...) {} }
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
			vk = std::make_unique<Vk>(device, IN_FLIGHT);
		} catch (const std::exception& e) {
			fprintf(stderr, "converter: %s\n", e.what());
			return false;
		}
		gpu_out = false;
		const char* e = getenv("VES_CONVERT_TO_GPU");
		if (gpu_out_wanted && vk->external() && !(e && atoi(e) == 0)) {
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
		VkDecodedFrame o; // gpu_out: the frame the converted picture goes into (declared before its lock, which outlives the try block)
		FrameLock in_lock, out_lock;
		try {
			if (shape_changed(shape, gf != nullptr)) {
				while (!order.empty()) if (take_oldest(true) < 0) return false; // the shared buffers change
				if (!setup(shape, L, gf != nullptr)) return false;
			}
			for (int pn = 0; pn < L.planes; pn++) cur_stride[pn] = plane_stride(shape, L, pn, gf != nullptr);
			vk->begin(k);
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
			if (gpu_out) {
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
			vk->submit(k);
			out_lock.release();
			in_lock.release();
		} catch (const std::exception& e) {
			fprintf(stderr, "converter: %s\n", e.what()); // a lost device has been reported (vk_compute.cpp's chk())
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
		return true;
	}

	int pop(Picture& o) override {
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
