// Converter on libplacebo's own API (Vulkan). Reads frames in whatever sample layout the source has (planar, NV12/P010, packed RGB,
// ...), scales them and converts them to planar YUV.
#include "convert.h"
#include "gpu.h"

#include <chrono>
#include <cmath>

#include <algorithm>
#include <cstdio>
#include <cstring>
#include <deque>
#include <map>
#include <memory>
#include <mutex>
#include <thread>
#include <vector>

extern "C" {
#include <libplacebo/colorspace.h>
#include <libplacebo/filters.h>
#include <libplacebo/gpu.h>
#include <libplacebo/log.h>
#include <libplacebo/renderer.h>
#include <libplacebo/utils/upload.h>
#include <libplacebo/vulkan.h>
}

namespace {

void log_cb(void*, enum pl_log_level level, const char* msg) {
	if (level <= PL_LOG_WARN) fprintf(stderr, "libplacebo: %s\n", msg);
}

// Plane sizes of a frame in a chroma layout.
int plane_w(int w, int chroma, int i) { return i == 0 || chroma == 444 ? w : (w + 1) / 2; }
int plane_h(int h, int chroma, int i) { return i == 0 || chroma != 420 ? h : (h + 1) / 2; }

// Colour tags (ITU-T H.273 numbers) to libplacebo's enums. Unknown stays unknown, which is fine as long as input and
// output agree, since libplacebo then does no colour management.
pl_color_system color_system(const Picture& p) {
	if (p.fmt.rgb) return PL_COLOR_SYSTEM_RGB;
	switch (p.color.space) {
	case 1: return PL_COLOR_SYSTEM_BT_709;
	case 5: case 6: return PL_COLOR_SYSTEM_BT_601;
	case 7: return PL_COLOR_SYSTEM_SMPTE_240M;
	case 8: return PL_COLOR_SYSTEM_YCGCO;
	case 9: return PL_COLOR_SYSTEM_BT_2020_NC;
	case 10: return PL_COLOR_SYSTEM_BT_2020_C;
	default: return pl_color_system_guess_ycbcr(p.width, p.height);
	}
}

pl_color_levels color_levels(const Picture& p) {
	return p.color.range == 1 ? PL_COLOR_LEVELS_LIMITED : p.color.range == 2 ? PL_COLOR_LEVELS_FULL : p.fmt.rgb ? PL_COLOR_LEVELS_FULL : PL_COLOR_LEVELS_LIMITED;
}

pl_color_primaries color_primaries(int v) {
	switch (v) {
	case 1: return PL_COLOR_PRIM_BT_709;
	case 5: return PL_COLOR_PRIM_BT_601_625;
	case 6: return PL_COLOR_PRIM_BT_601_525;
	case 9: return PL_COLOR_PRIM_BT_2020;
	default: return PL_COLOR_PRIM_UNKNOWN;
	}
}

pl_color_transfer color_transfer(int v) {
	switch (v) {
	case 1: case 6: case 14: case 15: return PL_COLOR_TRC_BT_1886;
	case 8: return PL_COLOR_TRC_LINEAR;
	case 13: return PL_COLOR_TRC_SRGB;
	case 16: return PL_COLOR_TRC_PQ;
	case 18: return PL_COLOR_TRC_HLG;
	default: return PL_COLOR_TRC_UNKNOWN;
	}
}

// One Vulkan device per GPU for every converter of this process, and its renderers shared as well. A device and a renderer of its own for each converter
// made a wave of n runs open 2n + 3 devices on the same GPU at once, each taking 1 to 2 s ("creating vulkan device (slow!)"), and 2n + 3 renderers, each
// holding intermediate frames at full size: together with the scorers they filled 16 GB of video memory until the driver reset the GPU (device lost).
// Every libplacebo call on the shared device is made with `use` held. libplacebo calls its Vulkan pl_gpu thread safe, but it is not quite: two threads that
// wait for its commands at the same time can both take the same finished command buffer, and one of them then recycles it while it runs again
// (vk_poll_commands; a stress test of 12 threads on one device crashed in the driver, the validation layer saying VUID-vkQueueSubmit2-commandBuffer-03875).
// Nothing waits with the lock held: a converter records and submits its frame, then waits for the download outside of it, so the GPU still works on the
// frames of many converters at once. A renderer keeps its intermediate frames for the sizes it last worked with, so a converter takes one of its own sizes
// for each frame; only as many exist as record at the same moment.
// A device that was lost is not handed out again: the next converter gets a new one.
struct SharedVulkan {
	pl_log log = nullptr;
	pl_vulkan vk = nullptr;
	std::mutex use;
	std::map<std::string, std::vector<pl_renderer>> idle; // by input and output size, output format and scalers

	pl_renderer borrow(const std::string& shape) { // with `use` held
		auto& list = idle[shape];
		if (list.empty()) return pl_renderer_create(log, vk->gpu);
		pl_renderer r = list.back();
		list.pop_back();
		return r;
	}
	void give_back(const std::string& shape, pl_renderer r) { idle[shape].push_back(r); } // with `use` held
	~SharedVulkan() {
		for (auto& [shape, list] : idle) for (auto& r : list) pl_renderer_destroy(&r);
		pl_vulkan_destroy(&vk);
		pl_log_destroy(&log);
	}
};

std::shared_ptr<SharedVulkan> shared_vulkan(const std::string& device) {
	static std::mutex m;
	static auto* devices = new std::map<std::string, std::shared_ptr<SharedVulkan>>; // never freed: converters may still be in use when the process ends
	std::lock_guard lock(m);
	auto& slot = (*devices)[device];
	if (slot && pl_gpu_is_failed(slot->vk->gpu)) slot.reset(); // lost: the converters that hold it let go of it as they end
	if (slot) return slot;
	auto s = std::make_shared<SharedVulkan>();
	struct pl_log_params lp = {};
	lp.log_cb = log_cb;
	lp.log_level = PL_LOG_WARN;
	s->log = pl_log_create(PL_API_VER, &lp);
	struct pl_vulkan_params vp = pl_vulkan_default_params;
	if (!device.empty()) vp.device_name = device.c_str();
	s->vk = pl_vulkan_create(s->log, &vp);
	if (!s->vk) return nullptr;
	slot = s;
	return slot;
}

struct Planes { // the memory of an output picture
	std::vector<uint8_t> mem[3];
};

// A frame on its way through the GPU: its planes were copied into `up`, and its result is downloaded into `dl`. Several are in flight per converter,
// so that the GPU converts one frame while the next is copied in and the one before is copied out, and the copies run without the device's lock:
// the buffers are mapped, so only recording and submitting the work needs it.
constexpr int IN_FLIGHT = 3;
struct Slot {
	pl_buf up[3]{}, dl[3]{};
	bool pending = false;
	int64_t pts = 0, duration = 1;
	Color color;
	bool rgb = false;
};

class PlConverter : public Converter {
	std::shared_ptr<SharedVulkan> dev;
	pl_vulkan vk = nullptr; // dev's
	std::string shape; // the renderers of its sizes, format and scalers are the ones it borrows
	pl_tex in_tex[3]{}, out_tex[3]{}; // shared by the frames in flight: libplacebo orders the work on them
	Slot slots[IN_FLIGHT];
	std::deque<int> order; // the slots in flight, oldest first
	int next_slot = 0;
	bool mapped_up = false; // upload through mapped buffers; otherwise libplacebo copies the planes itself, with the lock held
	const pl_filter_config *up = nullptr, *down = nullptr;
	int out_w = 0, out_h = 0;
	Fmt out_fmt;
	std::deque<Picture> ready;

	// Waits for the oldest frame in flight (if `wait`; otherwise only takes it if it is done) and copies it out. 1: taken, 0: not done yet, -1: failed.
	int take_oldest(bool wait) {
		Slot& s = slots[order.front()];
		std::unique_lock lock(dev->use);
		for (;;) {
			if (pl_gpu_is_failed(vk->gpu)) { lost(); return -1; }
			bool busy = false;
			for (auto& b : s.dl) busy = busy || pl_buf_poll(vk->gpu, b, 0);
			if (!busy) break;
			if (!wait) return 0;
			lock.unlock();
			std::this_thread::sleep_for(std::chrono::microseconds(250));
			lock.lock();
		}
		lock.unlock();
		auto planes = std::make_shared<Planes>();
		Picture out;
		out.width = out_w; out.height = out_h;
		out.fmt = out_fmt;
		for (int i = 0; i < 3; i++) {
			const size_t stride = stride_of(i);
			planes->mem[i].resize(stride * plane_h(out_h, out_fmt.chroma, i));
			if (s.dl[i]->data) memcpy(planes->mem[i].data(), s.dl[i]->data, planes->mem[i].size()); // done on the GPU: nothing else touches it now
			else {
				std::lock_guard again(dev->use);
				if (!pl_buf_read(vk->gpu, s.dl[i], 0, planes->mem[i].data(), planes->mem[i].size())) { fprintf(stderr, "libplacebo: download failed\n"); lost(); return -1; }
			}
			out.data[i] = planes->mem[i].data();
			out.stride[i] = (int)stride;
		}
		{
			std::lock_guard again(dev->use);
			if (lost()) return -1; // after a lost device (driver reset) calls can still report success and hand back stale memory
		}
		out.pts = s.pts;
		out.duration = s.duration;
		out.color = s.color;
		if (s.rgb) out.color.space = 1;
		if (target_range) out.color.range = target_range;
		else if (s.rgb) out.color.range = 1;
		out.owner = planes;
		ready.push_back(std::move(out));
		s.pending = false;
		order.pop_front();
		return 1;
	}

public:
	~PlConverter() override {
		if (vk) {
			std::lock_guard lock(dev->use);
			for (auto& t : in_tex) pl_tex_destroy(vk->gpu, &t);
			for (auto& t : out_tex) pl_tex_destroy(vk->gpu, &t);
			for (auto& s : slots) for (int i = 0; i < 3; i++) { pl_buf_destroy(vk->gpu, &s.up[i]); pl_buf_destroy(vk->gpu, &s.dl[i]); } // libplacebo frees them once the GPU is done
		}
	}

	// The device is lost (a driver reset or a GPU fault): this frame and every later one on it cannot be trusted.
	bool lost() {
		if (!pl_gpu_is_failed(vk->gpu)) return false;
		fprintf(stderr, "libplacebo: the GPU failed (device lost), the converted frame cannot be trusted\n");
		gpu_lost("Vulkan device lost while converting frames");
		return true;
	}

	int target_range = 0; // 0: as the input (RGB: limited), 1 limited, 2 full
	bool open(const Picture& first, Rational, int w, int h, Fmt out, const ScaleOptions& opt, int range) override {
		target_range = range;
		out_w = w; out_h = h; out_fmt = out;
		dev = shared_vulkan(opt.device);
		vk = dev ? dev->vk : nullptr;
		if (!vk) { fprintf(stderr, "libplacebo: no Vulkan GPU%s\n", opt.device.empty() ? "" : (" named '" + opt.device + "'").c_str()); return false; }
		shape = std::to_string(first.width) + "x" + std::to_string(first.height) + " to " + std::to_string(w) + "x" + std::to_string(h) + " " + std::to_string(out.chroma) + " " +
		        std::to_string(out.depth) + " " + opt.upscaler + "/" + opt.downscaler;
		up = pl_find_filter_config(opt.upscaler.c_str(), PL_FILTER_UPSCALING);
		down = pl_find_filter_config(opt.downscaler.c_str(), PL_FILTER_DOWNSCALING);
		if (!up || !down) { fprintf(stderr, "libplacebo: unknown scaler '%s' / '%s'\n", opt.upscaler.c_str(), opt.downscaler.c_str()); return false; }

		const int bits = out.depth > 8 ? 16 : 8;
		std::lock_guard lock(dev->use);
		pl_fmt fmt = pl_find_fmt(vk->gpu, PL_FMT_UNORM, 1, bits, bits, PL_FMT_CAP_RENDERABLE);
		if (!fmt) { fprintf(stderr, "libplacebo: this GPU cannot render %d-bit planes\n", bits); return false; }
		for (int i = 0; i < 3; i++) {
			struct pl_tex_params tp = {};
			tp.w = plane_w(w, out.chroma, i); tp.h = plane_h(h, out.chroma, i);
			tp.format = fmt;
			tp.renderable = true;
			tp.host_readable = true;
			if (!(out_tex[i] = pl_tex_create(vk->gpu, &tp))) { fprintf(stderr, "libplacebo: cannot create the output planes\n"); return false; }
			struct pl_buf_params bp = {};
			bp.size = stride_of(i) * tp.h;
			bp.host_readable = true;
			bp.host_mapped = bp.size <= vk->gpu->limits.max_mapped_size;
			bp.memory_type = PL_BUF_MEM_HOST;
			for (auto& s : slots) if (!(s.dl[i] = pl_buf_create(vk->gpu, &bp))) { fprintf(stderr, "libplacebo: cannot create the download buffers\n"); return false; }
		}
		mapped_up = vk->gpu->limits.buf_transfer && vk->gpu->limits.max_mapped_size > 0;
		return true;
	}

	// Rows of output plane i, 64 byte aligned.
	size_t stride_of(int i) const { return ((size_t)plane_w(out_w, out_fmt.chroma, i) * (out_fmt.depth > 8 ? 2 : 1) + 63) & ~(size_t)63; }

	// Rows of plane pn of p, and its bytes: every row but the last at its stride, the last one only as long as its samples (that is all a source has to have).
	static int plane_rows(const Picture* p, int pn) { return !p->fmt.rgb && pn > 0 && p->layout.log2_ch ? (p->height + (1 << p->layout.log2_ch) - 1) >> p->layout.log2_ch : p->height; }
	static size_t plane_bytes(const Picture* p, int pn) {
		const int w = !p->fmt.rgb && pn > 0 && p->layout.log2_cw ? (p->width + (1 << p->layout.log2_cw) - 1) >> p->layout.log2_cw : p->width;
		return (size_t)(plane_rows(p, pn) - 1) * (size_t)p->stride[pn] + (size_t)w * (size_t)p->layout.step[pn];
	}

	bool push(const Picture* p) override {
		if (!vk) return false;
		if (!p) { // everything in flight comes out
			while (!order.empty()) if (take_oldest(true) < 0) return false;
			std::lock_guard lock(dev->use);
			return !lost();
		}
		const Layout& L = p->layout;
		if (L.planes == 0) return false;
		if (slots[next_slot].pending && take_oldest(true) < 0) return false; // every slot is in flight: the oldest has to come out first
		Slot& fs = slots[next_slot];

		// The planes go into the slot's mapped upload buffers without the lock (nothing on the GPU reads them: the slot's last frame is done).
		bool via_buf = mapped_up;
		size_t need[3] = {};
		for (int pn = 0; pn < L.planes && pn < 3 && via_buf; pn++) {
			need[pn] = ((size_t)p->stride[pn] * (size_t)plane_rows(p, pn) + 3) & ~(size_t)3;
			if (p->stride[pn] <= 0 || L.step[pn] <= 0 || need[pn] > vk->gpu->limits.max_mapped_size || ((size_t)p->stride[pn] % (size_t)L.step[pn]) != 0) via_buf = false;
		}
		if (via_buf) {
			std::unique_lock lock(dev->use);
			for (int pn = 0; pn < L.planes && pn < 3; pn++) {
				if (fs.up[pn] && fs.up[pn]->params.size >= need[pn]) continue;
				pl_buf_destroy(vk->gpu, &fs.up[pn]);
				struct pl_buf_params bp = {};
				bp.size = need[pn];
				bp.host_writable = true;
				bp.host_mapped = true;
				bp.memory_type = PL_BUF_MEM_HOST;
				if (!(fs.up[pn] = pl_buf_create(vk->gpu, &bp)) || !fs.up[pn]->data) { via_buf = false; break; }
			}
			for (int pn = 0; via_buf && pn < L.planes && pn < 3; pn++) // the GPU may still read the buffer of a frame that failed halfway
				while (pl_buf_poll(vk->gpu, fs.up[pn], 0)) { lock.unlock(); std::this_thread::sleep_for(std::chrono::microseconds(250)); lock.lock(); }
		}
		if (via_buf) for (int pn = 0; pn < L.planes && pn < 3; pn++) memcpy(fs.up[pn]->data, p->data[pn], plane_bytes(p, pn));

		std::unique_lock lock(dev->use);
		if (lost()) return false;
		const pl_color_system sys = color_system(*p);

		// Read the source's own layout: every plane becomes one texture with the components it holds in memory order,
		// padding bytes (the 0 in bgr0, say) are filled in as unused components, and an alpha component is ignored.
		struct pl_frame image = {};
		struct pl_bit_encoding bits = {};
		static const int map_rgb[4] = {PL_CHANNEL_R, PL_CHANNEL_G, PL_CHANNEL_B, PL_CHANNEL_NONE};
		static const int map_yuv[4] = {PL_CHANNEL_Y, PL_CHANNEL_CB, PL_CHANNEL_CR, PL_CHANNEL_NONE};
		for (int pn = 0; pn < L.planes; pn++) {
			int idx[4], n = 0;
			for (int c = 0; c < 4; c++) if (L.comp[c].plane == pn) idx[n++] = c;
			for (int i = 1; i < n; i++) for (int j = i; j > 0 && L.comp[idx[j]].offset < L.comp[idx[j - 1]].offset; j--) std::swap(idx[j], idx[j - 1]);
			if (n == 0 || idx[0] == 3 || image.num_planes >= 3) continue; // nothing but alpha in this plane

			struct pl_plane_data d = {};
			d.type = PL_FMT_UNORM;
			const bool subsampled = !p->fmt.rgb && (idx[0] == 1 || idx[0] == 2);
			d.width = subsampled ? (p->width + (1 << L.log2_cw) - 1) >> L.log2_cw : p->width;
			d.height = subsampled ? (p->height + (1 << L.log2_ch) - 1) >> L.log2_ch : p->height;
			int slot = 0, cursor = 0;
			bool ok = true;
			auto add = [&](int size_bits, int map) {
				if (slot >= 4) { ok = false; return; }
				d.component_size[slot] = size_bits; d.component_map[slot] = map; slot++;
			};
			for (int k = 0; k < n; k++) {
				const Layout::Comp& c = L.comp[idx[k]];
				const int cbytes = c.depth + c.shift > 8 ? 2 : 1;
				if (c.depth < 8) ok = false; // sub-byte samples (rgb565, ...)
				if (c.offset > cursor) add(8 * (c.offset - cursor), PL_CHANNEL_NONE);
				add(8 * cbytes, idx[k] < 3 ? (p->fmt.rgb ? map_rgb[idx[k]] : map_yuv[idx[k]]) : PL_CHANNEL_NONE);
				cursor = c.offset + cbytes;
				if (idx[k] < 3 && bits.sample_depth == 0) { bits.sample_depth = 8 * cbytes; bits.color_depth = c.depth; bits.bit_shift = c.shift; } // the depth of the samples is told through repr.bits
			}
			if (cursor < L.step[pn]) add(8 * (L.step[pn] - cursor), PL_CHANNEL_NONE);
			if (!ok) { fprintf(stderr, "libplacebo: unsupported sample layout\n"); return false; }
			d.pixel_stride = L.step[pn];
			d.row_stride = p->stride[pn];
			if (via_buf) { d.buf = fs.up[pn]; d.buf_offset = 0; }
			else d.pixels = p->data[pn];
			d.swapped = L.big_endian;
			if (!pl_upload_plane(vk->gpu, &image.planes[image.num_planes], &in_tex[image.num_planes], &d)) { fprintf(stderr, "libplacebo: upload failed\n"); lost(); return false; }
			image.num_planes++;
		}
		image.repr.sys = sys;
		image.repr.levels = color_levels(*p);
		image.repr.alpha = PL_ALPHA_NONE;
		image.repr.bits = bits;
		image.color.primaries = color_primaries(p->color.primaries);
		image.color.transfer = color_transfer(p->color.trc);
		if (L.log2_cw || L.log2_ch) pl_frame_set_chroma_location(&image, PL_CHROMA_LEFT);

		const int obytes = out_fmt.depth > 8 ? 2 : 1;
		struct pl_frame target = {};
		target.num_planes = 3;
		for (int i = 0; i < 3; i++) {
			target.planes[i].texture = out_tex[i];
			target.planes[i].components = 1;
			target.planes[i].component_mapping[0] = i == 0 ? PL_CHANNEL_Y : i == 1 ? PL_CHANNEL_CB : PL_CHANNEL_CR;
		}
		// RGB has no matrix of its own, so converting it to YUV needs a choice: BT.709, limited range
		target.repr.sys = p->fmt.rgb ? PL_COLOR_SYSTEM_BT_709 : sys;
		target.repr.levels = target_range ? (target_range == 2 ? PL_COLOR_LEVELS_FULL : PL_COLOR_LEVELS_LIMITED) : p->fmt.rgb ? PL_COLOR_LEVELS_LIMITED : image.repr.levels;
		target.repr.alpha = PL_ALPHA_NONE;
		target.repr.bits.sample_depth = obytes * 8; target.repr.bits.color_depth = out_fmt.depth; target.repr.bits.bit_shift = 0;
		target.color = image.color;
		if (out_fmt.chroma != 444) pl_frame_set_chroma_location(&target, PL_CHROMA_LEFT);

		struct pl_render_params rp = pl_render_default_params;
		rp.upscaler = up;
		rp.downscaler = down;
		if (out_fmt.depth >= p->fmt.depth) rp.dither_params = nullptr; // only reducing the bit depth needs dithering, it would just add noise otherwise
		pl_renderer rr = dev->borrow(shape);
		if (!rr) { fprintf(stderr, "libplacebo: cannot create a renderer\n"); lost(); return false; }
		const bool rendered = pl_render_image(rr, &image, &target, &rp);
		dev->give_back(shape, rr);
		if (!rendered) { fprintf(stderr, "libplacebo: rendering failed\n"); lost(); return false; }

		for (int i = 0; i < 3; i++) {
			struct pl_tex_transfer_params tp = {};
			tp.tex = out_tex[i];
			tp.row_pitch = stride_of(i);
			tp.buf = fs.dl[i];
			if (!pl_tex_download(vk->gpu, &tp)) { fprintf(stderr, "libplacebo: download failed\n"); lost(); return false; }
		}
		pl_gpu_flush(vk->gpu); // submitted now, not when the next frame comes
		lock.unlock();
		fs.pending = true;
		fs.pts = p->pts;
		fs.duration = p->duration;
		fs.color = p->color;
		fs.rgb = p->fmt.rgb;
		order.push_back(next_slot);
		next_slot = (next_slot + 1) % IN_FLIGHT;
		// Whatever is done by now comes out; the rest when the slots are needed again, or at the end.
		for (int got; !order.empty() && (got = take_oldest(false)) != 0;) if (got < 0) return false;
		return true;
	}

	int pop(Picture& out) override {
		if (ready.empty()) return 0;
		out = std::move(ready.front());
		ready.pop_front();
		return 1;
	}
};

} // namespace

std::unique_ptr<Converter> make_converter() { return std::make_unique<PlConverter>(); }

bool converter_works(const ScaleOptions& opt, std::string& why) {
	// 64x64 8-bit 4:4:4: luma rises from left to right, Cb from top to bottom, Cr is flat; it goes to 32x32 4:2:0, which takes the same path as a real frame (scale, subsample, download).
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
	if (!conv->open(in, Rational{1, 60}, 32, 32, out_fmt, opt, 1)) { why = "it cannot convert at all"; return false; }
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
