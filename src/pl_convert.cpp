// Converter on libplacebo's own API (Vulkan). Takes planar frames (YUV or GBR), scales them and converts them to planar YUV.
#include "convert.h"

#include <cstdio>
#include <cstring>
#include <deque>
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

struct Planes { // the memory of an output picture
	std::vector<uint8_t> mem[3];
};

class PlConverter : public Converter {
	pl_log log = nullptr;
	pl_vulkan vk = nullptr;
	pl_renderer rr = nullptr;
	pl_tex in_tex[3]{}, out_tex[3]{};
	const pl_filter_config *up = nullptr, *down = nullptr;
	int out_w = 0, out_h = 0;
	Fmt out_fmt;
	std::deque<Picture> ready;

public:
	~PlConverter() override {
		if (vk) {
			for (auto& t : in_tex) pl_tex_destroy(vk->gpu, &t);
			for (auto& t : out_tex) pl_tex_destroy(vk->gpu, &t);
		}
		pl_renderer_destroy(&rr);
		pl_vulkan_destroy(&vk);
		pl_log_destroy(&log);
	}

	bool open(const Picture&, Rational, int w, int h, Fmt out, const ScaleOptions& opt) override {
		out_w = w; out_h = h; out_fmt = out;
		struct pl_log_params lp = {};
		lp.log_cb = log_cb;
		lp.log_level = PL_LOG_WARN;
		log = pl_log_create(PL_API_VER, &lp);
		struct pl_vulkan_params vp = pl_vulkan_default_params;
		if (!opt.device.empty()) vp.device_name = opt.device.c_str();
		vk = pl_vulkan_create(log, &vp);
		if (!vk) { fprintf(stderr, "libplacebo: no Vulkan GPU%s\n", opt.device.empty() ? "" : (" named '" + opt.device + "'").c_str()); return false; }
		rr = pl_renderer_create(log, vk->gpu);
		up = pl_find_filter_config(opt.upscaler.c_str(), PL_FILTER_UPSCALING);
		down = pl_find_filter_config(opt.downscaler.c_str(), PL_FILTER_DOWNSCALING);
		if (!rr || !up || !down) { fprintf(stderr, "libplacebo: unknown scaler '%s' / '%s'\n", opt.upscaler.c_str(), opt.downscaler.c_str()); return false; }

		const int bits = out.depth > 8 ? 16 : 8;
		pl_fmt fmt = pl_find_fmt(vk->gpu, PL_FMT_UNORM, 1, bits, bits, PL_FMT_CAP_RENDERABLE);
		if (!fmt) { fprintf(stderr, "libplacebo: this GPU cannot render %d-bit planes\n", bits); return false; }
		for (int i = 0; i < 3; i++) {
			struct pl_tex_params tp = {};
			tp.w = plane_w(w, out.chroma, i); tp.h = plane_h(h, out.chroma, i);
			tp.format = fmt;
			tp.renderable = true;
			tp.host_readable = true;
			if (!(out_tex[i] = pl_tex_create(vk->gpu, &tp))) { fprintf(stderr, "libplacebo: cannot create the output planes\n"); return false; }
		}
		return true;
	}

	bool push(const Picture* p) override {
		if (!p) return true; // nothing is ever held back
		if (!vk || !p->fmt.planar) return false;
		const int depth = p->fmt.depth, bytes = depth > 8 ? 2 : 1;
		const pl_color_system sys = color_system(*p);

		struct pl_frame image = {};
		image.num_planes = 3;
		for (int i = 0; i < 3; i++) {
			struct pl_plane_data d = {};
			d.type = PL_FMT_UNORM;
			d.width = plane_w(p->width, p->fmt.chroma, i);
			d.height = plane_h(p->height, p->fmt.chroma, i);
			d.component_size[0] = bytes * 8; // the depth in the sample is told through repr.bits
			d.component_map[0] = p->fmt.rgb ? (i == 0 ? PL_CHANNEL_G : i == 1 ? PL_CHANNEL_B : PL_CHANNEL_R) : (i == 0 ? PL_CHANNEL_Y : i == 1 ? PL_CHANNEL_CB : PL_CHANNEL_CR);
			d.pixel_stride = bytes;
			d.row_stride = p->stride[i];
			d.pixels = p->data[i];
			if (!pl_upload_plane(vk->gpu, &image.planes[i], &in_tex[i], &d)) { fprintf(stderr, "libplacebo: upload failed\n"); return false; }
		}
		image.repr.sys = sys;
		image.repr.levels = color_levels(*p);
		image.repr.alpha = PL_ALPHA_NONE;
		image.repr.bits.sample_depth = bytes * 8; image.repr.bits.color_depth = depth; image.repr.bits.bit_shift = 0;
		image.color.primaries = color_primaries(p->color.primaries);
		image.color.transfer = color_transfer(p->color.trc);
		if (!p->fmt.rgb && p->fmt.chroma != 444) pl_frame_set_chroma_location(&image, PL_CHROMA_LEFT);

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
		target.repr.levels = p->fmt.rgb ? PL_COLOR_LEVELS_LIMITED : image.repr.levels;
		target.repr.alpha = PL_ALPHA_NONE;
		target.repr.bits.sample_depth = obytes * 8; target.repr.bits.color_depth = out_fmt.depth; target.repr.bits.bit_shift = 0;
		target.color = image.color;
		if (out_fmt.chroma != 444) pl_frame_set_chroma_location(&target, PL_CHROMA_LEFT);

		struct pl_render_params rp = pl_render_default_params;
		rp.upscaler = up;
		rp.downscaler = down;
		if (out_fmt.depth >= depth) rp.dither_params = nullptr; // only reducing the bit depth needs dithering, it would just add noise otherwise
		if (!pl_render_image(rr, &image, &target, &rp)) { fprintf(stderr, "libplacebo: rendering failed\n"); return false; }

		auto planes = std::make_shared<Planes>();
		Picture out;
		out.width = out_w; out.height = out_h;
		out.fmt = out_fmt;
		for (int i = 0; i < 3; i++) {
			const int pw = plane_w(out_w, out_fmt.chroma, i), ph = plane_h(out_h, out_fmt.chroma, i);
			const size_t stride = ((size_t)pw * obytes + 63) & ~(size_t)63;
			planes->mem[i].resize(stride * ph);
			struct pl_tex_transfer_params tp = {};
			tp.tex = out_tex[i];
			tp.row_pitch = stride;
			tp.ptr = planes->mem[i].data();
			if (!pl_tex_download(vk->gpu, &tp)) { fprintf(stderr, "libplacebo: download failed\n"); return false; }
			out.data[i] = planes->mem[i].data();
			out.stride[i] = (int)stride;
		}
		out.pts = p->pts;
		out.duration = p->duration;
		out.color = p->color;
		if (p->fmt.rgb) { out.color.space = 1; out.color.range = 1; }
		out.owner = planes;
		ready.push_back(std::move(out));
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
