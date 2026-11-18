#include "ff.h"

extern "C" {
#include <libavutil/hwcontext.h>
}
#include "source.h"

#include <algorithm>
#include <atomic>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>
#include <thread>

#include <cstring>

bool describe_native(AVPixelFormat f, Fmt& out) {
	const AVPixFmtDescriptor* d = av_pix_fmt_desc_get(f);
	if (!d || d->nb_components < 3 || (d->flags & (AV_PIX_FMT_FLAG_HWACCEL | AV_PIX_FMT_FLAG_BITSTREAM | AV_PIX_FMT_FLAG_PAL | AV_PIX_FMT_FLAG_BAYER))) return false;
	out.rgb = d->flags & AV_PIX_FMT_FLAG_RGB;
	out.depth = d->comp[0].depth;
	if (out.depth > 12) return false;
	// shortcut: layouts libvmaf and the encoders lack (4:1:0, 4:1:1, 4:4:0) are treated as 4:2:0
	out.chroma = out.rgb || (!d->log2_chroma_w && !d->log2_chroma_h) ? 444 : d->log2_chroma_w == 1 && !d->log2_chroma_h ? 422 : 420;
	out.planar = (d->flags & AV_PIX_FMT_FLAG_PLANAR) && d->nb_components == 3 && !(d->flags & AV_PIX_FMT_FLAG_BE) &&
	                 d->comp[0].plane == 0 && d->comp[1].plane == 1 && d->comp[2].plane == 2 && // not NV12 and friends
	                 (d->log2_chroma_w + d->log2_chroma_h == (out.chroma == 420 ? 2 : out.chroma == 422 ? 1 : 0));
	return true;
}

AVPixelFormat av_planar_format(int chroma, int depth) {
	static const AVPixelFormat t[3][3] = {
		{AV_PIX_FMT_YUV420P, AV_PIX_FMT_YUV420P10LE, AV_PIX_FMT_YUV420P12LE},
		{AV_PIX_FMT_YUV422P, AV_PIX_FMT_YUV422P10LE, AV_PIX_FMT_YUV422P12LE},
		{AV_PIX_FMT_YUV444P, AV_PIX_FMT_YUV444P10LE, AV_PIX_FMT_YUV444P12LE},
	};
	int c = chroma == 420 ? 0 : chroma == 422 ? 1 : chroma == 444 ? 2 : -1;
	int d = depth == 8 ? 0 : depth == 10 ? 1 : depth == 12 ? 2 : -1;
	return c < 0 || d < 0 ? AV_PIX_FMT_NONE : t[c][d];
}

Layout layout_of(AVPixelFormat f) {
	Layout l;
	const AVPixFmtDescriptor* d = av_pix_fmt_desc_get(f);
	if (!d) return l;
	for (int i = 0; i < d->nb_components && i < 4; i++) {
		const AVComponentDescriptor& c = d->comp[i];
		l.comp[i] = {c.plane, c.offset, c.shift, c.depth};
		l.step[c.plane] = c.step;
		if (c.plane + 1 > l.planes) l.planes = c.plane + 1;
	}
	l.log2_cw = d->log2_chroma_w; l.log2_ch = d->log2_chroma_h;
	l.big_endian = d->flags & AV_PIX_FMT_FLAG_BE;
	return l;
}

Picture picture_from_frame(AVFrame* frame, const Fmt& fmt) {
	AVFrame* own = av_frame_alloc();
	av_frame_move_ref(own, frame);
	Picture p;
	p.width = own->width; p.height = own->height;
	p.fmt = fmt;
	p.native = own->format;
	p.layout = layout_of((AVPixelFormat)own->format);
	for (int i = 0; i < 4; i++) { p.data[i] = own->data[i]; p.stride[i] = own->linesize[i]; }
	p.pts = own->pts;
	p.duration = own->duration;
	p.color = {(int)own->color_range, (int)own->colorspace, (int)own->color_primaries, (int)own->color_trc};
	p.owner = std::shared_ptr<void>(own, [](void* f) { AVFrame* a = (AVFrame*)f; av_frame_free(&a); });
	return p;
}

static std::atomic<int> g_threads{0};

void set_decode_threads(int threads) { g_threads = threads; }

// All the cores by default: the pipeline never lets the decoder get more than a few frames ahead anyway.
int decode_threads() {
	if (const char* e = getenv("VES_DECODE_THREADS")) return atoi(e);
	if (g_threads > 0) return g_threads;
	return std::max(2, (int)std::thread::hardware_concurrency());
}

const AVCodec* hwaccel_decoder(AVCodecID id, const std::string& type) {
	const AVHWDeviceType t = av_hwdevice_find_type_by_name(type.c_str());
	if (t == AV_HWDEVICE_TYPE_NONE) return nullptr;
	// On Vulkan only the codecs of the GPU's video engine. FFmpeg also decodes FFV1, ProRes and others "on Vulkan" with compute shaders, and those are
	// not to be trusted: a second FFV1 decoder on a device that did other work after the first one was closed gives broken frames ("slices overread",
	// FFmpeg 8.1 on lavapipe, with nothing but an FFmpeg upload in between), while the first decoder, the one that is checked against software, is fine.
	if (t == AV_HWDEVICE_TYPE_VULKAN && id != AV_CODEC_ID_H264 && id != AV_CODEC_ID_HEVC && id != AV_CODEC_ID_AV1 && id != AV_CODEC_ID_VP9) return nullptr;
	void* it = nullptr;
	while (const AVCodec* c = av_codec_iterate(&it)) {
		if (c->id != id || !av_codec_is_decoder(c) || (c->capabilities & AV_CODEC_CAP_HARDWARE)) continue;
		for (int i = 0;; i++) {
			const AVCodecHWConfig* cfg = avcodec_get_hw_config(c, i);
			if (!cfg) break;
			if ((cfg->methods & AV_CODEC_HW_CONFIG_METHOD_HW_DEVICE_CTX) && cfg->device_type == t) return c;
		}
	}
	return nullptr;
}

const AVCodec* software_decoder(AVCodecID id) {
	void* it = nullptr;
	while (const AVCodec* c = av_codec_iterate(&it))
		if (c->id == id && av_codec_is_decoder(c) && !(c->capabilities & (AV_CODEC_CAP_HARDWARE | AV_CODEC_CAP_HYBRID))) return c;
	return nullptr;
}

// Semi-planar formats (NV12, P010, ...) keep U and V interleaved in one plane. The samples are unpacked to planar YUV exactly, on the CPU:
// the conversion libplacebo would do instead is not exact, it changes sample values by a little even when only the memory layout differs.
// This also makes a GPU's decode (which gives NV12/P010 back) reach the encoders the way a software decode does.
const SemiPlanar* semi_planar(AVPixelFormat f) {
	static const SemiPlanar table[] = {
		{AV_PIX_FMT_NV12, AV_PIX_FMT_YUV420P, 1, 1, false}, {AV_PIX_FMT_NV16, AV_PIX_FMT_YUV422P, 1, 0, false}, {AV_PIX_FMT_NV24, AV_PIX_FMT_YUV444P, 0, 0, false},
		{AV_PIX_FMT_P010LE, AV_PIX_FMT_YUV420P10LE, 1, 1, true}, {AV_PIX_FMT_P210LE, AV_PIX_FMT_YUV422P10LE, 1, 0, true}, {AV_PIX_FMT_P410LE, AV_PIX_FMT_YUV444P10LE, 0, 0, true},
	};
	for (auto& e : table) if (e.from == f) return &e;
	return nullptr;
}

// Returns `f` itself for planar formats, else `planar` filled with the unpacked samples.
AVFrame* unpack_semi_planar(AVFrame* f, AVFrame* planar) {
	const SemiPlanar* sp = semi_planar((AVPixelFormat)f->format);
	if (!sp) return f;
	av_frame_unref(planar);
	planar->format = sp->to;
	planar->width = f->width;
	planar->height = f->height;
	if (av_frame_get_buffer(planar, 0) < 0) return f;
	av_frame_copy_props(planar, f);
	const int cw = (f->width + (1 << sp->sx) - 1) >> sp->sx, ch = (f->height + (1 << sp->sy) - 1) >> sp->sy;
	for (int y = 0; y < f->height; y++) {
		if (!sp->high) memcpy(planar->data[0] + (size_t)y * planar->linesize[0], f->data[0] + (size_t)y * f->linesize[0], (size_t)f->width);
		else {
			const uint16_t* s = (const uint16_t*)(f->data[0] + (size_t)y * f->linesize[0]);
			uint16_t* d = (uint16_t*)(planar->data[0] + (size_t)y * planar->linesize[0]);
			for (int x = 0; x < f->width; x++) d[x] = (uint16_t)(s[x] >> 6); // the 10 bits sit in the top of 16
		}
	}
	for (int y = 0; y < ch; y++) {
		if (!sp->high) {
			const uint8_t* s = f->data[1] + (size_t)y * f->linesize[1];
			uint8_t *u = planar->data[1] + (size_t)y * planar->linesize[1], *v = planar->data[2] + (size_t)y * planar->linesize[2];
			for (int x = 0; x < cw; x++) { u[x] = s[2 * x]; v[x] = s[2 * x + 1]; }
		} else {
			const uint16_t* s = (const uint16_t*)(f->data[1] + (size_t)y * f->linesize[1]);
			uint16_t *u = (uint16_t*)(planar->data[1] + (size_t)y * planar->linesize[1]), *v = (uint16_t*)(planar->data[2] + (size_t)y * planar->linesize[2]);
			for (int x = 0; x < cw; x++) { u[x] = (uint16_t)(s[2 * x] >> 6); v[x] = (uint16_t)(s[2 * x + 1] >> 6); }
		}
	}
	return planar;
}

// Whether two pictures hold the same samples, whatever the memory layout (NV12 against planar, say).
bool same_pictures(const Picture& a, const Picture& b) {
	if (a.width != b.width || a.height != b.height || a.fmt.chroma != b.fmt.chroma || a.fmt.depth != b.fmt.depth) return false;
	std::vector<uint16_t> ra, rb;
	for (int c = 0; c < 3; c++) {
		const bool sub = c > 0 && !a.fmt.rgb;
		const int w = sub ? (a.width + (1 << a.layout.log2_cw) - 1) >> a.layout.log2_cw : a.width;
		const int h = sub ? (a.height + (1 << a.layout.log2_ch) - 1) >> a.layout.log2_ch : a.height;
		const Layout::Comp &ca = a.layout.comp[c], &cb = b.layout.comp[c];
		if (ca.plane == cb.plane && ca.offset == cb.offset && ca.shift == cb.shift && ca.depth == cb.depth && a.layout.step[ca.plane] == b.layout.step[cb.plane] && a.layout.step[ca.plane] == (ca.depth + ca.shift > 8 ? 2 : 1)) {
			// the same layout, one component per plane: whole rows can be compared as they are
			const size_t bytes = (size_t)w * a.layout.step[ca.plane];
			bool equal = true;
			for (int y = 0; y < h && equal; y++) equal = memcmp(a.data[ca.plane] + (size_t)y * a.stride[ca.plane], b.data[cb.plane] + (size_t)y * b.stride[cb.plane], bytes) == 0;
			if (!equal) return false;
			continue;
		}
		ra.resize((size_t)w); rb.resize((size_t)w);
		auto row = [&](const Picture& p, int y, std::vector<uint16_t>& out) {
			const Layout::Comp& cm = p.layout.comp[c];
			const uint8_t* line = p.data[cm.plane] + (size_t)y * p.stride[cm.plane] + cm.offset;
			const int step = p.layout.step[cm.plane];
			if (cm.depth + cm.shift > 8) for (int x = 0; x < w; x++) out[x] = (uint16_t)((line[(size_t)x * step] | line[(size_t)x * step + 1] << 8) >> cm.shift);
			else for (int x = 0; x < w; x++) out[x] = (uint16_t)(line[(size_t)x * step] >> cm.shift);
		};
		for (int y = 0; y < h; y++) {
			row(a, y, ra);
			row(b, y, rb);
			if (ra != rb) return false;
		}
	}
	return true;
}

// One row of component `c` as plain sample values, whatever the memory layout.
static void component_row(const Picture& p, int c, int y, int w, std::vector<uint16_t>& out) {
	const Layout::Comp& cm = p.layout.comp[c];
	const uint8_t* line = p.data[cm.plane] + (size_t)y * p.stride[cm.plane] + cm.offset;
	const int step = p.layout.step[cm.plane];
	out.resize((size_t)w);
	if (cm.depth + cm.shift > 8) for (int x = 0; x < w; x++) out[(size_t)x] = (uint16_t)((line[(size_t)x * step] | line[(size_t)x * step + 1] << 8) >> cm.shift);
	else for (int x = 0; x < w; x++) out[(size_t)x] = (uint16_t)(line[(size_t)x * step] >> cm.shift);
}

// Says how two pictures that are not the same differ, one line per component: how many samples, by how much and where. With VES_DUMP_FRAME=<folder> the pictures and
// their difference (times 16) are also written there as .pgm files, one per component, to look at.
void describe_difference(const Picture& a, const Picture& b, int frame) {
	fprintf(stderr, "  frame %d: software %dx%d %d-bit chroma %d, range %d matrix %d; hardware %dx%d %d-bit chroma %d, range %d matrix %d\n", frame, a.width, a.height, a.fmt.depth, a.fmt.chroma, a.color.range, a.color.space,
	        b.width, b.height, b.fmt.depth, b.fmt.chroma, b.color.range, b.color.space);
	if (a.width != b.width || a.height != b.height || a.fmt.chroma != b.fmt.chroma || a.fmt.depth != b.fmt.depth) return;
	const char* dump = getenv("VES_DUMP_FRAME");
	static const char* names[] = {"Y", "U", "V"};
	std::vector<uint16_t> ra, rb;
	for (int c = 0; c < 3; c++) {
		const bool sub = c > 0 && !a.fmt.rgb;
		const int w = sub ? (a.width + (1 << a.layout.log2_cw) - 1) >> a.layout.log2_cw : a.width;
		const int h = sub ? (a.height + (1 << a.layout.log2_ch) - 1) >> a.layout.log2_ch : a.height;
		long long different = 0, total = (long long)w * h, ones = 0, sum = 0;
		int biggest = 0, first_x = -1, first_y = -1, first_a = 0, first_b = 0;
		std::vector<uint8_t> pa, pb, pd;
		const int shift = a.fmt.depth > 8 ? a.fmt.depth - 8 : 0;
		for (int y = 0; y < h; y++) {
			component_row(a, c, y, w, ra);
			component_row(b, c, y, w, rb);
			for (int x = 0; x < w; x++) {
				const int d = std::abs((int)ra[(size_t)x] - (int)rb[(size_t)x]);
				if (d) {
					if (!different) { first_x = x; first_y = y; first_a = ra[(size_t)x]; first_b = rb[(size_t)x]; }
					different++; sum += d; ones += d == 1; biggest = std::max(biggest, d);
				}
				if (dump) { pa.push_back((uint8_t)(ra[(size_t)x] >> shift)); pb.push_back((uint8_t)(rb[(size_t)x] >> shift)); pd.push_back((uint8_t)std::min(255, d * 16)); }
			}
		}
		if (different) fprintf(stderr, "  %s: %lld of %lld samples differ (%.2f%%), largest difference %d, mean %.2f, %lld of them by 1; first at x=%d y=%d (software %d, hardware %d)\n", names[c], different, total,
		                       100.0 * different / total, biggest, (double)sum / different, ones, first_x, first_y, first_a, first_b);
		else fprintf(stderr, "  %s: the same\n", names[c]);
		if (dump) {
			const std::pair<const char*, const std::vector<uint8_t>*> files[] = {{"software", &pa}, {"hardware", &pb}, {"difference", &pd}};
			for (auto& [what, data] : files) {
				const std::string file = std::string(dump) + "/" + what + "_" + names[c] + ".pgm";
				if (FILE* f = fopen(file.c_str(), "wb")) {
					fprintf(f, "P5 %d %d 255\n", w, h);
					fwrite(data->data(), 1, data->size(), f);
					fclose(f);
				}
			}
		}
	}
	if (dump) fprintf(stderr, "  the frames are in %s\n", dump);
}
