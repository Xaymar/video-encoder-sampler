// Source on top of libavformat / libavcodec: any container and codec FFmpeg reads.
// Decoding is in software unless hardware decoding was asked for (set_hw_decode()). Then the source is decoded by the GPU, but only after
// its first frames have been checked against a software decode of the same file and found identical; anything else falls back to software.
#include "source.h"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <mutex>

#include "ff.h"

extern "C" {
#include <libavcodec/avcodec.h>
#include <libavformat/avformat.h>
#include <libavutil/hwcontext.h>
}

namespace {

// The deprecated "J" formats are plain planar YUV with full range; the range tag says so instead and the data is identical.
AVPixelFormat unj(AVPixelFormat f, bool& full) {
	full = true;
	switch (f) {
	case AV_PIX_FMT_YUVJ420P: return AV_PIX_FMT_YUV420P;
	case AV_PIX_FMT_YUVJ422P: return AV_PIX_FMT_YUV422P;
	case AV_PIX_FMT_YUVJ444P: return AV_PIX_FMT_YUV444P;
	case AV_PIX_FMT_YUVJ440P: return AV_PIX_FMT_YUV440P;
	default: full = false; return f;
	}
}

constexpr int VERIFY_FRAMES = 120;

// Semi-planar formats (NV12, P010, ...) keep U and V interleaved in one plane. The samples are unpacked to planar YUV exactly, on the CPU:
// the conversion libplacebo would do instead is not exact, it changes sample values by a little even when only the memory layout differs.
// This also makes a GPU's decode (which gives NV12/P010 back) reach the encoders the way a software decode does.
struct SemiPlanar { AVPixelFormat from, to; int sx, sy; bool high; };
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

class FfSource : public Source {
	AVFormatContext* ctx = nullptr;
	AVCodecContext* dec = nullptr;
	AVPacket* pkt = av_packet_alloc();
	AVFrame* frame = av_frame_alloc();
	AVFrame* sw_frame = av_frame_alloc(); // where a hardware frame is downloaded to
	AVFrame* planar_frame = av_frame_alloc(); // and where it is unpacked to if the download has interleaved chroma
	AVBufferRef* hw_dev = nullptr;
	AVPixelFormat hw_fmt = AV_PIX_FMT_NONE;
	bool hw_used = false;   // a frame really came from the hardware decoder
	bool fmt_known = true;  // false for hardware decoding until the first frame shows what the download looks like
	Picture pending;        // the first frame, read early to learn the format
	bool has_pending = false;
	int vi = -1;
	bool flushed = false;

	// Reads every video packet's timestamp (no decoding) to learn the real frame times.
	void scan(const std::string& path) {
		AVFormatContext* sc = nullptr;
		if (avformat_open_input(&sc, path.c_str(), nullptr, nullptr) < 0) return;
		AVPacket* pk = av_packet_alloc();
		bool missing = false;
		while (av_read_frame(sc, pk) >= 0) {
			if (pk->stream_index == vi) {
				int64_t ts = pk->pts != AV_NOPTS_VALUE ? pk->pts : pk->dts;
				if (ts == AV_NOPTS_VALUE) missing = true; else stamps.push_back(ts);
			}
			av_packet_unref(pk);
		}
		av_packet_free(&pk);
		avformat_close_input(&sc);
		if (missing || stamps.size() < 2) { stamps.clear(); return; } // raw streams and the like: assume constant rate
		std::sort(stamps.begin(), stamps.end());
		const int64_t first = stamps[0];
		for (auto& t : stamps) t -= first;

		// Constant frame rate if every frame sits within half a frame of where `framerate` puts it.
		const double fps = framerate.value(), unit = tb.value();
		double worst = 0;
		for (size_t i = 0; i < stamps.size(); i++) worst = std::max(worst, std::abs(stamps[i] * unit - i / fps));
		if (worst < 0.5 / fps) return;
		vfr = true;
		framerate = from_av(av_d2q((stamps.size() - 1) / (stamps.back() * unit), 1000000));
	}

	static AVPixelFormat pick_format(AVCodecContext* c, const AVPixelFormat* fmts) {
		const AVPixelFormat want = (AVPixelFormat)(intptr_t)c->opaque;
		for (const AVPixelFormat* p = fmts; *p != AV_PIX_FMT_NONE; p++) if (*p == want) return *p;
		return fmts[0]; // the hardware cannot do this stream: decode in software after all
	}

	// Fills `out` from the next decoded frame, downloading it first if the GPU decoded it.
	int pull(Picture& out) {
		for (;;) {
			int r = avcodec_receive_frame(dec, frame);
			if (r >= 0) {
				AVFrame* f = frame;
				if (hw_fmt != AV_PIX_FMT_NONE && frame->format == hw_fmt) {
					av_frame_unref(sw_frame);
					if (av_hwframe_transfer_data(sw_frame, frame, 0) < 0) return AVERROR_EXTERNAL;
					av_frame_copy_props(sw_frame, frame);
					av_frame_unref(frame);
					f = sw_frame;
					hw_used = true;
				}
				f = unpack_semi_planar(f, planar_frame);
				bool full = false;
				f->format = unj((AVPixelFormat)f->format, full);
				if (full) f->color_range = AVCOL_RANGE_JPEG;
				if (!fmt_known) {
					if (!describe_native((AVPixelFormat)f->format, fmt)) { fprintf(stderr, "unsupported source pixel format %s\n", av_get_pix_fmt_name((AVPixelFormat)f->format)); return AVERROR_INVALIDDATA; }
					fmt_known = true;
				}
				f->pts = read++;
				f->duration = 1;
				f->pict_type = AV_PICTURE_TYPE_NONE;
				out = picture_from_frame(f, fmt);
				return 1;
			}
			if (r == AVERROR_EOF) return 0;
			if (r != AVERROR(EAGAIN)) return r;
			if (flushed) return 0;
			if (av_read_frame(ctx, pkt) < 0) { avcodec_send_packet(dec, nullptr); flushed = true; continue; }
			if (pkt->stream_index == vi) avcodec_send_packet(dec, pkt);
			av_packet_unref(pkt);
		}
	}

public:
	std::string decoder_name;

	~FfSource() override {
		av_frame_free(&frame);
		av_frame_free(&sw_frame);
		av_frame_free(&planar_frame);
		av_packet_free(&pkt);
		avcodec_free_context(&dec);
		av_buffer_unref(&hw_dev);
		avformat_close_input(&ctx);
	}

	bool hardware() const { return hw_used; }

	// hw_type: the FFmpeg name of a hardware device type ("cuda", "d3d11va", ...), or empty for software.
	bool open(const std::string& path, const std::string& hw_type, bool do_scan) {
		char b[AV_ERROR_MAX_STRING_SIZE];
		int r = avformat_open_input(&ctx, path.c_str(), nullptr, nullptr);
		if (r < 0) { fprintf(stderr, "open '%s': %s\n", path.c_str(), av_make_error_string(b, sizeof b, r)); return false; }
		if ((r = avformat_find_stream_info(ctx, nullptr)) < 0) { fprintf(stderr, "stream info: %s\n", av_make_error_string(b, sizeof b, r)); return false; }
		vi = av_find_best_stream(ctx, AVMEDIA_TYPE_VIDEO, -1, -1, nullptr, 0);
		if (vi < 0) { fprintf(stderr, "'%s' has no video stream\n", path.c_str()); return false; }
		AVStream* st = ctx->streams[vi];
		const AVCodec* c = software_decoder(st->codecpar->codec_id);
		const AVCodec* dedicated = hw_type.empty() ? nullptr : avcodec_find_decoder_by_name(hw_type.c_str());
		if (dedicated) {
			if (dedicated->id != st->codecpar->codec_id) return false; // a decoder for some other codec
			c = dedicated;
		}
		if (!c) { fprintf(stderr, "no software decoder for this video\n"); return false; }
		dec = avcodec_alloc_context3(c);
		avcodec_parameters_to_context(dec, st->codecpar);
		if (dedicated) {
			dec->thread_count = 1; // the hardware does the work
			fmt_known = false;
		} else if (!hw_type.empty()) {
			const AVHWDeviceType type = av_hwdevice_find_type_by_name(hw_type.c_str());
			if (type == AV_HWDEVICE_TYPE_NONE) return false;
			for (int i = 0;; i++) { // the pixel format this decoder gives for that device type
				const AVCodecHWConfig* cfg = avcodec_get_hw_config(c, i);
				if (!cfg) return false;
				if ((cfg->methods & AV_CODEC_HW_CONFIG_METHOD_HW_DEVICE_CTX) && cfg->device_type == type) { hw_fmt = cfg->pix_fmt; break; }
			}
			{ // one device per type for the whole run: creating it is slow, and decoders can share it
				static std::mutex devices_mutex;
				static std::map<int, AVBufferRef*> devices;
				std::lock_guard lock(devices_mutex);
				AVBufferRef*& shared = devices[(int)type];
				if (!shared && av_hwdevice_ctx_create(&shared, type, nullptr, nullptr, 0) < 0) { shared = nullptr; return false; }
				hw_dev = av_buffer_ref(shared);
			}
			dec->hw_device_ctx = av_buffer_ref(hw_dev);
			dec->opaque = (void*)(intptr_t)hw_fmt;
			dec->get_format = &FfSource::pick_format;
			dec->thread_count = 1; // the GPU does the work
			fmt_known = false;
		} else {
			dec->thread_count = decode_threads(); // the library decodes on one thread unless told otherwise
			dec->thread_type = FF_THREAD_FRAME | FF_THREAD_SLICE;
		}
		if (avcodec_open2(dec, c, nullptr) < 0) { if (hw_type.empty()) fprintf(stderr, "decoder open failed\n"); return false; }
		width = dec->width; height = dec->height;
		framerate = from_av(av_guess_frame_rate(ctx, st, nullptr));
		tb = from_av(st->time_base);
		color = {(int)st->codecpar->color_range, (int)st->codecpar->color_space, (int)st->codecpar->color_primaries, (int)st->codecpar->color_trc};
		if (hw_type.empty()) {
			bool full = false;
			AVPixelFormat native = unj(dec->pix_fmt, full);
			if (const SemiPlanar* sp = semi_planar(native)) native = sp->to;
			if (full) color.range = AVCOL_RANGE_JPEG;
			if (!describe_native(native, fmt)) { fprintf(stderr, "unsupported source pixel format %s\n", av_get_pix_fmt_name(native)); return false; }
		} else {
			// The first frame tells what the download looks like (NV12, planar 4:4:4, ...) and whether the GPU took the stream at all.
			if (pull(pending) != 1) return false;
			if (dedicated) hw_used = true;
			if (!hw_used) return false;
			has_pending = true;
			if (pending.color.range == AVCOL_RANGE_JPEG) color.range = AVCOL_RANGE_JPEG;
		}
		if (do_scan) scan(path);
		decoder_name = std::string(c->name) + (hw_type.empty() ? " (software)" : dedicated ? " (hardware)" : " (" + hw_type + " hardware)");
		decoder = decoder_name;
		return true;
	}

	bool open(const std::string& path) override { return open(path, "", true); }

	int next(Picture& out) override {
		if (has_pending) {
			out = std::move(pending);
			has_pending = false;
			return 1;
		}
		return pull(out);
	}
};

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

// Decodes the first frames both ways and compares them.
bool verify_hardware(const std::string& path, const std::string& type) {
	FfSource sw, hw;
	if (!sw.open(path, "", false) || !hw.open(path, type, false)) return false;
	for (int i = 0; i < VERIFY_FRAMES; i++) {
		Picture a, b;
		const int ra = sw.next(a), rb = hw.next(b);
		if (ra != rb) return false;
		if (ra != 1) break;
		if (!same_pictures(a, b)) {
			fprintf(stderr, "hardware decoding (%s) differs from software at frame %d of '%s'\n", type.c_str(), i, path.c_str());
			return false;
		}
	}
	return true;
}

// The codec of the file's video stream, found once per file.
AVCodecID codec_id_of(const std::string& path) {
	static std::mutex m;
	static std::map<std::string, AVCodecID> known;
	std::lock_guard lock(m);
	if (auto it = known.find(path); it != known.end()) return it->second;
	AVCodecID id = AV_CODEC_ID_NONE;
	AVFormatContext* c = nullptr;
	if (avformat_open_input(&c, path.c_str(), nullptr, nullptr) >= 0) {
		if (avformat_find_stream_info(c, nullptr) >= 0) {
			const int v = av_find_best_stream(c, AVMEDIA_TYPE_VIDEO, -1, -1, nullptr, 0);
			if (v >= 0) id = c->streams[v]->codecpar->codec_id;
		}
		avformat_close_input(&c);
	}
	return known[path] = id;
}

std::mutex g_mutex;
std::string g_hw;                  // "", "auto" or a device type
std::map<std::string, bool> g_verdict; // path|type -> identical to software

} // namespace

void set_hw_decode(const std::string& type) {
	std::lock_guard lock(g_mutex);
	g_hw = type == "off" || type == "none" ? "" : type;
}

std::unique_ptr<Source> open_source(const std::string& path) {
	av_log_set_level(AV_LOG_ERROR); // warnings such as "UDTA parsing failed" repeat for every run that opens the file
	std::string want;
	{
		std::lock_guard lock(g_mutex);
		want = g_hw;
	}
	if (!want.empty()) {
		std::vector<std::string> types;
		if (want == "auto") {
			// The normal decoders on a hardware device first, then the dedicated hardware decoders of the stream's codec.
			types = {"cuda", "d3d11va", "d3d12va", "qsv", "vulkan"};
			const AVCodecID id = codec_id_of(path);
			for (const std::string suffix : {"_cuvid", "_qsv", "_amf"}) {
				void* it = nullptr;
				while (const AVCodec* c = av_codec_iterate(&it)) {
					const std::string name = c->name;
					if (av_codec_is_decoder(c) && c->id == id && (c->capabilities & AV_CODEC_CAP_HARDWARE) && name.size() > suffix.size() &&
					    name.compare(name.size() - suffix.size(), suffix.size(), suffix) == 0) types.push_back(name);
				}
			}
		} else types = {want};
		for (const auto& type : types) {
			const std::string key = path + "|" + type;
			bool known = false, good = false;
			{
				std::lock_guard lock(g_mutex);
				if (auto it = g_verdict.find(key); it != g_verdict.end()) { known = true; good = it->second; }
			}
			if (known && !good) continue;
			if (!known) {
				good = verify_hardware(path, type);
				if (good) fprintf(stderr, "hardware decoding (%s) matched software on the first %d frames of '%s'\n", type.c_str(), VERIFY_FRAMES, path.c_str());
				std::lock_guard lock(g_mutex);
				g_verdict[key] = good;
				if (!good) continue;
			}
			auto s = std::make_unique<FfSource>();
			if (s->open(path, type, true)) return s;
		}
		static std::mutex once;
		static std::map<std::string, bool> told;
		std::lock_guard lock(once);
		if (!told[path]) { told[path] = true; fprintf(stderr, "hardware decoding is not available for '%s': decoding in software\n", path.c_str()); }
	}
	auto s = std::make_unique<FfSource>();
	return s->open(path, "", true) ? std::unique_ptr<Source>(std::move(s)) : nullptr;
}
