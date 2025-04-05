// Source on top of libavformat / libavcodec: any container and codec FFmpeg reads.
#include "source.h"

#include <algorithm>
#include <cstdio>
#include <cstdlib>

#include "ff.h"

extern "C" {
#include <libavcodec/avcodec.h>
#include <libavformat/avformat.h>
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

class FfSource : public Source {
	AVFormatContext* ctx = nullptr;
	AVCodecContext* dec = nullptr;
	AVPacket* pkt = av_packet_alloc();
	AVFrame* frame = av_frame_alloc();
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

public:
	~FfSource() override {
		av_frame_free(&frame);
		av_packet_free(&pkt);
		avcodec_free_context(&dec);
		avformat_close_input(&ctx);
	}

	bool open(const std::string& path) override {
		char b[AV_ERROR_MAX_STRING_SIZE];
		int r = avformat_open_input(&ctx, path.c_str(), nullptr, nullptr);
		if (r < 0) { fprintf(stderr, "open '%s': %s\n", path.c_str(), av_make_error_string(b, sizeof b, r)); return false; }
		if ((r = avformat_find_stream_info(ctx, nullptr)) < 0) { fprintf(stderr, "stream info: %s\n", av_make_error_string(b, sizeof b, r)); return false; }
		vi = av_find_best_stream(ctx, AVMEDIA_TYPE_VIDEO, -1, -1, nullptr, 0);
		if (vi < 0) { fprintf(stderr, "'%s' has no video stream\n", path.c_str()); return false; }
		AVStream* st = ctx->streams[vi];
		const AVCodec* c = software_decoder(st->codecpar->codec_id);
		if (!c) { fprintf(stderr, "no software decoder for this video\n"); return false; }
		dec = avcodec_alloc_context3(c);
		avcodec_parameters_to_context(dec, st->codecpar);
		if (avcodec_open2(dec, c, nullptr) < 0) { fprintf(stderr, "decoder open failed\n"); return false; }
		width = dec->width; height = dec->height;
		framerate = from_av(av_guess_frame_rate(ctx, st, nullptr));
		tb = from_av(st->time_base);
		color = {(int)st->codecpar->color_range, (int)st->codecpar->color_space, (int)st->codecpar->color_primaries, (int)st->codecpar->color_trc};
		bool full = false;
		AVPixelFormat native = unj(dec->pix_fmt, full);
		if (full) color.range = AVCOL_RANGE_JPEG;
		if (!describe_native(native, fmt)) { fprintf(stderr, "unsupported source pixel format %s\n", av_get_pix_fmt_name(native)); return false; }
		scan(path);
		return true;
	}

	int next(Picture& out) override {
		for (;;) {
			int r = avcodec_receive_frame(dec, frame);
			if (r >= 0) {
				bool full = false;
				frame->format = unj((AVPixelFormat)frame->format, full);
				if (full) frame->color_range = AVCOL_RANGE_JPEG;
				frame->pts = read++;
				frame->duration = 1;
				frame->pict_type = AV_PICTURE_TYPE_NONE;
				out = picture_from_frame(frame, fmt);
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
};

} // namespace

std::unique_ptr<Source> open_source(const std::string& path) {
	auto s = std::make_unique<FfSource>();
	return s->open(path) ? std::unique_ptr<Source>(std::move(s)) : nullptr;
}
