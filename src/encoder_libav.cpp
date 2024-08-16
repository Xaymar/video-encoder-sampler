#include "encoder.h"

#include <cstdio>

extern "C" {
#include <libavcodec/avcodec.h>
#include <libavutil/opt.h>
}

namespace {

class Libav : public Encoder {
	std::string name;
	AVCodecContext* ctx = nullptr;
	AVPacket* pkt = av_packet_alloc();
	Sink sink;

public:
	explicit Libav(std::string n) : name(std::move(n)) {}
	~Libav() override { avcodec_free_context(&ctx); av_packet_free(&pkt); }

	bool open(const Params& p, Sink s) override {
		sink = std::move(s);
		const AVCodec* c = avcodec_find_encoder_by_name(name.c_str());
		if (!c) { fprintf(stderr, "no libavcodec encoder '%s'\n", name.c_str()); return false; }
		ctx = avcodec_alloc_context3(c);
		ctx->width = p.width; ctx->height = p.height;
		ctx->framerate = p.framerate; ctx->time_base = av_inv_q(p.framerate);
		ctx->bit_rate = p.bitrate; ctx->gop_size = p.gop;
		ctx->pix_fmt = AV_PIX_FMT_YUV420P; // shortcut: 8-bit 4:2:0 only, no conversion; add swscale when 10-bit tests land
		for (auto& [k, v] : p.options) {
			if (av_opt_set(ctx, k.c_str(), v.c_str(), AV_OPT_SEARCH_CHILDREN) < 0) {
				fprintf(stderr, "%s: bad option %s=%s\n", name.c_str(), k.c_str(), v.c_str());
				return false;
			}
		}
		int r = avcodec_open2(ctx, c, nullptr);
		if (r < 0) { char b[AV_ERROR_MAX_STRING_SIZE]; fprintf(stderr, "%s: open: %s\n", name.c_str(), av_make_error_string(b, sizeof b, r)); return false; }
		return true;
	}

	bool encode(const AVFrame* f) override {
		if (f && f->format != ctx->pix_fmt) { fprintf(stderr, "%s: input must be yuv420p\n", name.c_str()); return false; }
		if (avcodec_send_frame(ctx, f) < 0) return false;
		int r;
		while ((r = avcodec_receive_packet(ctx, pkt)) >= 0) {
			stats.packets++; stats.bytes += pkt->size;
			if (sink) sink(pkt->data, pkt->size, pkt->pts, pkt->flags & AV_PKT_FLAG_KEY);
			av_packet_unref(pkt);
		}
		return r == AVERROR(EAGAIN) || r == AVERROR_EOF;
	}
};

} // namespace

std::unique_ptr<Encoder> make_encoder_libav(const std::string& name) { return std::make_unique<Libav>(name); }
