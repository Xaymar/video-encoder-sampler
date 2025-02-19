// Generic fallback: any encoder libavcodec offers, by its FFmpeg name (e.g. libx264). The only encoder backend that
// depends on FFmpeg; the native backends do not.
#include "encoder.h"

#include <cstdio>

#include "ff.h"

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
	Fmt pix;

public:
	explicit Libav(std::string n) : name(std::move(n)) {}
	~Libav() override { avcodec_free_context(&ctx); av_packet_free(&pkt); }

	bool open(const Params& p, Sink s) override {
		sink = std::move(s);
		pix = p.pix;
		const AVCodec* c = avcodec_find_encoder_by_name(name.c_str());
		if (!c) { fprintf(stderr, "no libavcodec encoder '%s'\n", name.c_str()); return false; }
		ctx = avcodec_alloc_context3(c);
		ctx->width = p.width; ctx->height = p.height;
		ctx->framerate = to_av(p.framerate); ctx->time_base = to_av(p.framerate.inverse());
		ctx->bit_rate = p.bitrate; ctx->gop_size = p.gop;
		ctx->pix_fmt = av_planar_format(p.pix.chroma, p.pix.depth);
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

	bool encode(const Picture* p) override {
		AVFrame view;
		if (p) {
			if (p->fmt != pix) { fprintf(stderr, "%s: input must be %d-bit %d\n", name.c_str(), pix.depth, pix.chroma); return false; }
			frame_view(*p, view);
			view.format = ctx->pix_fmt;
		}
		if (avcodec_send_frame(ctx, p ? &view : nullptr) < 0) return false;
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

bool libav_encoder_supports(const std::string& encoder, int chroma, int depth) {
	const AVCodec* c = avcodec_find_encoder_by_name(encoder.c_str());
	if (!c) return false;
	const AVPixelFormat* fmts = nullptr;
	int n = 0;
	if (avcodec_get_supported_config(nullptr, c, AV_CODEC_CONFIG_PIX_FORMAT, 0, (const void**)&fmts, &n) < 0 || !fmts) return true;
	for (int i = 0; i < n; i++) if (fmts[i] == av_planar_format(chroma, depth)) return true;
	return false;
}

Codec libav_encoder_codec(const std::string& encoder) {
	const AVCodec* c = avcodec_find_encoder_by_name(encoder.c_str());
	if (!c) return Codec::None;
	return c->id == AV_CODEC_ID_H264 ? Codec::H264 : c->id == AV_CODEC_ID_HEVC ? Codec::HEVC : c->id == AV_CODEC_ID_AV1 ? Codec::AV1 : Codec::None;
}
