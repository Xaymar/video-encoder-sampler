// Decoder on top of libavcodec.
#include "decode.h"

#include <cstdio>
#include <cstring>

#include "ff.h"

extern "C" {
#include <libavcodec/avcodec.h>
}

namespace {

class FfDecoder : public Decoder {
	AVCodecContext* ctx = nullptr;
	AVPacket* pkt = av_packet_alloc();
	AVFrame* frame = av_frame_alloc();

public:
	~FfDecoder() override {
		av_frame_free(&frame);
		av_packet_free(&pkt);
		avcodec_free_context(&ctx);
	}

	bool open(Codec codec) override {
		AVCodecID id = codec == Codec::H264 ? AV_CODEC_ID_H264 : codec == Codec::HEVC ? AV_CODEC_ID_HEVC : codec == Codec::AV1 ? AV_CODEC_ID_AV1 : AV_CODEC_ID_NONE;
		const AVCodec* c = id == AV_CODEC_ID_NONE ? nullptr : software_decoder(id);
		ctx = c ? avcodec_alloc_context3(c) : nullptr;
		// Without VUI reorder info the H.264 decoder guesses the B-frame depth and can drop or misorder early frames;
		// strict mode makes it assume the maximum, so every frame comes out in display order.
		if (ctx) ctx->strict_std_compliance = FF_COMPLIANCE_STRICT;
		if (!ctx || avcodec_open2(ctx, c, nullptr) < 0) { fprintf(stderr, "no decoder for encoded stream\n"); return false; }
		return true;
	}

	bool send(const uint8_t* data, size_t size) override {
		av_new_packet(pkt, (int)size);
		memcpy(pkt->data, data, size);
		int r = avcodec_send_packet(ctx, pkt);
		av_packet_unref(pkt);
		return r >= 0;
	}

	bool flush() override { return avcodec_send_packet(ctx, nullptr) >= 0; }

	int receive(Picture& out) override {
		int r = avcodec_receive_frame(ctx, frame);
		if (r < 0) return r == AVERROR(EAGAIN) || r == AVERROR_EOF ? 0 : r;
		Fmt fmt;
		if (!describe_native((AVPixelFormat)frame->format, fmt)) { av_frame_unref(frame); return -1; }
		out = picture_from_frame(frame, fmt);
		return 1;
	}
};

} // namespace

std::unique_ptr<Decoder> make_decoder() { return std::make_unique<FfDecoder>(); }
