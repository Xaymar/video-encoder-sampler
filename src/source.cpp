#include "source.h"

#include <cstdio>

// The deprecated "J" formats are plain planar YUV with full range; the range tag says so instead and the data is identical.
static AVPixelFormat unj(AVPixelFormat f, bool& full) {
	full = true;
	switch (f) {
	case AV_PIX_FMT_YUVJ420P: return AV_PIX_FMT_YUV420P;
	case AV_PIX_FMT_YUVJ422P: return AV_PIX_FMT_YUV422P;
	case AV_PIX_FMT_YUVJ444P: return AV_PIX_FMT_YUV444P;
	case AV_PIX_FMT_YUVJ440P: return AV_PIX_FMT_YUV440P;
	default: full = false; return f;
	}
}

Source::~Source() {
	avcodec_free_context(&dec);
	avformat_close_input(&fmt);
	av_packet_free(&pkt);
}

bool Source::open(const std::string& path) {
	char b[AV_ERROR_MAX_STRING_SIZE];
	int r = avformat_open_input(&fmt, path.c_str(), nullptr, nullptr);
	if (r < 0) { fprintf(stderr, "open '%s': %s\n", path.c_str(), av_make_error_string(b, sizeof b, r)); return false; }
	if ((r = avformat_find_stream_info(fmt, nullptr)) < 0) { fprintf(stderr, "stream info: %s\n", av_make_error_string(b, sizeof b, r)); return false; }
	vi = av_find_best_stream(fmt, AVMEDIA_TYPE_VIDEO, -1, -1, nullptr, 0);
	if (vi < 0) { fprintf(stderr, "'%s' has no video stream\n", path.c_str()); return false; }
	AVStream* st = fmt->streams[vi];
	const AVCodec* c = avcodec_find_decoder(st->codecpar->codec_id);
	dec = avcodec_alloc_context3(c);
	avcodec_parameters_to_context(dec, st->codecpar);
	if (avcodec_open2(dec, c, nullptr) < 0) { fprintf(stderr, "decoder open failed\n"); return false; }
	width = dec->width; height = dec->height;
	framerate = av_guess_frame_rate(fmt, st, nullptr);
	range = st->codecpar->color_range;
	space = st->codecpar->color_space;
	primaries = st->codecpar->color_primaries;
	trc = st->codecpar->color_trc;
	bool full = false;
	pix = unj(dec->pix_fmt, full);
	if (full) range = AVCOL_RANGE_JPEG;
	return true;
}

int Source::next(AVFrame* out) {
	for (;;) {
		int r = avcodec_receive_frame(dec, out);
		if (r >= 0) {
			// The deprecated "J" format is plain yuv420p with full range; say so with the tag instead, the data is identical.
			bool full = false;
			out->format = unj((AVPixelFormat)out->format, full);
			if (full) out->color_range = AVCOL_RANGE_JPEG;
			out->pts = n++; out->pict_type = AV_PICTURE_TYPE_NONE;
			return 1;
		}
		if (r == AVERROR_EOF) return 0;
		if (r != AVERROR(EAGAIN)) return r;
		if (flushed) return 0;
		if (av_read_frame(fmt, pkt) < 0) { avcodec_send_packet(dec, nullptr); flushed = true; continue; }
		if (pkt->stream_index == vi) avcodec_send_packet(dec, pkt);
		av_packet_unref(pkt);
	}
}
