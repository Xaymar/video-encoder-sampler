#include "source.h"

#include <algorithm>
#include <cstdio>
#include <cstdlib>

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
	tb = st->time_base;
	scan(path);
	range = st->codecpar->color_range;
	space = st->codecpar->color_space;
	primaries = st->codecpar->color_primaries;
	trc = st->codecpar->color_trc;
	bool full = false;
	pix = unj(dec->pix_fmt, full);
	if (full) range = AVCOL_RANGE_JPEG;
	return true;
}

// Reads every video packet's timestamp (no decoding) to learn the real frame times.
void Source::scan(const std::string& path) {
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
	const double fps = av_q2d(framerate), unit = av_q2d(tb);
	double worst = 0;
	for (size_t i = 0; i < stamps.size(); i++) worst = std::max(worst, std::abs(stamps[i] * unit - i / fps));
	if (worst < 0.5 / fps) return;
	vfr = true;
	framerate = av_d2q((stamps.size() - 1) / (stamps.back() * unit), 1000000);
}

double Source::duration() const {
	if (stamps.size() < 2) return n / av_q2d(framerate);
	// the last frame lasts as long as the average one
	return (stamps.back() + (double)stamps.back() / (stamps.size() - 1)) * av_q2d(tb);
}

int64_t Source::slot(long long index, AVRational rate) const {
	if (index >= (long long)stamps.size()) return index * rate.num * framerate.den / ((int64_t)rate.den * framerate.num);
	// Timestamps are rounded to whole ticks, which can leave a frame just below its slot boundary (33 ms for 33.3 ms at
	// 30 fps); one tick of slack puts such frames in the slot they belong to.
	return (stamps[index] + 1) * tb.num * rate.num / ((int64_t)tb.den * rate.den);
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
