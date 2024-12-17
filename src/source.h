#pragma once
#include <string>

extern "C" {
#include <libavcodec/avcodec.h>
#include <libavformat/avformat.h>
}

// Sequential video decoder. Frames come out in display order with pts == frame index.
class Source {
	AVFormatContext* fmt = nullptr;
	AVCodecContext* dec = nullptr;
	AVPacket* pkt = av_packet_alloc();
	int vi = -1;
	bool flushed = false;
	long long n = 0;

public:
	int width = 0, height = 0;
	AVPixelFormat pix = AV_PIX_FMT_NONE; // as delivered by next(); full range "J" formats are reported as their plain equivalent
	AVRational framerate{30, 1};
	// Colour description of the stream, as tagged in the file.
	AVColorRange range = AVCOL_RANGE_UNSPECIFIED;
	AVColorSpace space = AVCOL_SPC_UNSPECIFIED;
	AVColorPrimaries primaries = AVCOL_PRI_UNSPECIFIED;
	AVColorTransferCharacteristic trc = AVCOL_TRC_UNSPECIFIED;

	~Source();
	bool open(const std::string& path);
	// 1: frame produced, 0: end of stream, <0: error
	int next(AVFrame* out);
};
