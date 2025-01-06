#pragma once
#include <string>
#include <vector>

extern "C" {
#include <libavcodec/avcodec.h>
#include <libavformat/avformat.h>
}

// Sequential video decoder. Frames come out in display order with pts == frame index.
// The real presentation times come from a scan of the file's packet timestamps, so variable frame rate
// sources keep their timing: `framerate` is the measured average and `duration()` the real length.
class Source {
	AVFormatContext* fmt = nullptr;
	AVCodecContext* dec = nullptr;
	AVPacket* pkt = av_packet_alloc();
	int vi = -1;
	bool flushed = false;
	long long n = 0;
	std::vector<int64_t> stamps; // presentation time of every frame in display order, in `tb` ticks from the first frame
	AVRational tb{1, 1000};

	void scan(const std::string& path);

public:
	int width = 0, height = 0;
	AVPixelFormat pix = AV_PIX_FMT_NONE; // as delivered by next(); full range "J" formats are reported as their plain equivalent
	AVRational framerate{30, 1}; // constant frame rate files: the file's; variable frame rate files: the measured average
	bool vfr = false;            // the timestamps do not follow `framerate`
	// Colour description of the stream, as tagged in the file.
	AVColorRange range = AVCOL_RANGE_UNSPECIFIED;
	AVColorSpace space = AVCOL_SPC_UNSPECIFIED;
	AVColorPrimaries primaries = AVCOL_PRI_UNSPECIFIED;
	AVColorTransferCharacteristic trc = AVCOL_TRC_UNSPECIFIED;

	~Source();
	bool open(const std::string& path);
	// 1: frame produced, 0: end of stream, <0: error
	int next(AVFrame* out);

	long long frame_count() const { return (long long)stamps.size(); } // 0 if the file carries no usable timestamps
	double duration() const;                                            // seconds, including the last frame's own length
	// Which slot of a constant `rate` output timeline frame `index` falls in: floor(time * rate).
	int64_t slot(long long index, AVRational rate) const;
};
