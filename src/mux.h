#pragma once
// Fragmented MP4 (libavformat) written packet by packet as the encoder produces them, and read back for scoring. The moov box comes
// first, then one fragment per key frame; only the packet being handled is in memory.
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "picture.h"

struct AVFormatContext;
struct AVStream;

class Mp4Writer {
	AVFormatContext* oc = nullptr;
	AVStream* st = nullptr;
	std::string path_;
	Codec codec_ = Codec::None;
	Shape enc_;
	Color color_;
	int hint_ = 0;                // B-frame depth, for encoders that report no decode times
	bool started = false, reported = false, warned = false, failed = false;
	int64_t shift = 0, index = 0, last_dts = -1;

	bool start(const uint8_t* first, size_t size);

public:
	~Mp4Writer();
	// `delay_hint` is Encoder::reorder_delay(); it is only used when the packets carry no DTS.
	bool open(const std::string& path, Codec codec, const Shape& enc, const Color& color, int delay_hint);
	// Packets in coding order (H.264 and HEVC as Annex B, AV1 as temporal units). `dts` is the encoder's decode time, or INT64_MIN.
	bool add(const uint8_t* data, size_t size, int64_t pts, bool key, int64_t dts);
	bool close();
};

class Mp4Reader {
	AVFormatContext* ic = nullptr;
	std::vector<uint8_t> extra_;

public:
	~Mp4Reader();
	bool open(const std::string& path);
	const std::vector<uint8_t>& extradata() const { return extra_; } // avcC, hvcC or av1C: the decoder needs it for this container's packets
	bool next(std::vector<uint8_t>& out);                              // packets in decode order; false at the end
};
