#pragma once
// Fragmented MP4 (libavformat) written packet by packet as the encoder produces them, and read back for scoring. The moov box comes
// first, then one fragment per key frame. Files are accessed through large buffers; muxing runs on a thread of its own so that a
// slow disk does not hold the encoder up, and only a bounded number of packets wait in memory.
#include <atomic>
#include <condition_variable>
#include <cstdint>
#include <deque>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

#include "picture.h"

struct AVFormatContext;
struct AVStream;

class Mp4Writer {
	struct Impl;
	struct Item {
		std::vector<uint8_t> data;
		int64_t pts, dts;
		bool key;
	};
	static constexpr size_t MAX_QUEUED = 32u << 20; // bytes of packets waiting for the disk

	Impl* impl;
	AVFormatContext* oc = nullptr;
	AVStream* st = nullptr;
	std::string path_;
	Codec codec_ = Codec::None;
	Shape enc_;
	Color color_;
	int hint_ = 0;                // B-frame depth, for encoders that report no decode times
	bool scratch_ = false;
	bool started = false, reported = false, warned = false;
	int64_t shift = 0, index = 0, last_dts = -1;

	std::thread th;
	std::mutex m;
	std::condition_variable cv;
	std::deque<Item> queue;
	size_t queued = 0;
	bool closing = false;
	std::atomic<bool> failed{false};

	bool start(const uint8_t* first, size_t size);
	bool write(const Item& it);
	void run();

public:
	Mp4Writer();
	~Mp4Writer();
	// `delay_hint` is Encoder::reorder_delay(); it is only used when the packets carry no DTS. A scratch file is one that is deleted
	// after scoring: it is marked temporary, so the system keeps it in memory unless that runs short.
	bool open(const std::string& path, Codec codec, const Shape& enc, const Color& color, int delay_hint, bool scratch);
	// Packets in coding order (H.264 and HEVC as Annex B, AV1 as temporal units). `dts` is the encoder's decode time, or INT64_MIN.
	bool add(const uint8_t* data, size_t size, int64_t pts, bool key, int64_t dts);
	bool close();
};

class Mp4Reader {
	struct Impl;
	Impl* impl;
	AVFormatContext* ic = nullptr;
	std::vector<uint8_t> extra_;

public:
	Mp4Reader();
	~Mp4Reader();
	bool open(const std::string& path);
	const std::vector<uint8_t>& extradata() const { return extra_; } // avcC, hvcC or av1C: the decoder needs it for this container's packets
	bool next(std::vector<uint8_t>& out);                              // packets in decode order; false at the end
};
