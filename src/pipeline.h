#pragma once
// Decoding, converting, encoding and scoring, arranged so that the expensive software decode of an input is done once for
// all the encoders that can share it. Runs that need the very same frames (same input, resolution, frame rate, pixel format)
// are encoded together as a "wave": one thread decodes and converts, every encoder has a thread that takes the frames from a
// queue of its own, and the decoder stays at most DECODE_AHEAD frames ahead of the slowest encoder. Scoring works the same way:
// one thread produces the original frames in the scoring format and every scorer takes them from its own queue.
#include <condition_variable>
#include <functional>
#include <mutex>
#include <string>
#include <vector>

#include "convert.h"
#include "encoder.h"
#include "picture.h"
#include "ui.h"
#include "vmaf.h"

// Count, size and a hash of an encoder's packets: equal hashes, equal streams.
struct OutputInfo {
	long long count = 0, bytes = 0;
	uint64_t hash = 1469598103934665603ull; // FNV-1a over every packet and its size
	void add(const uint8_t* d, size_t n) {
		for (int i = 0; i < 8; i++) hash = (hash ^ (uint8_t)((uint64_t)n >> (8 * i))) * 1099511628211ull;
		for (size_t i = 0; i < n; i++) hash = (hash ^ d[i]) * 1099511628211ull;
		count++;
		bytes += (long long)n;
	}
};

// How many converted frames may wait in front of an encoder (or scorer).
constexpr size_t DECODE_AHEAD = 8;

// What is needed to know about an input before decoding any of it.
struct SourceInfo {
	int width = 0, height = 0;
	Fmt fmt;
	Rational framerate;
	Color color;
	double duration = 0;       // seconds
	std::string decoder;       // what decodes the original (software, or which hardware)
	long long frame_count = 0; // 0 if the file carries no usable timestamps
};
bool source_info(const std::string& path, SourceInfo& out); // opens the file once and remembers the answer

// How one run is encoded, worked out from the input's properties alone.
struct Plan {
	bool ignored = false; // cannot work with this input or encoder; not an error
	bool expected = false; // ignored only because the input is smaller than the request (upscaling off, a higher frame rate): known beforehand, so not reported run by run
	std::string reason;
	Shape enc;            // resolution, frame rate and pixel format the frames are encoded at
	Fmt score_fmt;        // both sides are scored in this format
	bool convert_rate = false, rescale = false;
	int range = 0;        // the range the encoder gets: 0 the source's, 1 limited, 2 full (the "range" option)
	Color color;          // the colour of the frames the encoder gets: what the stream is tagged with
	Options options;      // the encoder's options, without chroma and bitdepth
	std::string feed_key() const; // runs with equal keys (and the same input) can share one decode
};
Plan make_plan(const SourceInfo& src, const std::string& encoder, const std::string& codec, const Options& options, const Shape& want);

struct EncodeJob {
	std::string encoder, codec;
	const Plan* plan = nullptr;
	int64_t bitrate = 0; // bits per second
	double kf = 2.0;     // key frame interval, seconds
	RunProgress* rp = nullptr;
	// results
	std::string file;          // the fragmented MP4 the packets are written to as they arrive; empty: they are only counted and hashed
	Codec id = Codec::None;
	Color color;
	OutputInfo output;         // what the encoder produced
	std::function<bool(const uint8_t*, size_t)> tee; // also gets every packet as it arrives (for scoring); false once nobody wants them
	double kbps = 0;
	bool ok = false;
};

// The frame map grows while the video is read, and scoring follows it.
class ShownMap {
	std::mutex m;
	std::condition_variable cv;
	std::vector<uint32_t> v;
	bool done = false;

public:
	void push(uint32_t n) { std::lock_guard lock(m); v.push_back(n); cv.notify_all(); }
	void finish() { std::lock_guard lock(m); done = true; cv.notify_all(); }
	// The encoded frame on screen at source frame k; waits until that is known. False once k is past the last frame.
	bool get(size_t k, uint32_t& n) {
		std::unique_lock lock(m);
		cv.wait(lock, [&] { return k < v.size() || done; });
		if (k >= v.size()) return false;
		n = v[k];
		return true;
	}
};

// What decoding and converting leaves for scoring; the same for every job of a wave.
struct FeedResult {
	ShownMap shown;       // shown[n]: the encoded frame that is on screen at source frame n
	long long frames = 0; // source frames
	bool ok = false;
};

// A run of a wave: its encoder, and the scoring that follows the encoder's output while it is still being written.
struct WaveRun {
	EncodeJob enc;
	Codec codec = Codec::None;
	RunProgress* score_rp = nullptr;
	std::vector<double> scores;
	bool score_ok = false;
};

// Encodes all runs (which must share a feed key) from one decode of `input`, and scores each against the original at the same time.
// Runs that compare in the same format share one decode of the original.
bool run_wave(const std::string& input, const SourceInfo& si, std::vector<WaveRun*>& runs, const ScaleOptions& so, const VmafOptions& vo, RunProgress* feed_rp, FeedResult& fr);
