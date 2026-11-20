#pragma once
// Decoding, converting, encoding and scoring, arranged so that the expensive software decode of an input is done once for
// all the encoders that can share it. Runs that need the very same frames (same input, resolution, frame rate, pixel format)
// are encoded together as a "wave": one thread decodes, one converts for the encoders, every encoder has a thread that takes the frames
// from a queue of its own, and the decoder stays at most DECODE_AHEAD frames ahead of the slowest encoder. The same decode serves the
// scoring: the original frames in each scoring format (the true score) and the frames the encoders got (the distorted score) are kept
// until every scorer has used them, and the decode waits for a scorer that falls far behind (see run_wave).
#include <condition_variable>
#include <cstring>
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
	// The same without the encoder's own settings text: x264 and x265 write their options into an SEI NAL unit, so two runs whose options changed
	// nothing in the pictures still differ there. Equal: the same pictures (what --check compares).
	uint64_t pictures = 1469598103934665603ull;
	void add(const uint8_t* d, size_t n) {
		for (int i = 0; i < 8; i++) hash = (hash ^ (uint8_t)((uint64_t)n >> (8 * i))) * 1099511628211ull;
		for (size_t i = 0; i < n; i++) hash = (hash ^ d[i]) * 1099511628211ull;
		count++;
		bytes += (long long)n;
		add_pictures(d, n);
	}
private:
	static bool has(const uint8_t* b, const uint8_t* e, const char* text) {
		const size_t len = strlen(text);
		for (; b + len <= e; b++) if (memcmp(b, text, len) == 0) return true;
		return false;
	}
	void add_pictures(const uint8_t* d, size_t n) {
		const uint8_t* end = d + n;
		auto mix = [&](const uint8_t* b, const uint8_t* e) { for (; b < e; b++) pictures = (pictures ^ *b) * 1099511628211ull; };
		if (!has(d, end, "options: ")) { mix(d, end); return; }
		// Annex B: the NAL units between start codes; the one with the settings text is left out.
		auto next_start = [&](const uint8_t* p) { for (; p + 3 <= end; p++) if (p[0] == 0 && p[1] == 0 && p[2] == 1) return p; return end; };
		for (const uint8_t* s = next_start(d); s < end;) {
			const uint8_t* e = next_start(s + 3);
			if (!has(s, e, "options: ")) mix(s, e);
			s = e;
		}
	}
};

// How many converted frames may wait in front of an encoder (or scorer).
constexpr size_t DECODE_AHEAD = 30;

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
	Fmt dist_fmt;         // and for the distorted score (the encode against the frames its encoder got) in this one
	bool convert_rate = false, rescale = false;
	int range = 0;        // the range the encoder gets: 0 the source's, 1 limited, 2 full (the "range" option)
	int trc = 0;          // the transfer curve the encoder gets (H.273 number, the "trc" option): 0 the source's; the frames are converted to it in linear light
	Color color;          // the colour of the frames the encoder gets: what the stream is tagged with
	Options options;      // the encoder's options, without chroma and bitdepth
	std::string feed_key() const; // runs with equal keys (and the same input) can share one decode
};
Plan make_plan(const SourceInfo& src, const std::string& encoder, const std::string& codec, const Options& options, const Shape& want);
// The lowest bit depth make_plan() scores at (vmaf_model_min_depth() of the model; 8 by default).
void set_min_score_depth(int depth);

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
	std::string error;    // why it failed, when it ran in an encoder process: what that process printed while it ran (see encoder_process.h)
	bool crashed = false; // the process it ran in ended under it (another run in that process may be what crashed it; see encoder_process.h)
	std::string unsupported; // the encoder does not know an option of the run (see Encoder::unsupported): why; the run is ignored, not failed
	bool no_session = false; // the encoder could not open: the GPU had no encode session left (see Encoder::no_session); the run is made again later
	bool out_of_vram = false; // it failed while its process said the GPU ran out of video memory (see encoder_process.h); made again with fewer runs at once
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
	VmafDetail detail; // the extra models' scores, the chroma values, the distorted scores
	bool score_ok = false; // both scores: against the original, and against the frames the encoder got (detail.distorted)
	bool dist_ok = false;  // run_wave's own: the distorted scores, before they go into `detail`
	std::vector<double> detail_distorted;
	std::vector<std::vector<double>> extra_distorted;
	std::string label, pool; // for the rows of the live display
};

// Encodes all runs (which must share a feed key) from one decode of `input`, and scores each against the original at the same time, and against the
// frames its encoder got (the distorted score: at the encode's size and frame rate). The input is decoded once for all of it.
bool run_wave(const std::string& input, const SourceInfo& si, std::vector<WaveRun*>& runs, const ScaleOptions& so, const VmafOptions& vo, RunProgress* feed_rp, FeedResult& fr);
