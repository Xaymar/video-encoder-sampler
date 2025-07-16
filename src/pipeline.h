#pragma once
// Decoding, converting, encoding and scoring, arranged so that the expensive software decode of an input is done once for
// all the encoders that can share it. Runs that need the very same frames (same input, resolution, frame rate, pixel format)
// are encoded together as a "wave": one thread decodes and converts, every encoder has a thread that takes the frames from a
// queue of its own, and the decoder stays at most DECODE_AHEAD frames ahead of the slowest encoder. Scoring works the same way:
// one thread produces the original frames in the scoring format and every scorer takes them from its own queue.
#include <string>
#include <vector>

#include "convert.h"
#include "encoder.h"
#include "picture.h"
#include "spool.h"
#include "ui.h"
#include "vmaf.h"

// How many converted frames may wait in front of an encoder (or scorer).
constexpr size_t DECODE_AHEAD = 8;

// What is needed to know about an input before decoding any of it.
struct SourceInfo {
	int width = 0, height = 0;
	Fmt fmt;
	Rational framerate;
	Color color;
	double duration = 0;       // seconds
	long long frame_count = 0; // 0 if the file carries no usable timestamps
};
bool source_info(const std::string& path, SourceInfo& out); // opens the file once and remembers the answer

// How one run is encoded, worked out from the input's properties alone.
struct Plan {
	bool ignored = false; // cannot work with this input or encoder; not an error
	std::string reason;
	Shape enc;            // resolution, frame rate and pixel format the frames are encoded at
	Fmt score_fmt;        // both sides are scored in this format
	bool convert_rate = false, rescale = false;
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
	std::string spool;         // where the packets are kept while the run is scored; empty: only counted and hashed
	SpoolInfo output;          // what the encoder produced
	double kbps = 0;
	bool ok = false;
};

// What decoding and converting leaves for scoring; the same for every job of a wave.
struct FeedResult {
	std::vector<uint32_t> shown; // shown[n]: the encoded frame that is on screen at source frame n
	long long frames = 0;        // source frames
	bool ok = false;
};

// Encodes all jobs (which must share a feed key) from one decode of `input`. `feed_rp` shows the decoder's progress.
bool encode_wave(const std::string& input, const SourceInfo& si, std::vector<EncodeJob*>& jobs, const ScaleOptions& so, RunProgress* feed_rp, FeedResult& fr);

struct ScoreJob {
	const EncodeJob* encoded = nullptr;
	const FeedResult* feed = nullptr;
	Codec codec = Codec::None;
	RunProgress* rp = nullptr;
	std::vector<double> scores;
	bool ok = false;
};

// Scores all jobs against one decode of the original.
void score_wave(const std::string& input, const SourceInfo& si, std::vector<ScoreJob*>& jobs, const VmafOptions& vo, const ScaleOptions& so);
