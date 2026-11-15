#include "pipeline.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <functional>
#include <cmath>
#include <condition_variable>
#include <cstdio>
#include <deque>
#include <map>
#include <memory>
#include <mutex>
#include <thread>

#include "ff.h"
#include "formats.h"
#include "gpu.h"
#include "mux.h"
#include "source.h"

bool source_info(const std::string& path, SourceInfo& out) {
	static std::mutex m;
	static std::map<std::string, SourceInfo> known;
	std::lock_guard lock(m);
	if (auto it = known.find(path); it != known.end()) { out = it->second; return true; }
	auto src = open_source(path);
	if (!src) return false;
	SourceInfo si;
	si.width = src->width; si.height = src->height;
	si.fmt = src->fmt;
	si.framerate = src->framerate;
	si.color = src->color;
	si.duration = src->duration();
	si.decoder = src->decoder;
	si.frame_count = src->frame_count();
	out = known[path] = si;
	return true;
}

std::string Plan::feed_key() const {
	char b[160];
	snprintf(b, sizeof b, "%dx%d@%d/%d rate=%d fmt=%d/%d range=%d", enc.w, enc.h, enc.fps.num, enc.fps.den, convert_rate, enc.pix.chroma, enc.pix.depth, range);
	return b;
}

static int min_score_depth = 8;
void set_min_score_depth(int depth) { min_score_depth = depth; }

Plan make_plan(const SourceInfo& src, const std::string& encoder, const std::string& codec, const Options& options, const Shape& want) {
	Plan pl;
	auto ignore = [&](std::string why) { pl.ignored = true; pl.reason = std::move(why); return pl; };
	auto beyond = [&](std::string why) { pl.expected = true; return ignore(std::move(why)); }; // the request is beyond what the input has
	Shape& enc = pl.enc;
	enc.w = want.w ? want.w : src.width;
	enc.h = want.h ? want.h : src.height;
	enc.fps = want.fps.num ? want.fps : src.framerate;
	if (!want.upscale && (enc.w > src.width || enc.h > src.height))
		return beyond("upscaling to " + std::to_string(enc.w) + "x" + std::to_string(enc.h) + " is off for this scenario (source: " + std::to_string(src.width) + "x" + std::to_string(src.height) + ")");
	// A source rate is a measurement (119.905 for a 120 fps clip with jittery timestamps), so a request within half a percent of it is the
	// source's own rate: no frames are dropped, and the encoder is told the rate that was asked for.
	const bool same_rate = std::fabs(enc.fps.value() / src.framerate.value() - 1) < 0.005;
	if (!same_rate && compare(enc.fps, src.framerate) > 0) return beyond("frame rate " + std::to_string(enc.fps.value()).substr(0, 6) + " is higher than the source's");

	// Encoder input format: "chroma" and "bitdepth" options if given, else the source's, else the closest the encoder takes.
	// What the config asks for is used as is; what it leaves open follows the source, falling back to the closest thing the
	// encoder takes: another chroma layout (4:2:0), then a lower depth.
	Options o = options;
	// An empty value means "leave it at the encoder's default" for every encoder (x264 and x265 read "tune": "" as no tune): the option is not passed on.
	for (auto it = o.begin(); it != o.end();) it = it->second.empty() ? o.erase(it) : std::next(it);
	const Fmt sf = src.fmt;
	std::vector<int> chromas{sf.chroma, 420}, depths{sf.depth, 10, 8};
	if (auto it = o.find("chroma"); it != o.end()) { chromas = {atoi(it->second.c_str())}; o.erase(it); }
	if (auto it = o.find("bitdepth"); it != o.end()) { depths = {atoi(it->second.c_str())}; o.erase(it); }
	int chroma = 0, depth = 0;
	const int gpu = options.count("gpu") ? atoi(options.at("gpu").c_str()) : 0; // the GPU of the run: a machine's first GPU is not always the one that encodes
	for (int c : chromas) for (int d : depths) {
		if (!chroma && (d <= sf.depth || depths.size() == 1) && encoder_supports(encoder, codec, c, d, gpu)) { chroma = c; depth = d; }
	}
	if (!chroma) return ignore(encoder + " cannot encode " + std::to_string(depths[0]) + "-bit " + std::to_string(chromas[0]) + " here");
	if (!want.upscale && (depth > sf.depth || chroma > sf.chroma))
		return beyond("upsampling to " + std::to_string(depth) + "-bit " + std::to_string(chroma) + " is off for this scenario (source: " + std::to_string(sf.depth) + "-bit " + std::to_string(sf.chroma) + ")");
	// "range": limited (TV) or full (PC) samples for the encoder, whatever the source has; the values are converted, not just tagged, and the
	// decoded stream is converted back to the source's range for scoring. RGB sources become limited BT.709 unless asked for full.
	if (auto it = o.find("range"); it != o.end()) {
		const std::string& v = it->second;
		if (v == "source") pl.range = 0;
		else if (v == "limited") pl.range = 1;
		else if (v == "full") pl.range = 2;
		else return ignore("range '" + v + "' is not source, limited or full");
		o.erase(it);
	}
	pl.color = src.color;
	if (src.fmt.rgb) pl.color.space = 1; // the converter makes BT.709 from RGB
	const int source_range = src.fmt.rgb ? 1 : (src.color.range ? src.color.range : 1);
	if (pl.range) pl.color.range = pl.range;
	else if (src.fmt.rgb) pl.color.range = 1;
	enc.pix = planar_fmt(chroma, depth);
	enc.has_pix = true;
	// Both sides are scored in the source's chroma layout at the higher of the two bit depths, and at least at the model's (10 bit for VMAF v1).
	// 4:2:2 sources are scored as 4:4:4, which holds all of their chroma.
	pl.score_fmt = planar_fmt(sf.chroma == 422 ? 444 : sf.chroma, std::max({sf.depth, depth, min_score_depth}));
	// The distorted score (the encode against the frames its encoder got) is scored in the encode's own chroma layout, at least at the model's depth.
	pl.dist_fmt = planar_fmt(chroma == 422 ? 444 : chroma, std::max(depth, min_score_depth));
	if (!valid(enc.pix) || !valid(pl.score_fmt) || !valid(pl.dist_fmt)) return ignore("unsupported chroma " + std::to_string(chroma) + " / bit depth " + std::to_string(depth));
	pl.convert_rate = want.fps.num != 0 && !same_rate;
	pl.rescale = enc.w != src.width || enc.h != src.height || enc.pix != src.fmt || (pl.range && pl.range != source_range); // scale, convert and/or change the range
	pl.options = std::move(o);
	return pl;
}

namespace {

// Hands frames from one thread to another; push() waits while the queue is full, pop() while it is empty.
class FrameQueue {
	std::mutex m;
	std::condition_variable cv;
	std::deque<Picture> q;
	const size_t cap;
	bool closed = false, cancelled = false;

public:
	explicit FrameQueue(size_t cap) : cap(cap) {}
	// Time spent waiting is added to the stage's counters, and the stage shows what it waits for.
	template <class Pred> void wait_for(std::unique_lock<std::mutex>& lock, Pred pred, RunProgress* rp, int state, std::atomic<int64_t> RunProgress::*counter) {
		if (pred()) return;
		const auto t = std::chrono::steady_clock::now();
		if (rp) rp->wait(state);
		cv.wait(lock, pred);
		if (rp) {
			rp->state = 0;
			(rp->*counter) += std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - t).count();
		}
	}

	bool push(Picture&& p, RunProgress* rp = nullptr) { // false once the consumer gave up; rp: the producer, backed up while the queue is full
		std::unique_lock lock(m);
		wait_for(lock, [&] { return q.size() < cap || cancelled; }, rp, 2, &RunProgress::blocked_ms);
		if (cancelled) return false;
		q.push_back(std::move(p));
		cv.notify_all();
		return true;
	}
	bool pop(Picture& p, RunProgress* rp = nullptr) { // false when nothing more will come; rp: the consumer, starved while the queue is empty
		std::unique_lock lock(m);
		wait_for(lock, [&] { return !q.empty() || closed; }, rp, 1, &RunProgress::starved_ms);
		if (q.empty()) return false;
		p = std::move(q.front());
		q.pop_front();
		cv.notify_all();
		return true;
	}
	void close() { std::lock_guard lock(m); closed = true; cv.notify_all(); }
	void cancel() { std::lock_guard lock(m); cancelled = closed = true; q.clear(); cv.notify_all(); }
};

// Wakes the stages that wait for room in a FrameHistory (the decode, the references' converters) when anything they wait on changes: a reader moved
// or left, a scorer started or stopped waiting for packets, an encoder took a frame or let one go. One for the process; a wake-up meant for another
// wave only makes a waiter look again. The waiter checks without holding this lock, so the locks of what it checks never nest inside it.
class Wakeup {
	std::mutex m;
	std::condition_variable cv;
	uint64_t gen = 0;

public:
	void notify() {
		{ std::lock_guard lock(m); gen++; }
		cv.notify_all();
	}
	template <class Pred> void wait_until(Pred done) {
		for (;;) {
			uint64_t seen;
			{ std::lock_guard lock(m); seen = gen; }
			if (done()) return;
			std::unique_lock lock(m);
			cv.wait(lock, [&] { return gen != seen; });
		}
	}
};
Wakeup changed;

// Encoded packets on their way to a scorer. Bounded by size, but always room for a few packets so that neither side can wait on the other.
class PacketQueue {
	static constexpr size_t MAX_BYTES = 64u << 20, MIN_PACKETS = 8;
	std::mutex m;
	std::condition_variable cv;
	std::deque<std::vector<uint8_t>> q;
	size_t bytes = 0;
	bool closed = false, cancelled = false;

public:
	std::atomic<bool> waiting{false}; // the scorer waits for the encoder's next packet: it has nothing to do until the encoder gets more frames

	bool push(const uint8_t* d, size_t n) { // false once the scorer gave up
		std::unique_lock lock(m);
		cv.wait(lock, [&] { return cancelled || bytes < MAX_BYTES || q.size() < MIN_PACKETS; });
		if (cancelled) return false;
		q.emplace_back(d, d + n);
		bytes += n;
		cv.notify_all();
		return true;
	}
	bool pop(std::vector<uint8_t>& out, RunProgress* rp = nullptr) { // false when no more will come
		std::unique_lock lock(m);
		if (q.empty() && !closed && !cancelled) {
			const auto t = std::chrono::steady_clock::now();
			if (rp) rp->wait(1);
			waiting = true;
			changed.notify();
			cv.wait(lock, [&] { return !q.empty() || closed || cancelled; });
			waiting = false;
			changed.notify();
			if (rp) {
				rp->state = 0;
				rp->starved_ms += std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - t).count();
			}
		}
		if (q.empty()) return false;
		out = std::move(q.front());
		q.pop_front();
		bytes -= out.size();
		cv.notify_all();
		return true;
	}
	void close() { std::lock_guard lock(m); closed = true; cv.notify_all(); }
	void cancel() { std::lock_guard lock(m); cancelled = closed = true; q.clear(); bytes = 0; cv.notify_all(); }
};

// One queue per consumer, so the producer is held back by the slowest of them. Frames are shared, not copied.
class Broadcast {
	std::vector<std::unique_ptr<FrameQueue>> qs;

public:
	explicit Broadcast(size_t n) { for (size_t i = 0; i < n; i++) qs.push_back(std::make_unique<FrameQueue>(DECODE_AHEAD)); }
	FrameQueue& at(size_t i) { return *qs[i]; }
	bool push(const Picture& p, RunProgress* rp = nullptr) { // false once every consumer has given up
		bool any = false;
		for (auto& q : qs) any |= q->push(Picture(p), rp);
		return any;
	}
	void close() { for (auto& q : qs) q->close(); }
};

// How many frames an encoder holds: given to it, no packet out for them yet. x264 and x265 hold their lookahead, their B-frames and one frame per frame
// thread (often a hundred frames and more); a scorer cannot be given those frames before the encoder lets go of them.
struct Holding {
	std::atomic<long long> fed{0}, out{0}; // changed.notify() after each change
	long long frames() const { return std::max(0LL, fed.load() - out.load()); }
};

// One encoder taking frames from its queue.
void consume(EncodeJob& job, const SourceInfo& si, FrameQueue& q, Holding* holding = nullptr) {
	const Plan& pl = *job.plan;
	Params p;
	p.codec = job.codec;
	p.color = job.color;
	p.width = pl.enc.w; p.height = pl.enc.h;
	p.framerate = pl.enc.fps;
	p.pix = pl.enc.pix;
	p.bitrate = job.bitrate;
	p.gop = std::max(1, (int)std::lround(job.kf * pl.enc.fps.value()));
	p.options = pl.options;
	auto codec = make_encoder(job.encoder);
	// Packets arrive in coding order, which is also the order the decoder needs them.
	// and are muxed to disk as they arrive; only the packet at hand is in memory.
	Mp4Writer mp4;
	bool write_failed = false;
	if (!codec || !codec->open(p, [&](const uint8_t* d, size_t n, int64_t pts, bool key, int64_t dts) {
		job.output.add(d, n);
		if (job.tee) job.tee(d, n);
		if (!job.file.empty() && !mp4.add(d, n, pts, key, dts)) write_failed = true;
	})) { q.cancel(); return; }
	if (!job.file.empty() && !mp4.open(job.file, job.id, pl.enc, job.color, codec->reorder_delay())) { fprintf(stderr, "cannot write %s\n", job.file.c_str()); q.cancel(); return; }
	bool ok = true;
	for (Picture pic; q.pop(pic, job.rp);) {
		const auto t = std::chrono::steady_clock::now();
		if (holding) { holding->fed++; changed.notify(); }
		if (ok && !(ok = codec->encode(&pic))) q.cancel(); // frees the producer if it is waiting for room
		else if (job.rp) job.rp->step();
		if (job.rp) job.rp->busy_ms += std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - t).count();
	}
	const auto t_flush = std::chrono::steady_clock::now();
	ok = ok && codec->encode(nullptr); // the encoder finishing what it still holds counts as busy
	if (job.rp) job.rp->busy_ms += std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - t_flush).count();
	if (!job.file.empty() && !(mp4.close() && !write_failed)) { fprintf(stderr, "cannot write %s\n", job.file.c_str()); ok = false; }
	job.ok = ok;
	if (job.rp) { job.rp->failed = !ok; job.rp->finish(); }
	job.kbps = codec->stats.bytes * 8 / si.duration / 1000;
}

// Frames that several readers take at their own pace, each through a cursor of its own; a frame is let go of once every reader is past it.
// The one decode of a wave reaches the encoders, the true scores and the distorted scores through these.
class FrameHistory {
	std::mutex m;
	std::condition_variable cv;
	std::deque<Picture> q;
	long long base = 0; // the index of q.front()
	bool closed = false;
	struct Reader {
		long long at = 0;
		bool gone = false;
		long long limit = 0;         // held(): how far behind it may fall while it is busy
		std::function<bool()> idle;  // true while the reader waits for something else than this history
		std::function<long long()> slack; // added to the limit: the frames an encoder holds that this reader waits for
	};
	std::vector<Reader> readers;

	void trim() { // with m held
		long long low = base + (long long)q.size();
		for (auto& r : readers) if (!r.gone) low = std::min(low, r.at);
		while (base < low && !q.empty()) { q.pop_front(); base++; }
	}

public:
	// A reader, before the first frame comes. `idle`, if given, says when the reader waits for something else (a scorer for the encoder's packets): such a
	// reader does not hold the decode back, however far behind it is, since its encoder may need more frames before it gives it anything to do. A busy
	// reader holds it back once it is `limit` frames behind (see held()), and `slack` more: the frames its encoder holds, which it cannot have yet.
	size_t add_reader(long long limit, std::function<bool()> idle = {}, std::function<long long()> slack = {}) {
		std::lock_guard lock(m);
		readers.push_back(Reader{0, false, limit, std::move(idle), std::move(slack)});
		return readers.size() - 1;
	}
	void push(Picture p) { { std::lock_guard lock(m); q.push_back(std::move(p)); cv.notify_all(); } changed.notify(); }
	void close() { { std::lock_guard lock(m); closed = true; cv.notify_all(); } changed.notify(); }
	// The reader's next frame: 1, or 0 when no more will come. Waits while it is not there yet; rp: the reader, starved meanwhile.
	int next(size_t r, Picture& out, RunProgress* rp = nullptr) {
		std::unique_lock lock(m);
		Reader& rd = readers[r];
		auto there = [&] { return rd.at < base + (long long)q.size() || closed; };
		if (!there()) {
			const auto t = std::chrono::steady_clock::now();
			if (rp) rp->wait(1);
			cv.wait(lock, there);
			if (rp) {
				rp->state = 0;
				rp->starved_ms += std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - t).count();
			}
		}
		if (rd.at >= base + (long long)q.size()) return 0;
		out = q[(size_t)(rd.at - base)];
		rd.at++;
		trim();
		lock.unlock();
		changed.notify();
		return 1;
	}
	void leave(size_t r) { { std::lock_guard lock(m); readers[r].gone = true; trim(); } changed.notify(); }
	bool read() { std::lock_guard lock(m); return std::any_of(readers.begin(), readers.end(), [](const Reader& r) { return !r.gone; }); } // by anyone still
	// How many frames the slowest reader is behind the newest one: of every reader, or of those that are busy (not idle) only.
	long long behind(bool busy_only) {
		std::lock_guard lock(m);
		const long long end = base + (long long)q.size();
		long long most = 0;
		for (auto& r : readers) if (!r.gone && !(busy_only && r.idle && r.idle())) most = std::max(most, end - r.at);
		return most;
	}
	// Whether any reader is busy (not idle), wherever it is.
	bool busy() {
		std::lock_guard lock(m);
		return std::any_of(readers.begin(), readers.end(), [](const Reader& r) { return !r.gone && !(r.idle && r.idle()); });
	}
	// Whether a busy reader is as far behind as it may be: then nothing more should be added for now.
	bool held() {
		std::lock_guard lock(m);
		const long long end = base + (long long)q.size();
		for (auto& r : readers) if (!r.gone && end - r.at >= r.limit + (r.slack ? r.slack() : 0) && !(r.idle && r.idle())) return true;
		return false;
	}
};

// How far a scorer that is busy may fall behind its encoder's output before the decode waits for it (and with it the encoders), in frames. The frames it
// has not looked at yet are kept for it, as they came from the decoder (the true score) or as the encoder got them (the distorted score); a scorer that
// waits for its encoder's packets never holds the decode back, since the encoder may need more frames before it gives the scorer anything to do. The
// frames the encoder holds (Holding) do not count: measured from the decode, x264's scorer is always that far behind, so the decode stopped whenever it
// scored, which fed x264 one frame per scored frame and left its threads with nothing to do.
constexpr long long SCORE_BEHIND = 30;

// Decodes the input once for the whole wave, while nothing that reads the frames holds it back (FrameHistory::held(): the encoders' feed DECODE_AHEAD frames
// behind, a busy scorer SCORE_BEHIND frames behind).
void decode_input(Source& src, FrameHistory& raw, const std::function<bool()>& held, bool& ok, RunProgress* rp) {
	Picture frame;
	long long n = 0;
	int r;
	for (;;) {
		if (held()) {
			const auto t = std::chrono::steady_clock::now();
			if (rp) rp->wait(2);
			changed.wait_until([&] { return !held() || !raw.read(); });
			if (rp) {
				rp->state = 0;
				rp->blocked_ms += std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - t).count();
			}
		}
		if (!raw.read()) { r = -1; break; } // everything that took frames has ended (failed): nothing to decode for
		if ((r = src.next(frame)) != 1) break;
		raw.push(std::move(frame));
		frame = Picture();
		n++;
		if (rp) rp->step();
	}
	ok = r == 0;
	if (r < 0 && raw.read()) fprintf(stderr, "decoding the input failed after %lld frames\n", n);
	if (ok && src.frame_count() && src.frame_count() != n) {
		fprintf(stderr, "the file lists %lld frames but %lld were decoded\n", src.frame_count(), n);
		ok = false;
	}
	if (rp) { rp->failed = !ok && raw.read(); rp->finish(); }
	raw.close();
}

// Takes the decoded frames, drops frames to the target rate, scales and converts, and hands the result to the encoders and to the distorted scores (`enc`).
// Frame rate conversion drops frames before scaling. Each source frame falls into a slot of the output timeline (its real presentation time times the rate,
// rounded down) and only the first frame in each slot is kept. `shown[n]` records which encoded frame is on screen at source frame n, for scoring.
void feed_encoders(const Source& src, FrameHistory& raw, size_t reader, const Plan& pl, const ScaleOptions& so, Broadcast& bc, FrameHistory& enc, FeedResult& fr, bool& ok, RunProgress* rp) {
	const Shape& e = pl.enc;
	auto scaler = make_converter();
	Picture frame, scaled;
	int64_t last_slot = -1;
	long long kept = 0;
	ok = true;
	auto hand_on = [&](const Picture& p) {
		enc.push(p);
		if (!bc.push(p, rp)) ok = false; // every encoder gave up
	};
	auto queue_scaled = [&] {
		int got;
		while (ok && (got = scaler->pop(scaled)) == 1) hand_on(scaled);
		ok = ok && got >= 0;
	};
	while (ok && raw.next(reader, frame, rp) == 1) {
		const int64_t slot = src.slot(fr.frames, e.fps);
		const bool keep = !pl.convert_rate || fr.frames == 0 || slot != last_slot; // no target rate: every frame stays
		last_slot = slot;
		fr.frames++;
		if (rp) rp->step();
		if (keep) {
			frame.pts = kept++; // encoded frames are numbered consecutively, whatever the gaps in the source timing
			frame.duration = 1; // one tick of the 1/enc.fps time base
			if (!pl.rescale) hand_on(frame);
			else {
				if (frame.pts == 0) ok = scaler->open(frame, e.fps.inverse(), e.w, e.h, e.pix, so, pl.range);
				ok = ok && scaler->push(&frame);
				queue_scaled();
			}
		}
		fr.shown.push((uint32_t)(kept - 1));
	}
	raw.leave(reader);
	if (ok && pl.rescale && kept) { ok = scaler->push(nullptr); queue_scaled(); }
	if (rp) rp->finish();
	fr.shown.finish();
	enc.close();
	bc.close();
}

// Takes the decoded frames and converts them to `fmt` for scoring (the true score's reference), for every scorer of that format, staying at most
// DECODE_AHEAD frames ahead of the slowest busy one: the decoded frames are what is kept for a scorer that is behind, not the converted ones (which are
// larger: scored at 10 bit or more). A scorer that waits for its encoder's packets does not hold it back: the busy scorers may need frames that encoder
// has not been given yet (an encoder that holds more frames back than the others), and waiting for it would wait forever. `waiting` is set while it
// waits for room.
void convert_reference(const SourceInfo& si, FrameHistory& raw, size_t reader, Fmt fmt, const ScaleOptions& so, FrameHistory& out, std::atomic<bool>& waiting, RunProgress& rp) {
	auto conv = make_converter();
	Picture t, o;
	bool ok = true, first = true;
	auto flush_converted = [&] {
		int got;
		while ((got = conv->pop(o)) == 1) out.push(std::move(o));
		if (got < 0) ok = false;
	};
	for (;;) {
		if (out.behind(true) >= (long long)DECODE_AHEAD) {
			const auto t = std::chrono::steady_clock::now();
			rp.wait(2);
			waiting = true;
			changed.notify();
			changed.wait_until([&] { return out.behind(true) < (long long)DECODE_AHEAD || !out.read(); });
			waiting = false;
			changed.notify();
			rp.state = 0;
			rp.blocked_ms += std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - t).count();
		}
		if (!out.read() || raw.next(reader, t) != 1) break; // the end, or every scorer of this format has ended
		rp.step();
		if (t.fmt == fmt) { out.push(t); continue; }
		t.duration = 1;
		if (first) { ok = conv->open(t, si.framerate.inverse(), si.width, si.height, fmt, so); first = false; }
		if (ok && !conv->push(&t)) ok = false;
		if (ok) flush_converted();
		if (!ok) break;
	}
	raw.leave(reader);
	if (ok && !first) { ok = conv->push(nullptr); flush_converted(); }
	if (!ok) fprintf(stderr, "the original frames could not be converted to %d-bit %d for scoring\n", fmt.depth, fmt.chroma);
	rp.finish();
	out.close();
}

// The frames an encoder got, as the reference of its distorted score, converted to the scoring format the way the decoded encode is, at the encode's
// size. has() tells whether one more frame comes, for the scorer's "shown" map, which is one to one here: encoded frame k is on screen at feed frame k.
class ConvertedFeed {
	FrameHistory& h;
	size_t reader;
	const Plan& pl;
	const ScaleOptions& so;
	std::unique_ptr<Converter> conv = make_converter();
	std::deque<Picture> ready;
	bool opened = false, ended = false, failed = false;

	bool fetch() { // one more frame into `ready`; false at the end or on an error
		while (ready.empty() && !ended) {
			Picture p;
			if (h.next(reader, p) != 1) {
				ended = true;
				if (opened && (!conv->push(nullptr) || !drain())) failed = true;
				break;
			}
			if (p.layout.planes == 0 && p.fmt.planar) p.layout = layout_of(av_planar_format(p.fmt.chroma, p.fmt.depth)); // the scaler's frames say only their format
			p.color = pl.color; // what the decoded encode is taken to be as well
			if (p.fmt == pl.dist_fmt) { ready.push_back(std::move(p)); break; }
			p.duration = 1;
			if (!opened && !(opened = conv->open(p, pl.enc.fps.inverse(), pl.enc.w, pl.enc.h, pl.dist_fmt, so))) { failed = ended = true; break; }
			if (!conv->push(&p) || !drain()) { failed = ended = true; break; }
		}
		return !ready.empty();
	}
	bool drain() {
		Picture o;
		int got;
		while ((got = conv->pop(o)) == 1) ready.push_back(std::move(o));
		return got >= 0;
	}

public:
	ConvertedFeed(FrameHistory& h, size_t reader, const Plan& pl, const ScaleOptions& so) : h(h), reader(reader), pl(pl), so(so) {}
	~ConvertedFeed() { h.leave(reader); }
	bool has() { return fetch(); }
	int next(Picture& out) {
		if (!fetch()) return failed ? -1 : 0;
		out = std::move(ready.front());
		ready.pop_front();
		return 1;
	}
	bool ok() const { return !failed; }
};

} // namespace

bool run_wave(const std::string& input, const SourceInfo& si, std::vector<WaveRun*>& runs, const ScaleOptions& so, const VmafOptions& vo, RunProgress* feed_rp, FeedResult& fr) {
	const size_t n = runs.size();
	auto src = open_source(input);
	if (!src) {
		fprintf(stderr, "cannot open %s\n", input.c_str());
		fr.shown.finish();
		for (auto* r : runs) { r->enc.ok = false; r->score_ok = false; }
		return false;
	}
	// One decode for everything: the encoders' feed takes the decoded frames, and so do the converters of the true scores' reference (one for each format
	// the runs score in); the frames the encoders got are kept for the distorted scores. Scorers read where they are, so nothing is decoded twice, and the
	// decode (and with it the encoders) waits for a scorer that falls far behind.
	FrameHistory raw, enc_frames;
	Broadcast bc(n); // the frames for the encoders
	std::vector<std::unique_ptr<PacketQueue>> packets, dist_packets;
	std::vector<std::unique_ptr<Holding>> holding;
	for (size_t i = 0; i < n; i++) {
		packets.push_back(std::make_unique<PacketQueue>());
		dist_packets.push_back(std::make_unique<PacketQueue>());
		holding.push_back(std::make_unique<Holding>());
		PacketQueue* pq = packets.back().get();
		PacketQueue* dq = dist_packets.back().get();
		Holding* hd = holding.back().get();
		runs[i]->enc.tee = [pq, dq, hd](const uint8_t* d, size_t size) { hd->out++; changed.notify(); const bool a = pq->push(d, size), b = dq->push(d, size); return a || b; };
		if (runs[i]->enc.rp) { // frames the encoder gets: fewer than the source's when the frame rate is lowered
			const Plan& pl = *runs[i]->enc.plan;
			runs[i]->enc.rp->begin(1, pl.convert_rate ? std::max<long long>(1, std::llround(si.duration * pl.enc.fps.value())) : std::max<long long>(1, si.frame_count));
		}
	}

	// The scoring formats, and a reference of each, converted from the decoded frames.
	std::vector<Fmt> formats;
	for (auto* r : runs) if (std::find(formats.begin(), formats.end(), r->enc.plan->score_fmt) == formats.end()) formats.push_back(r->enc.plan->score_fmt);
	struct Group {
		Fmt fmt;
		std::vector<size_t> members, readers;
		FrameHistory frames;
		size_t raw_reader = 0;
		std::atomic<bool> waiting{false}; // the converter waits for its scorers
		RunProgress ref_rp;
		int handle = 0;
	};
	std::vector<std::unique_ptr<Group>> groups;
	for (const Fmt& f : formats) {
		auto g = std::make_unique<Group>();
		g->fmt = f;
		for (size_t i = 0; i < n; i++) if (runs[i]->enc.plan->score_fmt == f) {
			g->members.push_back(i);
			PacketQueue* pq = packets[i].get();
			g->readers.push_back(g->frames.add_reader(DECODE_AHEAD, [pq] { return pq->waiting.load(); }));
		}
		// The converter holds the decode back like a busy scorer when it waits for one; when its scorers all wait for packets, it does not. Its slack is
		// what the members' encoders hold, in decoded frames (more of them than encoded ones when the frame rate is lowered).
		std::vector<std::pair<Holding*, double>> held_by;
		for (size_t i : g->members) {
			const Plan& pl = *runs[i]->enc.plan;
			held_by.push_back({holding[i].get(), pl.convert_rate && pl.enc.fps.value() > 0 ? std::max(1.0, si.framerate.value() / pl.enc.fps.value()) : 1.0});
		}
		g->raw_reader = raw.add_reader(SCORE_BEHIND, [gp = g.get()] { return gp->waiting.load() && !gp->frames.busy(); }, [held_by] {
			long long most = 0;
			for (auto& [h, per] : held_by) most = std::max(most, (long long)std::ceil(h->frames() * per));
			return most;
		});
		g->ref_rp.begin(6, std::max<long long>(1, si.frame_count));
		g->handle = ui::attach(&g->ref_rp, "reference to " + std::to_string(f.depth) + "-bit " + (f.rgb ? std::string("RGB") : f.chroma == 420 ? std::string("4:2:0") : f.chroma == 422 ? std::string("4:2:2") : std::string("4:4:4")), "convert");
		groups.push_back(std::move(g));
	}
	const size_t feed_reader = raw.add_reader(DECODE_AHEAD);
	std::vector<size_t> dist_readers;
	for (size_t i = 0; i < n; i++) {
		PacketQueue* dq = dist_packets[i].get();
		Holding* hd = holding[i].get();
		dist_readers.push_back(enc_frames.add_reader(SCORE_BEHIND, [dq] { return dq->waiting.load(); }, [hd] { return hd->frames(); }));
	}
	if (feed_rp) feed_rp->begin(6, std::max<long long>(1, si.frame_count)); // it steps once per decoded frame, kept or dropped
	// The one decode of the wave has a row of its own; the feed and the references are conversions of what it decoded.
	RunProgress decode_rp;
	decode_rp.begin(3, std::max<long long>(1, si.frame_count));
	const size_t slash = input.find_last_of("/\\");
	const int decode_handle = ui::attach(&decode_rp, input.substr(slash == std::string::npos ? 0 : slash + 1), "decode");

	bool decode_ok = false, feed_ok = false;
	std::thread decoder([&] { decode_input(*src, raw, [&] { return raw.held() || enc_frames.held(); }, decode_ok, &decode_rp); });
	std::thread feeder([&] { feed_encoders(*src, raw, feed_reader, *runs[0]->enc.plan, so, bc, enc_frames, fr, feed_ok, feed_rp); });
	std::vector<std::thread> threads;
	for (size_t i = 0; i < n; i++) threads.emplace_back([&, i] {
		consume(runs[i]->enc, si, bc.at(i), holding[i].get());
		packets[i]->close(); // the scorers have all there is
		dist_packets[i]->close();
	});

	// Scoring runs at the same time, following the encoders' output as it is written.
	RefInfo ref;
	ref.width = si.width; ref.height = si.height; ref.framerate = si.framerate; ref.color = si.color; ref.frames = si.frame_count;
	ref.score_range = si.fmt.rgb ? 1 : (si.color.range ? si.color.range : 1); // the reference side is converted to YUV without changing the range (RGB: limited)
	for (auto& gp : groups) {
		Group& g = *gp;
		threads.emplace_back([&, gptr = &g] { convert_reference(si, raw, gptr->raw_reader, gptr->fmt, so, gptr->frames, gptr->waiting, gptr->ref_rp); });
		VmafOptions group_vo = vo;
		if (group_vo.threads <= 0) group_vo.threads = std::max(2, (int)std::thread::hardware_concurrency() / (int)runs.size());
		for (size_t m = 0; m < g.members.size(); m++) threads.emplace_back([&, gptr = &g, m, group_vo] {
			Group& g = *gptr;
			WaveRun& run = *runs[g.members[m]];
			const size_t reader = g.readers[m];
			PacketQueue& pq = *packets[g.members[m]];
			const FrameFeed feed = [&g, reader, rp = run.score_rp](Picture& out) { return g.frames.next(reader, out, rp); };
			const PacketFeed next_packet = [&pq, rp = run.score_rp](std::vector<uint8_t>& out) { return pq.pop(out, rp); };
			const ShownFeed shown = [&fr](size_t k, uint32_t& v) { return fr.shown.get(k, v); };
			VmafOptions run_vo = group_vo; // scoring follows the run's encoder to its GPU (vmaf.gpu "auto")
			run_vo.encoder_vendor = encoder_vendor(run.enc.encoder);
			if (auto it = run.enc.plan->options.find("gpu"); it != run.enc.plan->options.end()) run_vo.encoder_gpu = atoi(it->second.c_str());
			run.score_ok = vmaf_score(ref, feed, run.codec, next_packet, {}, run.enc.plan->enc, run.enc.plan->color, shown, g.fmt, run_vo, so, run.scores, run.score_rp, &run.detail);
			if (run.score_ok && (long long)run.scores.size() != fr.frames) { fprintf(stderr, "%zu scores for %lld frames\n", run.scores.size(), fr.frames); run.score_ok = false; }
			if (run.score_rp) { run.score_rp->failed = !run.score_ok; run.score_rp->finish(); }
			g.frames.leave(reader); // done with the original frames: do not hold the decode back
			pq.cancel();            // and with the packets: do not hold the encoder back
		});
	}
	// The distorted score: each encode against the frames its encoder got, at their size and frame rate, with every model.
	for (size_t i = 0; i < n; i++) threads.emplace_back([&, i] {
		WaveRun& run = *runs[i];
		const Plan& pl = *run.enc.plan;
		PacketQueue& pq = *dist_packets[i];
		bool ok;
		{
			ConvertedFeed cf(enc_frames, dist_readers[i], pl, so);
			RefInfo dref;
			dref.width = pl.enc.w; dref.height = pl.enc.h; dref.framerate = pl.enc.fps; dref.color = pl.color;
			dref.frames = pl.convert_rate ? std::max<long long>(1, std::llround(si.duration * pl.enc.fps.value())) : si.frame_count;
			dref.score_range = pl.color.range ? pl.color.range : 1;
			const FrameFeed next_ref = [&cf](Picture& out) { return cf.next(out); };
			const PacketFeed next_packet = [&pq](std::vector<uint8_t>& out) { return pq.pop(out); };
			const ShownFeed shown = [&cf](size_t k, uint32_t& v) { v = (uint32_t)k; return cf.has(); };
			VmafOptions run_vo = vo;
			if (run_vo.threads <= 0) run_vo.threads = std::max(2, (int)std::thread::hardware_concurrency() / (int)runs.size());
			run_vo.encoder_vendor = encoder_vendor(run.enc.encoder);
			if (auto it = pl.options.find("gpu"); it != pl.options.end()) run_vo.encoder_gpu = atoi(it->second.c_str());
			std::vector<double> scores;
			VmafDetail detail;
			ok = vmaf_score(dref, next_ref, run.codec, next_packet, {}, pl.enc, pl.color, shown, pl.dist_fmt, run_vo, so, scores, nullptr, &detail) && cf.ok();
			if (ok) {
				run.detail_distorted = std::move(scores);
				for (auto& e : detail.extra) run.extra_distorted.push_back(std::move(e.scores));
			}
		} // the feed lets go of the encoders' frames here
		if (!ok) fprintf(stderr, "the distorted score (the encode against the frames its encoder got) failed\n");
		run.dist_ok = ok;
		pq.cancel();
	});

	decoder.join();
	feeder.join();
	for (auto& t : threads) t.join();
	for (auto& g : groups) ui::detach(g->handle);
	ui::detach(decode_handle);
	fr.ok = decode_ok && feed_ok;
	for (auto* r : runs) {
		r->enc.ok = r->enc.ok && fr.ok;
		// Both scores or none: a run without its distorted score counts as not scored.
		if (r->score_ok && (!r->dist_ok || r->extra_distorted.size() != r->detail.extra.size())) r->score_ok = false;
		if (r->score_ok) {
			r->detail.distorted = std::move(r->detail_distorted);
			for (size_t m = 0; m < r->detail.extra.size(); m++) r->detail.extra[m].distorted = std::move(r->extra_distorted[m]);
		}
	}
	return fr.ok;
}
