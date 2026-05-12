#include "pipeline.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <cstdio>
#include <deque>
#include <map>
#include <memory>
#include <mutex>
#include <thread>

#include "formats.h"
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
	snprintf(b, sizeof b, "%dx%d@%d/%d rate=%d fmt=%d/%d", enc.w, enc.h, enc.fps.num, enc.fps.den, convert_rate, enc.pix.chroma, enc.pix.depth);
	return b;
}

Plan make_plan(const SourceInfo& src, const std::string& encoder, const std::string& codec, const Options& options, const Shape& want) {
	Plan pl;
	auto ignore = [&](std::string why) { pl.ignored = true; pl.reason = std::move(why); return pl; };
	Shape& enc = pl.enc;
	enc.w = want.w ? want.w : src.width;
	enc.h = want.h ? want.h : src.height;
	enc.fps = want.fps.num ? want.fps : src.framerate;
	if (!want.upscale && (enc.w > src.width || enc.h > src.height))
		return ignore("upscaling to " + std::to_string(enc.w) + "x" + std::to_string(enc.h) + " is off for this scenario (source: " + std::to_string(src.width) + "x" + std::to_string(src.height) + ")");
	// A source rate is a measurement (119.905 for a 120 fps clip with jittery timestamps), so a request within half a percent of it is the
	// source's own rate: no frames are dropped, and the encoder is told the rate that was asked for.
	const bool same_rate = std::fabs(enc.fps.value() / src.framerate.value() - 1) < 0.005;
	if (!same_rate && compare(enc.fps, src.framerate) > 0) return ignore("frame rate " + std::to_string(enc.fps.value()).substr(0, 6) + " is higher than the source's");

	// Encoder input format: "chroma" and "bitdepth" options if given, else the source's, else the closest the encoder takes.
	// What the config asks for is used as is; what it leaves open follows the source, falling back to the closest thing the
	// encoder takes: another chroma layout (4:2:0), then a lower depth.
	Options o = options;
	const Fmt sf = src.fmt;
	std::vector<int> chromas{sf.chroma, 420}, depths{sf.depth, 10, 8};
	if (auto it = o.find("chroma"); it != o.end()) { chromas = {atoi(it->second.c_str())}; o.erase(it); }
	if (auto it = o.find("bitdepth"); it != o.end()) { depths = {atoi(it->second.c_str())}; o.erase(it); }
	int chroma = 0, depth = 0;
	for (int c : chromas) for (int d : depths) {
		if (!chroma && (d <= sf.depth || depths.size() == 1) && encoder_supports(encoder, codec, c, d)) { chroma = c; depth = d; }
	}
	if (!chroma) return ignore(encoder + " cannot encode " + std::to_string(depths[0]) + "-bit " + std::to_string(chromas[0]) + " here");
	if (!want.upscale && (depth > sf.depth || chroma > sf.chroma))
		return ignore("upsampling to " + std::to_string(depth) + "-bit " + std::to_string(chroma) + " is off for this scenario (source: " + std::to_string(sf.depth) + "-bit " + std::to_string(sf.chroma) + ")");
	enc.pix = planar_fmt(chroma, depth);
	enc.has_pix = true;
	// Both sides are scored in the source's chroma layout at the higher of the two bit depths.
	// 4:2:2 sources are scored as 4:4:4, which holds all of their chroma.
	pl.score_fmt = planar_fmt(sf.chroma == 422 ? 444 : sf.chroma, std::max(sf.depth, depth));
	if (!valid(enc.pix) || !valid(pl.score_fmt)) return ignore("unsupported chroma " + std::to_string(chroma) + " / bit depth " + std::to_string(depth));
	pl.convert_rate = want.fps.num != 0 && !same_rate;
	pl.rescale = enc.w != src.width || enc.h != src.height || enc.pix != src.fmt; // scale and/or convert
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
		if (rp) rp->state = state;
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

// Encoded packets on their way to a scorer. Bounded by size, but always room for a few packets so that neither side can wait on the other.
class PacketQueue {
	static constexpr size_t MAX_BYTES = 64u << 20, MIN_PACKETS = 8;
	std::mutex m;
	std::condition_variable cv;
	std::deque<std::vector<uint8_t>> q;
	size_t bytes = 0;
	bool closed = false, cancelled = false;

public:
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
			if (rp) rp->state = 1;
			cv.wait(lock, [&] { return !q.empty() || closed || cancelled; });
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

// Reads the input, drops frames to the target rate, scales and converts, and hands the result to the encoders.
// Frame rate conversion drops frames before scaling. Each source frame falls into a slot of the output timeline
// (its real presentation time times the rate, rounded down) and only the first frame in each slot is kept.
// `shown[n]` records which encoded frame is on screen at source frame n, for scoring.
void produce(const std::string& input, const SourceInfo& si, const Plan& pl, const ScaleOptions& so, Broadcast& bc, FeedResult& fr, RunProgress* rp) {
	auto src = open_source(input);
	if (!src) { fprintf(stderr, "cannot open %s\n", input.c_str()); fr.shown.finish(); bc.close(); return; }
	const Shape& enc = pl.enc;
	if (rp) rp->begin(3, pl.convert_rate ? std::max<long long>(1, std::llround(si.duration * enc.fps.value())) : std::max<long long>(1, si.frame_count));
	auto scaler = make_converter();
	Picture frame, scaled;
	int64_t last_slot = -1;
	long long kept = 0;
	bool ok = true;
	int r = 1;
	auto queue_scaled = [&] {
		int got;
		while (ok && (got = scaler->pop(scaled)) == 1) ok = bc.push(scaled, rp);
		ok = ok && got >= 0;
	};
	while (ok && (r = src->next(frame)) == 1) {
		const int64_t slot = src->slot(fr.frames, enc.fps);
		const bool keep = !pl.convert_rate || fr.frames == 0 || slot != last_slot; // no target rate: every frame stays
		last_slot = slot;
		fr.frames++;
		if (rp) rp->step();
		if (keep) {
			frame.pts = kept++; // encoded frames are numbered consecutively, whatever the gaps in the source timing
			frame.duration = 1; // one tick of the 1/enc.fps time base
			if (!pl.rescale) ok = bc.push(frame, rp);
			else {
				if (frame.pts == 0) ok = scaler->open(frame, enc.fps.inverse(), enc.w, enc.h, enc.pix, so);
				ok = ok && scaler->push(&frame);
				queue_scaled();
			}
		}
		fr.shown.push((uint32_t)(kept - 1));
	}
	if (ok && src->frame_count() && src->frame_count() != fr.frames) {
		fprintf(stderr, "the file lists %lld frames but %lld were decoded\n", src->frame_count(), fr.frames);
		ok = false;
	}
	if (ok && pl.rescale) { ok = scaler->push(nullptr); queue_scaled(); }
	fr.ok = ok && r >= 0;
	if (rp) rp->finish();
	fr.shown.finish();
	bc.close();
}

// One encoder taking frames from its queue.
void consume(EncodeJob& job, const SourceInfo& si, FrameQueue& q) {
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
		if (ok && !(ok = codec->encode(&pic))) q.cancel(); // frees the producer if it is waiting for room
		else if (job.rp) job.rp->step();
		if (job.rp) job.rp->busy_ms += std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - t).count();
	}
	const auto t_flush = std::chrono::steady_clock::now();
	ok = ok && codec->encode(nullptr); // the encoder finishing what it still holds counts as busy
	if (job.rp) job.rp->busy_ms += std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - t_flush).count();
	if (!job.file.empty() && !(mp4.close() && !write_failed)) { fprintf(stderr, "cannot write %s\n", job.file.c_str()); ok = false; }
	job.ok = ok;
	if (job.rp) job.rp->finish();
	job.kbps = codec->stats.bytes * 8 / si.duration / 1000;
}

} // namespace

bool run_wave(const std::string& input, const SourceInfo& si, std::vector<WaveRun*>& runs, const ScaleOptions& so, const VmafOptions& vo, RunProgress* feed_rp, FeedResult& fr) {
	const size_t n = runs.size();
	Broadcast bc(n);
	std::vector<std::unique_ptr<PacketQueue>> packets;
	for (size_t i = 0; i < n; i++) {
		packets.push_back(std::make_unique<PacketQueue>());
		PacketQueue* pq = packets.back().get();
		runs[i]->enc.tee = [pq](const uint8_t* d, size_t size) { return pq->push(d, size); };
		if (runs[i]->enc.rp) runs[i]->enc.rp->begin(1, std::max<long long>(1, si.frame_count));
	}

	// Encoding: one decode feeds every encoder.
	std::thread producer([&] { produce(input, si, *runs[0]->enc.plan, so, bc, fr, feed_rp); });
	std::vector<std::thread> threads;
	for (size_t i = 0; i < n; i++) threads.emplace_back([&, i] {
		consume(runs[i]->enc, si, bc.at(i));
		packets[i]->close(); // the scorer has all there is
	});

	// Scoring runs at the same time, following the encoders' output as it is written. Runs that compare in the same format share one decode of
	// the original, which is read as far as the slowest scorer has got.
	std::vector<Fmt> formats;
	for (auto* r : runs) if (std::find(formats.begin(), formats.end(), r->enc.plan->score_fmt) == formats.end()) formats.push_back(r->enc.plan->score_fmt);
	struct Group {
		Fmt fmt;
		std::vector<size_t> members;
		std::unique_ptr<Broadcast> bc;
		RunProgress ref_rp;
		int handle = 0;
	};
	std::vector<std::unique_ptr<Group>> groups;
	for (const Fmt& f : formats) {
		auto g = std::make_unique<Group>();
		g->fmt = f;
		for (size_t i = 0; i < n; i++) if (runs[i]->enc.plan->score_fmt == f) g->members.push_back(i);
		g->bc = std::make_unique<Broadcast>(g->members.size());
		g->ref_rp.begin(3, std::max<long long>(1, si.frame_count));
		g->handle = ui::attach(&g->ref_rp, "reference decode " + std::to_string(si.width) + "x" + std::to_string(si.height), "decode");
		groups.push_back(std::move(g));
	}
	RefInfo ref;
	ref.width = si.width; ref.height = si.height; ref.framerate = si.framerate; ref.color = si.color; ref.frames = si.frame_count;
	for (auto& gp : groups) {
		Group& g = *gp;
		threads.emplace_back([&, gptr = &g] {
			Group& g = *gptr;
			auto src = open_source(input);
			if (!src) { fprintf(stderr, "cannot open %s\n", input.c_str()); g.bc->close(); return; }
			const bool convert = src->fmt != g.fmt;
			auto conv = make_converter();
			Picture t, o;
			bool ok = true, first = true;
			int r;
			auto flush_converted = [&] {
				int got;
				while (ok && (got = conv->pop(o)) == 1) ok = g.bc->push(o, &g.ref_rp);
				ok = ok && got >= 0;
			};
			while (ok && (r = src->next(t)) == 1) {
				g.ref_rp.step();
				if (!convert) { ok = g.bc->push(t, &g.ref_rp); continue; }
				t.duration = 1;
				if (first) { ok = conv->open(t, src->framerate.inverse(), src->width, src->height, g.fmt, so); first = false; }
				ok = ok && conv->push(&t);
				flush_converted();
			}
			if (ok && convert) { ok = conv->push(nullptr); flush_converted(); }
			g.ref_rp.finish();
			g.bc->close();
		});
		VmafOptions group_vo = vo;
		if (group_vo.threads <= 0) group_vo.threads = std::max(2, (int)std::thread::hardware_concurrency() / (int)runs.size());
		for (size_t m = 0; m < g.members.size(); m++) threads.emplace_back([&, gptr = &g, m, group_vo] {
			Group& g = *gptr;
			WaveRun& run = *runs[g.members[m]];
			FrameQueue& rq = g.bc->at(m);
			PacketQueue& pq = *packets[g.members[m]];
			const FrameFeed feed = [&rq, rp = run.score_rp](Picture& out) { return rq.pop(out, rp) ? 1 : 0; };
			const PacketFeed next_packet = [&pq, rp = run.score_rp](std::vector<uint8_t>& out) { return pq.pop(out, rp); };
			const ShownFeed shown = [&fr](size_t k, uint32_t& v) { return fr.shown.get(k, v); };
			run.score_ok = vmaf_score(ref, feed, run.codec, next_packet, {}, run.enc.plan->enc, shown, g.fmt, group_vo, so, run.scores, run.score_rp);
			if (run.score_ok && (long long)run.scores.size() != fr.frames) { fprintf(stderr, "%zu scores for %lld frames\n", run.scores.size(), fr.frames); run.score_ok = false; }
			if (run.score_rp) run.score_rp->finish();
			rq.cancel(); // done with the original frames: do not hold the reference decode back
			pq.cancel(); // and with the packets: do not hold the encoder back
		});
	}

	producer.join();
	for (auto& t : threads) t.join();
	for (auto& g : groups) ui::detach(g->handle);
	for (auto* r : runs) r->enc.ok = r->enc.ok && fr.ok;
	return fr.ok;
}
