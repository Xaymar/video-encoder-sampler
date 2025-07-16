#include "pipeline.h"

#include <algorithm>
#include <cmath>
#include <condition_variable>
#include <cstdio>
#include <deque>
#include <map>
#include <memory>
#include <mutex>
#include <thread>

#include "formats.h"
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
	bool push(Picture&& p) { // false once the consumer gave up
		std::unique_lock lock(m);
		cv.wait(lock, [&] { return q.size() < cap || cancelled; });
		if (cancelled) return false;
		q.push_back(std::move(p));
		cv.notify_all();
		return true;
	}
	bool pop(Picture& p) { // false when nothing more will come
		std::unique_lock lock(m);
		cv.wait(lock, [&] { return !q.empty() || closed; });
		if (q.empty()) return false;
		p = std::move(q.front());
		q.pop_front();
		cv.notify_all();
		return true;
	}
	void close() { std::lock_guard lock(m); closed = true; cv.notify_all(); }
	void cancel() { std::lock_guard lock(m); cancelled = closed = true; q.clear(); cv.notify_all(); }
};

// One queue per consumer, so the producer is held back by the slowest of them. Frames are shared, not copied.
class Broadcast {
	std::vector<std::unique_ptr<FrameQueue>> qs;

public:
	explicit Broadcast(size_t n) { for (size_t i = 0; i < n; i++) qs.push_back(std::make_unique<FrameQueue>(DECODE_AHEAD)); }
	FrameQueue& at(size_t i) { return *qs[i]; }
	bool push(const Picture& p) { // false once every consumer has given up
		bool any = false;
		for (auto& q : qs) any |= q->push(Picture(p));
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
	if (!src) { fprintf(stderr, "cannot open %s\n", input.c_str()); bc.close(); return; }
	const Shape& enc = pl.enc;
	if (rp) rp->begin(1, pl.convert_rate ? std::max<long long>(1, std::llround(si.duration * enc.fps.value())) : std::max<long long>(1, si.frame_count));
	auto scaler = make_converter();
	Picture frame, scaled;
	int64_t last_slot = -1;
	long long kept = 0;
	bool ok = true;
	int r = 1;
	auto queue_scaled = [&] {
		int got;
		while (ok && (got = scaler->pop(scaled)) == 1) ok = bc.push(scaled);
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
			if (!pl.rescale) ok = bc.push(frame);
			else {
				if (frame.pts == 0) ok = scaler->open(frame, enc.fps.inverse(), enc.w, enc.h, enc.pix, so);
				ok = ok && scaler->push(&frame);
				queue_scaled();
			}
		}
		fr.shown.push_back((uint32_t)(kept - 1));
	}
	if (ok && src->frame_count() && src->frame_count() != fr.frames) {
		fprintf(stderr, "the file lists %lld frames but %lld were decoded\n", src->frame_count(), fr.frames);
		ok = false;
	}
	if (ok && pl.rescale) { ok = scaler->push(nullptr); queue_scaled(); }
	fr.ok = ok && r >= 0;
	bc.close();
}

// One encoder taking frames from its queue.
void consume(EncodeJob& job, const SourceInfo& si, FrameQueue& q) {
	const Plan& pl = *job.plan;
	Params p;
	p.codec = job.codec;
	p.width = pl.enc.w; p.height = pl.enc.h;
	p.framerate = pl.enc.fps;
	p.pix = pl.enc.pix;
	p.bitrate = job.bitrate;
	p.gop = std::max(1, (int)std::lround(job.kf * pl.enc.fps.value()));
	p.options = pl.options;
	auto codec = make_encoder(job.encoder);
	// Packets arrive in coding order, which is also the order the decoder needs them.
	if (!codec || !codec->open(p, [&](const uint8_t* d, size_t n, int64_t pts, bool key) { job.packets.emplace_back(d, d + n); job.pts.push_back(pts); job.key.push_back(key); })) { q.cancel(); return; }
	bool ok = true;
	for (Picture pic; q.pop(pic);) {
		if (ok && !(ok = codec->encode(&pic))) q.cancel(); // frees the producer if it is waiting for room
		else if (job.rp) job.rp->step();
	}
	job.ok = ok && codec->encode(nullptr);
	job.kbps = codec->stats.bytes * 8 / si.duration / 1000;
}

} // namespace

bool encode_wave(const std::string& input, const SourceInfo& si, std::vector<EncodeJob*>& jobs, const ScaleOptions& so, RunProgress* feed_rp, FeedResult& fr) {
	Broadcast bc(jobs.size());
	for (auto* j : jobs) if (j->rp) j->rp->begin(1, std::max<long long>(1, si.frame_count));
	std::thread producer([&] { produce(input, si, *jobs[0]->plan, so, bc, fr, feed_rp); });
	std::vector<std::thread> consumers;
	for (size_t i = 0; i < jobs.size(); i++) consumers.emplace_back([&, i] { consume(*jobs[i], si, bc.at(i)); });
	for (auto& c : consumers) c.join();
	producer.join();
	for (auto* j : jobs) j->ok = j->ok && fr.ok;
	return fr.ok;
}

void score_wave(const std::string& input, const SourceInfo& si, std::vector<ScoreJob*>& jobs, const VmafOptions& vo, const ScaleOptions& so) {
	// Jobs that compare in the same format share one decode of the original.
	std::vector<Fmt> formats;
	for (auto* j : jobs) if (std::find(formats.begin(), formats.end(), j->encoded->plan->score_fmt) == formats.end()) formats.push_back(j->encoded->plan->score_fmt);
	for (const Fmt& score_fmt : formats) {
		std::vector<ScoreJob*> group;
		for (auto* j : jobs) if (j->encoded->plan->score_fmt == score_fmt) group.push_back(j);
		Broadcast bc(group.size());
		std::thread ref_producer([&] {
			auto src = open_source(input);
			if (!src) { fprintf(stderr, "cannot open %s\n", input.c_str()); bc.close(); return; }
			const bool convert = src->fmt != score_fmt;
			auto conv = make_converter();
			Picture t, o;
			bool ok = true, first = true;
			int r;
			auto flush_converted = [&] {
				int got;
				while (ok && (got = conv->pop(o)) == 1) ok = bc.push(o);
				ok = ok && got >= 0;
			};
			while (ok && (r = src->next(t)) == 1) {
				if (!convert) { ok = bc.push(t); continue; }
				t.duration = 1;
				if (first) { ok = conv->open(t, src->framerate.inverse(), src->width, src->height, score_fmt, so); first = false; }
				ok = ok && conv->push(&t);
				flush_converted();
			}
			if (ok && convert) { ok = conv->push(nullptr); flush_converted(); }
			bc.close();
		});
		VmafOptions group_vo = vo;
		if (group_vo.threads <= 0) group_vo.threads = std::max(2, (int)std::thread::hardware_concurrency() / (int)group.size());
		RefInfo ref;
		ref.width = si.width; ref.height = si.height; ref.framerate = si.framerate; ref.color = si.color;
		std::vector<std::thread> scorers;
		for (size_t i = 0; i < group.size(); i++) scorers.emplace_back([&, i] {
			ScoreJob& j = *group[i];
			FrameQueue& q = bc.at(i);
			const FrameFeed feed = [&q](Picture& out) { return q.pop(out) ? 1 : 0; };
			j.ok = vmaf_score(ref, feed, j.codec, j.encoded->packets, j.encoded->plan->enc, j.feed->shown, score_fmt, group_vo, so, j.scores, j.rp);
			if (j.ok && (long long)j.scores.size() != j.feed->frames) { fprintf(stderr, "%zu scores for %lld frames\n", j.scores.size(), j.feed->frames); j.ok = false; }
			q.cancel(); // done with the original frames: do not hold the producer back
		});
		for (auto& s : scorers) s.join();
		ref_producer.join();
	}
}
