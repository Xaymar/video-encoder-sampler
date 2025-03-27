// Sweep encoder option combinations: for each one, encode in-process, decode back and score every
// frame with VMAF, then report which configurations are the best in at least 95% of frames.
#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <map>
#include <mutex>
#include <numeric>
#include <optional>
#include <sstream>
#include <string>
#include <thread>
#include <tuple>

#include "cache.h"
#include "encoder.h"
#include "formats.h"
#include "gpu.h"
#include "convert.h"
#include "json.h"
#include "source.h"
#include "vmaf.h"

namespace fs = std::filesystem;

static constexpr double BEST_FRAME_SHARE = 0.95;

// Running under a CBR target is the encoder's own loss, so only overshoot is penalized.
// Default slope: the 3000 -> 6000 kbps sweep gains roughly 2.5 VMAF over +100%, i.e. ~0.025 per percent.
static double over_penalty = 0.025; // VMAF points per percent over target

struct Axis { // one swept option; a missing value (JSON null) leaves the option unset
	std::string key;
	std::vector<std::optional<std::string>> values;
};

struct Job {
	std::string encoder, codec = "h264";
	std::string pool;  // runs in one pool share a device; at most `parallel` of them run at once
	std::vector<std::string> scenarios; // names of the scenarios this encoder takes part in; empty: all of them
	int parallel = 0;  // 0: default for the pool (NVENC 2, everything else 1 since CPU encoders use every core)
	std::vector<Axis> axes;
};

// One sub-sweep: every combination of its lists is tested with every encoder. Several scenarios let a setup
// say that some combinations only make sense together (e.g. 1080p60 only at high bitrates).
struct Scenario {
	std::string name;
	std::vector<int64_t> bitrates;   // kbps
	std::vector<double> kfs{2.0};    // key frame intervals, seconds
	std::vector<Shape> shapes{Shape{}}; // resolution / frame rate to encode at; zero fields mean "as the source"
};

struct Config {
	std::vector<std::string> inputs;
	std::vector<Scenario> scenarios;
	VmafOptions vo;
	ScaleOptions scale;
	std::string cache_dir = "cache"; // finished runs are kept here and not repeated; empty turns it off
	std::vector<Job> jobs;
};

struct Run {
	std::string input, label;
	int64_t bitrate;        // target, kbps
	double kf;
	Shape shape;
	double kbps = 0;        // actual
	std::vector<double> scores; // per frame, display order
	bool cached = false;    // taken from the result cache instead of being run
	std::string gpu;        // name and driver of the GPU a GPU encoder ran on
	double penalty = 0;     // VMAF points deducted for exceeding the target bitrate
	double best_share = 0;  // share of frames where this run is at least as good as every other run of its group
};

static std::vector<std::string> split(const std::string& s) {
	std::vector<std::string> out;
	std::stringstream ss(s);
	for (std::string t; std::getline(ss, t, ',');) out.push_back(t);
	return out;
}

static Rational parse_fps(const std::string& s) { // "30", "29.97" or "30000/1001"
	int n, d;
	if (sscanf(s.c_str(), "%d/%d", &n, &d) == 2) return {n, d};
	return {(int)std::lround(atof(s.c_str()) * 1000), 1000}; // "29.97" -> 29970/1000, close enough to 30000/1001 for slot arithmetic
}

// Resolutions x frame rates; either list may be empty (= source).
static std::vector<Shape> make_shapes(const std::vector<std::string>& res, const std::vector<std::string>& fps) {
	std::vector<Shape> out;
	for (size_t i = 0; i < std::max<size_t>(res.size(), 1); i++) for (size_t j = 0; j < std::max<size_t>(fps.size(), 1); j++) {
		Shape s;
		if (i < res.size() && sscanf(res[i].c_str(), "%dx%d", &s.w, &s.h) != 2) throw std::runtime_error("bad resolution '" + res[i] + "', expected WxH");
		if (j < fps.size()) s.fps = parse_fps(fps[j]);
		out.push_back(s);
	}
	return out;
}

static std::vector<Options> expand(const Job& j) {
	std::vector<Options> combos(1);
	for (auto& ax : j.axes) {
		if (ax.values.empty()) continue;
		// gpu=all means every GPU of the encoder's vendor
		std::vector<std::optional<std::string>> values;
		for (auto& v : ax.values) {
			if (ax.key == "gpu" && v && *v == "all") {
				size_t n = std::max<size_t>(list_gpus(encoder_vendor(j.encoder)).size(), 1);
				for (size_t i = 0; i < n; i++) values.push_back(std::to_string(i));
			} else values.push_back(v);
		}
		std::vector<Options> next;
		for (auto& c : combos) for (auto& v : values) { Options n = c; if (v) n[ax.key] = *v; next.push_back(n); }
		combos = std::move(next);
	}
	return combos;
}

static void add_inputs(Config& cfg, const std::string& path) {
	if (!fs::exists(path)) throw std::runtime_error("input '" + path + "' does not exist");
	if (!fs::is_directory(path)) { cfg.inputs.push_back(path); return; }
	std::vector<std::string> found;
	for (auto& e : fs::directory_iterator(path)) {
		std::string ext = e.path().extension().string();
		if (ext == ".mkv" || ext == ".mp4" || ext == ".mov") found.push_back(e.path().string());
	}
	std::sort(found.begin(), found.end());
	cfg.inputs.insert(cfg.inputs.end(), found.begin(), found.end());
}

static void load_config(const std::string& path, Config& cfg) {
	std::ifstream f(path);
	if (!f) throw std::runtime_error("cannot read '" + path + "'");
	std::stringstream ss; ss << f.rdbuf();
	Json root = Json::parse(ss.str());

	auto list = [&](const char* k, auto fn) {
		if (auto* v = root.find(k)) { if (v->type != Json::Arr) throw std::runtime_error(std::string("'") + k + "' must be an array"); for (auto& e : v->a) fn(e); }
	};
	list("inputs", [&](const Json& e) { add_inputs(cfg, e.text()); });
	// Either a "scenarios" list, or the top level itself is the one and only scenario.
	// Key frame intervals set at the top level are the default for scenarios that do not list their own.
	std::vector<double> default_kfs{2.0};
	if (auto* v = root.find("kfintervals")) { default_kfs.clear(); for (auto& e : v->a) default_kfs.push_back(atof(e.text().c_str())); }
	auto scenario = [&](const Json& o) {
		Scenario sc;
		sc.kfs = default_kfs;
		if (auto* n = o.find("name")) sc.name = n->text();
		std::vector<std::string> res, fps;
		auto strs = [&](const char* k, auto fn) {
			if (auto* v = o.find(k)) { if (v->type != Json::Arr) throw std::runtime_error(std::string("'") + k + "' must be an array"); for (auto& e : v->a) fn(e.text()); }
		};
		strs("bitrates", [&](const std::string& t) { sc.bitrates.push_back(atoll(t.c_str())); });
		if (o.find("kfintervals")) { sc.kfs.clear(); strs("kfintervals", [&](const std::string& t) { sc.kfs.push_back(atof(t.c_str())); }); }
		strs("resolutions", [&](const std::string& t) { res.push_back(t); });
		strs("framerates", [&](const std::string& t) { fps.push_back(t); });
		sc.shapes = make_shapes(res, fps);
		if (sc.bitrates.empty()) throw std::runtime_error("a scenario needs bitrates");
		cfg.scenarios.push_back(std::move(sc));
	};
	if (auto* v = root.find("scenarios")) { for (auto& e : v->a) scenario(e); } else scenario(root);
	if (auto* v = root.find("scaling")) {
		if (auto* u = v->find("upscaler")) cfg.scale.upscaler = u->text();
		if (auto* d = v->find("downscaler")) cfg.scale.downscaler = d->text();
		if (auto* g = v->find("gpu")) cfg.scale.device = g->text();
	}
	if (auto* v = root.find("cache")) cfg.cache_dir = v->type == Json::Bool || v->type == Json::Null ? (v->s == "1" ? "cache" : "") : v->text();
	if (auto* v = root.find("over_penalty")) over_penalty = atof(v->text().c_str());
	if (auto* v = root.find("vmaf")) {
		if (auto* m = v->find("model")) cfg.vo.model = m->text();
		if (auto* t = v->find("threads")) cfg.vo.threads = atoi(t->text().c_str());
	}
	list("encoders", [&](const Json& e) {
		if (auto* en = e.find("enabled"); en && en->text() == "0") return;
		Job j;
		auto* name = e.find("encoder");
		if (!name) throw std::runtime_error("encoder entry without 'encoder'");
		j.encoder = name->text();
		if (auto* c = e.find("codec")) j.codec = c->text();
		if (auto* p = e.find("pool")) j.pool = p->text();
		if (auto* p = e.find("parallel")) j.parallel = atoi(p->text().c_str());
		if (auto* sn = e.find("scenarios")) for (auto& x : sn->a) j.scenarios.push_back(x.text());
		if (auto* o = e.find("options")) for (auto& [k, v] : o->o) {
			Axis ax{k, {}};
			if (v.type == Json::Arr) { for (auto& x : v.a) ax.values.push_back(x.type == Json::Null ? std::nullopt : std::optional(x.text())); }
			else ax.values.push_back(v.type == Json::Null ? std::nullopt : std::optional(v.text()));
			j.axes.push_back(std::move(ax));
		}
		cfg.jobs.push_back(std::move(j));
	});
}

// Legacy one-liner: ves <input> <encoder> <codec> <kbps[,..]> [key=a,b ...]
static bool parse_cli(int argc, char** argv, Config& cfg) {
	add_inputs(cfg, argv[1]);
	Job j;
	j.encoder = argv[2]; j.codec = argv[3];
	Scenario sc;
	std::vector<std::string> res, fps;
	for (auto& b : split(argv[4])) sc.bitrates.push_back(atoll(b.c_str()));
	for (int i = 5; i < argc; i++) {
		std::string a = argv[i];
		size_t eq = a.find('=');
		if (eq == std::string::npos) { fprintf(stderr, "bad option '%s', expected key=value\n", argv[i]); return false; }
		std::string k = a.substr(0, eq), v = a.substr(eq + 1);
		if (k == "vmaf.model") cfg.vo.model = v;
		else if (k == "vmaf.threads") cfg.vo.threads = atoi(v.c_str());
		else if (k == "over.penalty") over_penalty = atof(v.c_str());
		else if (k == "cache") cfg.cache_dir = v == "off" ? "" : v;
		else if (k == "pool") j.pool = v;
		else if (k == "parallel") j.parallel = atoi(v.c_str());
		else if (k == "kf") { sc.kfs.clear(); for (auto& x : split(v)) sc.kfs.push_back(atof(x.c_str())); }
		else if (k == "res") res = split(v);
		else if (k == "fps") fps = split(v);
		else { Axis ax{k, {}}; for (auto& x : split(v)) ax.values.push_back(x); j.axes.push_back(std::move(ax)); }
	}
	sc.shapes = make_shapes(res, fps);
	cfg.scenarios.push_back(std::move(sc));
	cfg.jobs.push_back(std::move(j));
	return true;
}

static Codec codec_of(const std::string& encoder, const std::string& codec) {
	if (encoder == "x264") return Codec::H264;
	if (encoder == "x265") return Codec::HEVC;
	if (encoder == "svtav1") return Codec::AV1;
	if (is_native_encoder(encoder)) return codec == "h264" ? Codec::H264 : codec == "hevc" ? Codec::HEVC : codec == "av1" ? Codec::AV1 : Codec::None;
	return Codec::None;
}

static bool run_one(const std::string& input, const std::string& encoder, Codec id, Params p, double kf, const Shape& want,
                    const VmafOptions& vo, const ScaleOptions& so, Run& out) {
	auto src_ptr = open_source(input);
	if (!src_ptr) return false;
	Source& src = *src_ptr;
	Shape enc;
	enc.w = want.w ? want.w : src.width;
	enc.h = want.h ? want.h : src.height;
	enc.fps = want.fps.num ? want.fps : src.framerate;
	if (enc.w > src.width || enc.h > src.height) { fprintf(stderr, "%dx%d is larger than the source (%dx%d)\n", enc.w, enc.h, src.width, src.height); return false; }
	if (compare(enc.fps, src.framerate) > 0) { fprintf(stderr, "frame rate %.3f is higher than the source (%.3f)\n", enc.fps.value(), src.framerate.value()); return false; }
	p.width = enc.w; p.height = enc.h;
	p.framerate = enc.fps;
	p.gop = std::max(1, (int)std::lround(kf * enc.fps.value()));

	// Encoder input format: "chroma" and "bitdepth" options if given, else the source's, else the closest the encoder takes.
	const Fmt sf = src.fmt;
	// What the config asks for (chroma / bitdepth options) is used as is; what it leaves open follows the source,
	// falling back to the closest thing the encoder takes: another chroma layout (4:2:0), then a lower depth.
	std::vector<int> chromas{sf.chroma, 420}, depths{sf.depth, 10, 8};
	if (auto it = p.options.find("chroma"); it != p.options.end()) { chromas = {atoi(it->second.c_str())}; p.options.erase(it); }
	if (auto it = p.options.find("bitdepth"); it != p.options.end()) { depths = {atoi(it->second.c_str())}; p.options.erase(it); }
	int chroma = 0, depth = 0;
	for (int c : chromas) for (int d : depths) {
		if (!chroma && (d <= sf.depth || depths.size() == 1) && encoder_supports(encoder, p.codec, c, d)) { chroma = c; depth = d; }
	}
	if (!chroma) { fprintf(stderr, "%s cannot encode %d-bit %d here\n", encoder.c_str(), depths[0], chromas[0]); return false; }
	p.pix = enc.pix = planar_fmt(chroma, depth);
	enc.has_pix = true;
	// Both sides are scored in the source's chroma layout at the higher of the two bit depths.
	// 4:2:2 sources are scored as 4:4:4, which holds all of their chroma.
	const Fmt score_fmt = planar_fmt(sf.chroma == 422 ? 444 : sf.chroma, std::max(sf.depth, depth));
	if (!valid(p.pix) || !valid(score_fmt)) { fprintf(stderr, "unsupported chroma %d / bit depth %d\n", chroma, depth); return false; }

	std::vector<std::vector<uint8_t>> packets;
	auto codec = make_encoder(encoder);
	// Packets arrive in coding order, which is also the order the decoder needs them.
	if (!codec->open(p, [&](const uint8_t* d, size_t n, int64_t, bool) { packets.emplace_back(d, d + n); })) return false;

	// Frame rate conversion drops frames before scaling. Each source frame falls into a slot of the output timeline
	// (its real presentation time times enc.fps, rounded down) and only the first frame in each slot is kept.
	// `shown[n]` records which encoded frame is on screen at source frame n, for scoring.
	std::vector<uint32_t> shown;
	const bool convert_rate = want.fps.num != 0;
	int64_t last_slot = -1;
	long long kept = 0;
	const bool rescale = enc.w != src.width || enc.h != src.height || enc.pix != src.fmt; // scale and/or convert
	auto scaler = make_converter();
	Picture frame, scaled;
	long long frames = 0; // source frames
	bool ok = true;
	int r;
	auto encode_scaled = [&] {
		int got;
		while (ok && (got = scaler->pop(scaled)) == 1) ok = codec->encode(&scaled);
		ok = ok && got >= 0;
	};
	while (ok && (r = src.next(frame)) == 1) {
		const int64_t slot = src.slot(frames, enc.fps);
		const bool keep = !convert_rate || frames == 0 || slot != last_slot; // no target rate: every frame stays
		last_slot = slot;
		frames++;
		if (keep) {
			frame.pts = kept++; // encoded frames are numbered consecutively, whatever the gaps in the source timing
			frame.duration = 1; // one tick of the 1/enc.fps time base
			if (!rescale) ok = codec->encode(&frame);
			else {
				if (frame.pts == 0) ok = scaler->open(frame, enc.fps.inverse(), enc.w, enc.h, enc.pix, so);
				ok = ok && scaler->push(&frame);
				encode_scaled();
			}
		}
		shown.push_back((uint32_t)(kept - 1));
	}
	if (ok && src.frame_count() && src.frame_count() != frames) {
		fprintf(stderr, "the file lists %lld frames but %lld were decoded\n", src.frame_count(), frames);
		ok = false;
	}
	if (ok && rescale) { ok = scaler->push(nullptr); encode_scaled(); }
	ok = ok && r >= 0 && codec->encode(nullptr);
	if (!ok) return false;
	out.kbps = codec->stats.bytes * 8 / src.duration() / 1000;
	codec.reset();

	auto ref = open_source(input);
	if (!ref || !vmaf_score(*ref, id, packets, enc, shown, score_fmt, vo, so, out.scores)) return false;
	if ((long long)out.scores.size() != frames) { fprintf(stderr, "%zu scores for %lld frames\n", out.scores.size(), frames); return false; }
	return true;
}

int main(int argc, char** argv) {
	Config cfg;
	try {
		if (argc >= 5) { if (!parse_cli(argc, argv, cfg)) return 1; }
		// (config errors, including missing inputs, throw and are reported below)
		else if (argc <= 2) load_config(argc == 2 ? argv[1] : "config.json", cfg);
		else {
			fprintf(stderr, "usage: ves [config.json]\n"
			                "       ves <input> <encoder> <codec> <bitrate-kbps[,..]> [key=value[,value..] ...]\n"
			                "  encoder: nvenc | amf | x264 | x265 | svtav1 (codec is ignored for the CPU encoders)\n"
			                "  comma separated values are swept; every combination is tested\n"
			                "  kf=<seconds>, vmaf.model=<spec>, vmaf.threads=<n>, over.penalty=<vmaf points per %% over target>;\n"
			                "  other keys go to the encoder\n");
			return 1;
		}
	} catch (const std::exception& e) { fprintf(stderr, "config: %s\n", e.what()); return 1; }
	if (cfg.inputs.empty() || cfg.scenarios.empty() || cfg.jobs.empty()) { fprintf(stderr, "config needs inputs, bitrates and at least one enabled encoder\n"); return 1; }

	struct Task {
		Run r;
		const Job* job;
		Codec id;
		Params p;
		bool ok = false;
	};
	struct Pool {
		std::vector<size_t> tasks;
		std::atomic<size_t> next{0};
		int parallel = 1;
	};
	std::vector<Task> tasks;
	std::map<std::string, Pool> pools;
	int failed = 0;
	for (auto& job : cfg.jobs) for (auto& want : job.scenarios) {
		bool known = false;
		for (auto& sc : cfg.scenarios) known |= sc.name == want;
		if (!known) { fprintf(stderr, "config: encoder '%s' lists unknown scenario '%s'\n", job.encoder.c_str(), want.c_str()); return 1; }
	}
	auto takes_part = [](const Job& job, const Scenario& sc) {
		return job.scenarios.empty() || std::find(job.scenarios.begin(), job.scenarios.end(), sc.name) != job.scenarios.end();
	};

	for (auto& input : cfg.inputs) for (auto& job : cfg.jobs) {
		const Codec id = codec_of(job.encoder, job.codec);
		if (id == Codec::None) { fprintf(stderr, "unknown encoder/codec '%s'/'%s'\n", job.encoder.c_str(), job.codec.c_str()); failed++; continue; }
			for (auto& sc : cfg.scenarios) if (takes_part(job, sc)) for (int64_t br : sc.bitrates) for (double kf : sc.kfs) for (auto& shape : sc.shapes) for (auto& opts : expand(job)) {
			Task t;
			t.job = &job; t.id = id;
				t.r.input = input; t.r.bitrate = br; t.r.kf = kf; t.r.shape = shape;
				t.r.label = is_gpu_encoder(job.encoder) ? job.encoder + "/" + job.codec : job.encoder;
				if (shape.w) t.r.label += " " + std::to_string(shape.w) + "x" + std::to_string(shape.h);
				if (shape.fps.num) t.r.label += " @" + std::to_string(shape.fps.value()).substr(0, 6);
			for (auto& [k, v] : opts) t.r.label += " " + k + "=" + v;
			t.p.codec = job.codec; t.p.bitrate = br * 1000; t.p.options = opts;
			// Every GPU is a pool of its own, so GPUs work in parallel; CPU encoders share one pool.
			std::string gpu_index = opts.count("gpu") ? opts.at("gpu") : "0";
			if (is_gpu_encoder(job.encoder)) t.r.gpu = gpu_description(encoder_vendor(job.encoder), atoi(gpu_index.c_str()));
			std::string pool = !job.pool.empty() ? job.pool : is_gpu_encoder(job.encoder) ? job.encoder + ":" + gpu_index : "cpu";
			Pool& pl = pools[pool];
			pl.parallel = std::max(pl.parallel, job.parallel ? job.parallel : (is_gpu_encoder(job.encoder) ? 2 : 1));
			pl.tasks.push_back(tasks.size());
			tasks.push_back(std::move(t));
		}
	}

	// Each pool gets `parallel` workers; pools run side by side. Concurrency only affects speed, never the scores.
	std::mutex out_mutex;
	std::vector<std::thread> workers;
	for (auto& [name, pl] : pools) for (int w = 0; w < pl.parallel; w++) workers.emplace_back([&, pp = &pl] {
		for (size_t n; (n = pp->next++) < pp->tasks.size();) {
			Task& t = tasks[pp->tasks[n]];
			Run& r = t.r;
			std::string key;
			CachedRun hit;
			if (!cfg.cache_dir.empty()) {
				key = cache_key(r.input, t.job->encoder, t.job->codec, t.p.options, r.bitrate, r.kf, r.shape, cfg.scale, cfg.vo);
				if (cache_load(cfg.cache_dir, key, hit)) { r.kbps = hit.kbps; r.scores = std::move(hit.scores); r.cached = t.ok = true; }
			}
			if (!r.cached) {
				t.ok = run_one(r.input, t.job->encoder, t.id, t.p, r.kf, r.shape, cfg.vo, cfg.scale, r);
				if (t.ok && !key.empty()) cache_store(cfg.cache_dir, key, CachedRun{r.kbps, r.scores});
			}
			std::lock_guard lock(out_mutex);
			if (!t.ok) { fprintf(stderr, "[%s %lld kbps kf %.1f] %s: FAILED\n", r.input.c_str(), (long long)r.bitrate, r.kf, r.label.c_str()); continue; }
			r.penalty = std::max(0.0, r.kbps / r.bitrate - 1) * 100 * over_penalty;
			std::vector<double> s = r.scores;
			std::sort(s.begin(), s.end());
			printf("[%s %lld kbps kf %.1f] %s: %.0f kbps, vmaf mean %.3f min %.3f p5 %.3f%s%s%s\n", r.input.c_str(), (long long)r.bitrate, r.kf, r.label.c_str(), r.kbps,
			       std::accumulate(s.begin(), s.end(), 0.0) / s.size(), s.front(), s[(size_t)(s.size() * 0.05)], r.cached ? " (cached)" : "",
			       r.gpu.empty() ? "" : " on ", r.gpu.c_str());
			fflush(stdout);
		}
	});
	for (auto& w : workers) w.join();

	// Collect in task order so the ranking below does not depend on which run finished first.
	std::vector<Run> runs;
	for (auto& t : tasks) { if (t.ok) runs.push_back(std::move(t.r)); else failed++; }

	// A group is every run that shares input, bitrate target and key frame interval.
	auto key = [](const Run& r) { return std::make_tuple(r.input, -r.bitrate, r.kf); };
	std::stable_sort(runs.begin(), runs.end(), [&](const Run& a, const Run& b) { return key(a) < key(b); });
	for (size_t g = 0; g < runs.size();) {
		size_t e = g;
		while (e < runs.size() && key(runs[e]) == key(runs[g])) e++;
		size_t frames = runs[g].scores.size();
		// Count the frames on which a run is at least as good as all others (after the overshoot penalty).
		for (size_t i = g; i < e; i++) {
			size_t best = 0;
			for (size_t f = 0; f < frames; f++) {
				double top = 0;
				for (size_t j = g; j < e; j++) top = std::max(top, runs[j].scores[f] - runs[j].penalty);
				best += runs[i].scores[f] - runs[i].penalty >= top - 1e-6;
			}
			runs[i].best_share = (double)best / frames;
		}
		std::sort(runs.begin() + g, runs.begin() + e, [](const Run& a, const Run& b) { return a.best_share > b.best_share; });
		printf("\n== %s, %lld kbps, kf %.1f: best in >= %.0f%% of frames ==\n", runs[g].input.c_str(), (long long)runs[g].bitrate, runs[g].kf, BEST_FRAME_SHARE * 100);
		bool any = false;
		for (size_t i = g; i < e; i++) {
			bool win = runs[i].best_share >= BEST_FRAME_SHARE;
			any |= win;
			printf("%s %5.1f%%  %s  (%.0f kbps, penalty %.2f%s%s)\n", win ? "*" : " ", runs[i].best_share * 100, runs[i].label.c_str(), runs[i].kbps, runs[i].penalty,
			       runs[i].gpu.empty() ? "" : ", ", runs[i].gpu.c_str());
		}
		if (!any) printf("  no configuration reaches %.0f%%; the top entry is the closest\n", BEST_FRAME_SHARE * 100);
		g = e;
	}
	return failed ? 2 : 0;
}
