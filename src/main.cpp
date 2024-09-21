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

#include "encoder.h"
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
	int parallel = 0;  // 0: default for the pool (NVENC 2, everything else 1 since CPU encoders use every core)
	std::vector<Axis> axes;
};

struct Config {
	std::vector<std::string> inputs;
	std::vector<int64_t> bitrates;   // kbps
	std::vector<double> kfs{2.0};    // key frame intervals, seconds
	VmafOptions vo;
	std::vector<Job> jobs;
};

struct Run {
	std::string input, label;
	int64_t bitrate;        // target, kbps
	double kf;
	double kbps = 0;        // actual
	std::vector<double> scores; // per frame, display order
	double penalty = 0;     // VMAF points deducted for exceeding the target bitrate
	double best_share = 0;  // share of frames where this run is at least as good as every other run of its group
};

static std::vector<std::string> split(const std::string& s) {
	std::vector<std::string> out;
	std::stringstream ss(s);
	for (std::string t; std::getline(ss, t, ',');) out.push_back(t);
	return out;
}

static std::vector<Options> expand(const Job& j) {
	std::vector<Options> combos(1);
	for (auto& ax : j.axes) {
		if (ax.values.empty()) continue;
		std::vector<Options> next;
		for (auto& c : combos) for (auto& v : ax.values) { Options n = c; if (v) n[ax.key] = *v; next.push_back(n); }
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
	list("bitrates", [&](const Json& e) { cfg.bitrates.push_back(atoll(e.text().c_str())); });
	if (root.find("kfintervals")) { cfg.kfs.clear(); list("kfintervals", [&](const Json& e) { cfg.kfs.push_back(atof(e.text().c_str())); }); }
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
	for (auto& b : split(argv[4])) cfg.bitrates.push_back(atoll(b.c_str()));
	for (int i = 5; i < argc; i++) {
		std::string a = argv[i];
		size_t eq = a.find('=');
		if (eq == std::string::npos) { fprintf(stderr, "bad option '%s', expected key=value\n", argv[i]); return false; }
		std::string k = a.substr(0, eq), v = a.substr(eq + 1);
		if (k == "vmaf.model") cfg.vo.model = v;
		else if (k == "vmaf.threads") cfg.vo.threads = atoi(v.c_str());
		else if (k == "over.penalty") over_penalty = atof(v.c_str());
		else if (k == "pool") j.pool = v;
		else if (k == "parallel") j.parallel = atoi(v.c_str());
		else if (k == "kf") { cfg.kfs.clear(); for (auto& x : split(v)) cfg.kfs.push_back(atof(x.c_str())); }
		else { Axis ax{k, {}}; for (auto& x : split(v)) ax.values.push_back(x); j.axes.push_back(std::move(ax)); }
	}
	cfg.jobs.push_back(std::move(j));
	return true;
}

static bool codec_id(const std::string& encoder, const std::string& codec, AVCodecID& id) {
	if (encoder == "x264" || encoder == "x265") {
		id = encoder == "x264" ? AV_CODEC_ID_H264 : AV_CODEC_ID_HEVC;
	} else if (is_native_encoder(encoder)) {
		id = codec == "h264" ? AV_CODEC_ID_H264 : codec == "hevc" ? AV_CODEC_ID_HEVC : codec == "av1" ? AV_CODEC_ID_AV1 : AV_CODEC_ID_NONE;
	} else {
		const AVCodec* c = avcodec_find_encoder_by_name(encoder.c_str());
		id = c ? c->id : AV_CODEC_ID_NONE;
	}
	return id != AV_CODEC_ID_NONE;
}

static bool run_one(const std::string& input, const std::string& encoder, AVCodecID id, Params p, double kf, const VmafOptions& vo, Run& out) {
	Source src;
	if (!src.open(input)) return false;
	p.width = src.width; p.height = src.height;
	p.framerate = src.framerate;
	p.gop = std::max(1, (int)std::lround(kf * av_q2d(p.framerate)));

	std::vector<std::vector<uint8_t>> packets;
	auto enc = make_encoder(encoder);
	// Packets arrive in coding order, which is also the order the decoder needs them.
	if (!enc->open(p, [&](const uint8_t* d, size_t n, int64_t, bool) { packets.emplace_back(d, d + n); })) return false;

	AVFrame* frame = av_frame_alloc();
	long long frames = 0;
	bool ok = true;
	int r;
	while (ok && (r = src.next(frame)) == 1) { ok = enc->encode(frame); av_frame_unref(frame); frames++; }
	ok = ok && r >= 0 && enc->encode(nullptr);
	av_frame_free(&frame);
	if (!ok) return false;
	out.kbps = enc->stats.bytes * 8 / (frames / av_q2d(p.framerate)) / 1000;
	enc.reset();

	Source ref;
	if (!ref.open(input) || !vmaf_score(ref, id, packets, vo, out.scores)) return false;
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
			                "  encoder: nvenc | any libavcodec encoder name (codec is ignored for the latter)\n"
			                "  comma separated values are swept; every combination is tested\n"
			                "  kf=<seconds>, vmaf.model=<spec>, vmaf.threads=<n>, over.penalty=<vmaf points per %% over target>;\n"
			                "  other keys go to the encoder\n");
			return 1;
		}
	} catch (const std::exception& e) { fprintf(stderr, "config: %s\n", e.what()); return 1; }
	if (cfg.inputs.empty() || cfg.bitrates.empty() || cfg.jobs.empty()) { fprintf(stderr, "config needs inputs, bitrates and at least one enabled encoder\n"); return 1; }

	struct Task {
		Run r;
		const Job* job;
		AVCodecID id;
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
	for (auto& input : cfg.inputs) for (auto& job : cfg.jobs) {
		AVCodecID id;
		if (!codec_id(job.encoder, job.codec, id)) { fprintf(stderr, "unknown encoder/codec '%s'/'%s'\n", job.encoder.c_str(), job.codec.c_str()); failed++; continue; }
		std::string pool = job.pool.empty() ? (is_gpu_encoder(job.encoder) ? job.encoder : "cpu") : job.pool;
		Pool& pl = pools[pool];
		pl.parallel = std::max(pl.parallel, job.parallel ? job.parallel : (is_gpu_encoder(job.encoder) ? 2 : 1));
		for (int64_t br : cfg.bitrates) for (double kf : cfg.kfs) for (auto& opts : expand(job)) {
			Task t;
			t.job = &job; t.id = id;
			t.r.input = input; t.r.bitrate = br; t.r.kf = kf;
			t.r.label = is_gpu_encoder(job.encoder) ? job.encoder + "/" + job.codec : job.encoder;
			for (auto& [k, v] : opts) t.r.label += " " + k + "=" + v;
			t.p.codec = job.codec; t.p.bitrate = br * 1000; t.p.options = opts;
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
			t.ok = run_one(r.input, t.job->encoder, t.id, t.p, r.kf, cfg.vo, r);
			std::lock_guard lock(out_mutex);
			if (!t.ok) { fprintf(stderr, "[%s %lld kbps kf %.1f] %s: FAILED\n", r.input.c_str(), (long long)r.bitrate, r.kf, r.label.c_str()); continue; }
			r.penalty = std::max(0.0, r.kbps / r.bitrate - 1) * 100 * over_penalty;
			std::vector<double> s = r.scores;
			std::sort(s.begin(), s.end());
			printf("[%s %lld kbps kf %.1f] %s: %.0f kbps, vmaf mean %.3f min %.3f p5 %.3f\n", r.input.c_str(), (long long)r.bitrate, r.kf, r.label.c_str(), r.kbps,
			       std::accumulate(s.begin(), s.end(), 0.0) / s.size(), s.front(), s[(size_t)(s.size() * 0.05)]);
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
			printf("%s %5.1f%%  %s  (%.0f kbps, penalty %.2f)\n", win ? "*" : " ", runs[i].best_share * 100, runs[i].label.c_str(), runs[i].kbps, runs[i].penalty);
		}
		if (!any) printf("  no configuration reaches %.0f%%; the top entry is the closest\n", BEST_FRAME_SHARE * 100);
		g = e;
	}
	return failed ? 2 : 0;
}
