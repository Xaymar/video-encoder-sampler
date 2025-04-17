// Sweep encoder option combinations: for each one, encode in-process, decode back and score every
// frame with VMAF, then report which configurations are the best in at least 95% of frames.
#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <cstdio>
#include <cstdlib>
#include <deque>
#include <filesystem>
#include <fstream>
#include <map>
#include <mutex>
#include <numeric>
#include <optional>
#include <set>
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

// Console output from several worker threads goes through this.
static std::mutex print_mutex;

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
	bool prune = true;                 // leave out options that change nothing, found by comparing output bytes
	std::set<std::string> fixed;       // options that are never left out, and every combination of them is screened on its own (preset, ...)
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
	std::string output_dir = "output"; // encoded streams and result files go here; empty: nothing is written
	bool save_videos = true;
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
	bool same = false;      // its encoder produced the same bytes as an earlier run, so that run's scores were reused
	std::string group;      // what runs must share for identical output to mean identical scores
	std::string encoder, codec;
	Options options;
	std::string video;      // the saved stream, relative to the output directory; empty if none was saved
	std::string video_file; // where run_one() writes it
	std::string tag;        // "[run 12/240]", for the progress lines
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

// Every combination of the axes on top of `base`.
static std::vector<Options> expand(const std::string& encoder, const std::vector<Axis>& axes, const Options& base = {}) {
	std::vector<Options> combos{base};
	for (auto& ax : axes) {
		if (ax.values.empty()) continue;
		// gpu=all means every GPU of the encoder's vendor
		std::vector<std::optional<std::string>> values;
		for (auto& v : ax.values) {
			if (ax.key == "gpu" && v && *v == "all") {
				size_t n = std::max<size_t>(list_gpus(encoder_vendor(encoder)).size(), 1);
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
	if (auto* v = root.find("output")) cfg.output_dir = v->type == Json::Bool || v->type == Json::Null ? (v->s == "1" ? "output" : "") : v->text();
	if (auto* v = root.find("videos")) cfg.save_videos = !(v->text() == "0" || v->text() == "false");
	if (auto* v = root.find("over_penalty")) over_penalty = atof(v->text().c_str());
	if (auto* v = root.find("vmaf")) {
		if (auto* m = v->find("model")) cfg.vo.model = m->text();
		if (auto* t = v->find("threads")) cfg.vo.threads = atoi(t->text().c_str());
	}
	auto off = [](const Json* v) { return v && (v->text() == "0" || v->text() == "false"); };
	const bool default_prune = !off(root.find("prune"));
	list("encoders", [&](const Json& e) {
		if (off(e.find("enabled"))) return;
		Job j;
		j.prune = default_prune && !off(e.find("prune"));
		if (auto* f = e.find("fixed")) for (auto& x : f->a) j.fixed.insert(x.text());
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

static const char* VERSION = "0.2.0";

static void print_help(FILE* f) {
	fputs(R"HELP(ves - tries encoder option combinations, scores every frame with VMAF and reports the configurations
that are the best in at least 95% of frames.

usage: ves [options]                  run a config file (config.json unless --config says otherwise)
       ves -enc <encoder> -br <kbps> -i <input> [--option key=a,b ...]    a quick run without a config file
       ves <input> <encoder> <codec> <kbps[,..]> [key=a,b ...]           the same, short form

Paths
  -cfg, --config <file>     config file (default: config.json)
  -i,   --input <path>      an input video or a directory with videos (.mkv, .mp4, .mov); replaces the
                            inputs of the config file; can be given more than once
  -o,   --output <dir>      where the results go (default: output): the encoded stream of every run in
                            <dir>/<input>/ (.h264, .h265 or .ivf) and the result file <dir>/<input>.json;
                            "off" writes nothing
        --no-videos         write the result files but not the encoded streams
  -c,   --cache <dir>       finished runs are kept here and not repeated (default: cache); "off" disables it

Scoring
        --vmaf-model <spec> libvmaf model, e.g. path=models/vmaf_4k_v0.6.1.json
        --vmaf-threads <n>
        --over-penalty <p>  VMAF points deducted per percent over the target bitrate (default: 0.025)

Quick run (replaces the encoders and scenarios of the config file)
  -enc, --encoder <name>    nvenc | amf | qsv | x264 | x265 | svtav1
  -cd,  --codec <name>      h264 | hevc | av1, for the GPU encoders
  -br,  --bitrate <kbps[,..]>
  -res, --resolution <WxH[,..]>    -fps, --framerate <n[,..]>    --kf <seconds[,..]>
        --option <key=a,b>  an encoder option, can be given more than once; comma separated values are swept
                            and every combination is tested; "null" leaves the option unset
        --fixed <key[,..]>  options that are never left out when pruning
        --no-prune          test options even when they change nothing
        --pool <name>  --parallel <n>

  -h, --help                this text
  -v, --version
)HELP", f);
}

// Command line made of --flags. Sets `quit` when there is nothing left to do (--help, --version).
static void parse_flags(int argc, char** argv, Config& cfg, bool& quit);

// Legacy one-liner: ves <input> <encoder> <codec> <kbps[,..]> [key=a,b ...]
static bool parse_cli(int argc, char** argv, Config& cfg) {
	cfg.output_dir.clear();
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
		else if (k == "output") cfg.output_dir = v;
		else if (k == "prune") j.prune = v != "off";
		else if (k == "fixed") { for (auto& x : split(v)) j.fixed.insert(x); }
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

static void parse_flags(int argc, char** argv, Config& cfg, bool& quit) {
	std::string config_path = "config.json";
	std::vector<std::string> inputs, bitrates, res, fps, kfs;
	std::optional<std::string> output, cache, model;
	std::optional<int> threads;
	std::optional<double> penalty;
	bool videos = true, quick = false;
	Job job;
	for (int i = 1; i < argc; i++) {
		std::string a = argv[i], v;
		bool inline_value = false;
		if (a.rfind("--", 0) == 0) {
			if (size_t eq = a.find('='); eq != std::string::npos) { v = a.substr(eq + 1); a.resize(eq); inline_value = true; }
		}
		auto value = [&]() -> std::string {
			if (inline_value) return v;
			if (i + 1 >= argc) throw std::runtime_error("missing value after " + a);
			return argv[++i];
		};
		if (a == "-h" || a == "--help") { print_help(stdout); quit = true; return; }
		else if (a == "-v" || a == "--version") { printf("ves %s\n", VERSION); quit = true; return; }
		else if (a == "-cfg" || a == "--config") config_path = value();
		else if (a == "-i" || a == "--input") inputs.push_back(value());
		else if (a == "-o" || a == "--output") output = value();
		else if (a == "--no-videos") videos = false;
		else if (a == "-c" || a == "--cache") cache = value();
		else if (a == "--vmaf-model") model = value();
		else if (a == "--vmaf-threads") threads = atoi(value().c_str());
		else if (a == "--over-penalty") penalty = atof(value().c_str());
		else if (a == "-enc" || a == "--encoder") { job.encoder = value(); quick = true; }
		else if (a == "-cd" || a == "--codec") job.codec = value();
		else if (a == "-br" || a == "--bitrate") bitrates = split(value());
		else if (a == "-res" || a == "--resolution") res = split(value());
		else if (a == "-fps" || a == "--framerate") fps = split(value());
		else if (a == "--kf") kfs = split(value());
		else if (a == "--option") {
			const std::string o = value();
			const size_t eq = o.find('=');
			if (eq == std::string::npos) throw std::runtime_error("bad option '" + o + "', expected key=value");
			Axis ax{o.substr(0, eq), {}};
			for (auto& x : split(o.substr(eq + 1))) ax.values.push_back(x == "null" ? std::nullopt : std::optional<std::string>(x));
			job.axes.push_back(std::move(ax));
		}
		else if (a == "--fixed") { for (auto& x : split(value())) job.fixed.insert(x); }
		else if (a == "--no-prune") job.prune = false;
		else if (a == "--pool") job.pool = value();
		else if (a == "--parallel") job.parallel = atoi(value().c_str());
		else throw std::runtime_error("unknown option '" + a + "', see --help");
	}
	if (quick) {
		if (bitrates.empty()) throw std::runtime_error("--encoder needs --bitrate");
		Scenario sc;
		for (auto& b : bitrates) sc.bitrates.push_back(atoll(b.c_str()));
		if (!kfs.empty()) { sc.kfs.clear(); for (auto& k : kfs) sc.kfs.push_back(atof(k.c_str())); }
		sc.shapes = make_shapes(res, fps);
		cfg.scenarios.push_back(std::move(sc));
		cfg.jobs.push_back(std::move(job));
	} else load_config(config_path, cfg);
	if (!inputs.empty()) { cfg.inputs.clear(); for (auto& i : inputs) add_inputs(cfg, i); }
	if (output) cfg.output_dir = *output == "off" ? "" : *output;
	cfg.save_videos = cfg.save_videos && videos;
	if (cache) cfg.cache_dir = *cache == "off" ? "" : *cache;
	if (model) cfg.vo.model = *model;
	if (threads) cfg.vo.threads = *threads;
	if (penalty) over_penalty = *penalty;
}

static Codec codec_of(const std::string& encoder, const std::string& codec) {
	if (encoder == "x264") return Codec::H264;
	if (encoder == "x265") return Codec::HEVC;
	if (encoder == "svtav1") return Codec::AV1;
	if (is_native_encoder(encoder)) return codec == "h264" ? Codec::H264 : codec == "hevc" ? Codec::HEVC : codec == "av1" ? Codec::AV1 : Codec::None;
	return Codec::None;
}

// What encoding one configuration leaves behind, ready to be decoded and scored.
struct Encoded {
	std::vector<std::vector<uint8_t>> packets;
	Shape enc;
	std::vector<uint32_t> shown;
	long long frames = 0; // source frames
	double kbps = 0;
	Fmt score_fmt;
};

// How many converted frames may wait between decoding and the encoder.
static constexpr size_t DECODE_AHEAD = 8;

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

static bool encode_one(const std::string& input, const std::string& encoder, Params p, double kf, const Shape& want, const ScaleOptions& so, Encoded& e) {
	auto src_ptr = open_source(input);
	if (!src_ptr) return false;
	Source& src = *src_ptr;
	Shape& enc = e.enc;
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
	const Fmt score_fmt = e.score_fmt = planar_fmt(sf.chroma == 422 ? 444 : sf.chroma, std::max(sf.depth, depth));
	if (!valid(p.pix) || !valid(score_fmt)) { fprintf(stderr, "unsupported chroma %d / bit depth %d\n", chroma, depth); return false; }

	auto& packets = e.packets;
	auto codec = make_encoder(encoder);
	// Packets arrive in coding order, which is also the order the decoder needs them.
	if (!codec->open(p, [&](const uint8_t* d, size_t n, int64_t, bool) { packets.emplace_back(d, d + n); })) return false;

	// Frame rate conversion drops frames before scaling. Each source frame falls into a slot of the output timeline
	// (its real presentation time times enc.fps, rounded down) and only the first frame in each slot is kept.
	// `shown[n]` records which encoded frame is on screen at source frame n, for scoring.
	auto& shown = e.shown;
	const bool convert_rate = want.fps.num != 0;
	int64_t last_slot = -1;
	long long kept = 0;
	const bool rescale = enc.w != src.width || enc.h != src.height || enc.pix != src.fmt; // scale and/or convert
	long long& frames = e.frames;

	// Decoding and converting (this thread's producer) and submitting to the encoder (this thread) are separate stages joined by a queue
	// that holds a few frames, so a fast decoder cannot run ahead of a slow encoder. The encoder's own receiving is separate where it is asynchronous.
	FrameQueue queue(DECODE_AHEAD);
	bool produced = true;
	std::thread producer([&] {
		auto scaler = make_converter();
		Picture frame, scaled;
		bool ok = true;
		int r;
		auto queue_scaled = [&] {
			int got;
			while (ok && (got = scaler->pop(scaled)) == 1) ok = queue.push(std::move(scaled));
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
				if (!rescale) ok = queue.push(Picture(frame));
				else {
					if (frame.pts == 0) ok = scaler->open(frame, enc.fps.inverse(), enc.w, enc.h, enc.pix, so);
					ok = ok && scaler->push(&frame);
					queue_scaled();
				}
			}
			shown.push_back((uint32_t)(kept - 1));
		}
		if (ok && src.frame_count() && src.frame_count() != frames) {
			fprintf(stderr, "the file lists %lld frames but %lld were decoded\n", src.frame_count(), frames);
			ok = false;
		}
		if (ok && rescale) { ok = scaler->push(nullptr); queue_scaled(); }
		produced = ok && r >= 0;
		queue.close();
	});
	bool ok = true;
	for (Picture pic; queue.pop(pic);) {
		if (ok && !(ok = codec->encode(&pic))) queue.cancel(); // frees the producer if it is waiting for room
	}
	producer.join();
	ok = ok && produced && codec->encode(nullptr);
	if (!ok) return false;
	e.kbps = codec->stats.bytes * 8 / src.duration() / 1000;
	return true;
}

// H.264 and HEVC are written as plain Annex B streams, AV1 as IVF; players and FFmpeg read both.
static bool save_stream(const std::string& path, Codec id, const Encoded& e) {
	std::ofstream f(path, std::ios::binary);
	if (!f) return false;
	auto put = [&](uint64_t v, int n) { for (int i = 0; i < n; i++) f.put((char)(v >> (8 * i))); };
	if (id == Codec::AV1) {
		f.write("DKIF", 4); put(0, 2); put(32, 2); f.write("AV01", 4);
		put(e.enc.w, 2); put(e.enc.h, 2); put(e.enc.fps.num, 4); put(e.enc.fps.den, 4); put(e.packets.size(), 4); put(0, 4);
	}
	uint64_t n = 0;
	for (auto& p : e.packets) {
		if (id == Codec::AV1) { put(p.size(), 4); put(n++, 8); }
		f.write((const char*)p.data(), (std::streamsize)p.size());
	}
	return (bool)f;
}

static std::string sanitize(std::string s, size_t max = 100) {
	for (char& c : s) if (!isalnum((unsigned char)c) && c != '-' && c != '_' && c != '.' && c != '=') c = '_';
	if (s.size() > max) s.resize(max);
	return s;
}

static uint64_t hash_packets(const Encoded& e) { // FNV-1a over every packet and its size
	uint64_t h = 1469598103934665603ull;
	auto mix = [&](uint8_t b) { h = (h ^ b) * 1099511628211ull; };
	for (auto& p : e.packets) { for (int i = 0; i < 8; i++) mix((uint8_t)((uint64_t)p.size() >> (8 * i))); for (uint8_t b : p) mix(b); }
	return h;
}

// Runs whose encoders produced the very same bytes decode to the very same frames, so they share one set of scores.
static std::mutex same_mutex;
static std::map<std::string, std::vector<double>> same_output;

static bool run_one(const std::string& input, const std::string& encoder, Codec id, Params p, double kf, const Shape& want,
                    const VmafOptions& vo, const ScaleOptions& so, Run& out) {
	Encoded e;
	if (!encode_one(input, encoder, p, kf, want, so, e)) return false;
	out.kbps = e.kbps;
	if (!out.video_file.empty() && !save_stream(out.video_file, id, e)) { fprintf(stderr, "cannot write %s\n", out.video_file.c_str()); out.video.clear(); }
	char hex[17];
	snprintf(hex, sizeof hex, "%016llx", (unsigned long long)hash_packets(e));
	const std::string same_key = out.group + "|" + hex;
	{
		std::lock_guard lock(same_mutex);
		if (auto it = same_output.find(same_key); it != same_output.end()) { out.scores = it->second; out.same = true; return true; }
	}
	{
		std::lock_guard lock(print_mutex);
		printf("%s scoring   %s\n", out.tag.c_str(), out.label.c_str());
		fflush(stdout);
	}
	auto ref = open_source(input);
	if (!ref || !vmaf_score(*ref, id, e.packets, e.enc, e.shown, e.score_fmt, vo, so, out.scores)) return false;
	if ((long long)out.scores.size() != e.frames) { fprintf(stderr, "%zu scores for %lld frames\n", out.scores.size(), e.frames); return false; }
	std::lock_guard lock(same_mutex);
	same_output[same_key] = out.scores;
	return true;
}

// Which of the swept options change anything. The baseline has every option at its default (unset when its list has a null,
// else its first value). Options are tested in groups, always moving the whole group away from the baseline: a group whose output
// bytes stay identical does nothing, so all of it is left out; otherwise the group is split in half and each half is tested again.
// If both halves do nothing alone but the group does something, its options only matter together and all of them stay.
// Whatever was left out is then tested again with the options found so far switched on.
// The encoders are deterministic, so identical bytes mean the options did nothing in this setup. A run that fails counts as a difference.
using Probe = std::function<std::optional<uint64_t>(const Options&)>; // hash of the output bytes; nullopt when the run failed

static std::vector<Axis> screen(const std::vector<Axis>& axes, const Options& base, const Probe& probe, std::string& dropped) {
	struct Info { std::optional<std::string> base; std::vector<std::optional<std::string>> alts; };
	std::vector<Info> info(axes.size());
	std::vector<size_t> swept;
	size_t rounds = 0;
	for (size_t i = 0; i < axes.size(); i++) {
		auto& v = axes[i].values;
		if (!v.empty() && std::find(v.begin(), v.end(), std::nullopt) == v.end()) info[i].base = v[0];
		for (auto& x : v) if (x != info[i].base) info[i].alts.push_back(x);
		if (!info[i].alts.empty()) { swept.push_back(i); rounds = std::max(rounds, info[i].alts.size()); }
	}
	std::map<std::string, std::optional<uint64_t>> seen;
	auto hash_with = [&](const std::vector<size_t>& set, size_t round) {
		Options o = base;
		for (size_t i = 0; i < axes.size(); i++) {
			const std::optional<std::string>* v = &info[i].base;
			if (std::find(set.begin(), set.end(), i) != set.end()) v = &info[i].alts[std::min(round, info[i].alts.size() - 1)];
			if (*v) o[axes[i].key] = **v;
		}
		std::string k;
		for (auto& [a, b] : o) k += a + "=" + b + ";";
		auto it = seen.find(k);
		if (it == seen.end()) it = seen.emplace(k, probe(o)).first;
		return it->second;
	};
	if (!hash_with({}, 0)) return axes; // nothing to compare against
	// Whether moving `set` away from its defaults changes the output, with the options in `ctx` already moved.
	auto inert = [&](std::vector<size_t> set, const std::vector<size_t>& ctx) {
		const auto base_set = ctx;
		set.insert(set.end(), ctx.begin(), ctx.end());
		for (size_t j = 0; j < rounds; j++) {
			auto h = hash_with(set, j), h0 = hash_with(base_set, j);
			if (!h || !h0 || *h != *h0) return false;
		}
		return true;
	};
	std::function<std::vector<size_t>(const std::vector<size_t>&, const std::vector<size_t>&)> solve =
	    [&](const std::vector<size_t>& set, const std::vector<size_t>& ctx) -> std::vector<size_t> {
		if (inert(set, ctx)) return {};
		if (set.size() == 1) return set;
		std::vector<size_t> a(set.begin(), set.begin() + set.size() / 2), b(set.begin() + set.size() / 2, set.end());
		std::vector<size_t> ka = solve(a, ctx), kb = solve(b, ctx);
		if (ka.empty() && kb.empty()) return set;
		ka.insert(ka.end(), kb.begin(), kb.end());
		return ka;
	};
	// An option that did nothing next to the defaults may still do something once another one is on (weightb needs b-frames),
	// so what was left out is tried again with everything found so far moved, until nothing new turns up.
	std::vector<size_t> keep;
	for (;;) {
		std::vector<size_t> rest;
		for (size_t i : swept) if (std::find(keep.begin(), keep.end(), i) == keep.end()) rest.push_back(i);
		std::vector<size_t> found = rest.empty() ? rest : solve(rest, keep);
		if (found.empty()) break;
		keep.insert(keep.end(), found.begin(), found.end());
	}
	std::vector<Axis> out = axes;
	for (size_t i : swept) {
		if (std::find(keep.begin(), keep.end(), i) != keep.end()) continue;
		out[i].values = {info[i].base}; // left out: stays at its default
		dropped += (dropped.empty() ? "" : ", ") + axes[i].key;
	}
	return out;
}

// The option combinations to run for one input and scenario. Pruning is judged at the scenario's first bitrate, key frame interval
// and shape, once for every combination of the `fixed` options (preset, ...), since those change what the others do.
// shortcut: an option that only matters at another bitrate or shape, or only together with one in a different group, can be missed; use prune=off then.
static std::vector<Options> option_sets(const Config& cfg, const std::string& input, const Job& job, const Scenario& sc) {
	if (!job.prune || !is_deterministic(job.encoder)) return expand(job.encoder, job.axes);
	std::vector<Axis> fixed, free;
	bool any = false;
	for (auto& ax : job.axes) {
		const bool pinned = job.fixed.count(ax.key) || ax.key == "gpu"; // gpu does not change the bytes, but is exactly what is being swept
		(pinned ? fixed : free).push_back(ax);
		any |= !pinned && ax.values.size() > 1;
	}
	if (!any) return expand(job.encoder, job.axes);
	std::vector<Options> out;
	for (auto& base : expand(job.encoder, fixed)) {
		const Probe probe = [&](const Options& o) -> std::optional<uint64_t> {
			Params p;
			p.codec = job.codec; p.bitrate = sc.bitrates[0] * 1000; p.options = o;
			Encoded e;
			if (!encode_one(input, job.encoder, p, sc.kfs[0], sc.shapes[0], cfg.scale, e)) return std::nullopt;
			return hash_packets(e);
		};
		std::string dropped;
		auto kept = screen(free, base, probe, dropped);
		if (!dropped.empty()) {
			std::string ctx;
			for (auto& [k, v] : base) ctx += " " + k + "=" + v;
			printf("[%s] %s/%s%s: no effect on the output, left out: %s\n", sc.name.c_str(), job.encoder.c_str(), job.codec.c_str(), ctx.c_str(), dropped.c_str());
			fflush(stdout);
		}
		auto sets = expand(job.encoder, kept, base);
		out.insert(out.end(), sets.begin(), sets.end());
	}
	return out;
}

static std::string jq(const std::string& s) { // JSON string
	std::string o = "\"";
	for (unsigned char c : s) {
		if (c == '"' || c == '\\') { o += '\\'; o += (char)c; }
		else if (c < 0x20) { char b[8]; snprintf(b, sizeof b, "\\u%04x", c); o += b; }
		else o += (char)c;
	}
	return o + "\"";
}

int main(int argc, char** argv) {
	Config cfg;
	try {
		// (config errors, including missing inputs, throw and are reported below)
		if (argc >= 2 && argv[1][0] == '-') {
			bool quit = false;
			parse_flags(argc, argv, cfg, quit);
			if (quit) return 0;
		}
		else if (argc >= 5) { if (!parse_cli(argc, argv, cfg)) return 1; }
		else if (argc == 1) load_config("config.json", cfg);
		else if (argc == 2 && !fs::is_directory(argv[1]) && fs::path(argv[1]).extension() == ".json") load_config(argv[1], cfg);
		else { print_help(stderr); return 1; }
	} catch (const std::exception& e) { fprintf(stderr, "error: %s\n", e.what()); return 1; }
	if (cfg.inputs.empty() || cfg.scenarios.empty() || cfg.jobs.empty()) { fprintf(stderr, "error: need inputs, bitrates and at least one enabled encoder (see --help)\n"); return 1; }

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
			for (auto& sc : cfg.scenarios) if (takes_part(job, sc)) {
				{ std::lock_guard lock(print_mutex); printf("Preparing %s/%s, scenario %s, %s\n", job.encoder.c_str(), job.codec.c_str(), sc.name.empty() ? "(unnamed)" : sc.name.c_str(), fs::path(input).filename().string().c_str()); fflush(stdout); }
				const std::vector<Options> sets = option_sets(cfg, input, job, sc);
			for (int64_t br : sc.bitrates) for (double kf : sc.kfs) for (auto& shape : sc.shapes) for (auto& opts : sets) {
			Task t;
			t.job = &job; t.id = id;
				t.r.input = input; t.r.bitrate = br; t.r.kf = kf; t.r.shape = shape;
				t.r.label = is_gpu_encoder(job.encoder) ? job.encoder + "/" + job.codec : job.encoder;
				if (shape.w) t.r.label += " " + std::to_string(shape.w) + "x" + std::to_string(shape.h);
				if (shape.fps.num) t.r.label += " @" + std::to_string(shape.fps.value()).substr(0, 6);
			for (auto& [k, v] : opts) t.r.label += " " + k + "=" + v;
			t.p.codec = job.codec; t.p.bitrate = br * 1000; t.p.options = opts;
			t.r.encoder = job.encoder; t.r.codec = job.codec; t.r.options = opts;
			if (!cfg.output_dir.empty() && cfg.save_videos) {
				char idx[16];
				snprintf(idx, sizeof idx, "%04zu", tasks.size());
				t.r.video = sanitize(fs::path(input).stem().string()) + "/" + idx + "_" + sanitize(t.r.label) + (id == Codec::AV1 ? ".ivf" : id == Codec::HEVC ? ".h265" : ".h264");
				t.r.video_file = (fs::path(cfg.output_dir) / t.r.video).string();
				std::error_code ec;
				fs::create_directories(fs::path(t.r.video_file).parent_path(), ec);
			}
			// Every GPU is a pool of its own, so GPUs work in parallel; CPU encoders share one pool.
			std::string gpu_index = opts.count("gpu") ? opts.at("gpu") : "0";
			t.r.group = input + "|" + job.encoder + "/" + job.codec + "|" + std::to_string(br) + "|" + std::to_string(kf) + "|" + std::to_string(shape.w) + "x" + std::to_string(shape.h) + "@" + std::to_string(shape.fps.num) + "/" + std::to_string(shape.fps.den) + "|" + gpu_index;
			if (is_gpu_encoder(job.encoder)) t.r.gpu = gpu_description(encoder_vendor(job.encoder), atoi(gpu_index.c_str()));
			std::string pool = !job.pool.empty() ? job.pool : is_gpu_encoder(job.encoder) ? job.encoder + ":" + gpu_index : "cpu";
			Pool& pl = pools[pool];
			pl.parallel = std::max(pl.parallel, job.parallel ? job.parallel : (is_gpu_encoder(job.encoder) ? 2 : 1));
			pl.tasks.push_back(tasks.size());
			tasks.push_back(std::move(t));
		}
		}
	}

	// What is about to happen.
	{
		printf("ves %s\n\nInputs (%zu)\n", VERSION, cfg.inputs.size());
		for (auto& in : cfg.inputs) printf("  %s\n", in.c_str());
		printf("Scenarios\n");
		for (auto& sc : cfg.scenarios) {
			printf("  %s: ", sc.name.empty() ? "(unnamed)" : sc.name.c_str());
			for (size_t k = 0; k < sc.bitrates.size(); k++) printf("%s%lld", k ? "," : "", (long long)sc.bitrates[k]);
			printf(" kbps, key frames every ");
			for (size_t k = 0; k < sc.kfs.size(); k++) printf("%s%g", k ? "," : "", sc.kfs[k]);
			printf(" s");
			for (auto& sh : sc.shapes) if (sh.w || sh.fps.num) printf(", %dx%d@%g", sh.w, sh.h, sh.fps.num ? sh.fps.value() : 0.0);
			printf("\n");
		}
		printf("Encoders\n");
		for (auto& job : cfg.jobs) {
			size_t runs_here = 0;
			for (auto& t : tasks) runs_here += t.job == &job;
			printf("  %s%s%s: %zu runs%s\n", job.encoder.c_str(), is_gpu_encoder(job.encoder) ? "/" : "", is_gpu_encoder(job.encoder) ? job.codec.c_str() : "", runs_here, job.prune ? "" : " (pruning off)");
		}
		printf("Pools:");
		for (auto& [name, pl] : pools) printf(" %s x%d (%zu runs)", name.c_str(), pl.parallel, pl.tasks.size());
		printf("\nCache: %s   Output: %s%s\n\n%zu runs in total\n\n", cfg.cache_dir.empty() ? "off" : cfg.cache_dir.c_str(), cfg.output_dir.empty() ? "off" : cfg.output_dir.c_str(),
		       cfg.output_dir.empty() || !cfg.save_videos ? "" : " (with encoded streams)", tasks.size());
		fflush(stdout);
	}

	// Each pool gets `parallel` workers; pools run side by side. Concurrency only affects speed, never the scores.
	const auto t0 = std::chrono::steady_clock::now();
	std::atomic<int> started{0}, finished{0};
	const int total = (int)tasks.size();
	auto span = [](double sec) {
		char b[32];
		if (sec < 90) snprintf(b, sizeof b, "%.0fs", sec);
		else if (sec < 5400) snprintf(b, sizeof b, "%.0fm%02.0fs", std::floor(sec / 60), std::fmod(sec, 60));
		else snprintf(b, sizeof b, "%.0fh%02.0fm", std::floor(sec / 3600), std::fmod(sec / 60, 60));
		return std::string(b);
	};
	std::vector<std::thread> workers;
	for (auto& [name, pl] : pools) for (int w = 0; w < pl.parallel; w++) workers.emplace_back([&, pp = &pl, pool_name = name] {
		for (size_t n; (n = pp->next++) < pp->tasks.size();) {
			Task& t = tasks[pp->tasks[n]];
			Run& r = t.r;
			char tag[48];
			snprintf(tag, sizeof tag, "[run %d/%d]", ++started, total);
			r.tag = tag;
			std::string key;
			CachedRun hit;
			if (!cfg.cache_dir.empty()) {
				key = cache_key(r.input, t.job->encoder, t.job->codec, t.p.options, r.bitrate, r.kf, r.shape, cfg.scale, cfg.vo);
				if (cache_load(cfg.cache_dir, key, hit)) { r.kbps = hit.kbps; r.scores = std::move(hit.scores); r.cached = t.ok = true; }
			}
			if (r.cached) { r.video.clear(); r.video_file.clear(); } // nothing was encoded
			else {
				{
					std::lock_guard lock(print_mutex);
					printf("%s encoding  %s  (%s, %lld kbps, kf %.1f s, pool %s)\n", r.tag.c_str(), r.label.c_str(), fs::path(r.input).filename().string().c_str(), (long long)r.bitrate, r.kf, pool_name.c_str());
					fflush(stdout);
				}
				t.ok = run_one(r.input, t.job->encoder, t.id, t.p, r.kf, r.shape, cfg.vo, cfg.scale, r);
				if (t.ok && !key.empty()) cache_store(cfg.cache_dir, key, CachedRun{r.kbps, r.scores});
			}
			std::lock_guard lock(print_mutex);
			const int done = ++finished;
			const double elapsed = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
			char progress[96];
			snprintf(progress, sizeof progress, "%d/%d done, %s, ~%s left", done, total, span(elapsed).c_str(), done < total ? span(elapsed / done * (total - done)).c_str() : "0s");
			if (!t.ok) { fprintf(stderr, "%s FAILED  %s  (%s, %lld kbps)  [%s]\n", r.tag.c_str(), r.label.c_str(), r.input.c_str(), (long long)r.bitrate, progress); continue; }
			r.penalty = std::max(0.0, r.kbps / r.bitrate - 1) * 100 * over_penalty;
			std::vector<double> sc = r.scores;
			std::sort(sc.begin(), sc.end());
			printf("%s finished  %s: %.0f kbps, vmaf mean %.3f min %.3f p5 %.3f%s%s%s  [%s]\n", r.tag.c_str(), r.label.c_str(), r.kbps,
			       std::accumulate(sc.begin(), sc.end(), 0.0) / sc.size(), sc.front(), sc[(size_t)(sc.size() * 0.05)], r.cached ? " (cached)" : r.same ? " (same output as an earlier run)" : "",
			       r.gpu.empty() ? "" : " on ", r.gpu.c_str(), progress);
			fflush(stdout);
		}
	});
	for (auto& w : workers) w.join();

	// Collect in task order so the ranking below does not depend on which run finished first.
	std::vector<Run> runs;
	std::map<std::string, std::vector<std::string>> failed_runs, group_json; // by input
	for (auto& t : tasks) { if (t.ok) runs.push_back(std::move(t.r)); else { failed++; failed_runs[t.r.input].push_back(t.r.label); } }

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

		std::ostringstream o;
		o << "{\"bitrate_kbps\":" << runs[g].bitrate << ",\"keyframe_interval_s\":" << runs[g].kf << ",\"frames\":" << frames << ",\"best_frame_share_required\":" << BEST_FRAME_SHARE << ",\"runs\":[";
		for (size_t i = g; i < e; i++) {
			const Run& r = runs[i];
			std::vector<double> sc = r.scores;
			std::sort(sc.begin(), sc.end());
			o << (i > g ? "," : "") << "{\"label\":" << jq(r.label) << ",\"encoder\":" << jq(r.encoder) << ",\"codec\":" << jq(r.codec) << ",\"options\":{";
			bool first = true;
			for (auto& [k, v] : r.options) { o << (first ? "" : ",") << jq(k) << ":" << jq(v); first = false; }
			o << "},\"width\":" << r.shape.w << ",\"height\":" << r.shape.h << ",\"fps\":" << (r.shape.fps.num ? r.shape.fps.value() : 0.0)
			  << ",\"kbps\":" << r.kbps << ",\"over_target_penalty\":" << r.penalty << ",\"best_frame_share\":" << r.best_share << ",\"winner\":" << (r.best_share >= BEST_FRAME_SHARE ? "true" : "false")
			  << ",\"vmaf\":{\"mean\":" << std::accumulate(sc.begin(), sc.end(), 0.0) / sc.size() << ",\"min\":" << sc.front() << ",\"p5\":" << sc[(size_t)(sc.size() * 0.05)] << "}"
			  << ",\"gpu\":" << (r.gpu.empty() ? "null" : jq(r.gpu)) << ",\"cached\":" << (r.cached ? "true" : "false") << ",\"same_output_as_earlier_run\":" << (r.same ? "true" : "false")
			  << ",\"video\":" << (r.video.empty() ? "null" : jq(r.video)) << "}";
		}
		o << "]}";
		group_json[runs[g].input].push_back(o.str());
		g = e;
	}

	if (!cfg.output_dir.empty()) {
		std::error_code ec;
		fs::create_directories(cfg.output_dir, ec);
		for (auto& input : cfg.inputs) {
			const fs::path file = fs::path(cfg.output_dir) / (sanitize(fs::path(input).stem().string()) + ".json");
			std::ofstream f(file);
			f << "{\"tool\":\"ves\",\"version\":" << jq(VERSION) << ",\"input\":" << jq(input) << ",\"vmaf_model\":" << jq(cfg.vo.model) << ",\"over_penalty\":" << over_penalty << ",\"failed\":[";
			for (size_t i = 0; i < failed_runs[input].size(); i++) f << (i ? "," : "") << jq(failed_runs[input][i]);
			f << "],\"groups\":[";
			for (size_t i = 0; i < group_json[input].size(); i++) f << (i ? "," : "") << group_json[input][i];
			f << "]}\n";
			if (f) printf("%sResults: %s\n", &input == &cfg.inputs.front() ? "\n" : "", file.string().c_str());
			else fprintf(stderr, "cannot write %s\n", file.string().c_str());
		}
	}
	return failed ? 2 : 0;
}
