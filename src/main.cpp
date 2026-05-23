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
#include <random>
#include <set>
#include <sstream>
#include <string>
#include <thread>
#include <tuple>

#include "cache.h"
#include "hash.h"
#include "upload.h"
#include "priority.h"
#include "encoder.h"
#include "formats.h"
#include "gpu.h"
#include "convert.h"
#include "download.h"
#include "json.h"
#include "source.h"
#include "mux.h"
#include "pipeline.h"
#include "ui.h"
#include "vmaf.h"

namespace fs = std::filesystem;

static constexpr double BEST_FRAME_SHARE = 0.95;

// Running under a CBR target is the encoder's own loss, so only overshoot is penalized.
// Default slope: the 3000 -> 6000 kbps sweep gains roughly 2.5 VMAF over +100%, i.e. ~0.025 per percent.
static double over_penalty = 0.025; // VMAF points per percent over target, for a hair over; it steepens from there
// The same curve as the website's over_penalty(): slope * width * (e^(x / width) - 1) with x the percent over target (2% over loses 0.07, 10% 1.9, 20% 59), at most 100.
static double over_penalty_points(double kbps, double target) {
	const double width = 3.0;
	if (target <= 0 || kbps <= target) return 0;
	return std::min(100.0, over_penalty * width * (std::exp((kbps / target - 1) * 100 / width) - 1));
}

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
	std::vector<std::string> chroma, bitdepth, range; // pixel format to encode in (420, 444; 8, 10; source, limited, full); empty: the source's
};

struct Config {
	std::vector<std::string> inputs;
	std::vector<Scenario> scenarios;
	VmafOptions vo;
	ScaleOptions scale;
	std::string cache_dir = "cache"; // finished runs are kept here and not repeated; empty turns it off
	std::string output_dir = "output"; // encoded streams and result files go here; empty: nothing is written
	bool save_videos = true;
	bool best_videos = false; // keep only the stream of the best run so far of each group
	UploadOptions upload; // where the results go after a sweep; no address: nowhere
	bool plain = false; // no live display, just lines
	std::string hw_decode = "auto"; // "" software only, "auto", or a device type or decoder name: see set_hw_decode()
	int decode_threads = 0;  // 0: as many as the cores allow within a memory budget
	std::string priority = "low"; // "low": idle CPU and lowest disk priority, see priority.h; "normal" leaves the system's default
	bool low_memory = false; // score a wave before encoding the next, and fewer VMAF threads
	std::vector<Job> jobs;
};

static const std::string NULL_JSON = "null";

struct Run {
	std::string input, label;
	int64_t bitrate;        // target, kbps
	double kf;
	Shape shape;
	double kbps = 0;        // actual
	std::vector<double> scores; // per frame, display order
	bool cached = false;    // taken from the result cache instead of being run
	std::vector<std::string> identical; // labels of the runs that produced exactly the same stream (left out of the ranking, see below)
	std::string group;      // what runs must share for identical output to mean identical scores
	std::string encoder, codec;
	Options options;
	std::string video;      // the saved stream, relative to the output directory; empty if none was saved
	std::string video_file; // where run_one() writes it
	std::string video_sha256; // SHA-256 of the encoded stream
	std::string result;     // its result file (options, bitrates, per-frame VMAF), relative to the output directory; empty if none
	std::string tag;        // "[run 12/240]", for the progress lines
	bool ignored = false;   // incompatible with the input or the encoder, so it was not run
	bool expected = false;  // because the input is smaller than the request: not reported run by run
	std::string reason;
	std::shared_ptr<RunProgress> progress, score_progress; // the encoding and the scoring of this run
	double encode_s = 0, score_s = 0; // wall time of the encoding wave this run was in, and of the scoring wave
	double enc_busy_s = 0, enc_starved_s = 0, decode_blocked_s = 0, score_starved_s = 0; // where that time went
	std::string gpu;        // name and driver of the GPU a GPU encoder ran on
	const std::string *gpu_json = &NULL_JSON, *caps_json = &NULL_JSON; // its PCI ids, and what its encoder reports about itself (see encoder_caps_json); shared by all runs on the GPU
	double penalty = 0;     // VMAF points deducted for exceeding the target bitrate
	double best_share = 0;  // share of frames where this run is at least as good as every other run of its group
};

static std::vector<std::string> split(const std::string& s) {
	std::vector<std::string> out;
	std::stringstream ss(s);
	for (std::string t; std::getline(ss, t, ',');) out.push_back(t);
	return out;
}

static Rational parse_fps(const std::string& s) { // "30", "29.97" (= 30000/1001) or "30000/1001"
	int n, d;
	if (sscanf(s.c_str(), "%d/%d", &n, &d) == 2) {
		if (n <= 0 || d <= 0) throw std::runtime_error("bad frame rate '" + s + "'");
		return {n, d};
	}
	const double v = atof(s.c_str());
	if (!(v > 0)) throw std::runtime_error("bad frame rate '" + s + "'");
	// The NTSC rates are written as decimals (23.976, 29.97, 59.94) but are n*1000/1001 exactly.
	for (int k : {24, 30, 48, 60, 120, 240}) if (std::fabs(v - k * 1000.0 / 1001) < 0.0005 * k) return {k * 1000, 1001};
	return {(int)std::lround(v * 1000), 1000};
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
	if (is_url(path)) { cfg.inputs.push_back(path); return; } // fetched when the first run that needs it comes up, see local_path() in main()
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
		strs("chroma", [&](const std::string& t) { sc.chroma.push_back(t); });
		strs("bitdepth", [&](const std::string& t) { sc.bitdepth.push_back(t); });
		strs("range", [&](const std::string& t) { sc.range.push_back(t); });
		if (auto* u = o.find("upscale"); u && (u->text() == "1" || u->text() == "true")) for (auto& sh : sc.shapes) sh.upscale = true;
		if (sc.bitrates.empty()) throw std::runtime_error("a scenario needs bitrates");
		cfg.scenarios.push_back(std::move(sc));
	};
	if (auto* v = root.find("scenarios")) { for (auto& e : v->a) scenario(e); } else scenario(root);
	if (auto* v = root.find("scaling")) {
		if (auto* u = v->find("upscaler")) cfg.scale.upscaler = u->text();
		if (auto* d = v->find("downscaler")) cfg.scale.downscaler = d->text();
		if (auto* g = v->find("gpu")) cfg.scale.device = g->text();
	}
	// These live in "settings"; older files have them at the top level, which still works.
	auto setting = [&](const char* k) -> const Json* {
		if (auto* g = root.find("settings")) if (auto* v = g->find(k)) return v;
		return root.find(k);
	};
	if (auto* v = setting("cache")) cfg.cache_dir = v->type == Json::Bool || v->type == Json::Null ? (v->s == "1" ? "cache" : "") : v->text();
	if (auto* v = root.find("output")) cfg.output_dir = v->type == Json::Bool || v->type == Json::Null ? (v->s == "1" ? "output" : "") : v->text();
	if (auto* v = setting("hw_decode")) cfg.hw_decode = v->type == Json::Bool || v->type == Json::Null ? (v->s == "1" ? "auto" : "") : v->text();
	if (auto* v = setting("decode_threads")) cfg.decode_threads = atoi(v->text().c_str());
	if (auto* v = setting("priority")) cfg.priority = v->text();
	if (auto* v = setting("low_memory")) cfg.low_memory = v->text() == "1" || v->text() == "true";
	if (auto* u = root.find("upload")) {
		if (auto* x = u->find("url")) cfg.upload.url = x->text();
		if (auto* x = u->find("key")) cfg.upload.key = x->text();
		if (auto* x = u->find("videos")) cfg.upload.videos = !(x->text() == "0" || x->text() == "false");
	}
	if (auto* v = setting("videos")) {
		cfg.save_videos = !(v->text() == "0" || v->text() == "false");
		cfg.best_videos = v->text() == "best";
	}
	if (auto* v = root.find("over_penalty")) over_penalty = atof(v->text().c_str()); // where older files have it; it belongs in "vmaf" now
	if (auto* v = root.find("vmaf")) {
		if (auto* m = v->find("model")) cfg.vo.model = m->text();
		if (auto* t = v->find("threads")) cfg.vo.threads = atoi(t->text().c_str());
		if (auto* p = v->find("over_penalty")) over_penalty = atof(p->text().c_str());
	}
	auto off = [](const Json* v) { return v && (v->text() == "0" || v->text() == "false"); };
	list("encoders", [&](const Json& e) {
		if (off(e.find("enabled"))) return;
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

static const char* VERSION = "0.2.0";

// The config file used when none is named: config.jsonc, or config.json if that is what there is.
static std::string default_config() { return !fs::exists("config.jsonc") && fs::exists("config.json") ? "config.json" : "config.jsonc"; }

static void print_help(FILE* f) {
	fputs(R"HELP(ves - tries encoder option combinations, scores every frame with VMAF and reports the configurations
that are the best in at least 95% of frames.

usage: ves [options]                  run a config file (config.jsonc, or config.json, unless --config says otherwise)
       ves -enc <encoder> -br <kbps> -i <input> [--option key=a,b ...]    a quick run without a config file
       ves <input> <encoder> <codec> <kbps[,..]> [key=a,b ...]           the same, short form

Paths
  -cfg, --config <file>     config file, with comments allowed (default: config.jsonc, else config.json)
  -i,   --input <path>      an input video or a directory with videos (.mkv, .mp4, .mov); replaces the
                            inputs of the config file; can be given more than once
  -o,   --output <dir>      where the results go (default: output): the encoded stream of every run in
                            <dir>/<input>/ (fragmented .mp4) and the result file <dir>/<input>.json;
                            "off" writes nothing
        --no-videos         write the result files but not the encoded streams
        --best-videos       keep only the stream of the best run so far (highest mean VMAF after the overshoot penalty) of each input, encoder, bitrate,
                            key frame interval and size; the others are deleted once they are scored; the kept one is uploaded when its group is complete
        --upload <url>      after the sweep, upload the results (and videos) to this VES website, e.g. https://ves.example.com ("off" disables it)
        --upload-key <key>  the API key from the website's account page; better in the config ("upload": {"key": ...}) or the VES_API_KEY environment variable
        --no-upload-videos  upload the results but not the encoded videos
        --hw-decode <auto|off|name>  decode the sources on the GPU when their first frames match a software decode exactly (default: auto);
                            a name is a device type (cuda, d3d11va, d3d12va, dxva2, qsv, amf, vaapi, vulkan) or a hardware decoder (hevc_cuvid, h264_qsv, ...)
        --priority <low|normal>  low (default): idle CPU priority and the lowest disk priority, so ves never gets in the way of other programs; normal: the system default
        --decode-threads <n> threads per decoder (default: all cores; 4 with --low-memory)
        --low-memory        score each wave before the next is encoded and use fewer VMAF threads: slower, takes much less memory
        --plain             plain lines instead of the live display (also used when stdout is not a terminal)
  -c,   --cache <dir>       finished runs are kept here and not repeated (default: cache); "off" disables it

Scoring
        --vmaf-model <spec> libvmaf model, e.g. path=models/vmaf_4k_v0.6.1.json
        --vmaf-threads <n>
        --over-penalty <p>  VMAF points deducted per percent for a run just over the target bitrate (default: 0.025); it grows exponentially, like the website's

Quick run (replaces the encoders and scenarios of the config file)
  -enc, --encoder <name>    nvenc | amf | qsv | x264 | x265 | svtav1
  -cd,  --codec <name>      h264 | hevc | av1, for the GPU encoders
  -br,  --bitrate <kbps[,..]>
  -res, --resolution <WxH[,..]>    -fps, --framerate <n[,..]>    --kf <seconds[,..]>
        --option <key=a,b>  an encoder option, can be given more than once; comma separated values are swept
                            and every combination is tested; "null" leaves the option unset
        --upscale           allow encoding above the source's resolution, chroma layout and bit depth
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
	bool upscale = false;
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
		else if (k == "pool") j.pool = v;
		else if (k == "parallel") j.parallel = atoi(v.c_str());
		else if (k == "kf") { sc.kfs.clear(); for (auto& x : split(v)) sc.kfs.push_back(atof(x.c_str())); }
		else if (k == "res") res = split(v);
		else if (k == "fps") fps = split(v);
		else if (k == "upscale") upscale = v == "on" || v == "1" || v == "true";
		else { Axis ax{k, {}}; for (auto& x : split(v)) ax.values.push_back(x); j.axes.push_back(std::move(ax)); }
	}
	sc.shapes = make_shapes(res, fps);
	for (auto& sh : sc.shapes) sh.upscale = upscale;
	cfg.scenarios.push_back(std::move(sc));
	cfg.jobs.push_back(std::move(j));
	return true;
}

static void parse_flags(int argc, char** argv, Config& cfg, bool& quit) {
	std::string config_path = default_config();
	std::vector<std::string> inputs, bitrates, res, fps, kfs;
	std::optional<std::string> output, cache, model, upload_url, upload_key;
	std::optional<int> threads;
	std::optional<double> penalty;
	bool videos = true, best_videos = false, upload_videos = true, quick = false, upscale = false;
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
		else if (a == "--best-videos") best_videos = true;
		else if (a == "--upload") upload_url = value();
		else if (a == "--upload-key") upload_key = value();
		else if (a == "--no-upload-videos") upload_videos = false;
		else if (a == "--plain") cfg.plain = true;
		else if (a == "--low-memory") cfg.low_memory = true;
		else if (a == "--decode-threads") cfg.decode_threads = atoi(value().c_str());
		else if (a == "--hw-decode") cfg.hw_decode = value();
		else if (a == "--priority") cfg.priority = value();
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
		else if (a == "--upscale") upscale = true;
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
		for (auto& sh : sc.shapes) sh.upscale = upscale;
		cfg.scenarios.push_back(std::move(sc));
		cfg.jobs.push_back(std::move(job));
	} else load_config(config_path, cfg);
	if (!inputs.empty()) { cfg.inputs.clear(); for (auto& i : inputs) add_inputs(cfg, i); }
	if (output) cfg.output_dir = *output == "off" ? "" : *output;
	cfg.save_videos = cfg.save_videos && videos;
	cfg.best_videos = (cfg.best_videos || best_videos) && cfg.save_videos;
	if (upload_url) cfg.upload.url = *upload_url == "off" ? "" : *upload_url;
	if (upload_key) cfg.upload.key = *upload_key;
	cfg.upload.videos = cfg.upload.videos && upload_videos;
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

static std::string sanitize(std::string s, size_t max = 100) {
	for (char& c : s) if (!isalnum((unsigned char)c) && c != '-' && c != '_' && c != '.' && c != '=') c = '_';
	if (s.size() > max) s.resize(max);
	return s;
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

// Result files name inputs and model files without the folders they sit in: the paths are the user's own, and results are matched by hash.
static std::string file_name(const std::string& path) { return is_url(path) ? url_file_name(path) : fs::path(path).filename().string(); }
static std::string model_name(const std::string& model) { return model.rfind("path=", 0) == 0 ? "path=" + file_name(model.substr(5)) : model; }

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
		else if (argc == 1) load_config(default_config(), cfg);
		else if (argc == 2 && !fs::is_directory(argv[1]) && (fs::path(argv[1]).extension() == ".json" || fs::path(argv[1]).extension() == ".jsonc")) load_config(argv[1], cfg);
		else { print_help(stderr); return 1; }
	} catch (const std::exception& e) { fprintf(stderr, "error: %s\n", e.what()); return 1; }
	if (cfg.inputs.empty() || cfg.scenarios.empty() || cfg.jobs.empty()) { fprintf(stderr, "error: need inputs, bitrates and at least one enabled encoder (see --help)\n"); return 1; }

	if (cfg.low_memory && cfg.vo.threads <= 0) cfg.vo.threads = 2;
	if (cfg.priority != "low" && cfg.priority != "normal") { fprintf(stderr, "error: priority must be low or normal\n"); return 1; }
	if (cfg.priority == "low" && !set_low_priority()) fprintf(stderr, "warning: could not lower the process priority\n");
	set_hw_decode(cfg.hw_decode);
	set_decode_threads(cfg.decode_threads > 0 ? cfg.decode_threads : cfg.low_memory ? 4 : 0);
	ui::start(cfg.plain);
	struct UiGuard { ~UiGuard() { ui::stop(); } } ui_guard; // the live display is cleared on every way out

	struct Task {
		Run r;
		const Job* job;
		Codec id;
		Params p;
		bool ok = false;
		std::string key, pool; // what the run's outcome depends on (see cache_key); the pool it counts against
		std::string result_file; // where its result is kept, and looked for when the same run comes up again; empty: not at all
		Plan plan;
	};
	struct Pool {
		std::vector<size_t> tasks;
		std::atomic<size_t> next{0};
		int parallel = 1;
		std::string name;
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

	for (auto& job : cfg.jobs) for (auto& ax : job.axes) for (auto& sc : cfg.scenarios) {
		const bool in_scenario = (ax.key == "chroma" && !sc.chroma.empty()) || (ax.key == "bitdepth" && !sc.bitdepth.empty()) || (ax.key == "range" && !sc.range.empty());
		if (in_scenario && takes_part(job, sc)) {
			fprintf(stderr, "config: '%s' is listed by encoder '%s' and by scenario '%s'; list it in one of them\n", ax.key.c_str(), job.encoder.c_str(), sc.name.c_str());
			return 1;
		}
	}

	// What does not depend on the input is built once, for each encoder and scenario: the option sets merged with the scenario's chroma, bit depth and
	// range, each with its label, the part of the cache key it decides, and the GPU and pool it runs on. The inputs, bitrates and key frame intervals
	// and geometries are then put around them, which is all that is left to do for a run.
	struct Variant {
		Options opts;
		std::string label;     // " name=value ..."
		std::string key;       // the encoder, its environment and these options, as the cache key has them
		std::string gpu_index;
		std::string gpu;
		const std::string *gpu_json = &NULL_JSON, *caps_json = &NULL_JSON;
		Pool* pool = nullptr;
	};
	std::map<std::pair<const Job*, const Scenario*>, std::vector<Variant>> variants;
	auto variants_of = [&](const Job& job, const Scenario& sc) -> const std::vector<Variant>& {
		auto found = variants.find({&job, &sc});
		if (found != variants.end()) return found->second;
		std::vector<Variant>& out = variants[{&job, &sc}];
		auto axis = [](const std::vector<std::string>& v) { return v.empty() ? std::vector<std::string>{""} : v; }; // "": the source's
		for (const Options& base : expand(job.encoder, job.axes))
			for (auto& chroma : axis(sc.chroma)) for (auto& bitdepth : axis(sc.bitdepth)) for (auto& range : axis(sc.range)) {
				Variant v;
				v.opts = base;
				if (!chroma.empty()) v.opts["chroma"] = chroma;
				if (!bitdepth.empty()) v.opts["bitdepth"] = bitdepth;
				if (!range.empty()) v.opts["range"] = range;
				for (auto& [k, val] : v.opts) v.label += " " + k + "=" + val;
				v.key = key_encoder(job.encoder, job.codec, v.opts);
				// Every GPU is a pool of its own, so GPUs work in parallel; CPU encoders share one pool.
				v.gpu_index = v.opts.count("gpu") ? v.opts.at("gpu") : "0";
				if (is_gpu_encoder(job.encoder)) {
					v.gpu = gpu_description(encoder_vendor(job.encoder), atoi(v.gpu_index.c_str()));
					static std::map<std::string, std::string> gpu_known; // asked once per vendor and GPU; the runs only point at the text
					const std::string gpu_key = job.encoder + "/" + v.gpu_index;
					auto gpu_at = gpu_known.find(gpu_key);
					if (gpu_at == gpu_known.end()) gpu_at = gpu_known.emplace(gpu_key, gpu_json(encoder_vendor(job.encoder), atoi(v.gpu_index.c_str()))).first;
					v.gpu_json = &gpu_at->second;
					static std::map<std::string, std::string> caps_known; // asked once per encoder, codec and GPU
					const std::string caps_key = job.encoder + "/" + job.codec + "/" + v.gpu_index;
					auto known = caps_known.find(caps_key);
					if (known == caps_known.end()) known = caps_known.emplace(caps_key, encoder_caps_json(job.encoder, job.codec, atoi(v.gpu_index.c_str()))).first;
					v.caps_json = &known->second;
				}
				const std::string pool = !job.pool.empty() ? job.pool : is_gpu_encoder(job.encoder) ? job.encoder + ":" + v.gpu_index : "cpu";
				v.pool = &pools[pool];
				v.pool->parallel = std::max(v.pool->parallel, job.parallel ? job.parallel : (is_gpu_encoder(job.encoder) ? 2 : 1));
				v.pool->name = pool;
				out.push_back(std::move(v));
			}
		return out;
	};
	const std::string key_tail = key_settings(cfg.scale, cfg.vo);
	const char sep = (char)fs::path::preferred_separator;

	for (auto& input : cfg.inputs) {
		const std::string key_head = key_input(input);
		const std::string folder = sanitize(fs::path(file_name(input)).stem().string());
		const std::string folder_path = cfg.output_dir.empty() ? std::string() : (fs::path(cfg.output_dir) / folder).string() + sep;
		bool folder_made = false;
		for (auto& job : cfg.jobs) {
			const Codec id = codec_of(job.encoder, job.codec);
			if (id == Codec::None) { fprintf(stderr, "unknown encoder/codec '%s'/'%s'\n", job.encoder.c_str(), job.codec.c_str()); failed++; continue; }
			const std::string encoder_label = is_gpu_encoder(job.encoder) ? job.encoder + "/" + job.codec : job.encoder;
			for (auto& sc : cfg.scenarios) if (takes_part(job, sc)) {
				ui::log("Preparing " + job.encoder + "/" + job.codec + ", scenario " + (sc.name.empty() ? "(unnamed)" : sc.name) + ", " + file_name(input));
				const std::vector<Variant>& sets = variants_of(job, sc);
				for (int64_t br : sc.bitrates) for (double kf : sc.kfs) {
					const std::string kf_text = std::to_string((double)kf).substr(0, std::to_string((double)kf).find('.') + 2);
					for (auto& shape : sc.shapes) {
						std::string head = encoder_label; // the part of the label the options come after
						if (shape.w) head += " " + std::to_string(shape.w) + "x" + std::to_string(shape.h);
						if (shape.fps.num) head += " @" + std::to_string(shape.fps.value()).substr(0, 6);
						const std::string key_run_text = key_run(br, kf, shape);
						const std::string group_head = input + "|" + job.encoder + "/" + job.codec + "|" + std::to_string(br) + "|" + std::to_string(kf) + "|" + std::to_string(shape.w) + "x" + std::to_string(shape.h) + "@" + std::to_string(shape.fps.num) + "/" + std::to_string(shape.fps.den) + "|";
						for (const Variant& v : sets) {
							Task t;
							t.job = &job; t.id = id;
							t.r.input = input; t.r.bitrate = br; t.r.kf = kf; t.r.shape = shape;
							t.r.label = head + v.label;
							t.p.codec = job.codec; t.p.bitrate = br * 1000; t.p.options = v.opts;
							t.r.encoder = job.encoder; t.r.codec = job.codec; t.r.options = v.opts;
							// The run's result (.json) and stream (.mp4) share a name made from the run and a hash of everything its outcome depends on, so a
							// run that comes up again finds its result straight away. They go to the output folder; without one, results are kept in the cache.
							t.key = key_head + v.key + key_run_text + key_tail;
							if (!cfg.output_dir.empty()) {
								const std::string base = result_name(t.r.label + " " + std::to_string(br) + "k kf" + kf_text, t.key);
								t.result_file = folder_path + base + ".json";
								t.r.result = folder + "/" + base + ".json";
								if (cfg.save_videos) {
									t.r.video = folder + "/" + base + ".mp4";
									t.r.video_file = folder_path + base + ".mp4";
								}
								if (!folder_made) { std::error_code ec; fs::create_directories(fs::path(cfg.output_dir) / folder, ec); folder_made = true; }
							} else if (!cfg.cache_dir.empty()) t.result_file = (fs::path(cfg.cache_dir) / "results" / (key_hash(t.key) + ".json")).string();
							t.r.group = group_head + v.gpu_index;
							t.r.gpu = v.gpu; t.r.gpu_json = v.gpu_json; t.r.caps_json = v.caps_json;
							t.pool = v.pool->name;
							v.pool->tasks.push_back(tasks.size());
							tasks.push_back(std::move(t));
						}
					}
				}
			}
		}
	}

	// What is about to happen.
	{
		std::string text;
		auto put = [&](const char* fmt, auto... args) {
			char b[1024];
			snprintf(b, sizeof b, fmt, args...);
			text += b;
		};
		put("ves %s\n\nInputs (%zu)\n", VERSION, cfg.inputs.size());
		for (auto& in : cfg.inputs) put("  %s\n", in.c_str());
		put("Scenarios\n");
		for (auto& sc : cfg.scenarios) {
			put("  %s: ", sc.name.empty() ? "(unnamed)" : sc.name.c_str());
			for (size_t k = 0; k < sc.bitrates.size(); k++) put("%s%lld", k ? "," : "", (long long)sc.bitrates[k]);
			put(" kbps, key frames every ");
			for (size_t k = 0; k < sc.kfs.size(); k++) put("%s%g", k ? "," : "", sc.kfs[k]);
			put(" s");
			for (auto& sh : sc.shapes) if (sh.w || sh.fps.num) put(", %dx%d@%g", sh.w, sh.h, sh.fps.num ? sh.fps.value() : 0.0);
			for (auto* list : {&sc.chroma, &sc.bitdepth, &sc.range}) if (!list->empty()) {
				put(", %s ", list == &sc.chroma ? "chroma" : list == &sc.bitdepth ? "bitdepth" : "range");
				for (size_t k = 0; k < list->size(); k++) put("%s%s", k ? "," : "", (*list)[k].c_str());
			}
			put("\n");
		}
		put("Encoders\n");
		for (auto& job : cfg.jobs) {
			size_t runs_here = 0;
			for (auto& t : tasks) runs_here += t.job == &job;
			put("  %s%s%s: %zu runs\n", job.encoder.c_str(), is_gpu_encoder(job.encoder) ? "/" : "", is_gpu_encoder(job.encoder) ? job.codec.c_str() : "", runs_here);
		}
		put("Pools:");
		for (auto& [name, pl] : pools) put(" %s x%d (%zu runs)", name.c_str(), pl.parallel, pl.tasks.size());
		put("\nCache: %s   Output: %s%s\n\n%zu runs in total\n", cfg.cache_dir.empty() ? "off" : cfg.cache_dir.c_str(), cfg.output_dir.empty() ? "off" : cfg.output_dir.c_str(),
		    cfg.output_dir.empty() || !cfg.save_videos ? "" : " (with encoded streams)", tasks.size());
		ui::log(text);
	}

	// A run is finished when its result is known, however it came about: print it, count it.
	std::atomic<int> finished{0};
	const int total = (int)tasks.size();
	ui::overall(0, total);
	std::map<std::string, SourceInfo> infos;
	// The file behind an input. A URL is downloaded when it is first needed (its folder is under the cache folder, or the temp folder if there is
	// none), with the progress in the live display like the runs; an input that cannot be had throws.
	std::map<std::string, std::string> local_paths;
	std::set<std::string> source_failed; // inputs that could not be had: said once, not tried again
	auto local_path = [&](const std::string& input) -> std::string {
		if (!is_url(input)) return input;
		if (auto it = local_paths.find(input); it != local_paths.end()) return it->second;
		const std::string dir = (cfg.cache_dir.empty() ? fs::temp_directory_path() / "ves-downloads" : fs::path(cfg.cache_dir) / "downloads").string();
		RunProgress rp;
		rp.begin(5, 0);
		const int handle = ui::attach(&rp, file_name(input), "input");
		struct Detach { RunProgress& rp; int h; ~Detach() { rp.finish(); ui::detach(h); } } detach{rp, handle};
		return local_paths[input] = fetch_url(input, dir, &rp);
	};
	// SHA-256 of an input file, read once per input and only when a result needs it.
	std::map<std::string, std::string> reference_hashes;
	auto reference_hash = [&](const std::string& input) -> const std::string& {
		auto it = reference_hashes.find(input);
		if (it != reference_hashes.end()) return it->second;
		std::string hash;
		if (!source_failed.count(input)) try { hash = sha256_file(local_path(input)); } catch (const std::exception& e) { ui::log("Input " + file_name(input) + ": " + e.what()); }
		return reference_hashes.emplace(input, hash).first->second;
	};
	// What goes into a run's result file besides its key, bitrate and scores.
	auto run_details = [&](const Task& t) {
		const Run& r = t.r;
		std::ostringstream o;
		o << "\"input\":" << jq(file_name(r.input)) << ",\"input_url\":" << (is_url(r.input) ? jq(r.input) : std::string("null"))
		  << ",\"source_decoder\":" << jq(infos.count(r.input) ? infos.at(r.input).decoder : std::string()) << ",\"encoder\":" << jq(r.encoder) << ",\"codec\":" << jq(r.codec) << ",\"options\":{";
		bool first = true;
		for (auto& [k, v] : r.options) { o << (first ? "" : ",") << jq(k) << ":" << jq(v); first = false; }
		o << "},\"width\":" << r.shape.w << ",\"height\":" << r.shape.h << ",\"fps\":" << (r.shape.fps.num ? r.shape.fps.value() : 0.0)
		  << ",\"target_kbps\":" << r.bitrate << ",\"keyframe_interval_s\":" << r.kf << ",\"gpu\":" << (r.gpu.empty() ? std::string("null") : jq(r.gpu))
		  << ",\"gpu_info\":" << *r.gpu_json << ",\"encoder_caps\":" << *r.caps_json << ",\"video\":" << (r.video.empty() ? std::string("null") : jq(r.video))
		  << ",\"video_sha256\":" << (r.video_sha256.empty() ? std::string("null") : jq(r.video_sha256)) << ",\"reference_sha256\":" << jq(reference_hash(r.input));
		return o.str();
	};
	// Results go out as the runs are scored, from a thread of their own. Runs whose streams are identical to one already sent are one result (see below).
	std::unique_ptr<Uploader> uploader;
	std::set<std::string> sent_streams;
	if (!cfg.upload.url.empty() && !cfg.output_dir.empty()) {
		UploadOptions up = cfg.upload;
		if (up.key.empty()) if (const char* k = getenv("VES_API_KEY")) up.key = k;
		uploader = std::make_unique<Uploader>(up, [](const std::string& line) { ui::log(line); });
	}
	std::map<std::string, std::pair<Task*, double>> best_video; // group -> the run whose stream is kept, and its mean VMAF less the overshoot penalty
	std::map<std::string, int> group_left; // group -> runs not finished yet; its kept stream is uploaded when this reaches 0
	if (cfg.best_videos) for (auto& t : tasks) group_left[t.r.group]++;
	auto finish_run = [&](Task& t) {
		Run& r = t.r;
		ui::overall(++finished, total);
		char line[1024];
		if (r.ignored) { if (!r.expected) { snprintf(line, sizeof line, "%s IGNORED  %s: %s", r.tag.c_str(), r.label.c_str(), r.reason.c_str()); ui::log(line); } return; }
		if (!t.ok) { snprintf(line, sizeof line, "%s FAILED  %s  (%s, %lld kbps)", r.tag.c_str(), r.label.c_str(), r.input.c_str(), (long long)r.bitrate); ui::log(line); return; }
		r.penalty = over_penalty_points(r.kbps, (double)r.bitrate);
		std::vector<double> sc = r.scores;
		std::sort(sc.begin(), sc.end());
		const double mean = std::accumulate(sc.begin(), sc.end(), 0.0) / sc.size();
		if (!t.result_file.empty() && !r.cached) result_store(t.result_file, t.key, CachedRun{r.kbps, r.scores}, run_details(t));
		if (uploader && !r.result.empty() && (r.video_sha256.empty() || sent_streams.insert(r.group + "|" + r.video_sha256).second)) {
			UploadItem it;
			it.result_file = (fs::path(cfg.output_dir) / r.result).string();
			it.label = r.label;
			if (!cfg.best_videos && !r.video.empty() && fs::exists(fs::path(cfg.output_dir) / r.video)) it.video_file = (fs::path(cfg.output_dir) / r.video).string();
			uploader->add(std::move(it));
		}
		// Only the stream of the best run so far of each group stays; the others are deleted. The kept one is uploaded when the group is complete.
		if (cfg.best_videos && !r.video.empty() && fs::exists(fs::path(cfg.output_dir) / r.video)) {
			const double score = mean - r.penalty;
			auto [at, fresh] = best_video.emplace(r.group, std::make_pair(&t, score));
			Task* loser = &t;
			if (!fresh && score > at->second.second) { loser = at->second.first; at->second = {&t, score}; }
			std::error_code ec;
			fs::remove(fs::path(cfg.output_dir) / loser->r.video, ec);
			loser->r.video.clear();
		}
		snprintf(line, sizeof line, "%s finished  %s: %.0f kbps, vmaf mean %.3f min %.3f p5 %.3f%s%s%s  [wave %.1fs: encoder busy %.1f, starved %.1f; decoder backed up %.1f; scorer starved %.1f]", r.tag.c_str(), r.label.c_str(), r.kbps,
		         mean, sc.front(), sc[(size_t)(sc.size() * 0.05)], r.cached ? " (cached)" : "",
		         r.gpu.empty() ? "" : " on ", r.gpu.c_str(), r.encode_s, r.enc_busy_s, r.enc_starved_s, r.decode_blocked_s, r.score_starved_s);
		ui::log(line);
	};
	auto finish = [&](Task& t) {
		finish_run(t);
		if (!cfg.best_videos || --group_left[t.r.group] > 0) return;
		auto kept = best_video.find(t.r.group);
		if (kept == best_video.end() || !uploader) return;
		const Run& k = kept->second.first->r;
		UploadItem it;
		it.result_file = (fs::path(cfg.output_dir) / k.result).string();
		it.video_file = (fs::path(cfg.output_dir) / k.video).string();
		it.label = k.label;
		it.late = true;
		uploader->add(std::move(it));
	};

	// The source files, probed once each. A run that asks for no particular resolution or frame rate gets the source's, which is what its result records.
	auto source_of = [&](const std::string& input) -> const SourceInfo* {
		auto it = infos.find(input);
		if (it != infos.end()) return &it->second;
		if (source_failed.count(input)) return nullptr;
		SourceInfo si;
		try {
			if (!source_info(local_path(input), si)) { source_failed.insert(input); return nullptr; }
		} catch (const std::exception& e) { ui::log("Input " + file_name(input) + ": " + e.what()); source_failed.insert(input); return nullptr; }
		it = infos.emplace(input, si).first;
		ui::log("Source " + file_name(input) + ": " + std::to_string(si.width) + "x" + std::to_string(si.height) + " decoded by " + si.decoder);
		return &it->second;
	};
	auto fit_shape = [](Task& t) {
		if (!t.r.shape.w) { t.r.shape.w = t.plan.enc.w; t.r.shape.h = t.plan.enc.h; }
		if (!t.r.shape.fps.num) t.r.shape.fps = t.plan.enc.fps;
	};
	// Whatever cannot work, or was done before, is finished right away. The rest is sorted by the frames it needs: runs with the
	// same input, resolution, frame rate and pixel format share one decode.
	std::map<std::string, std::vector<size_t>> uncached; // by input: the runs that have to be done, planned when their input comes up
	std::vector<std::string> input_order;
	for (size_t i = 0; i < tasks.size(); i++) {
		Task& t = tasks[i];
		Run& r = t.r;
		char tag[48];
		snprintf(tag, sizeof tag, "[run %zu/%d]", i + 1, total);
		r.tag = tag;
		r.progress = std::make_shared<RunProgress>();
		if (!t.result_file.empty()) {
			CachedRun hit;
			if (result_load(t.result_file, t.key, hit)) {
				r.kbps = hit.kbps; r.scores = std::move(hit.scores); r.video_sha256 = hit.video_sha256; r.cached = t.ok = true;
				if (!hit.reference_sha256.empty()) reference_hashes.emplace(r.input, hit.reference_sha256); // known: the file is not needed to say it
				if (!r.shape.w && hit.width > 0 && hit.height > 0 && hit.fps > 0) { // the geometry the run was encoded at, for the summary: from its result file
					r.shape.w = hit.width; r.shape.h = hit.height;
					r.shape.fps = Rational{(int)std::lround(hit.fps * 1000), 1000};
				} else if (!r.shape.w) if (const SourceInfo* src = source_of(r.input)) { // an older result file: ask the source (which has to be had for that)
					t.plan = make_plan(*src, t.job->encoder, t.job->codec, t.p.options, r.shape);
					fit_shape(t);
				}
				const bool have_video = !r.video_file.empty() && fs::exists(r.video_file);
				if (!have_video) r.video.clear(); // the stream of the earlier run, if it is still there
				// Result files from before the hashes existed get them now, as far as the files they describe are still there.
				const bool new_video_hash = r.video_sha256.empty() && have_video && !(r.video_sha256 = sha256_file(r.video_file)).empty();
				if ((new_video_hash || hit.reference_sha256.empty() || hit.input != file_name(r.input) || (hit.width <= 0 && r.shape.w > 0)) && !reference_hash(r.input).empty())
					result_store(t.result_file, t.key, CachedRun{r.kbps, r.scores}, run_details(t));
				r.video_file.clear();
				finish(t);
				continue;
			}
		}
		if (!uncached.count(r.input)) input_order.push_back(r.input);
		uncached[r.input].push_back(i);
	}

	// A wave is the runs that are encoded at the same time from one decode: up to `parallel` of each pool. Each run is scored while it is
	// being encoded, from the packets as they come out, so a wave is done when its slowest encoder and scorer are.
	// Encoded packets are muxed to disk as they arrive; scratch files go next to the results if there are any, else in the temp folder.
	// A folder of its own for the scratch streams. Several ves processes can run at once (and start in the same moment), so the name is random and the
	// folder is created in one step that fails when the name is taken: a clock value as the name gave two processes the same folder now and then.
	const fs::path spool_dir = [&] {
		const fs::path parent = cfg.output_dir.empty() ? fs::temp_directory_path() : fs::path(cfg.output_dir);
		std::error_code ec;
		fs::create_directories(parent, ec);
		std::random_device rd;
		for (int attempt = 0; attempt < 100; attempt++) {
			char name[40];
			snprintf(name, sizeof name, ".ves-tmp-%08x%08x", rd(), rd());
			if (fs::create_directory(parent / name, ec)) return parent / name;
		}
		return parent / ".ves-tmp";
	}();
	for (const std::string& input_name : input_order) {
	// Plan the runs of this input: opening the source is where a URL is downloaded. Runs that cannot work are finished right away; the rest is
	// sorted by the frames it needs, and runs with the same resolution, frame rate and pixel format share one decode.
	std::map<std::string, std::vector<Task*>> by_feed;
	std::vector<std::string> feed_order;
	for (size_t i : uncached[input_name]) {
		Task& t = tasks[i];
		Run& r = t.r;
		const SourceInfo* src = source_of(r.input);
		if (!src) { finish(t); continue; }
		t.plan = make_plan(*src, t.job->encoder, t.job->codec, t.p.options, r.shape);
		if (t.plan.ignored) { r.ignored = true; r.expected = t.plan.expected; r.reason = t.plan.reason; finish(t); continue; }
		fit_shape(t);
		const std::string fk = r.input + "|" + t.plan.feed_key();
		if (!by_feed.count(fk)) feed_order.push_back(fk);
		by_feed[fk].push_back(&t);
	}
	for (auto& fk : feed_order) {
		std::map<std::string, std::vector<Task*>> by_pool;
		for (Task* t : by_feed[fk]) by_pool[t->pool].push_back(t);
		std::map<std::string, size_t> taken;
		for (;;) {
			std::vector<Task*> wave;
			for (auto& [pool, list] : by_pool) {
				for (int k = 0; k < pools[pool].parallel && taken[pool] < list.size(); k++) wave.push_back(list[taken[pool]++]);
			}
			if (wave.empty()) break;
			const std::string& input = wave[0]->r.input;
			const SourceInfo& si = infos.at(input);
			std::vector<WaveRun> wr(wave.size());
			std::vector<WaveRun*> ptrs;
			std::vector<int> handles;
			for (size_t i = 0; i < wave.size(); i++) {
				Task& t = *wave[i];
				WaveRun& run = wr[i];
				EncodeJob& ej = run.enc;
				t.r.score_progress = std::make_shared<RunProgress>();
				ej.encoder = t.job->encoder; ej.codec = t.job->codec; ej.plan = &t.plan;
				ej.bitrate = t.r.bitrate * 1000; ej.kf = t.r.kf; ej.rp = t.r.progress.get();
				ej.file = !t.r.video_file.empty() ? t.r.video_file : (spool_dir / ("run" + std::to_string(&t - tasks.data()) + ".mp4")).string();
				ej.id = t.id; ej.color = t.plan.color;
				run.codec = t.id; run.score_rp = t.r.score_progress.get();
				ptrs.push_back(&run);
				ui::log_plain(t.r.tag + " encoding and scoring  " + t.r.label + "  (" + file_name(input) + ", " + std::to_string(t.r.bitrate) + " kbps, pool " + t.pool + ")");
				handles.push_back(ui::attach(t.r.progress.get(), t.r.label, t.pool));
				handles.push_back(ui::attach(t.r.score_progress.get(), t.r.label, t.pool));
			}
			RunProgress feed_rp;
			feed_rp.begin(3, 1);
			const int feed_handle = ui::attach(&feed_rp, "decode+convert " + file_name(input) + " to " + std::to_string(wave[0]->plan.enc.w) + "x" + std::to_string(wave[0]->plan.enc.h), "decode");
			FeedResult fr;
			const auto t_wave = std::chrono::steady_clock::now();
			run_wave(local_path(input), si, ptrs, cfg.scale, cfg.vo, &feed_rp, fr);
			const double wave_s = std::chrono::duration<double>(std::chrono::steady_clock::now() - t_wave).count();
			ui::detach(feed_handle);
			for (int h : handles) ui::detach(h);
			for (size_t i = 0; i < wave.size(); i++) {
				Task& t = *wave[i];
				Run& r = t.r;
				r.encode_s = wave_s;
				r.enc_busy_s = r.progress->busy_ms / 1000.0;
				r.enc_starved_s = r.progress->starved_ms / 1000.0;
				r.decode_blocked_s = feed_rp.blocked_ms / 1000.0;
				r.score_starved_s = r.score_progress->starved_ms / 1000.0;
				if (wr[i].enc.ok && wr[i].score_ok) {
					r.kbps = wr[i].enc.kbps;
					r.scores = std::move(wr[i].scores);
					t.ok = true;
				}
				std::error_code ec;
				if (t.ok) r.video_sha256 = sha256_file(wr[i].enc.file);
					if (r.video_file.empty()) fs::remove(wr[i].enc.file, ec); // a temporary file
				finish(t);
			}
		}
	}
	}
	{ std::error_code ec; fs::remove_all(spool_dir, ec); }
	if (uploader) uploader->finish(); // the live display stays up while the last uploads go out
	ui::stop();

	// Collect in task order so the ranking below does not depend on which run finished first.
	std::vector<Run> runs;
	std::map<std::string, std::vector<std::string>> failed_runs, ignored_runs, group_json; // by input
	int ignored = 0;
	std::map<std::string, size_t> first_with; // group and stream hash -> the run that has it
	int identical = 0;
	for (auto& t : tasks) {
		if (t.ok) {
			// Options that changed nothing show up here, after encoding: the stream is byte for byte the same as another run's of this group, so it
			// has the same scores and would only tie with it. It stays out of the ranking; its result file is still written.
			if (!t.r.video_sha256.empty()) {
				auto [at, fresh] = first_with.emplace(t.r.group + "|" + t.r.video_sha256, runs.size());
				if (!fresh) { runs[at->second].identical.push_back(t.r.label); identical++; continue; }
			}
			runs.push_back(std::move(t.r));
		}
		else if (t.r.ignored) { ignored++; ignored_runs[t.r.input].push_back("{\"label\":" + jq(t.r.label) + ",\"reason\":" + jq(t.r.reason) + "}"); }
		else { failed++; failed_runs[t.r.input].push_back(t.r.label); }
	}

	if (identical) printf("\n%d runs produced exactly the same stream as another run of their group and are left out of the ranking (listed as identical_runs in the result files)\n", identical);

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
			  << ",\"gpu\":" << (r.gpu.empty() ? "null" : jq(r.gpu)) << ",\"gpu_info\":" << *r.gpu_json << ",\"encoder_caps\":" << *r.caps_json << ",\"cached\":" << (r.cached ? "true" : "false")
			  << ",\"video\":" << (r.video.empty() ? "null" : jq(r.video)) << ",\"video_sha256\":" << (r.video_sha256.empty() ? "null" : jq(r.video_sha256)) << ",\"identical_runs\":[" << [&] { std::string l; for (auto& x : r.identical) l += (l.empty() ? "" : ",") + jq(x); return l; }() << "],\"result\":" << (r.result.empty() ? "null" : jq(r.result)) << "}";
		}
		o << "]}";
		group_json[runs[g].input].push_back(o.str());
		g = e;
	}

	if (!cfg.output_dir.empty()) {
		std::error_code ec;
		fs::create_directories(cfg.output_dir, ec);
		for (auto& input : cfg.inputs) {
			const fs::path file = fs::path(cfg.output_dir) / (sanitize(fs::path(file_name(input)).stem().string()) + ".json");
			std::ofstream f(file);
			f << "{\"tool\":\"ves\",\"version\":" << jq(VERSION) << ",\"input\":" << jq(file_name(input)) << ",\"input_url\":" << (is_url(input) ? jq(input) : std::string("null")) << ",\"reference_sha256\":" << jq(reference_hash(input)) << ",\"source_decoder\":" << jq(infos.count(input) ? infos.at(input).decoder : std::string()) << ",\"vmaf_model\":" << jq(model_name(cfg.vo.model)) << ",\"over_penalty\":" << over_penalty << ",\"failed\":[";
			for (size_t i = 0; i < failed_runs[input].size(); i++) f << (i ? "," : "") << jq(failed_runs[input][i]);
			f << "],\"ignored\":[";
			for (size_t i = 0; i < ignored_runs[input].size(); i++) f << (i ? "," : "") << ignored_runs[input][i];
			f << "],\"groups\":[";
			for (size_t i = 0; i < group_json[input].size(); i++) f << (i ? "," : "") << group_json[input][i];
			f << "]}\n";
			if (f) printf("%sResults: %s\n", &input == &cfg.inputs.front() ? "\n" : "", file.string().c_str());
			else fprintf(stderr, "cannot write %s\n", file.string().c_str());
		}
	}
	printf("\nPeak memory use: %lld MB\n", ui::peak_memory_mb());
	if (ignored) printf("\n%d runs were ignored: they cannot work with the input or the encoder (see the result files)\n", ignored);
	return failed ? 2 : 0;
}
