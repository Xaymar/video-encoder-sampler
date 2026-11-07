// Sweep encoder option combinations: for each one, encode in-process, decode back and score every
// frame with VMAF, then report which configurations are the best in at least 95% of frames.
#include <algorithm>
#if defined(_WIN32) && defined(_DEBUG)
#include <crtdbg.h>
#include <stdlib.h>
#endif
#include <atomic>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <filesystem>
#include <fstream>
#include <list>
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
#include <unordered_map>
#include <unordered_set>

#include "cache.h"
#include "hash.h"
#include "upload.h"
#include "priority.h"
#include "encoder.h"
#include "encoder_process.h"
#include "formats.h"
#include "gpu.h"
#include "combos.h"
#include "convert.h"
#include "download.h"
#include "json.h"
#include "source.h"
#include "mux.h"
#include "pipeline.h"
#include "ui.h"
#include "vk_compute.h"
#include "remote.h"
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

struct Job {
	std::string encoder, codec = "h264";
	std::string pool;  // runs in one pool share a device; at most `parallel` of them run at once
	std::vector<std::string> scenarios; // names of the scenarios this encoder takes part in; empty: all of them
	int parallel = 0;  // 0: default for the pool (NVENC 2, everything else 1 since CPU encoders use every core)
	std::vector<Axis> axes;
	bool prune = true;                 // skip runs of an option that shows no effect (judged on the scores of the runs themselves)
	std::set<std::string> fixed;       // options that are never pruned
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
	bool short_names = false;            // --short-names: result files and the input's folder are named by hashes (a worker's: its names are not seen, and the path stays short)
	bool only_runs_set = false;          // --only-runs: a worker runs only what the host asked for
	std::set<std::string> only_runs;      // the option lines (see run_options_of) of those runs
	std::vector<std::string> inputs;
	std::vector<Scenario> scenarios;
	VmafOptions vo;
	ScaleOptions scale;
	std::string cache_dir = "cache"; // finished runs are kept here and not repeated; empty turns it off
	std::string output_dir = "output"; // encoded streams and result files go here; empty: nothing is written
	bool save_videos = true;
	bool best_videos = true; // keep only the stream of the best run so far of each group
	UploadOptions upload; // where the results go after a sweep; no address: nowhere
	bool plain = false; // no live display, just lines
	std::string hw_decode = "auto"; // "" software only, "auto", or a device type or decoder name: see set_hw_decode()
	int decode_threads = 0;  // 0: as many as the cores allow within a memory budget
	std::string priority = "low"; // "low": idle CPU and lowest disk priority, see priority.h; "normal" leaves the system's default
	bool low_memory = false; // score a wave before encoding the next, and fewer VMAF threads
	bool ceiling = true;     // score each feed once with no encoder in between, so every run also has its score against what its size, frame rate and format allow
	bool in_process = false; // the encoders run inside this process instead of one process per GPU and CPU encoder (encoder_process.h)
	int host_port = 0;       // distributed: listen on this port and hand the runs that are not done to the workers that connect (remote.h); 0: run everything here
	std::string token;       // what a worker must present to the host
	int min_workers = 1;     // the host starts handing out jobs when this many workers have connected
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
	std::shared_ptr<const std::vector<double>> ceiling; // per frame: what its feed scores with no encoder (shared by the runs of the feed); null if unknown
	bool ceiling_new = false; // the ceiling was not in its result file yet
	VmafDetail detail;          // the extra models' scores and the chroma values
	bool cached = false;    // taken from the result cache instead of being run
	bool uploaded = false;  // handed to the uploader when the worker sent it
	bool imported = false;  // its result came from a worker: written again with this sweep's details (the input's name and URL are the host's, not the worker's copy)
	std::vector<std::string> identical; // labels of the runs that produced exactly the same stream (left out of the ranking, see below)
	std::string group;      // what runs must share for identical output to mean identical scores
	std::string encoder, codec;
	const Options* options = nullptr; // the run's encoder options: the variant's, shared by every run of it
	std::string video;      // the saved stream, relative to the output directory; empty if none was saved
	std::string video_file; // where run_one() writes it
	std::string video_sha256; // SHA-256 of the encoded stream
	std::string result;     // its result file (options, bitrates, per-frame VMAF), relative to the output directory; empty if none
	std::string tag;        // "[run 12/240]", for the progress lines
	bool pruned = false;    // not run: an option showed no effect, see reason
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

// Mean, lowest and 5th percentile of per-frame values.
struct FrameStats { double mean = 0, min = 0, p5 = 0; };
static FrameStats frame_stats(std::vector<double> v) {
	FrameStats st;
	if (v.empty()) return st;
	std::sort(v.begin(), v.end());
	st.mean = std::accumulate(v.begin(), v.end(), 0.0) / v.size();
	st.min = v.front();
	st.p5 = v[(size_t)(v.size() * 0.05)];
	return st;
}
// A run against its ceiling, frame by frame: score minus ceiling (0: all its size, frame rate and format allow; below: what its encoder lost).
static std::vector<double> vs_ceiling(const Run& r) {
	std::vector<double> d;
	if (!r.ceiling || r.ceiling->size() != r.scores.size()) return d;
	for (size_t f = 0; f < r.scores.size(); f++) d.push_back(r.scores[f] - (*r.ceiling)[f]);
	return d;
}

// What the GPU's encoder says it cannot do, against what a run asks of it: the reason the run cannot work there, or empty. Such a run is ignored, not failed.
// The capabilities are the ones the encoder reports about itself (encoder_caps_json: NVENC's NV_ENC_CAPS_*, AMF's caps properties).
static std::string caps_refusal(const std::string& encoder, const std::string& codec, const Options& opts, const std::string* caps_text) {
	if (!caps_text || *caps_text == "null" || caps_text->empty()) return {};
	static std::mutex m;
	static std::map<const std::string*, Json> parsed; // the texts are shared by every run on the GPU and live as long as the sweep
	const Json* caps;
	{
		std::lock_guard lock(m);
		auto it = parsed.find(caps_text);
		if (it == parsed.end()) {
			Json j;
			try { j = Json::parse(*caps_text); } catch (const std::exception&) {}
			it = parsed.emplace(caps_text, std::move(j)).first;
		}
		caps = &it->second;
	}
	auto cap = [&](const char* name, long long& v) {
		const Json* c = caps->find(name);
		if (!c || (c->type != Json::Num && c->type != Json::Bool)) return false;
		v = atoll(c->s.c_str());
		return true;
	};
	auto opt = [&](const char* key, std::string& v) { auto it = opts.find(key); if (it == opts.end()) return false; v = it->second; return true; };
	auto num = [](const std::string& v) { return atoll(v.c_str()); };
	std::string v;
	long long c;
	// An option that is on (above 0) where the capability is 0.
	auto needs = [&](const char* key, const char* name) -> std::string {
		if (opt(key, v) && num(v) > 0 && cap(name, c) && c == 0) return std::string(key) + "=" + v + " needs " + name + ", which this GPU's encoder does not have";
		return {};
	};
	auto at_most = [&](const char* key, const char* name, long long add = 0) -> std::string {
		if (opt(key, v) && cap(name, c) && num(v) + add > c) return std::string(key) + "=" + v + " is above this GPU's " + name + " (" + std::to_string(c) + ")";
		return {};
	};
	auto in_range = [&](const char* key, const char* lo, const char* hi) -> std::string {
		if (!opt(key, v)) return {};
		if (cap(lo, c) && num(v) < c) return std::string(key) + "=" + v + " is below this GPU's " + lo + " (" + std::to_string(c) + ")";
		if (cap(hi, c) && num(v) > c) return std::string(key) + "=" + v + " is above this GPU's " + hi + " (" + std::to_string(c) + ")";
		return {};
	};
	std::string why;
	auto first = [&](std::string w) { if (why.empty()) why = std::move(w); };
	if (encoder == "nvenc") {
		first(at_most("max_b_frames", "NUM_MAX_BFRAMES"));
		first(needs("lookahead_frames", "SUPPORT_LOOKAHEAD"));
		first(needs("lookahead_level", "SUPPORT_LOOKAHEAD_LEVEL"));
		first(needs("temporal_aq", "SUPPORT_TEMPORAL_AQ"));
		first(needs("hierarchical_p_frames", "SUPPORT_HIERARCHICAL_PFRAMES"));
		first(needs("hierarchical_b_frames", "SUPPORT_HIERARCHICAL_BFRAMES"));
		first(needs("enable_weighted_prediction", "SUPPORT_WEIGHTED_PREDICTION"));
		first(needs("enable_uni_directional_b", "SUPPORT_UNIDIRECTIONAL_B"));
		first(needs("tf_level", "SUPPORT_TEMPORAL_FILTER"));
		first(at_most("num_ltr_frames", "NUM_MAX_LTR_FRAMES"));
		// NV_ENC_CAPS_SUPPORT_BFRAME_REF_MODE: bit 0 each, bit 1 middle
		if (opt("b_reference_mode", v) && (v == "each" || v == "middle") && cap("SUPPORT_BFRAME_REF_MODE", c) && !(c & (v == "each" ? 1 : 2)))
			first("b_reference_mode=" + v + " is not in this GPU's SUPPORT_BFRAME_REF_MODE (" + std::to_string(c) + ")");
	} else if (encoder == "amf") {
		const bool h264 = codec == "h264", hevc = codec == "hevc", av1 = codec == "av1";
		const char* bframes = h264 ? "BFrames" : av1 ? "AV1BFrames" : nullptr;
		if (bframes) for (const char* key : {"max_b_frames", "b_pictures_pattern", "max_consecutive_bpictures"}) first(needs(key, bframes));
		const char* lo = h264 ? "MinReferenceFrames" : hevc ? "HevcMinReferenceFrames" : nullptr;
		const char* hi = h264 ? "MaxReferenceFrames" : hevc ? "HevcMaxReferenceFrames" : nullptr;
		if (lo) for (const char* key : {"max_ref_frames", "max_num_ref_frames"}) first(in_range(key, lo, hi));
	}
	return why;
}

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

// "requires" and "unless" of an option: a name ("max_b_frames": it is set), a list of names, or an object of name to what it must be:
// true (set), a value or a list of values, or a comparison with a number (">=2").
static void parse_conds(const Json& j, std::vector<Cond>& out, const std::string& option) {
	if (j.type == Json::Arr) { for (auto& x : j.a) parse_conds(x, out, option); return; }
	if (j.type != Json::Obj) { Cond c; c.key = j.text(); out.push_back(c); return; }
	for (auto& [name, spec] : j.o) {
		Cond c;
		c.key = name;
		if (spec.type == Json::Arr) for (auto& x : spec.a) c.any_of.push_back(x.text());
		else if (spec.type == Json::Bool) { if (spec.text() != "1" && spec.text() != "true") throw std::runtime_error(spec.where() + "option '" + option + "': a condition on '" + name + "' says false; leave it out"); }
		else {
			const std::string t = spec.text();
			size_t n = 0;
			for (const char* op : {">=", "<=", "==", "!=", ">", "<"}) if (t.compare(0, strlen(op), op) == 0) { c.op = op; n = strlen(op); break; }
			if (!c.op.empty()) {
				if (!Cond::number(t.substr(n), c.num)) throw std::runtime_error(spec.where() + "option '" + option + "': '" + t + "' is not a comparison with a number");
			} else c.any_of.push_back(t);
		}
		out.push_back(c);
	}
}

// Every combination of the axes on top of `base`.
static std::vector<Options> expand(const std::string& encoder, const std::vector<Axis>& axes, const Options& base = {}) {
	// An option with conditions is placed after the options its conditions name, so the combinations that cannot be are never made: where its condition fails it
	// stays unset (one branch, not one per value). The rest keeps the order it was written in. Conditions that go in circles are checked against what is placed.
	std::vector<const Axis*> order;
	{
		std::vector<bool> placed(axes.size());
		auto names_unplaced = [&](const Axis& ax) {
			for (auto* list : {&ax.requires_all, &ax.unless_any}) for (auto& c : *list)
				for (size_t k = 0; k < axes.size(); k++) if (!placed[k] && axes[k].key == c.key && &axes[k] != &ax) return true;
			return false;
		};
		while (order.size() < axes.size()) {
			bool progress = false;
			for (size_t k = 0; k < axes.size(); k++) if (!placed[k] && !names_unplaced(axes[k])) { placed[k] = true; order.push_back(&axes[k]); progress = true; }
			if (!progress) for (size_t k = 0; k < axes.size(); k++) if (!placed[k]) { placed[k] = true; order.push_back(&axes[k]); } // a circle
		}
	}
	std::vector<Options> combos{base};
	for (const Axis* axp : order) {
		const Axis& ax = *axp;
		if (ax.values.empty()) continue;
		// gpu=all means every GPU of the encoder's vendor
		std::vector<std::string> values;
		for (auto& v : ax.values) {
			if (ax.key == "gpu" && v == "all") {
				size_t n = std::max<size_t>(list_gpus(encoder_vendor(encoder)).size(), 1);
				for (size_t i = 0; i < n; i++) values.push_back(std::to_string(i));
			} else values.push_back(v);
		}
		const bool ruled = !ax.requires_all.empty() || !ax.unless_any.empty();
		std::vector<Options> next;
		next.reserve(combos.size() * (ruled ? 1 : values.size()));
		for (auto& c : combos) {
			if (ruled) {
				bool applies = true;
				for (auto& r : ax.requires_all) applies &= r.holds(c);
				for (auto& u : ax.unless_any) applies &= !u.holds(c);
				if (!applies) { next.push_back(std::move(c)); continue; }
			}
			for (auto& v : values) { Options n = c; n[ax.key] = v; next.push_back(std::move(n)); }
		}
		combos = std::move(next);
	}
	return combos;
}

// "a,b,c": the first model ranks, the others are scored with it.
static void set_models(VmafOptions& vo, const std::string& list) {
	std::vector<std::string> ms;
	for (size_t p = 0;;) {
		const size_t c = list.find(',', p);
		std::string m = list.substr(p, c == std::string::npos ? std::string::npos : c - p);
		if (!m.empty()) ms.push_back(m);
		if (c == std::string::npos) break;
		p = c + 1;
	}
	if (ms.empty()) throw std::runtime_error("no VMAF model given");
	vo.model = ms[0];
	vo.extra_models.assign(ms.begin() + 1, ms.end());
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

static void load_config_json(const Json& root, Config& cfg);
// Errors name the file, and the line where the JSON says where (Json::where()).
static void load_config(const std::string& path, Config& cfg) {
	std::ifstream f(path);
	if (!f) throw std::runtime_error("cannot read '" + path + "'");
	std::stringstream ss; ss << f.rdbuf();
	try { load_config_json(Json::parse(ss.str()), cfg); }
	catch (const std::runtime_error& e) { throw std::runtime_error(path + ": " + e.what()); }
}

static void load_config_json(const Json& root, Config& cfg) {

	auto list = [&](const char* k, auto fn) {
		if (auto* v = root.find(k)) { if (v->type != Json::Arr) throw std::runtime_error(v->where() + "'" + k + "' must be an array"); for (auto& e : v->a) fn(e); }
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
			if (auto* v = o.find(k)) { if (v->type != Json::Arr) throw std::runtime_error(v->where() + "'" + k + "' must be an array"); for (auto& e : v->a) fn(e.text()); }
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
		if (sc.bitrates.empty()) throw std::runtime_error(o.where() + "a scenario needs bitrates");
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
	if (auto* v = setting("encoder_processes")) cfg.in_process = v->text() == "0" || v->text() == "false";
	if (auto* u = root.find("upload")) {
		if (auto* x = u->find("url")) cfg.upload.url = x->text();
		if (auto* x = u->find("key")) cfg.upload.key = x->text();
		if (auto* x = u->find("videos")) cfg.upload.videos = !(x->text() == "0" || x->text() == "false");
	}
	if (auto* v = setting("videos")) {
		cfg.save_videos = !(v->text() == "0" || v->text() == "false");
		cfg.best_videos = v->text() != "all";
	}
	if (auto* v = root.find("over_penalty")) over_penalty = atof(v->text().c_str()); // where older files have it; it belongs in "vmaf" now
	if (auto* v = root.find("vmaf")) {
		if (auto* m = v->find("model")) { // one model, or a list: the first one ranks, the others are scored with it
			if (m->type == Json::Arr) {
				if (m->a.empty()) throw std::runtime_error(m->where() + "vmaf.model is an empty list");
				cfg.vo.model = m->a[0].text();
				cfg.vo.extra_models.clear();
				for (size_t i = 1; i < m->a.size(); i++) cfg.vo.extra_models.push_back(m->a[i].text());
			} else { cfg.vo.model = m->text(); cfg.vo.extra_models.clear(); }
		}
		if (auto* t = v->find("threads")) cfg.vo.threads = atoi(t->text().c_str());
		if (auto* d = v->find("device")) {
			cfg.vo.device = d->text();
			if (cfg.vo.device != "cpu" && cfg.vo.device != "gpu" && cfg.vo.device != "auto") throw std::runtime_error(d->where() + "vmaf.device must be cpu, gpu or auto");
		}
		if (auto* g = v->find("gpu")) cfg.vo.gpu = g->text();
		if (auto* p = v->find("over_penalty")) over_penalty = atof(p->text().c_str());
		if (auto* c = v->find("ceiling")) cfg.ceiling = !(c->text() == "0" || c->text() == "false");
	}
	auto off = [](const Json* v) { return v && (v->text() == "0" || v->text() == "false"); };
	const bool default_prune = !off(setting("prune"));
	list("encoders", [&](const Json& e) {
		if (off(e.find("enabled"))) return;
		Job j;
		j.prune = default_prune && !off(e.find("prune"));
		if (auto* f = e.find("fixed")) for (auto& x : f->a) j.fixed.insert(x.text());
		for (auto& [k, v] : e.o)
			if (k != "enabled" && k != "encoder" && k != "codec" && k != "pool" && k != "parallel" && k != "scenarios" && k != "options" && k != "fixed" && k != "prune")
				fprintf(stderr, "warning: %sencoder entry key '%s' is not known and is ignored\n", v.where().c_str(), k.c_str());
		auto* name = e.find("encoder");
		if (!name) throw std::runtime_error(e.where() + "encoder entry without 'encoder'");
		j.encoder = name->text();
		if (auto* c = e.find("codec")) j.codec = c->text();
		if (auto* p = e.find("pool")) j.pool = p->text();
		if (auto* p = e.find("parallel")) j.parallel = atoi(p->text().c_str());
		if (auto* sn = e.find("scenarios")) for (auto& x : sn->a) j.scenarios.push_back(x.text());
		if (auto* o = e.find("options")) for (auto& [k, v] : o->o) {
			Axis ax{k, {}, {}, {}};
			const Json* list = &v;
			if (v.type == Json::Obj) { // { "values": [...], "requires": "other" or ["a", "b"], "unless": "other" or [...] }
				for (auto& [rk, rv] : v.o) if (rk != "values" && rk != "requires" && rk != "unless") throw std::runtime_error(rv.where() + "option '" + k + "' has an unknown key '" + rk + "' (it takes values, requires and unless)");
				list = v.find("values");
				if (!list) throw std::runtime_error(v.where() + "option '" + k + "' needs 'values'");
				for (auto* rule : {&ax.requires_all, &ax.unless_any}) {
					const Json* conds = v.find(rule == &ax.requires_all ? "requires" : "unless");
					if (conds) parse_conds(*conds, *rule, k);
				}
			}
			bool dropped_null = false;
			auto add_value = [&](const Json& x) { if (x.type == Json::Null) dropped_null = true; else ax.values.push_back(x.text()); };
			if (list->type == Json::Arr) { for (auto& x : list->a) add_value(x); } else add_value(*list);
			if (dropped_null) fprintf(stderr, "config: %s option '%s' lists null, which is no longer a value (it meant the preset's own, which ves cannot know); it is left out. To run without the option, do not list it\n", j.encoder.c_str(), k.c_str());
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
        --all-videos        keep every encoded stream. By default only the winner (the run that is best in the most frames after the overshoot penalty, as in the ranking) of each input,
                            encoder, bitrate, key frame interval and size keeps its stream; the others are deleted when the group is complete, and the kept
                            one is uploaded then (a stream an earlier sweep sent that is beaten now is removed from the site)
        --upload <url>      after the sweep, upload the results (and videos) to this VES website, e.g. https://ves.example.com ("off" disables it)
        --upload-key <key>  the API key from the website's account page; better in the config ("upload": {"key": ...}) or the VES_API_KEY environment variable
        --no-upload-videos  upload the results but not the encoded videos
        --hw-decode <auto|off|name>  decode the sources on the GPU when their first frames match a software decode exactly (default: auto);
                            a name is a device type (cuda, d3d11va, d3d12va, dxva2, qsv, amf, vaapi, vulkan) or a hardware decoder (hevc_cuvid, h264_qsv, ...)
        --priority <low|normal>  low (default): idle CPU priority and the lowest disk priority, so ves never gets in the way of other programs; normal: the system default
        --decode-threads <n> threads per decoder (default: all cores; 4 with --low-memory)
        --low-memory        score each wave before the next is encoded and use fewer VMAF threads: slower, takes much less memory
        --plain             plain lines instead of the live display (also used when stdout is not a terminal)
        --in-process        run everything inside ves itself. By default every GPU gets a process of its own that decodes, encodes and scores its runs
                            on that GPU (the CPU encoders share one), so a driver or library that crashes fails only the runs that were in that process, and
                            the sweep goes on ("encoder_processes": false in the config)
  -c,   --cache <dir>       finished runs are kept here and not repeated (default: cache); "off" disables it

Scoring
        --vmaf-model <spec>[,<spec>...]  VMAF models, scored in one pass; the first one ranks, the others' scores go into the result files
                            (default: the four VMAF v1 models in models/, vmaf_v1.0.16_1d5h_2160 first)
        --vmaf-threads <n>
        --vmaf-device <cpu|gpu|auto>  where VMAF is computed (default: cpu). gpu: on the GPU with Vulkan compute (8 and 10 bit input; v0 and v1 models; scores within 1e-9 of libvmaf's);
                            auto: the GPU where it can be used (8 or 10 bit YUV, a model file, a Vulkan GPU), else the CPU. These scores are kept apart from libvmaf's in the cache
        --vmaf-gpu <auto|n|name>  the Vulkan device that scores (default: auto, the GPU that encodes the run, else the first discrete GPU); n: Vulkan device n;
                            name: the first device whose name contains it. --vmaf-gpu list shows the Vulkan devices
        --no-ceiling        do not score the ceiling of each size, frame rate and format (the original against its own scaled and converted frames, no encoder).
                            With it (the default), every run also has its score relative to that ceiling: 0 is all that size could reach
        --over-penalty <p>  VMAF points deducted per percent for a run just over the target bitrate (default: 0.025); it grows exponentially, like the website's

Distributed (see docs/distributed.md)
        --host <port>       decide what is run, and hand the runs that are not done to the workers that connect to this port. The host runs nothing
                            itself, except what no worker could do; it needs an output or cache folder
        --trust-downloads   use inputs downloaded earlier as they are, without asking the server whether they changed (a worker sets it for its jobs)
        --progress          once a second, print the numbers of the runs in flight on a line of their own (a worker reads them from the `ves` that runs its job)
        --token <secret>    what workers must present (or the VES_TOKEN environment variable); use it on both sides
        --min-workers <n>   jobs start when this many workers have connected (default: 1)
        --worker <host:port or file>  be a worker: connect to a host and run the jobs it sends, until stopped. A config file holds the host, the token and
                            the resources the worker offers with how many encodes run on each at once (see docs/distributed.md). Every other flag (--priority, --hw-decode,
                            --vmaf-device, ...) is passed on to the runs; --worker-dir <dir> keeps fetched inputs and scratch files (default: ves-worker);
                            --worker-gpu <n> offers only NVIDIA GPU n, so a machine with several GPUs runs one worker per GPU;
                            --worker-jobs <n> sets how many encodes run at once on a resource that is not in the config file (default: 1 on the CPU, 2 on a GPU)

Quick run (replaces the encoders and scenarios of the config file)
  -enc, --encoder <name>    nvenc | amf | qsv | x264 | x265 | svtav1
  -cd,  --codec <name>      h264 | hevc | av1, for the GPU encoders
  -br,  --bitrate <kbps[,..]>
  -res, --resolution <WxH[,..]>    -fps, --framerate <n[,..]>    --kf <seconds[,..]>
        --option <key=a,b>  an encoder option, can be given more than once; comma separated values are swept
                            and every combination is tested
        --fixed <key[,..]>  options that are never pruned
        --no-prune          run every combination even when an option shows no effect
        --upscale           allow encoding above the source's resolution, chroma layout and bit depth
        --pool <name>  --parallel <n>   (with a config file, these apply to every encoder entry of it)

  -h, --help                this text
  -v, --version
)HELP", f);
}

// Command line made of --flags. Sets `quit` when there is nothing left to do (--help, --version).
// `--vmaf-gpu list`: the Vulkan devices, by the numbers --vmaf-gpu takes.
static void list_vulkan_devices() {
	std::string why;
	const auto all = Vk::devices(why);
	if (all.empty()) { printf("no Vulkan devices%s%s\n", why.empty() ? "" : ": ", why.c_str()); return; }
	for (auto& d : all) printf("%d: %s (%s%s)\n", d.index, d.name.c_str(), d.discrete ? "discrete" : "not discrete", d.usable ? "" : ", cannot score: no 64 bit integers and floats");
}

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
		if (k == "vmaf.model") set_models(cfg.vo, v);
		else if (k == "vmaf.threads") cfg.vo.threads = atoi(v.c_str());
		else if (k == "vmaf.device") cfg.vo.device = v;
		else if (k == "vmaf.gpu") cfg.vo.gpu = v;
		else if (k == "vmaf.ceiling") cfg.ceiling = !(v == "off" || v == "0" || v == "false");
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
	std::optional<std::string> vmaf_device, vmaf_gpu;
	bool videos = true, all_videos = false, upload_videos = true, quick = false, upscale = false;
	std::set<std::string> only_lines; // --only-runs
	bool short_names = false;
	bool only_set = false;
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
		else if (a == "--all-videos") all_videos = true;
		else if (a == "--upload") upload_url = value();
		else if (a == "--upload-key") upload_key = value();
		else if (a == "--no-upload-videos") upload_videos = false;
		else if (a == "--plain") cfg.plain = true;
		else if (a == "--low-memory") cfg.low_memory = true;
		else if (a == "--in-process") cfg.in_process = true;
		else if (a == "--no-ceiling") cfg.ceiling = false;
		else if (a == "--host") cfg.host_port = atoi(value().c_str());
		else if (a == "--progress") ui::set_progress_stdout(true);
		else if (a == "--trust-downloads") set_trust_downloads(true);
		else if (a == "--short-names") short_names = true;
		else if (a == "--only-runs") { std::ifstream f(value()); for (std::string l; std::getline(f, l);) { if (!l.empty() && l.back() == '\r') l.pop_back(); only_lines.insert(l); } only_set = true; }
		else if (a == "--token") cfg.token = value();
		else if (a == "--min-workers") cfg.min_workers = std::max(1, atoi(value().c_str()));
		else if (a == "--decode-threads") cfg.decode_threads = atoi(value().c_str());
		else if (a == "--hw-decode") cfg.hw_decode = value();
		else if (a == "--priority") cfg.priority = value();
		else if (a == "-c" || a == "--cache") cache = value();
		else if (a == "--vmaf-model") model = value();
		else if (a == "--vmaf-threads") threads = atoi(value().c_str());
		else if (a == "--vmaf-device") { // kept until the config is read: the flag wins over its vmaf.device (a worker's job config carries the host's)
			vmaf_device = value();
			if (*vmaf_device != "cpu" && *vmaf_device != "gpu" && *vmaf_device != "auto") throw std::runtime_error("--vmaf-device must be cpu, gpu or auto");
		}
		else if (a == "--vmaf-gpu") {
			vmaf_gpu = value();
			if (*vmaf_gpu == "list") { list_vulkan_devices(); quit = true; return; }
		}
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
			Axis ax{o.substr(0, eq), {}, {}, {}};
			for (auto& x : split(o.substr(eq + 1))) { if (x == "null") fprintf(stderr, "option %s: null is not a value any more and is left out%s", ax.key.c_str(), "\n"); else ax.values.push_back(x); }
			job.axes.push_back(std::move(ax));
		}
		else if (a == "--fixed") { for (auto& x : split(value())) job.fixed.insert(x); }
		else if (a == "--no-prune") job.prune = false;
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
	} else {
		load_config(config_path, cfg);
		// --parallel and --pool given with a config file override every encoder entry of it.
		for (auto& j : cfg.jobs) {
			if (job.parallel > 0) j.parallel = job.parallel;
			if (!job.pool.empty()) j.pool = job.pool;
		}
	}
	if (!inputs.empty()) { cfg.inputs.clear(); for (auto& i : inputs) add_inputs(cfg, i); }
	if (output) cfg.output_dir = *output == "off" ? "" : *output;
	cfg.save_videos = cfg.save_videos && videos;
	cfg.best_videos = cfg.best_videos && !all_videos && cfg.save_videos;
	if (upload_url) cfg.upload.url = *upload_url == "off" ? "" : *upload_url;
	if (upload_key) cfg.upload.key = *upload_key;
	cfg.upload.videos = cfg.upload.videos && upload_videos;
	if (cache) cfg.cache_dir = *cache == "off" ? "" : *cache;
	if (model) set_models(cfg.vo, *model);
	if (only_set) { cfg.only_runs_set = true; cfg.only_runs = only_lines; }
	if (short_names) cfg.short_names = true;
	if (threads) cfg.vo.threads = *threads;
	if (vmaf_device) cfg.vo.device = *vmaf_device;
	if (vmaf_gpu) cfg.vo.gpu = *vmaf_gpu;
	if (penalty) over_penalty = *penalty;
	// A relative model is looked for where the program is as well (a worker's jobs run wherever the worker was started), and in the folder above it, where a build sits.
	auto locate = [](std::string& model) {
		if (model.rfind("path=", 0) != 0) return;
		const fs::path rel = model.substr(5);
		std::error_code ec;
		if (rel.is_relative() && !fs::exists(rel, ec)) {
			const fs::path exe = fs::path(executable_path()).parent_path();
			for (const fs::path& base : {exe, exe.parent_path()})
				if (fs::exists(base / rel, ec)) { model = "path=" + (base / rel).string(); break; }
		}
	};
	locate(cfg.vo.model);
	for (auto& m : cfg.vo.extra_models) locate(m);
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

std::string Cond::json() const {
	if (!op.empty()) { char b[32]; snprintf(b, sizeof b, "%g", num); return jq(op + b); }
	if (any_of.empty()) return "true";
	std::string r = "[";
	for (size_t i = 0; i < any_of.size(); i++) r += (i ? "," : "") + jq(any_of[i]);
	return r + "]";
}

// Result files name inputs and model files without the folders they sit in: the paths are the user's own, and results are matched by hash.
static std::string file_name(const std::string& path) { return is_url(path) ? url_file_name(path) : fs::path(path).filename().string(); }
static std::string model_name(const std::string& model) { return model.rfind("path=", 0) == 0 ? "path=" + file_name(model.substr(5)) : model; }

// The config of one remote job (remote.h): the host's option lists for one encoder entry, and a scenario with a single bitrate, key frame interval and size.
// What the worker decides for itself (hardware decoding, priority, threads) is left out, so the worker's own flags apply.
static std::string sub_config(const Config& cfg, const Job& job, const Scenario& sc, int64_t bitrate, double kf, const Shape& shape) {
	std::ostringstream o;
	auto list = [](const std::vector<std::string>& v) {
		std::string s = "[";
		for (size_t i = 0; i < v.size(); i++) s += (i ? "," : "") + jq(v[i]);
		return s + "]";
	};
	o << "{\"scenarios\":[{\"name\":" << jq(sc.name) << ",\"bitrates\":[" << bitrate << "],\"kfintervals\":[" << kf << "]";
	if (shape.w) o << ",\"resolutions\":[" << jq(std::to_string(shape.w) + "x" + std::to_string(shape.h)) << "]";
	if (shape.fps.num) o << ",\"framerates\":[" << jq(std::to_string(shape.fps.num) + "/" + std::to_string(shape.fps.den)) << "]";
	if (!sc.chroma.empty()) o << ",\"chroma\":" << list(sc.chroma);
	if (!sc.bitdepth.empty()) o << ",\"bitdepth\":" << list(sc.bitdepth);
	if (!sc.range.empty()) o << ",\"range\":" << list(sc.range);
	if (shape.upscale) o << ",\"upscale\":true";
	o << "}],\"scaling\":{\"upscaler\":" << jq(cfg.scale.upscaler) << ",\"downscaler\":" << jq(cfg.scale.downscaler);
	if (!cfg.scale.device.empty()) o << ",\"gpu\":" << jq(cfg.scale.device);
	o << "},\"vmaf\":{\"model\":[";
	for (size_t i = 0, n = vmaf_models(cfg.vo).size(); i < n; i++) o << (i ? "," : "") << jq(vmaf_models(cfg.vo)[i]);
	o << "],\"device\":" << jq(cfg.vo.device) << ",\"ceiling\":" << (cfg.ceiling ? "true" : "false") << "},\"encoders\":[{\"encoder\":" << jq(job.encoder) << ",\"codec\":" << jq(job.codec);
	o << ",\"prune\":false,\"options\":{";
	bool first = true;
	if (is_gpu_encoder(job.encoder)) { o << "\"gpu\":[\"@GPU@\"]"; first = false; } // the host puts the worker's GPU index in
	for (auto& ax : job.axes) {
		if (ax.key == "gpu" && is_gpu_encoder(job.encoder)) continue;
		const bool rules = !ax.requires_all.empty() || !ax.unless_any.empty(); // the worker expands the same combinations as the host does
		o << (first ? "" : ",") << jq(ax.key) << ":" << (rules ? "{\"values\":[" : "[");
		first = false;
		bool first_value = true;
		for (auto& v : ax.values) {
			if (ax.key == "gpu" && v == "all") { // as expand() does on the host: every GPU of the vendor
				for (size_t i = 0; i < std::max<size_t>(list_gpus(encoder_vendor(job.encoder)).size(), 1); i++) { o << (first_value ? "" : ",") << jq(std::to_string(i)); first_value = false; }
				continue;
			}
			o << (first_value ? "" : ",") << jq(v);
			first_value = false;
		}
		o << "]";
		if (rules) {
			for (auto* rule : {&ax.requires_all, &ax.unless_any}) {
				if (rule->empty()) continue;
				o << ",\"" << (rule == &ax.requires_all ? "requires" : "unless") << "\":{";
				for (size_t i = 0; i < rule->size(); i++) o << (i ? "," : "") << jq((*rule)[i].key) << ":" << (*rule)[i].json();
				o << "}";
			}
			o << "}";
		}
	}
	o << "}}]}";
	return o.str();
}

// A source's details as JSON, and back: the first job of an input tells the host what the input is, and the host leaves out the jobs that ask for more.
static std::string source_json(const SourceInfo& si) {
	std::ostringstream o;
	o << "{\"w\":" << si.width << ",\"h\":" << si.height << ",\"chroma\":" << si.fmt.chroma << ",\"depth\":" << si.fmt.depth << ",\"rgb\":" << si.fmt.rgb << ",\"planar\":" << si.fmt.planar
	  << ",\"fps_num\":" << si.framerate.num << ",\"fps_den\":" << si.framerate.den << ",\"range\":" << si.color.range << ",\"space\":" << si.color.space << ",\"primaries\":" << si.color.primaries
	  << ",\"trc\":" << si.color.trc << ",\"duration\":" << si.duration << ",\"frames\":" << si.frame_count << ",\"decoder\":" << jq(si.decoder) << "}";
	return o.str();
}

static SourceInfo source_from_json(const std::string& text) {
	Json j = Json::parse(text);
	auto num = [&](const char* k) { return atof(j.find(k)->text().c_str()); };
	SourceInfo si;
	si.width = (int)num("w");
	si.height = (int)num("h");
	si.fmt.chroma = (int)num("chroma");
	si.fmt.depth = (int)num("depth");
	si.fmt.rgb = num("rgb") != 0;
	si.fmt.planar = num("planar") != 0;
	si.framerate = Rational{(int)num("fps_num"), (int)num("fps_den")};
	si.color.range = (int)num("range");
	si.color.space = (int)num("space");
	si.color.primaries = (int)num("primaries");
	si.color.trc = (int)num("trc");
	si.duration = num("duration");
	si.frame_count = (long long)num("frames");
	si.decoder = j.find("decoder")->text();
	return si;
}

// `ves --worker <host:port or config file>`: runs the jobs a host sends, until it is stopped. The config file says where the host is and what the worker offers:
//   { "host": "10.0.0.1:5599", "token": "...", "dir": "ves-worker",
//     "resources": { "cpu": { "parallel": 4 }, "nvenc": [ { "gpu": 0, "parallel": 10 } ], "amf": { "gpu": 0, "parallel": 2 } },
//     "vmaf": { "device": "gpu", "gpu": "0" } }
// A resource that is left out is detected and gets the default (1 encode on the CPU, 2 on a GPU); "parallel": 0 does not offer it. Every other flag is passed on to the `ves`
// that runs a job, so --priority, --hw-decode, --vmaf-device and so on work as they do for a sweep started by hand.
static int run_worker(int argc, char** argv) {
	// Settings of the runs from the worker config, as the flags they stand for; a flag on the command line is passed on as it is and wins.
	static const std::pair<const char*, const char*> run_settings[] = {
		{"vmaf_gpu", "--vmaf-gpu"}, {"vmaf_device", "--vmaf-device"}, {"hw_decode", "--hw-decode"}, {"decode_threads", "--decode-threads"}, {"priority", "--priority"}};
	std::vector<std::pair<std::string, std::string>> from_file; // flag, value
	int only_gpu = -1, default_parallel = 0;
	std::string target, dir, token = getenv("VES_TOKEN") ? getenv("VES_TOKEN") : "";
	bool dir_given = false, token_given = false;
	std::vector<std::string> pass;
	for (int i = 1; i < argc; i++) {
		const std::string a = argv[i];
		auto value = [&]() -> std::string { if (i + 1 >= argc) throw std::runtime_error("missing value for " + a); return argv[++i]; };
		try {
			if (a == "--worker") target = value();
			else if (a == "--worker-dir") { dir = value(); dir_given = true; }
			else if (a == "--worker-gpu") only_gpu = atoi(value().c_str());
			else if (a == "--worker-jobs") default_parallel = std::max(1, atoi(value().c_str()));
			else if (a == "--token") { token = value(); token_given = true; }
			else pass.push_back(a);
		} catch (const std::exception& e) { fprintf(stderr, "error: %s\n", e.what()); return 1; }
	}
	std::vector<WorkerResource> resources;
	if (target.find(':') == std::string::npos || fs::exists(target)) { // a config file
		const std::string config_file = target; // target becomes the host it names
		try {
			std::ifstream f(target);
			if (!f) throw std::runtime_error("cannot read '" + target + "'");
			std::stringstream ss; ss << f.rdbuf();
			Json root = Json::parse(ss.str());
			if (auto* v = root.find("host")) target = v->text(); else throw std::runtime_error("the worker config needs a \"host\" (host:port)");
			if (auto* v = root.find("token"); v && !token_given) token = v->text();
			if (auto* v = root.find("dir"); v && !dir_given) { dir = v->text(); dir_given = true; }
			if (auto* v = root.find("gpu"); v && only_gpu < 0) only_gpu = atoi(v->text().c_str());
			for (auto& [key, flag] : run_settings) if (auto* v = root.find(key)) from_file.push_back({flag, v->text()});
			if (auto* v = root.find("vmaf")) { // "vmaf": { "device": ..., "gpu": ... }, as in a sweep config
				if (v->type != Json::Obj) throw std::runtime_error(v->where() + "\"vmaf\" must be an object ({ \"device\": ..., \"gpu\": ... })");
				for (auto& [name, x] : v->o) {
					if (name == "device") from_file.push_back({"--vmaf-device", x.text()});
					else if (name == "gpu") from_file.push_back({"--vmaf-gpu", x.text()});
					else if (name == "threads") from_file.push_back({"--vmaf-threads", x.text()});
					else throw std::runtime_error(x.where() + "a worker's \"vmaf\" takes device, gpu and threads (the models come from the host)");
				}
			}
			if (auto* v = root.find("parallel"); v && !default_parallel) default_parallel = std::max(1, atoi(v->text().c_str()));
			if (auto* rs = root.find("resources")) {
				for (auto& [name, v] : rs->o) {
					if (name != "cpu" && !is_gpu_encoder(name)) throw std::runtime_error("unknown resource '" + name + "' (cpu, nvenc, amf, qsv)");
					auto one = [&](const Json& r) {
						WorkerResource w;
						w.encoder = name;
						if (auto* g = r.find("gpu")) w.index = atoi(g->text().c_str());
						if (auto* p = r.find("parallel")) w.parallel = std::max(0, atoi(p->text().c_str()));
						resources.push_back(w);
					};
					if (v.type == Json::Arr) for (auto& r : v.a) one(r); else one(v);
				}
			}
		} catch (const std::exception& e) { fprintf(stderr, "error: %s: %s\n", config_file.c_str(), e.what()); return 1; }
	}
	if (dir.empty()) dir = "ves-worker";
	for (auto& [flag, v] : from_file) if (std::find(pass.begin(), pass.end(), flag) == pass.end()) { pass.push_back(flag); pass.push_back(v); }
	const size_t colon = target.rfind(':');
	if (colon == std::string::npos) { fprintf(stderr, "error: --worker takes host:port or a config file\n"); return 1; }
	// The worker has the same display as a sweep: its log lines above, and the rows of the jobs it runs. --plain turns it off.
	ui::start(std::find(pass.begin(), pass.end(), "--plain") != pass.end());
	return remote_worker(target.substr(0, colon), atoi(target.c_str() + colon + 1), token, dir, only_gpu, default_parallel, resources, pass, [](const std::string& line) { ui::log(line); });
}

int main(int argc, char** argv) {
	ui::end_on_interrupt();
#if defined(_WIN32) && defined(_DEBUG)
	// A failed assertion of the debug runtime goes to stderr and aborts, instead of opening a dialog that offers to start a debugger.
	_set_error_mode(_OUT_TO_STDERR);
	_CrtSetReportMode(_CRT_ASSERT, _CRTDBG_MODE_FILE);
	_CrtSetReportFile(_CRT_ASSERT, _CRTDBG_FILE_STDERR);
	_CrtSetReportMode(_CRT_ERROR, _CRTDBG_MODE_FILE);
	_CrtSetReportFile(_CRT_ERROR, _CRTDBG_FILE_STDERR);
	_set_abort_behavior(0, _WRITE_ABORT_MSG | _CALL_REPORTFAULT);
#endif
	// `ves --probe-amf <codec> <gpu>`: whether AMD GPU <gpu> can encode the codec, and its capabilities, on one line. A worker asks this in a process of its own: AMD's runtime
	// has crashed on GPUs a newer driver no longer supports, and that must not take the worker with it.
	// The answer is flushed and the process ends without winding anything down: AMD's runtime has crashed on the way out, after answering, and a crash reads as "no".
	if (argc == 4 && std::string(argv[1]) == "--probe-amf") {
		std::string caps;
		amf_dedicate_process(atoi(argv[3]));
		const bool ok = amf_probe(argv[2], atoi(argv[3]), caps);
		if (!amf_runtime().empty()) printf("runtime %s\n", amf_runtime().c_str());
		if (ok) printf("ok %s\n", caps.c_str()); else printf("no\n");
		exit_now(0);
	}
	// `ves --encoder-process <encoder> <gpu>`: the process the encoders of one GPU (or one CPU encoder) run in, started by a sweep (see encoder_process.h).
	if (argc >= 4 && std::string(argv[1]) == "--encoder-process") return encoder_process_main(argc, argv);
	for (int i = 1; i < argc; i++) if (std::string(argv[i]) == "--worker") return run_worker(argc, argv);
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

	if (cfg.host_port) {
		if (cfg.output_dir.empty() && cfg.cache_dir.empty()) { fprintf(stderr, "error: --host needs an output or a cache folder to keep the workers' results in\n"); return 1; }
		if (cfg.token.empty()) if (const char* t = getenv("VES_TOKEN")) cfg.token = t;
	}
	if (cfg.low_memory && cfg.vo.threads <= 0) cfg.vo.threads = 2;
	{ // a model libvmaf cannot use would fail every run
		std::string why;
		int depth = 8;
		for (auto& m : vmaf_models(cfg.vo)) {
			if (!vmaf_check_model(m, why)) { fprintf(stderr, "error: %s\n", why.c_str()); return 1; }
			depth = std::max(depth, vmaf_model_min_depth(m));
		}
		set_min_score_depth(depth);
	}
	if (cfg.priority != "low" && cfg.priority != "normal") { fprintf(stderr, "error: priority must be low or normal\n"); return 1; }
	if (cfg.priority == "low" && !set_low_priority()) fprintf(stderr, "warning: could not lower the process priority\n");
	set_isolated_encoders(!cfg.in_process);
	{ // the processes decode on their GPUs as asked; this one only reads what the inputs are, which needs no GPU
		std::vector<std::string> args{"--hw-decode", cfg.hw_decode.empty() ? "off" : cfg.hw_decode, "--decode-threads", std::to_string(cfg.decode_threads > 0 ? cfg.decode_threads : cfg.low_memory ? 4 : 0)};
		if (cfg.priority == "low") { args.push_back("--priority"); args.push_back("low"); }
		set_encoder_process_args(args);
	}
	set_hw_decode(cfg.in_process ? cfg.hw_decode : "");
	set_decode_threads(cfg.decode_threads > 0 ? cfg.decode_threads : cfg.low_memory ? 4 : 0);
	ui::start(cfg.plain);
	struct UiGuard { ~UiGuard() { ui::stop(); } } ui_guard; // the live display is cleared on every way out
	struct EncoderProcesses { ~EncoderProcesses() { stop_encoder_processes(); } } encoder_processes; // ended before the live display is, so that what they say still shows

	// Distributed: workers connect first, because what GPU models they have decides what is planned for GPU encoders.
	std::unique_ptr<RemoteHost> remote;
	std::deque<std::string> remote_texts; // the GPU details of workers that runs point at
	if (cfg.host_port) {
		RemoteHostOptions ro;
		ro.port = cfg.host_port;
		ro.token = cfg.token;
		remote = std::make_unique<RemoteHost>(ro, [](const std::string& line) { ui::log(line); });
		std::string err;
		if (!remote->start(err)) { fprintf(stderr, "error: %s\n", err.c_str()); return 1; }
		remote->wait_for_workers(cfg.min_workers);
	}

	struct LazyGroup;
	struct Task {
		Run r;
		const Job* job;
		Codec id;
		const Options* opts = nullptr;   // its encoder options: the variant's, shared by every run of it (a task is made millions of times for a big config)
		bool ok = false;
		bool retried = false;            // its encoder process crashed under it once, and it was run again on its own
		// What the run's outcome depends on (see cache_key), put together when it is needed: a big config plans hundreds of thousands of runs, and the parts are
		// shared by the runs of an input, size and bit rate (head, and the run's part with the settings) and by the runs of a variant (its options).
		struct KeyParts { std::string head, run_tail; };
		std::shared_ptr<const KeyParts> key_parts;
		const std::string* variant_key = nullptr;
		std::string key() const { return key_parts->head + *variant_key + key_parts->run_tail; }
		std::string pool; // the pool it counts against
		std::string result_file; // where its result is kept, and looked for when the same run comes up again; empty: not at all
		Plan plan;
		bool done = false;               // finished, however it came about
		std::vector<int> mid;            // pruning: the lines along which this run is neither the lowest nor the highest value (see `lines`)
		const Scenario* sc = nullptr;    // the scenario it comes from, and which of its sizes: where a remote job is cut
		std::string model;               // distributed GPU runs: the GPU model of the worker that runs it
		size_t shape_i = 0;
		size_t index = 0;                // its place in `tasks`
		LazyGroup* group = nullptr;      // lazy planning: the group of runs it was made for, and which combination of its entry it is
		ComboTable::Id combo = 0;
	};
	struct Pool {
		std::vector<size_t> tasks;
		std::atomic<size_t> next{0};
		int parallel = 1;
		int limit = 0;     // the lowest "parallel" an encoder entry asked for; it beats the defaults of the other entries in the pool
		int fallback = 1;  // the highest default of the entries that asked for nothing
		std::string name;
	};
	std::deque<Task> tasks; // a deque: runs are made between rounds, and pointers to tasks stay valid
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
		std::string model;     // distributed: the GPU model of the worker that runs it
	};
	// What an entry's options are made of in a scenario: the scenario only adds its chroma, bit depth and range, so scenarios that list the same ones share the
	// entry's variants and option table (the sizes and bit rates are put around them later).
	using ShareKey = std::tuple<const Job*, std::vector<std::string>, std::vector<std::string>, std::vector<std::string>>;
	auto share_key = [](const Job& job, const Scenario& sc) { return ShareKey{&job, sc.chroma, sc.bitdepth, sc.range}; };
	std::map<ShareKey, std::vector<Variant>> variants;
	// One variant of an encoder entry from its options (the scenario's chroma, bit depth and range already among them): the label, the cache key, the GPU it runs
	// on and the pool it counts against.
	auto build_variant = [&](const Job& job, Options opts) {
		Variant v;
		v.opts = std::move(opts);
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
		// A "parallel" that is written down wins over the defaults of other entries of the pool (several codecs of one GPU share it).
		if (job.parallel > 0) v.pool->limit = v.pool->limit ? std::min(v.pool->limit, job.parallel) : job.parallel;
		else v.pool->fallback = std::max(v.pool->fallback, is_gpu_encoder(job.encoder) ? 2 : 1);
		v.pool->parallel = v.pool->limit ? v.pool->limit : v.pool->fallback;
		v.pool->name = pool;
		return v;
	};
	auto variants_of = [&](const Job& job, const Scenario& sc) -> const std::vector<Variant>& {
		auto found = variants.find(share_key(job, sc));
		if (found != variants.end()) return found->second;
		std::vector<Variant>& out = variants[share_key(job, sc)];
		auto axis = [](const std::vector<std::string>& v) { return v.empty() ? std::vector<std::string>{""} : v; }; // "": the source's
		// A worker's job: the host listed the runs it still needs by their options (see run_options_of), which already carry the scenario's chroma, bit depth and
		// range. Those are the variants, with the GPU the host put in, instead of every combination of the entry's options: a job of a big config has hundreds of
		// thousands of runs out of millions of combinations.
		if (cfg.only_runs_set) {
			std::vector<std::string> gpus;
			for (auto& ax : job.axes) if (ax.key == "gpu") gpus = ax.values;
			for (const std::string& line : cfg.only_runs) {
				Options base;
				for (size_t at = 0; at < line.size();) {
					size_t end = line.find(';', at);
					if (end == std::string::npos) end = line.size();
					const size_t eq = line.find('=', at);
					if (eq != std::string::npos && eq < end) base[line.substr(at, eq - at)] = line.substr(eq + 1, end - eq - 1);
					at = end + 1;
				}
				for (auto& gpu : axis(gpus)) {
					Options opts = base;
					if (!gpu.empty()) opts["gpu"] = gpu;
					out.push_back(build_variant(job, std::move(opts)));
				}
			}
			return out;
		}
		// Distributed: a GPU encoder's runs are planned once for every GPU model the workers have that can encode the codec, and the "gpu" option (which GPU of the
		// host) has no meaning.
		const bool remote_gpu = remote && is_gpu_encoder(job.encoder);
		std::vector<RemoteHost::Model> models;
		std::vector<Axis> axes_no_gpu;
		if (remote_gpu) {
			models = remote->models(job.encoder, job.codec);
			for (auto& ax : job.axes) if (ax.key != "gpu") axes_no_gpu.push_back(ax);
			if (models.empty()) ui::log("no worker has a GPU that can encode " + job.encoder + "/" + job.codec + ": those runs are skipped");
		}
		for (const Options& base : expand(job.encoder, remote_gpu ? axes_no_gpu : job.axes))
			for (auto& chroma : axis(sc.chroma)) for (auto& bitdepth : axis(sc.bitdepth)) for (auto& range : axis(sc.range)) {
				Options opts = base;
				if (!chroma.empty()) opts["chroma"] = chroma;
				if (!bitdepth.empty()) opts["bitdepth"] = bitdepth;
				if (!range.empty()) opts["range"] = range;
				Variant v = build_variant(job, std::move(opts));
				if (!remote_gpu) { out.push_back(std::move(v)); continue; }
				for (auto& m : models) {
					Variant w = v;
					w.model = m.description;
					w.key = key_encoder(job.encoder, job.codec, w.opts, m.description);
					w.gpu = m.description;
					remote_texts.push_back(m.info);
					w.gpu_json = &remote_texts.back();
					remote_texts.push_back(m.caps);
					w.caps_json = &remote_texts.back();
					out.push_back(std::move(w));
				}
			}
		return out;
	};
	const std::string key_tail = key_settings(cfg.scale, cfg.vo);
	const char sep = (char)fs::path::preferred_separator;

	// The software encoders make one codec each, whatever the entry says: results and the site go by the codec, and an entry without one carries the default ("h264").
	for (auto& job : cfg.jobs) {
		if (job.encoder == "x264") job.codec = "h264";
		else if (job.encoder == "x265") job.codec = "hevc";
		else if (job.encoder == "svtav1") job.codec = "av1";
	}

	// Lazy planning. A big config has hundreds of thousands of combinations of an entry's options, for every input, bit rate and size, and pruning leaves a few of
	// them. So the combinations are not made into runs up front: first the probes (each prunable option at the ends of its range) and everything of the options that
	// are not pruned. A combination with options in the middle of their range is made into a run when, for each of those options, the two ends of its line scored
	// differently (see advance below); until then it costs nothing but a row of integers.
	struct LazyTable {
		ComboTable table;
		std::map<std::pair<ComboTable::Id, size_t>, const Variant*> made; // the variants made so far, by combination and worker GPU model (see LazyGroup::model)
		LazyTable(const std::string& encoder, const std::vector<Axis>& axes, const std::vector<Axis>& free_axes, const std::set<std::string>& fixed, bool prune) : table(encoder, axes, free_axes, fixed, prune) {}
	};
	struct LazyGroup {
		LazyTable* lt = nullptr;
		const Job* job = nullptr;
		std::string input;
		size_t model = 0;                              // distributed GPU encoders: which of the workers' GPU models (see remote_models) its runs are for, plus one; 0: none
		std::unordered_map<ComboTable::Id, uint8_t> status; // per combination made or judged: 1 ran, 2 left out or failed; missing or 0: not run (yet)
		std::unordered_map<ComboTable::Id, size_t> task;    // combination -> its task
		size_t layer = 0;                              // the layers made into runs so far
		bool expanding = true;                         // more runs of it may come: its streams are not judged yet
		std::set<std::string> groups;                  // the groups (see Run::group) its runs are in
		std::function<bool(const Variant&, LazyGroup*, ComboTable::Id)> make; // makes the task of a combination: false if it is not wanted
	};
	std::map<ShareKey, std::unique_ptr<LazyTable>> lazy_tables; // shared by the scenarios with the same chroma, bit depth and range
	std::deque<Variant> lazy_variants;                 // stable addresses: a task points at its variant's options
	std::deque<LazyGroup> lazy_groups;
	std::map<std::string, std::vector<LazyGroup*>> lazy_of_input;
	std::map<std::string, std::map<std::string, size_t>> left_out; // input -> option -> combinations never made into runs because it showed no effect
	auto is_lazy = [&](const Job& job) { return job.prune && !cfg.only_runs_set; };
	// Distributed: a GPU encoder's runs are made for every GPU model the workers have that can encode the codec (as in variants_of), and its "gpu" option (which GPU of
	// the host) has no meaning. Each model is a lazy group of its own: the runs of two GPU models are pruned by their own scores.
	std::map<const Job*, std::vector<RemoteHost::Model>> remote_models;
	auto models_of = [&](const Job& job) -> const std::vector<RemoteHost::Model>& {
		auto it = remote_models.find(&job);
		if (it != remote_models.end()) return it->second;
		auto& m = remote_models[&job];
		if (remote && is_gpu_encoder(job.encoder)) {
			m = remote->models(job.encoder, job.codec);
			if (m.empty()) ui::log("no worker has a GPU that can encode " + job.encoder + "/" + job.codec + ": those runs are skipped");
		}
		return m;
	};
	auto lazy_table_of = [&](const Job& job, const Scenario& sc) -> LazyTable& {
		auto& slot = lazy_tables[share_key(job, sc)];
		if (!slot) {
			std::vector<Axis> free_axes;
			if (!sc.chroma.empty()) free_axes.push_back(Axis{"chroma", sc.chroma, {}, {}});
			if (!sc.bitdepth.empty()) free_axes.push_back(Axis{"bitdepth", sc.bitdepth, {}, {}});
			if (!sc.range.empty()) free_axes.push_back(Axis{"range", sc.range, {}, {}});
			std::vector<Axis> axes;
			for (auto& ax : job.axes) if (!(remote && is_gpu_encoder(job.encoder) && ax.key == "gpu")) axes.push_back(ax);
			slot = std::make_unique<LazyTable>(job.encoder, axes, free_axes, job.fixed, job.prune);
		}
		return *slot;
	};
	auto lazy_variant = [&](LazyTable& lt, const Job& job, ComboTable::Id combo, size_t model) -> const Variant& {
		auto it = lt.made.find({combo, model});
		if (it != lt.made.end()) return *it->second;
		Variant v = build_variant(job, lt.table.options(combo));
		if (model) { // a worker's GPU model, as variants_of does it
			const RemoteHost::Model& m = models_of(job)[model - 1];
			v.model = m.description;
			v.key = key_encoder(job.encoder, job.codec, v.opts, m.description);
			v.gpu = m.description;
			remote_texts.push_back(m.info);
			v.gpu_json = &remote_texts.back();
			remote_texts.push_back(m.caps);
			v.caps_json = &remote_texts.back();
		}
		lazy_variants.push_back(std::move(v));
		return *(lt.made[{combo, model}] = &lazy_variants.back());
	};
	int total = 0;

	// The runs of one input. With lazy planning only the first input is made up front and each next one when its turn comes: a big config has over a million runs
	// at the start, and holding those of every input at once costs gigabytes.
	auto create_input_tasks = [&](const std::string& input) {
		const std::string key_head = key_input(input);
		const std::string folder = cfg.short_names ? key_hash(key_input(input)).substr(0, 8) : sanitize(fs::path(file_name(input)).stem().string());
		const std::string folder_path = cfg.output_dir.empty() ? std::string() : (fs::path(cfg.output_dir) / folder).string() + sep;
		const auto folder_made = std::make_shared<bool>(false);
		for (auto& job : cfg.jobs) {
			const Codec id = codec_of(job.encoder, job.codec);
			if (id == Codec::None) { fprintf(stderr, "unknown encoder/codec '%s'/'%s'\n", job.encoder.c_str(), job.codec.c_str()); failed++; continue; }
			const std::string encoder_label = is_gpu_encoder(job.encoder) ? job.encoder + "/" + job.codec : job.encoder;
			for (auto& sc : cfg.scenarios) if (takes_part(job, sc)) {
				ui::log("Preparing " + job.encoder + "/" + job.codec + ", scenario " + (sc.name.empty() ? "(unnamed)" : sc.name) + ", " + file_name(input));
				const bool lazy_job = is_lazy(job);
				LazyTable* lt = lazy_job ? &lazy_table_of(job, sc) : nullptr;
				const std::vector<Variant>* sets = lazy_job ? nullptr : &variants_of(job, sc);
				for (int64_t br : sc.bitrates) for (double kf : sc.kfs) {
					const std::string kf_text = std::to_string((double)kf).substr(0, std::to_string((double)kf).find('.') + 2);
					for (auto& shape : sc.shapes) {
						std::string head = encoder_label; // the part of the label the options come after
						if (shape.w) head += " " + std::to_string(shape.w) + "x" + std::to_string(shape.h);
						if (shape.fps.num) head += " @" + std::to_string(shape.fps.value()).substr(0, 6);
						const std::string key_run_text = key_run(br, kf, shape);
						const std::string group_head = input + "|" + job.encoder + "/" + job.codec + "|" + std::to_string(br) + "|" + std::to_string(kf) + "|" + std::to_string(shape.w) + "x" + std::to_string(shape.h) + "@" + std::to_string(shape.fps.num) + "/" + std::to_string(shape.fps.den) + "|";
						const auto key_parts = std::make_shared<const Task::KeyParts>(Task::KeyParts{key_head, key_run_text + key_tail});
						// Makes the run of one variant. A lazy group keeps it, to make more runs later.
						auto make_task = [jobp = &job, scp = &sc, shape_i = (size_t)(&shape - sc.shapes.data()), id, input, br, kf, shape, head, key_parts, kf_text, group_head, folder, folder_path, folder_made, &tasks, &cfg](const Variant& v, LazyGroup* lg, ComboTable::Id combo) {
							const Job& job = *jobp;
							Task t;
							t.job = jobp; t.id = id; t.sc = scp; t.shape_i = shape_i; t.model = v.model;
							t.r.input = input; t.r.bitrate = br; t.r.kf = kf; t.r.shape = shape;
							t.r.label = head + v.label;
							t.opts = &v.opts;
							t.r.encoder = job.encoder; t.r.codec = job.codec; t.r.options = &v.opts;
							// The run's result (.json) and stream (.mp4) share a name made from the run and a hash of everything its outcome depends on, so a
							// run that comes up again finds its result straight away. They go to the output folder; without one, results are kept in the cache.
							t.key_parts = key_parts; t.variant_key = &v.key;
							const std::string key = t.key();
							if (cfg.only_runs_set && !cfg.only_runs.count(run_options_of(key))) return false; // the host has this one already
							if (!cfg.output_dir.empty()) {
								const std::string base = cfg.short_names ? key_hash(key).substr(0, 16) : result_name(t.r.label + " " + std::to_string(br) + "k kf" + kf_text, key);
								t.result_file = folder_path + base + ".json";
								t.r.result = folder + "/" + base + ".json";
								if (cfg.save_videos) {
									t.r.video = folder + "/" + base + ".mp4";
									t.r.video_file = folder_path + base + ".mp4";
								}
								if (!*folder_made) { std::error_code ec; fs::create_directories(fs::path(cfg.output_dir) / folder, ec); *folder_made = true; }
							} else if (!cfg.cache_dir.empty()) t.result_file = (fs::path(cfg.cache_dir) / "results" / (key_hash(key) + ".json")).string();
							t.r.group = group_head + v.gpu_index + (v.model.empty() ? std::string() : "|" + v.model);
							t.r.gpu = v.gpu; t.r.gpu_json = v.gpu_json; t.r.caps_json = v.caps_json;
							t.pool = v.pool->name;
							t.group = lg; t.combo = combo;
							if (lg) { lg->task[combo] = tasks.size(); lg->groups.insert(t.r.group); }
							t.index = tasks.size();
							v.pool->tasks.push_back(tasks.size());
							tasks.push_back(std::move(t));
							return true;
						};
						if (!lazy_job) { for (const Variant& v : *sets) make_task(v, nullptr, 0); continue; }
						// One group, or one per worker GPU model for a GPU encoder in a distributed sweep.
						const size_t groups_here = remote && is_gpu_encoder(job.encoder) ? models_of(job).size() : 1;
						for (size_t m = 0; m < groups_here; m++) {
							lazy_groups.emplace_back();
							LazyGroup& g = lazy_groups.back();
							g.lt = lt; g.job = &job; g.input = input; g.make = make_task;
							g.model = remote && is_gpu_encoder(job.encoder) ? m + 1 : 0;
							lazy_of_input[input].push_back(&g);
							for (ComboTable::Id c : lt->table.layer(0)) {
								if (!g.make(lazy_variant(*lt, job, c, g.model), &g, c)) g.status[c] = 2;
							}
						}
					}
				}
			}
		}
	};
	const bool all_inputs_now = std::any_of(cfg.jobs.begin(), cfg.jobs.end(), [&](const Job& job) { return !is_lazy(job); });
	std::set<std::string> created_inputs;
	for (auto& input : cfg.inputs) {
		if (!all_inputs_now && !created_inputs.empty()) break;
		if (created_inputs.insert(input).second) create_input_tasks(input);
	}
	total = (int)tasks.size();

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
			put("  %s%s%s: %zu%s runs\n", job.encoder.c_str(), is_gpu_encoder(job.encoder) ? "/" : "", is_gpu_encoder(job.encoder) ? job.codec.c_str() : "", runs_here, is_lazy(job) ? "+" : "");
		}
		put("Pools:");
		for (auto& [name, pl] : pools) put(" %s x%d (%zu runs)", name.c_str(), pl.parallel, pl.tasks.size());
		put("\nCache: %s   Output: %s%s\n\n%zu%s runs in total\n", cfg.cache_dir.empty() ? "off" : cfg.cache_dir.c_str(), cfg.output_dir.empty() ? "off" : cfg.output_dir.c_str(),
		    cfg.output_dir.empty() || !cfg.save_videos ? "" : " (with encoded streams)", tasks.size(), lazy_groups.empty() ? "" : "+");
		ui::log(text);
	}

	// Which result files there are, by folder: one listing of a folder instead of opening the result file of every run (a big config plans hundreds of
	// thousands of runs per input, and on Windows each open of a file that is not there costs about as much as listing a folder). A result written by this
	// process is added; the listings are dropped when the workers have sent theirs and when the next input comes up, so files written meanwhile are seen.
	std::unordered_map<std::string, std::unordered_set<std::string>> result_dirs;
	auto split_file = [](const std::string& file) {
		const size_t cut = file.find_last_of("/\\");
		return cut == std::string::npos ? std::make_pair(std::string("."), file) : std::make_pair(file.substr(0, cut), file.substr(cut + 1));
	};
	auto result_listed = [&](const std::string& file) {
		auto [dir, name] = split_file(file);
		auto it = result_dirs.find(dir);
		if (it == result_dirs.end()) {
			std::unordered_set<std::string> names;
			std::error_code ec;
			for (fs::directory_iterator e(dir, ec), end; !ec && e != end; e.increment(ec)) names.insert(e->path().filename().string());
			it = result_dirs.emplace(dir, std::move(names)).first;
		}
		return it->second.count(name) != 0;
	};
	auto cached_of = [](const Run& r) {
		CachedRun c{r.kbps, r.scores, r.detail};
		if (r.ceiling) c.ceiling = *r.ceiling;
		return c;
	};
	// Gives a finished run its ceiling (see ceiling_wave): from memory, the cache folder, or scored now, once for each feed. Set further down.
	std::function<void(Task&)> attach_ceiling = [](Task&) {};
	auto store_result = [&](const Task& t, const std::string& file, const CachedRun& run, const std::string& details) {
		result_store(file, t.key(), run, details, cfg.vo);
		auto [dir, name] = split_file(file);
		if (auto it = result_dirs.find(dir); it != result_dirs.end()) it->second.insert(name);
	};

	// A run is finished when its result is known, however it came about: print it, count it.
	std::atomic<int> finished{0};
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
		for (auto& [k, v] : *r.options) { o << (first ? "" : ",") << jq(k) << ":" << jq(v); first = false; }
		o << "},\"width\":" << r.shape.w << ",\"height\":" << r.shape.h << ",\"fps\":" << (r.shape.fps.num ? r.shape.fps.value() : 0.0)
		  << ",\"target_kbps\":" << r.bitrate << ",\"keyframe_interval_s\":" << r.kf << ",\"gpu\":" << (r.gpu.empty() ? std::string("null") : jq(r.gpu))
		  << ",\"encoder_version\":" << (is_gpu_encoder(r.encoder) ? std::string("null") : jq(encoder_library_version(r.encoder))) // the library behind a software encoder (a GPU encoder has gpu_info)
		  << ",\"gpu_info\":" << *r.gpu_json << ",\"encoder_caps\":" << *r.caps_json << ",\"video\":" << (r.video.empty() ? std::string("null") : jq(r.video))
		  << ",\"video_sha256\":" << (r.video_sha256.empty() ? std::string("null") : jq(r.video_sha256)) << ",\"reference_sha256\":" << jq(reference_hash(r.input));
		return o.str();
	};
	// Results go out as the runs are scored, from a thread of their own. Runs whose streams are identical to one already sent are one result (see below).
	std::unique_ptr<Uploader> uploader;
	std::set<std::string> sent_streams;
	if (cfg.upload.url.empty()) ui::log("Upload: off (no \"upload\": {\"url\": ...} in the config and no --upload)");
	else if (cfg.output_dir.empty()) ui::log("Upload: off, there is no output folder (\"output\" in the config or --output); results are only kept in the cache");
	else {
		UploadOptions up = cfg.upload;
		if (up.key.empty()) if (const char* k = getenv("VES_API_KEY")) up.key = k;
		uploader = std::make_unique<Uploader>(up, [](const std::string& line) { ui::log(line); });
	}
	std::map<std::string, std::vector<Task*>> group_ok; // group -> its runs that were scored, for picking the stream to keep
	std::map<std::string, int> group_left; // group -> runs not finished yet; its kept stream is uploaded when this reaches 0
	if (cfg.best_videos) for (auto& t : tasks) group_left[t.r.group]++;
	auto finish_run = [&](Task& t) {
		Run& r = t.r;
		ui::overall(++finished, total);
		char line[1024];
		if (r.pruned) { snprintf(line, sizeof line, "%s pruned  %s: %s", r.tag.c_str(), r.label.c_str(), r.reason.c_str()); ui::log(line); return; }
		if (r.ignored) { if (!r.expected) { snprintf(line, sizeof line, "%s IGNORED  %s: %s", r.tag.c_str(), r.label.c_str(), r.reason.c_str()); ui::log(line); } return; }
		if (!t.ok) {
			snprintf(line, sizeof line, "%s FAILED  %s  (%s, %lld kbps)", r.tag.c_str(), r.label.c_str(), r.input.c_str(), (long long)r.bitrate);
			ui::log(std::string(line) + (r.reason.empty() ? "" : ": " + r.reason));
			return;
		}
		r.penalty = over_penalty_points(r.kbps, (double)r.bitrate);
		attach_ceiling(t);
		std::vector<double> sc = r.scores;
		std::sort(sc.begin(), sc.end());
		const double mean = std::accumulate(sc.begin(), sc.end(), 0.0) / sc.size();
		char rel[96] = "";
		if (const auto d = vs_ceiling(r); !d.empty()) { const FrameStats st = frame_stats(d); snprintf(rel, sizeof rel, ", vs ceiling mean %.3f p5 %.3f", st.mean, st.p5); }
		if (!t.result_file.empty() && (!r.cached || r.imported || r.ceiling_new)) store_result(t, t.result_file, cached_of(r), run_details(t));
		r.detail = VmafDetail{}; // in the result file now; nothing else reads it, and a sweep holds many runs
		if (uploader && r.result.empty()) ui::log("Upload " + r.label + ": not sent, the run has no result file in the output folder");
		else if (uploader && !r.uploaded && !r.video_sha256.empty() && !sent_streams.insert(r.group + "|" + r.video_sha256).second)
			ui::log("Upload " + r.label + ": not sent, its stream is identical to one of its group that was sent already");
		else if (uploader && !r.uploaded) {
			UploadItem it;
			it.result_file = (fs::path(cfg.output_dir) / r.result).string();
			it.label = r.label;
			if (!cfg.best_videos && !r.video.empty() && fs::exists(fs::path(cfg.output_dir) / r.video)) it.video_file = (fs::path(cfg.output_dir) / r.video).string();
			uploader->add(std::move(it));
		}
		// Only the stream of the group's winner stays, decided when the group is complete (see `finish`).
		if (cfg.best_videos) group_ok[r.group].push_back(&t);
		snprintf(line, sizeof line, "%s finished  %s: %.0f kbps, vmaf mean %.3f min %.3f p5 %.3f%s%s%s%s  [wave %.1fs: encoder busy %.1f, starved %.1f; decoder backed up %.1f; scorer starved %.1f]", r.tag.c_str(), r.label.c_str(), r.kbps,
		         mean, sc.front(), sc[(size_t)(sc.size() * 0.05)], rel, r.cached ? " (cached)" : "",
		         r.gpu.empty() ? "" : " on ", r.gpu.c_str(), r.encode_s, r.enc_busy_s, r.enc_starved_s, r.decode_blocked_s, r.score_starved_s);
		ui::log(line);
	};
	// The group is complete: the stream of its winner stays, the others go (see below).
	auto complete_group = [&](const std::string& group) {
		// The winner is ranked as in the summary: the run that is at least as good as all the others (after the overshoot penalty)
		// in the most frames; ties go to the higher mean. Only runs that still have a stream can keep it.
		auto& runs_ok = group_ok[group];
		const size_t frames = runs_ok.empty() ? 0 : runs_ok.front()->r.scores.size();
		std::vector<double> top(frames, 0.0);
		for (Task* x : runs_ok) if (x->r.scores.size() == frames) for (size_t f = 0; f < frames; f++) top[f] = std::max(top[f], x->r.scores[f] - x->r.penalty);
		Task* kept = nullptr;
		double kept_share = -1, kept_mean = 0;
		for (Task* x : runs_ok) {
			if (x->r.video.empty() || !fs::exists(fs::path(cfg.output_dir) / x->r.video)) continue;
			size_t best = 0;
			double sum = 0;
			for (size_t f = 0; f < x->r.scores.size(); f++) {
				sum += x->r.scores[f];
				best += x->r.scores.size() == frames && x->r.scores[f] - x->r.penalty >= top[f] - 1e-6;
			}
			const double share = frames ? (double)best / frames : 0, mean = sum / std::max<size_t>(1, x->r.scores.size()) - x->r.penalty;
			if (share > kept_share || (share == kept_share && mean > kept_mean)) { kept = x; kept_share = share; kept_mean = mean; }
		}
		for (Task* x : runs_ok) {
			if (x == kept || x->r.video.empty()) continue;
			std::error_code ec;
			fs::remove(fs::path(cfg.output_dir) / x->r.video, ec);
			x->r.video.clear();
			if (uploader && x->r.cached && !x->r.result.empty()) { // an earlier sweep may have sent it to the site as its best
				UploadItem drop;
				drop.result_file = (fs::path(cfg.output_dir) / x->r.result).string();
				drop.label = x->r.label;
				drop.drop_video = true;
				uploader->add(std::move(drop));
			}
		}
		if (!kept || !uploader) return;
		UploadItem it;
		it.result_file = (fs::path(cfg.output_dir) / kept->r.result).string();
		it.video_file = (fs::path(cfg.output_dir) / kept->r.video).string();
		it.label = kept->r.label;
		it.late = true;
		uploader->add(std::move(it));
	};
	auto finish = [&](Task& t) {
		t.done = true;
		if (t.group) t.group->status[t.combo] = t.ok ? 1 : 2;
		finish_run(t);
		if (!cfg.best_videos || --group_left[t.r.group] > 0) return;
		if (t.group && t.group->expanding) return; // more runs of its group may come; they are judged when it is done
		complete_group(t.r.group);
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
		ui::log("Source " + file_name(input) + ": " + std::to_string(si.width) + "x" + std::to_string(si.height) + (cfg.in_process ? " decoded by " + si.decoder : std::string()));
		ui::emit_record((char)29, source_json(it->second));
		return &it->second;
	};
	auto fit_shape = [](Task& t) {
		if (!t.r.shape.w) { t.r.shape.w = t.plan.enc.w; t.r.shape.h = t.plan.enc.h; }
		if (!t.r.shape.fps.num) t.r.shape.fps = t.plan.enc.fps;
	};
	// One ceiling per model, index 0 the first (vmaf_models order); each model's is cached on its own (see ceiling_key), so a model added later
	// scores only its own. By the first model's ceiling_key; null: could not be scored (not tried again).
	using Ceilings = std::vector<std::vector<double>>;
	std::map<std::string, std::shared_ptr<const Ceilings>> ceilings;
	const std::vector<std::string> all_models = vmaf_models(cfg.vo);
	auto model_opts = [&](const std::string& m) { VmafOptions o = cfg.vo; o.model = m; o.extra_models.clear(); return o; };
	attach_ceiling = [&](Task& t) {
		Run& r = t.r;
		if (!cfg.ceiling || !t.ok) return;
		const bool have_extra = r.detail.extra.size() == all_models.size() - 1 && std::all_of(r.detail.extra.begin(), r.detail.extra.end(), [](const ModelScores& m) { return !m.ceiling.empty(); });
		if (r.ceiling && have_extra) return;
		Plan pl = t.plan;
		if (!pl.enc.w) { // a run taken from its result file: planned now, from the shape it asked for (the file's frame rate is rounded)
			const SourceInfo* src = source_of(r.input);
			if (!src) return;
			pl = make_plan(*src, t.job->encoder, t.job->codec, *t.opts, t.sc && t.shape_i < t.sc->shapes.size() ? t.sc->shapes[t.shape_i] : r.shape);
			if (pl.ignored) return;
		}
		std::vector<std::string> keys;
		for (auto& m : all_models) keys.push_back(ceiling_key(r.input, pl.feed_key(), pl.score_fmt, cfg.scale, model_opts(m)));
		auto it = ceilings.find(keys[0]);
		if (it == ceilings.end()) {
			auto file_of = [&](const std::string& key) { return cfg.cache_dir.empty() ? std::string() : (fs::path(cfg.cache_dir) / "ceilings" / (key_hash(key) + ".json")).string(); };
			Ceilings sc(all_models.size());
			std::vector<size_t> missing;
			for (size_t m = 0; m < all_models.size(); m++) { const std::string file = file_of(keys[m]); if (file.empty() || !ceiling_load(file, keys[m], sc[m])) missing.push_back(m); }
			bool ok = true;
			if (!missing.empty()) {
				ok = false;
				if (const SourceInfo* src = source_of(r.input)) {
					const std::string what = file_name(r.input) + " at " + std::to_string(pl.enc.w) + "x" + std::to_string(pl.enc.h) + " " + pl.feed_key().substr(pl.feed_key().find(' ') + 1);
					VmafOptions o = model_opts(all_models[missing[0]]); // only the models whose ceiling is not cached yet
					for (size_t k = 1; k < missing.size(); k++) o.extra_models.push_back(all_models[missing[k]]);
					RunProgress rp;
					const int h = ui::attach(&rp, "ceiling " + what, "decode");
					const auto t0 = std::chrono::steady_clock::now();
					std::vector<std::vector<double>> extra;
					ok = ceiling_wave(local_path(r.input), *src, pl, cfg.scale, o, sc[missing[0]], &rp, &extra);
					ui::detach(h);
					for (size_t k = 1; ok && k < missing.size(); k++) sc[missing[k]] = std::move(extra[k - 1]);
					char line[512];
					if (ok) snprintf(line, sizeof line, "Ceiling %s: vmaf mean %.3f min %.3f  (%.1fs)", what.c_str(), frame_stats(sc[0]).mean, frame_stats(sc[0]).min, std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count());
					else snprintf(line, sizeof line, "Ceiling %s could not be scored; its runs have no score against it", what.c_str());
					ui::log(line);
					if (ok) for (size_t m : missing) if (const std::string file = file_of(keys[m]); !file.empty()) ceiling_store(file, keys[m], sc[m]);
				}
			}
			it = ceilings.emplace(keys[0], ok ? std::make_shared<const Ceilings>(std::move(sc)) : nullptr).first;
		}
		if (!it->second || (*it->second)[0].size() != r.scores.size()) return;
		if (!r.ceiling) { r.ceiling = std::shared_ptr<const std::vector<double>>(it->second, &(*it->second)[0]); r.ceiling_new = true; }
		if (r.detail.extra.size() == all_models.size() - 1)
			for (size_t m = 1; m < all_models.size(); m++)
				if (ModelScores& e = r.detail.extra[m - 1]; e.ceiling.empty() && (*it->second)[m].size() == e.scores.size()) { e.ceiling = (*it->second)[m]; r.ceiling_new = true; }
	};
	// Distributed: the runs of a round (an input's runs made so far, then each next layer of its lazy groups) that are not done yet are cut into jobs (one input,
	// encoder entry, scenario, bitrate, key frame interval, size and GPU model each) for the workers that connect to --host. What they send back is stored as result
	// files with this run's keys, so settle() finds those runs finished and their scores decide the next layer. Whatever the workers could not do (runs that cannot
	// work with their input, a job given up) is run here as usual.
	auto remote_round = [&](const std::vector<size_t>& todo) {
		if (!remote) return;
		std::map<std::string, RemoteJobSpec> groups;
		std::vector<Task*> asked; // the runs that were handed to workers
		std::vector<std::string> order;
		std::map<std::string, std::vector<std::pair<Options, Shape>>> job_runs; // the options and size of each run of a job, for telling whether the input allows any
		for (size_t i : todo) {
			Task& t = tasks[i];
			if (t.done || t.result_file.empty() || result_listed(t.result_file)) continue;
			if (std::string why = caps_refusal(t.job->encoder, t.job->codec, *t.opts, t.r.caps_json); !why.empty()) { t.r.ignored = true; t.r.reason = why; finish(t); continue; }
			const std::string g = t.r.input + "|" + std::to_string(t.job - cfg.jobs.data()) + "|" + std::to_string(t.sc - cfg.scenarios.data()) + "|" + std::to_string(t.r.bitrate) + "|" +
			                      std::to_string(t.r.kf) + "|" + std::to_string(t.shape_i) + "|" + t.model;
			auto it = groups.find(g);
			if (it == groups.end()) {
				RemoteJobSpec spec;
				spec.config = sub_config(cfg, *t.job, *t.sc, t.r.bitrate, t.r.kf, t.r.shape);
				spec.input = t.r.input;
				spec.encoder = t.job->encoder;
				spec.codec = t.job->codec;
				spec.model = t.model;
				spec.name = file_name(t.r.input) + " " + t.job->encoder + (is_gpu_encoder(t.job->encoder) ? "/" + t.job->codec : std::string()) + " " + std::to_string(t.r.bitrate) + "k kf" + std::to_string(t.r.kf).substr(0, std::to_string(t.r.kf).find('.') + 2) +
				            (t.r.shape.w ? " " + std::to_string(t.r.shape.w) + "x" + std::to_string(t.r.shape.h) : std::string()) + (t.r.shape.fps.num ? "@" + std::to_string(t.r.shape.fps.value()).substr(0, 6) : std::string());
				order.push_back(g);
				it = groups.emplace(g, std::move(spec)).first;
			}
			it->second.runs.push_back({t.result_file, t.key()});
			asked.push_back(&t);
			job_runs[g].push_back({*t.opts, t.r.shape});
		}
		std::vector<RemoteJobSpec> jobs;
		for (auto& g : order) {
			RemoteJobSpec spec = std::move(groups[g]);
			// A URL input is downloaded here once and streamed to the workers, which then all have the same bytes as the host.
			if (is_url(spec.input)) {
				try { spec.path = local_path(spec.input); } catch (const std::exception& e) {
					if (!source_failed.count(spec.input)) ui::log("Input " + file_name(spec.input) + ": " + e.what());
					source_failed.insert(spec.input);
					continue;
				}
			}
			// The job is left out when every run of it asks for more than the input has (a higher frame rate, or a size, chroma or bit depth above it without "upscale").
			// Other reasons a run cannot work (this GPU, a codec) are for the worker to find out.
			spec.possible = [encoder = spec.encoder, codec = spec.codec, runs = job_runs[g]](const std::string& json, std::string& why) {
				SourceInfo si;
				try { si = source_from_json(json); } catch (const std::exception&) { return true; }
				for (auto& [opts, shape] : runs) {
					const Plan plan = make_plan(si, encoder, codec, opts, shape);
					if (!(plan.ignored && plan.expected)) return true;
					why = plan.reason;
				}
				return false;
			};
			jobs.push_back(std::move(spec));
		}
		if (jobs.empty()) return;
		for (auto& j : jobs) j.name += " (" + std::to_string(j.runs.size()) + " runs)";
		const size_t already = (size_t)finished.load(); // runs that are finished so far
		ui::overall((int)already, total);
		std::map<std::string, Task*> by_file;
		for (Task* t : asked) by_file[t->result_file] = t;
		// A result is written again with this sweep's details (the worker's copy of the input has another name and no URL) and goes to the site as soon as it is here, not
		// when the whole sweep is through.
		const size_t stored = remote->run_jobs(jobs, [&](size_t n, const std::string& file) {
			ui::overall((int)(already + n), total);
			auto it = by_file.find(file);
			if (it == by_file.end()) { ui::log("remote: a worker's result " + file + " belongs to no run of this sweep; it is not used or uploaded"); return; }
			Task& t = *it->second;
			CachedRun hit;
			if (!result_load(file, t.key(), hit)) { ui::log("remote: the worker's result for " + t.r.label + " cannot be read back (" + file + "); it is not uploaded now"); return; }
			t.r.kbps = hit.kbps; t.r.scores = std::move(hit.scores); t.r.detail = std::move(hit.detail);
			if (!hit.ceiling.empty()) t.r.ceiling = std::make_shared<const std::vector<double>>(std::move(hit.ceiling));
			store_result(t, file, cached_of(t.r), run_details(t));
			t.r.imported = false; // already rewritten
			if (uploader && !t.r.result.empty()) {
				UploadItem it2;
				it2.result_file = (fs::path(cfg.output_dir) / t.r.result).string();
				it2.label = t.r.label;
				uploader->add(std::move(it2));
				t.r.uploaded = true;
			}
		});
		ui::log("remote: " + std::to_string(stored) + " results from the workers");
		for (Task* t : asked) t->r.imported = fs::exists(t->result_file);
		result_dirs.clear(); // the workers' results are there now
	};

	// Whatever cannot work, or was done before, is finished right away. The rest is sorted by the frames it needs: runs with the
	// same input, resolution, frame rate and pixel format share one decode.
	std::map<std::string, std::vector<size_t>> uncached; // by input: the runs that have to be done, planned when their input comes up
	std::vector<std::string> input_order;
	auto settle = [&](size_t i) -> bool { // true: the run was done before, and is finished from its result
		Task& t = tasks[i];
		Run& r = t.r;
		char tag[48];
		snprintf(tag, sizeof tag, "[run %zu/%d]", i + 1, total);
		r.tag = tag;
		if (!t.result_file.empty() && result_listed(t.result_file)) {
			CachedRun hit;
			if (result_load(t.result_file, t.key(), hit)) {
				r.kbps = hit.kbps; r.scores = std::move(hit.scores); r.detail = std::move(hit.detail); r.video_sha256 = hit.video_sha256; r.cached = t.ok = true;
				if (!hit.ceiling.empty()) r.ceiling = std::make_shared<const std::vector<double>>(std::move(hit.ceiling));
				if (!hit.reference_sha256.empty()) reference_hashes.emplace(r.input, hit.reference_sha256); // known: the file is not needed to say it
				if (!r.shape.w && hit.width > 0 && hit.height > 0 && hit.fps > 0) { // the geometry the run was encoded at, for the summary: from its result file
					r.shape.w = hit.width; r.shape.h = hit.height;
					r.shape.fps = Rational{(int)std::lround(hit.fps * 1000), 1000};
				} else if (!r.shape.w) if (const SourceInfo* src = source_of(r.input)) { // an older result file: ask the source (which has to be had for that)
					t.plan = make_plan(*src, t.job->encoder, t.job->codec, *t.opts, r.shape);
					fit_shape(t);
				}
				const bool have_video = !r.video_file.empty() && fs::exists(r.video_file);
				if (!have_video) r.video.clear(); // the stream of the earlier run, if it is still there
				// Result files from before the hashes existed get them now, as far as the files they describe are still there.
				const bool new_video_hash = r.video_sha256.empty() && have_video && !(r.video_sha256 = sha256_file(r.video_file)).empty();
				if ((new_video_hash || hit.reference_sha256.empty() || hit.input != file_name(r.input) || (hit.width <= 0 && r.shape.w > 0)) && !reference_hash(r.input).empty())
					store_result(t, t.result_file, cached_of(r), run_details(t));
				r.video_file.clear();
				finish(t);
				return true;
			}
		}
		return false;
	};
	for (size_t i = 0; i < tasks.size(); i++) if (!settle(i)) {
		const std::string& in = tasks[i].r.input;
		if (!uncached.count(in)) input_order.push_back(in);
		uncached[in].push_back(i);
	}

	if (!all_inputs_now) input_order = cfg.inputs; // the others are made when their turn comes

	// Pruning, after encoding: an option that shows no effect is not swept further. Within a group (and with every other option at the same values) the
	// runs that differ only in one option form a line. Its lowest and its highest value run first; when their scores match on every
	// frame, the option does nothing there and the values in between are not run. When they differ, or a probe failed, the line runs in full.
	// The probes are runs of the sweep itself, with the other options as they actually are, so an option that only matters in some situations
	// (B-frame settings when B-frames are on) is found where it matters. Cached runs count as probes too.
	struct Line { Task* lo = nullptr; Task* hi = nullptr; int verdict = 0; std::string axis; }; // verdict: 0 not known yet, 1 has an effect, 2 has none
	std::vector<Line> lines;
	{
		std::unordered_map<std::string, int> line_ids;
		std::map<const Job*, std::vector<std::tuple<size_t, std::string, std::string>>> samples; // per job: axis, lowest, highest
		for (auto& t : tasks) {
			const Job& job = *t.job;
			if (!job.prune || t.group) continue;
			auto sm = samples.find(&job);
			if (sm == samples.end()) {
				sm = samples.emplace(&job, decltype(sm->second){}).first;
				for (size_t a = 0; a < job.axes.size(); a++) {
					const Axis& ax = job.axes[a];
					if (job.fixed.count(ax.key) || ax.key == "gpu" || ax.key == "chroma" || ax.key == "bitdepth" || ax.key == "range") continue;
					std::vector<std::string> vals;
					bool numeric = true;
					for (auto& v : ax.values) {
						if (std::find(vals.begin(), vals.end(), v) != vals.end()) continue;
						vals.push_back(v);
						char* end = nullptr;
						strtod(v.c_str(), &end);
						numeric &= !v.empty() && *end == 0;
					}
					if (vals.size() < 3) continue; // nothing in between to leave out
					auto less = [](const std::string& x, const std::string& y) { return atof(x.c_str()) < atof(y.c_str()); };
					if (numeric) sm->second.emplace_back(a, *std::min_element(vals.begin(), vals.end(), less), *std::max_element(vals.begin(), vals.end(), less));
					else sm->second.emplace_back(a, vals.front(), vals.back()); // not numbers (presets): the order of the list
				}
			}
			for (auto& [a, lo, hi] : sm->second) {
				const std::string& name = job.axes[a].key;
				const auto have = t.opts->find(name);
				std::string key = std::to_string((uintptr_t)&job) + "|" + t.r.group + "|" + name + "|";
				for (auto& [k, v] : *t.opts) if (k != name) key += k + "=" + v + ";";
				auto [at, fresh] = line_ids.emplace(key, (int)lines.size());
				if (fresh) { lines.emplace_back(); lines.back().axis = name; }
				Line& L = lines[at->second];
				if (have != t.opts->end() && have->second == lo) L.lo = &t;
				else if (have != t.opts->end() && have->second == hi) L.hi = &t;
				else t.mid.push_back(at->second);
			}
		}
	}
	auto verdict = [&](Line& L) {
		if (L.verdict) return L.verdict;
		if (!L.lo || !L.hi) return L.verdict = 1;
		const int a = L.lo->ok ? 1 : L.lo->done ? 2 : 0, b = L.hi->ok ? 1 : L.hi->done ? 2 : 0; // ran, did not run, still to come
		if (a == 2 || b == 2) return L.verdict = 1;
		if (!a || !b) return 0;
		const Run &x = L.lo->r, &y = L.hi->r;
		bool same = x.scores.size() == y.scores.size() && std::abs(x.kbps - y.kbps) <= 1e-6 * std::max(1.0, x.kbps);
		for (size_t f = 0; same && f < x.scores.size(); f++) same = std::abs(x.scores[f] - y.scores[f]) <= 1e-6;
		return L.verdict = same ? 2 : 1;
	};
	// 0: can run now, 1: pruned (`why` says by which option), 2: waits for probes
	auto prune_state = [&](Task& t, std::string& why) {
		int state = 0;
		for (int id : t.mid) {
			const int v = verdict(lines[id]);
			if (v == 2) { why = lines[id].axis + " has no effect here: its lowest and highest values scored the same on every frame"; return 1; }
			if (v == 0) state = 2;
		}
		return state;
	};

	// Lazy planning, the next layer of a group: the combinations with one more prunable option in the middle of its range. Each is on a line along every one
	// of those options, with the two ends of the line in the layer before. It is made into a run when none of the lines shows "no effect" (both ends ran and scored the
	// same on every frame, as with pruning of the whole list); an end that did not run or is missing counts as an effect. The new runs that are not already done go
	// to `added`. False when the group has no more layers.
	auto status_of = [](const LazyGroup& g, ComboTable::Id c) { auto it = g.status.find(c); return it == g.status.end() ? 0 : (int)it->second; };
	auto advance = [&](LazyGroup& g, std::vector<size_t>& added) -> bool {
		if (g.layer + 1 >= g.lt->table.layers()) return false;
		const size_t k = ++g.layer;
		std::vector<ComboTable::Line> lines_of;
		for (ComboTable::Id c : g.lt->table.layer(k)) {
			g.lt->table.mid_lines(c, lines_of);
			const char* no_effect = nullptr;
			for (auto& L : lines_of) {
				if (L.lo < 0 || L.hi < 0 || status_of(g, (ComboTable::Id)L.lo) != 1 || status_of(g, (ComboTable::Id)L.hi) != 1) continue; // an end is missing or did not run: the line is run in full
				const Run &x = tasks[g.task.at((ComboTable::Id)L.lo)].r, &y = tasks[g.task.at((ComboTable::Id)L.hi)].r;
				bool same = x.scores.size() == y.scores.size() && std::abs(x.kbps - y.kbps) <= 1e-6 * std::max(1.0, x.kbps);
				for (size_t f = 0; same && f < x.scores.size(); f++) same = std::abs(x.scores[f] - y.scores[f]) <= 1e-6;
				if (same) { no_effect = g.lt->table.axis_key(L.axis).c_str(); break; }
			}
			if (no_effect) { g.status[c] = 2; left_out[g.input][no_effect]++; continue; }
			const size_t at = tasks.size();
			if (!g.make(lazy_variant(*g.lt, *g.job, c, g.model), &g, c)) { g.status[c] = 2; continue; }
			total = (int)tasks.size();
			if (cfg.best_videos) group_left[tasks[at].r.group]++;
			if (!settle(at)) added.push_back(at);
		}
		return true;
	};

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
	if (created_inputs.insert(input_name).second) { // lazy planning: the runs of this input are made now
		result_dirs.clear();
		const size_t first = tasks.size();
		create_input_tasks(input_name);
		total = (int)tasks.size();
		for (size_t i = first; i < tasks.size(); i++) {
			if (cfg.best_videos) group_left[tasks[i].r.group]++;
			if (!settle(i)) uncached[input_name].push_back(i);
		}
	}
	auto run_round = [&](const std::vector<size_t>& todo) {
	// Plan the runs of this input: opening the source is where a URL is downloaded. Runs that cannot work are finished right away; the rest is
	// sorted by the frames it needs, and runs with the same resolution, frame rate and pixel format share one decode.
	std::map<std::string, std::vector<Task*>> by_feed;
	std::vector<std::string> feed_order;
	for (size_t i : todo) {
		Task& t = tasks[i];
		Run& r = t.r;
		if (!t.model.empty()) { r.ignored = true; r.expected = true; r.reason = "no worker produced it (on " + t.model + ")"; finish(t); continue; } // not on this machine's GPU: the key says another one
		const SourceInfo* src = source_of(r.input);
		if (!src) { finish(t); continue; }
		t.plan = make_plan(*src, t.job->encoder, t.job->codec, *t.opts, r.shape);
		if (t.plan.ignored) { r.ignored = true; r.expected = t.plan.expected; r.reason = t.plan.reason; finish(t); continue; }
		if (std::string why = caps_refusal(t.job->encoder, t.job->codec, *t.opts, r.caps_json); !why.empty()) { r.ignored = true; r.reason = why; finish(t); continue; }
		fit_shape(t);
		const std::string fk = r.input + "|" + t.plan.feed_key();
		if (!by_feed.count(fk)) feed_order.push_back(fk);
		by_feed[fk].push_back(&t);
	}
	// The smallest uncompressed data rate of the encoded format (resolution, frame rate, chroma, bit depth) goes first: the cheap ones finish early.
	auto raw_rate = [&](const std::string& fk) {
		const Shape& s = by_feed[fk][0]->plan.enc;
		return (double)s.w * s.h * s.fps.value() * (s.pix.chroma == 444 ? 3.0 : 1.5) * s.pix.depth;
	};
	std::stable_sort(feed_order.begin(), feed_order.end(), [&](const std::string& a, const std::string& b) { return raw_rate(a) < raw_rate(b); });
	for (auto& fk : feed_order) {
		// The runs of a pool in order, each group's probes before the rest of it.
		std::map<std::string, std::list<Task*>> by_pool;
		{
			std::map<std::string, std::vector<Task*>> sorted;
			std::map<std::string, int> group_at;
			for (Task* t : by_feed[fk]) { group_at.emplace(t->r.group, (int)group_at.size()); sorted[t->pool].push_back(t); }
			for (auto& [pool, list] : sorted) {
				std::stable_sort(list.begin(), list.end(), [&](Task* a, Task* b) { return std::make_pair(group_at[a->r.group], a->mid.size()) < std::make_pair(group_at[b->r.group], b->mid.size()); });
				by_pool[pool].assign(list.begin(), list.end());
			}
		}
		// Runs whose encoder process crashed while they were in it are run once more, each on its own: the crash may have been another run's, and a run that
		// crashes alone is the one that does.
		std::deque<Task*> alone;
		for (;;) {
			std::vector<Task*> wave;
			if (!alone.empty()) { wave.push_back(alone.front()); alone.pop_front(); }
			else for (auto& [pool, list] : by_pool) {
				int room = pools[pool].parallel;
				for (auto it = list.begin(); it != list.end() && room > 0;) {
					Task& t = **it;
					std::string why;
					const int state = prune_state(t, why);
					if (state == 1) { t.r.pruned = true; t.r.reason = why; it = list.erase(it); finish(t); }
					else if (state == 0) { wave.push_back(&t); it = list.erase(it); room--; }
					else ++it;
				}
			}
			if (wave.empty()) { // only runs that wait for probes that will never come are left: run them
				for (auto& [pool, list] : by_pool) if (!list.empty()) { wave.push_back(list.front()); list.pop_front(); }
				if (wave.empty()) break;
			}
			const std::string& input = wave[0]->r.input;
			const SourceInfo& si = infos.at(input);
			std::vector<WaveRun> wr(wave.size());
			std::vector<WaveRun*> ptrs;
			std::vector<int> handles;
			for (size_t i = 0; i < wave.size(); i++) {
				Task& t = *wave[i];
				WaveRun& run = wr[i];
				EncodeJob& ej = run.enc;
				t.r.progress = std::make_shared<RunProgress>();
				t.r.score_progress = std::make_shared<RunProgress>();
				ej.encoder = t.job->encoder; ej.codec = t.job->codec; ej.plan = &t.plan;
				ej.bitrate = t.r.bitrate * 1000; ej.kf = t.r.kf; ej.rp = t.r.progress.get();
				ej.file = !t.r.video_file.empty() ? t.r.video_file : (spool_dir / ("run" + std::to_string(t.index) + ".mp4")).string();
				ej.id = t.id; ej.color = t.plan.color;
				run.codec = t.id; run.score_rp = t.r.score_progress.get();
				ptrs.push_back(&run);
				ui::log_plain(t.r.tag + " encoding and scoring  " + t.r.label + "  (" + file_name(input) + ", " + std::to_string(t.r.bitrate) + " kbps, pool " + t.pool + ")");
				run.label = t.r.label;
				run.pool = t.pool;
				if (cfg.in_process) { // otherwise the processes show these rows
					handles.push_back(ui::attach(t.r.progress.get(), t.r.label, t.pool));
					handles.push_back(ui::attach(t.r.score_progress.get(), t.r.label, t.pool));
				}
			}
			RunProgress feed_rp;
			feed_rp.begin(3, 1);
			const int feed_handle = cfg.in_process ? ui::attach(&feed_rp, "decode+convert " + file_name(input) + " to " + std::to_string(wave[0]->plan.enc.w) + "x" + std::to_string(wave[0]->plan.enc.h), "decode") : -1;
			FeedResult fr;
			const auto t_wave = std::chrono::steady_clock::now();
			if (cfg.in_process) run_wave(local_path(input), si, ptrs, cfg.scale, cfg.vo, &feed_rp, fr);
			else {
				std::string decoders;
				run_wave_isolated(local_path(input), si, ptrs, cfg.scale, cfg.vo, &feed_rp, fr, decoders);
				if (!decoders.empty() && infos.at(input).decoder != decoders) { // what the result file says decoded the input
					infos.at(input).decoder = decoders;
					ui::log("Source " + file_name(input) + " decoded by " + decoders);
				}
			}
			const double wave_s = std::chrono::duration<double>(std::chrono::steady_clock::now() - t_wave).count();
			if (feed_handle >= 0) ui::detach(feed_handle);
			for (int h : handles) ui::detach(h);
			const size_t crashed_runs = (size_t)std::count_if(wr.begin(), wr.end(), [](const WaveRun& w) { return w.enc.crashed; });
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
					r.detail = std::move(wr[i].detail);
					t.ok = true;
				}
				if (!t.ok) r.reason = wr[i].enc.error;
				std::error_code ec;
				if (!t.ok && wr[i].enc.crashed && !t.retried && crashed_runs > 1) { // others were in the process with it
					t.retried = true;
					fs::remove(wr[i].enc.file, ec);
					ui::log(t.r.tag + " " + t.r.label + ": its encoder process crashed; it is run again on its own");
					alone.push_back(&t);
					continue;
				}
				if (t.ok) r.video_sha256 = sha256_file(wr[i].enc.file);
					if (r.video_file.empty()) fs::remove(wr[i].enc.file, ec); // a temporary file
				finish(t);
			}
		}
	}
	};
	// Round after round: the runs of this input that were made so far, then the next layer of every group of it, until no group has a run to make.
	std::vector<size_t> todo = uncached[input_name];
	for (;;) {
		if (remote) { // the workers first; what they did is finished from its result, the rest is run here
			remote_round(todo);
			std::vector<size_t> left;
			for (size_t i : todo) if (!tasks[i].done && !settle(i)) left.push_back(i);
			todo = std::move(left);
		}
		run_round(todo);
		todo.clear();
		for (bool progress = true; progress && todo.empty();) {
			progress = false;
			for (LazyGroup* g : lazy_of_input[input_name]) progress |= advance(*g, todo);
		}
		if (todo.empty()) break;
	}
	for (LazyGroup* g : lazy_of_input[input_name]) { // nothing more to make: the streams of the groups can be judged
		g->expanding = false;
		if (cfg.best_videos) for (const std::string& grp : g->groups) if (group_left[grp] == 0) complete_group(grp);
	}
	}
	{ std::error_code ec; fs::remove_all(spool_dir, ec); }
	stop_encoder_processes();
	if (uploader) uploader->finish(); // the live display stays up while the last uploads go out
	ui::stop();

	// Collect in task order so the ranking below does not depend on which run finished first.
	std::vector<Run> runs;
	std::map<std::string, std::vector<std::string>> failed_runs, ignored_runs, pruned_runs, group_json; // by input
	int ignored = 0, pruned = 0;
	std::map<std::string, int> ignored_why; // reason -> how many runs
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
		else if (t.r.pruned) { pruned++; pruned_runs[t.r.input].push_back("{\"label\":" + jq(t.r.label) + ",\"reason\":" + jq(t.r.reason) + "}"); }
		else if (t.r.ignored) { ignored++; ignored_why[t.r.reason]++; ignored_runs[t.r.input].push_back("{\"label\":" + jq(t.r.label) + ",\"reason\":" + jq(t.r.reason) + "}"); }
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
			for (auto& [k, v] : *r.options) { o << (first ? "" : ",") << jq(k) << ":" << jq(v); first = false; }
			o << "},\"width\":" << r.shape.w << ",\"height\":" << r.shape.h << ",\"fps\":" << (r.shape.fps.num ? r.shape.fps.value() : 0.0)
			  << ",\"kbps\":" << r.kbps << ",\"over_target_penalty\":" << r.penalty << ",\"best_frame_share\":" << r.best_share << ",\"winner\":" << (r.best_share >= BEST_FRAME_SHARE ? "true" : "false")
			  << ",\"vmaf\":{\"mean\":" << std::accumulate(sc.begin(), sc.end(), 0.0) / sc.size() << ",\"min\":" << sc.front() << ",\"p5\":" << sc[(size_t)(sc.size() * 0.05)] << "}"
			  << ",\"ceiling\":" << [&] { if (!r.ceiling) return std::string("null"); const FrameStats st = frame_stats(*r.ceiling); std::ostringstream c; c << "{\"mean\":" << st.mean << ",\"min\":" << st.min << ",\"p5\":" << st.p5 << "}"; return c.str(); }()
			  << ",\"vs_ceiling\":" << [&] { const auto d = vs_ceiling(r); if (d.empty()) return std::string("null"); const FrameStats st = frame_stats(d); std::ostringstream c; c << "{\"mean\":" << st.mean << ",\"min\":" << st.min << ",\"p5\":" << st.p5 << "}"; return c.str(); }()
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
			f << "],\"pruned\":[";
			for (size_t i = 0; i < pruned_runs[input].size(); i++) f << (i ? "," : "") << pruned_runs[input][i];
			f << "],\"left_out\":{";
			bool first_option = true;
			for (auto& [option, n] : left_out[input]) { f << (first_option ? "" : ",") << jq(option) << ":" << n; first_option = false; }
			f << "},\"groups\":[";
			for (size_t i = 0; i < group_json[input].size(); i++) f << (i ? "," : "") << group_json[input][i];
			f << "]}\n";
			if (f) printf("%sResults: %s\n", &input == &cfg.inputs.front() ? "\n" : "", file.string().c_str());
			else fprintf(stderr, "cannot write %s\n", file.string().c_str());
		}
	}
	printf("\nPeak memory use: %lld MB\n", ui::peak_memory_mb());
	if (pruned) printf("\n%d runs were not run: an option showed no effect (see \"pruned\" in the result files)\n", pruned);
	{
		std::map<std::string, size_t> by_option;
		size_t left = 0;
		for (auto& [in, options] : left_out) for (auto& [option, n] : options) { by_option[option] += n; left += n; }
		if (left) {
			printf("\n%zu runs were never made: an option showed no effect, its lowest and highest value scored the same on every frame (see \"left_out\" in the result files)\n", left);
			for (auto& [option, n] : by_option) printf("  %s: %zu\n", option.c_str(), n);
		}
	}
	if (ignored) {
		printf("\n%d runs were ignored: they cannot work with the input or the encoder (see the result files)\n", ignored);
		for (auto& [why, n] : ignored_why) printf("  %d: %s\n", n, why.c_str());
	}
	return failed ? 2 : 0;
}
