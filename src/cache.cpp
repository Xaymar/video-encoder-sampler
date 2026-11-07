#include "cache.h"

#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <map>
#include <mutex>
#include <sstream>

#include "download.h"
#include "gpu.h"
#include "json.h"

namespace fs = std::filesystem;

static uint64_t fnv1a(const std::string& s) {
	uint64_t h = 1469598103934665603ull;
	for (unsigned char c : s) { h ^= c; h *= 1099511628211ull; }
	return h;
}

static std::string hex64(uint64_t v) {
	char b[17];
	snprintf(b, sizeof b, "%016llx", (unsigned long long)v);
	return b;
}

// Version of whatever does the encoding for this encoder name.
static std::string encoder_environment(const std::string& encoder, const Options& options, const std::string& gpu_override) {
	if (!gpu_override.empty() && (encoder == "nvenc" || encoder == "amf" || encoder == "qsv")) return encoder + " " + gpu_override; // the GPU of a worker
	auto gpu = [&](unsigned vendor) {
		auto it = options.find("gpu");
		return gpu_description(vendor, it == options.end() ? 0 : atoi(it->second.c_str()));
	};
	if (encoder == "nvenc") return "nvenc " + gpu(VENDOR_NVIDIA);
	if (encoder == "amf") return "amf " + gpu(VENDOR_AMD);
	if (encoder == "qsv") return "qsv " + gpu(VENDOR_INTEL);
	return encoder_library_version(encoder);
}

// The key in parts, so that what stays the same from run to run is built once: the input, the encoder with its options, the bitrate, key frame interval and
// geometry, and the settings of scaler, decoder and scorer. cache_key() is these one after the other.
std::string key_input(const std::string& input) {
	std::ostringstream k;
	std::error_code ec;
	k << "ves-cache-1\n";
	// A URL input is identified by its address, so that results can be found without downloading the file; a file by its name, size and time.
	if (is_url(input)) k << "input=" << url_file_name(input) << "|url=" << input << '\n';
	else k << "input=" << fs::path(input).filename().string() << '|' << fs::file_size(input, ec) << '|' << fs::last_write_time(input, ec).time_since_epoch().count() << '\n';
	return k.str();
}

std::string key_encoder(const std::string& encoder, const std::string& codec, const Options& options, const std::string& gpu_override) {
	std::ostringstream k;
	k << "encoder=" << encoder << '/' << codec << '\n';
	k << "environment=" << encoder_environment(encoder, options, gpu_override) << '\n';
	k << "options=";
	for (auto& [name, value] : options) k << name << '=' << value << ';';
	return k.str();
}

std::string key_run(long long bitrate_kbps, double kf, const Shape& shape) {
	char buf[128];
	snprintf(buf, sizeof buf, "\nbitrate=%lld kf=%.6g shape=%dx%d@%d/%d\n", bitrate_kbps, kf, shape.w, shape.h, shape.fps.num, shape.fps.den);
	return buf;
}

// The key's text for a model: an absolute path (the user's own folders) is reduced to the file name, since the key goes into the result files; the
// content hash (model_id()) is what identifies a model file. Relative paths are kept as they are, which keeps the keys of earlier results valid.
static std::string model_key_text(const std::string& model) {
	if (model.rfind("path=", 0) == 0 && fs::path(model.substr(5)).is_absolute()) return "path=" + fs::path(model.substr(5)).filename().string();
	return model;
}

std::string model_id(const std::string& model) {
	if (model.rfind("path=", 0) != 0) return "";
	static std::mutex m;
	static std::map<std::string, std::string> model_hash; // the model file does not change during a run
	std::lock_guard lock(m);
	auto it = model_hash.find(model);
	if (it == model_hash.end()) {
		std::ifstream f(model.substr(5), std::ios::binary);
		std::stringstream ss; ss << f.rdbuf();
		it = model_hash.emplace(model, hex64(fnv1a(ss.str()))).first;
	}
	return it->second;
}

std::string key_settings(const ScaleOptions& scale, const VmafOptions& vmaf) {
	std::ostringstream k;
	k << "scaling=" << scale.upscaler << '/' << scale.downscaler << ' ' << scale.device << ' ' << scaler_library_version() << '\n';
	k << "decoder=" << decoder_library_version() << '\n';
	// the scorer: library version, model, and the model file's contents when it is a path
	k << "vmaf=" << vmaf_library_version() << ' ' << model_key_text(vmaf.model);
	if (const std::string id = model_id(vmaf.model); !id.empty()) k << ' ' << id;
	// The GPU scorer's scores differ from libvmaf's in the fourth decimal; "cpu" adds nothing, which keeps the keys of earlier results valid.
	if (vmaf.device != "cpu") k << " scorer=" << vmaf.device << "-1";
	k << '\n';
	// the models scored besides it (only when there are any, which keeps the keys of earlier results valid)
	const std::vector<std::string> all = vmaf_models(vmaf);
	if (all.size() > 1) {
		k << "vmaf_extra=";
		for (size_t i = 1; i < all.size(); i++) {
			k << (i > 1 ? ";" : "") << model_key_text(all[i]);
			if (const std::string id = model_id(all[i]); !id.empty()) k << ' ' << id;
		}
		k << '\n';
	}
	return k.str();
}

std::string cache_key(const std::string& input, const std::string& encoder, const std::string& codec, const Options& options,
                      long long bitrate_kbps, double kf, const Shape& shape, const ScaleOptions& scale, const VmafOptions& vmaf) {
	return key_input(input) + key_encoder(encoder, codec, options) + key_run(bitrate_kbps, kf, shape) + key_settings(scale, vmaf);
}

static std::string json_escape(const std::string& s) {
	std::string o;
	for (unsigned char c : s) {
		if (c == '"' || c == '\\') { o += '\\'; o += (char)c; }
		else if (c == '\n') o += "\\n";
		else if (c < 0x20) o += ' ';
		else o += (char)c;
	}
	return o;
}

std::string key_hash(const std::string& key) { return hex64(fnv1a(key)); }

std::string result_name(const std::string& label, const std::string& key) {
	std::string name;
	for (unsigned char c : label) name += (isalnum(c) || c == '-' || c == '_' || c == '.' || c == '=') ? (char)c : '_';
	if (name.size() > 100) name.resize(100);
	return name + "_" + key_hash(key).substr(0, 8);
}

bool result_load(const std::string& file, const std::string& key, CachedRun& out) {
	std::ifstream f(file);
	if (!f) return false;
	try {
		std::stringstream ss; ss << f.rdbuf();
		Json j = Json::parse(ss.str());
		auto* k = j.find("key");
		auto* kbps = j.find("kbps");
		auto* vmaf = j.find("vmaf");
		auto* scores = vmaf ? vmaf->find("scores") : j.find("scores");
		if (!k || !kbps || !scores || k->text() != key) return false; // another configuration, a hash collision or a damaged file counts as a miss
		out.kbps = atof(kbps->text().c_str());
		auto* sha = j.find("video_sha256");
		out.video_sha256 = sha && sha->type == Json::Str ? sha->text() : std::string();
		auto* wd = j.find("width");
		out.width = wd && wd->type == Json::Num ? atoi(wd->s.c_str()) : 0;
		auto* ht = j.find("height");
		out.height = ht && ht->type == Json::Num ? atoi(ht->s.c_str()) : 0;
		auto* fp = j.find("fps");
		out.fps = fp && fp->type == Json::Num ? atof(fp->s.c_str()) : 0.0;
		auto* in = j.find("input");
		out.input = in && in->type == Json::Str ? in->text() : std::string();
		auto* ref = j.find("reference_sha256");
		out.reference_sha256 = ref && ref->type == Json::Str ? ref->text() : std::string();
		auto list = [](const Json* a, std::vector<double>& v) { v.clear(); if (a && a->type == Json::Arr) for (auto& s : a->a) v.push_back(atof(s.text().c_str())); };
		list(scores, out.scores);
		list(vmaf ? vmaf->find("ceiling") : nullptr, out.ceiling);
		if (out.ceiling.size() != out.scores.size()) out.ceiling.clear();
		out.detail = VmafDetail{};
		if (vmaf) {
			list(vmaf->find("chroma_u"), out.detail.chroma_u);
			list(vmaf->find("chroma_v"), out.detail.chroma_v);
			if (const Json* ex = vmaf->find("extra_models"); ex && ex->type == Json::Arr)
				for (auto& e : ex->a) {
					ModelScores m;
					list(e.find("scores"), m.scores); list(e.find("chroma_u"), m.chroma_u); list(e.find("chroma_v"), m.chroma_v); list(e.find("ceiling"), m.ceiling);
					if (m.ceiling.size() != m.scores.size()) m.ceiling.clear();
					out.detail.extra.push_back(std::move(m));
				}
		}
		return !out.scores.empty();
	} catch (const std::exception&) { return false; }
}

void result_store(const std::string& file, const std::string& key, const CachedRun& run, const std::string& details, const VmafOptions& vo) {
	std::error_code ec;
	fs::create_directories(fs::path(file).parent_path(), ec);
	fs::path tmp = file;
	tmp += ".tmp";
	{
		std::ofstream f(tmp, std::ios::binary);
		char b[32];
		snprintf(b, sizeof b, "%.17g", run.kbps);
		f << "{\"key\":\"" << json_escape(key) << "\",\"kbps\":" << b;
		if (!details.empty()) f << "," << details;
		auto list = [&](const char* name, const std::vector<double>& v, const char* fmt) {
			f << ",\"" << name << "\":[";
			for (size_t i = 0; i < v.size(); i++) { snprintf(b, sizeof b, fmt, v[i]); f << (i ? "," : "") << b; }
			f << "]";
		};
		auto model = [&](const std::string& spec, const std::vector<double>& scores, const std::vector<double>& u, const std::vector<double>& v) {
			f << "\"model\":\"" << json_escape(vmaf_model_display_name(spec)) << "\",\"model_id\":\"" << json_escape(model_id(spec)) << "\",\"model_generation\":\""
			  << json_escape(vmaf_model_generation(spec)) << "\"";
			list("scores", scores, "%.17g");
			if (!u.empty()) { list("chroma_u", u, "%.9g"); list("chroma_v", v, "%.9g"); }
		};
		f << ",\"vmaf\":{";
		model(vo.model, run.scores, run.detail.chroma_u, run.detail.chroma_v);
		if (!run.ceiling.empty() && run.ceiling.size() == run.scores.size()) list("ceiling", run.ceiling, "%.17g"); // what the frames could score at all at this size, frame rate and format
		const std::vector<std::string> all = vmaf_models(vo);
		if (all.size() > 1 && run.detail.extra.size() == all.size() - 1) {
			f << ",\"extra_models\":[";
			for (size_t i = 1; i < all.size(); i++) {
				const ModelScores& e = run.detail.extra[i - 1];
				f << (i > 1 ? "," : "") << "{";
				model(all[i], e.scores, e.chroma_u, e.chroma_v);
				if (!e.ceiling.empty() && e.ceiling.size() == e.scores.size()) list("ceiling", e.ceiling, "%.17g");
				f << "}";
			}
			f << "]";
		}
		f << "}}\n";
		if (!f) { fprintf(stderr, "cannot write the result %s (is the path too long?)\n", file.c_str()); fs::remove(tmp, ec); return; }
	}
	fs::rename(tmp, file, ec); // atomic, so an interrupted run never leaves a half-written file
}

std::string ceiling_key(const std::string& input, const std::string& feed_key, const Fmt& score_fmt, const ScaleOptions& scale, const VmafOptions& vmaf) {
	std::ostringstream k;
	VmafOptions first = vmaf; // one model's ceiling: each model's is kept on its own, so adding or dropping a model keeps the others
	first.extra_models.clear();
	k << key_input(input) << "ceiling-1 feed=" << feed_key << " score=" << score_fmt.chroma << '/' << score_fmt.depth << '\n' << key_settings(scale, first);
	return k.str();
}

bool ceiling_load(const std::string& file, const std::string& key, std::vector<double>& scores) {
	std::ifstream f(file);
	if (!f) return false;
	try {
		std::stringstream ss; ss << f.rdbuf();
		Json j = Json::parse(ss.str());
		auto* k = j.find("key");
		auto* s = j.find("scores");
		if (!k || !s || k->text() != key || s->type != Json::Arr) return false;
		scores.clear();
		for (auto& v : s->a) scores.push_back(atof(v.text().c_str()));
		return !scores.empty();
	} catch (const std::exception&) { return false; }
}

void ceiling_store(const std::string& file, const std::string& key, const std::vector<double>& scores) {
	std::error_code ec;
	fs::create_directories(fs::path(file).parent_path(), ec);
	fs::path tmp = file;
	tmp += ".tmp";
	{
		std::ofstream f(tmp, std::ios::binary);
		char b[32];
		f << "{\"key\":\"" << json_escape(key) << "\",\"scores\":[";
		for (size_t i = 0; i < scores.size(); i++) { snprintf(b, sizeof b, "%.17g", scores[i]); f << (i ? "," : "") << b; }
		f << "]}\n";
		if (!f) { fs::remove(tmp, ec); return; }
	}
	fs::rename(tmp, file, ec);
}
