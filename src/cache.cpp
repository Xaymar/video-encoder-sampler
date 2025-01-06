#include "cache.h"

#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <sstream>

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
static std::string encoder_environment(const std::string& encoder, const Options& options) {
	auto gpu = [&](unsigned vendor) {
		auto it = options.find("gpu");
		return gpu_description(vendor, it == options.end() ? 0 : atoi(it->second.c_str()));
	};
	if (encoder == "nvenc") return "nvenc " + gpu(VENDOR_NVIDIA);
	if (encoder == "amf") return "amf " + gpu(VENDOR_AMD);
	return encoder_library_version(encoder);
}

std::string cache_key(const std::string& input, const std::string& encoder, const std::string& codec, const Options& options,
                      long long bitrate_kbps, double kf, const Shape& shape, const ScaleOptions& scale, const VmafOptions& vmaf) {
	std::ostringstream k;
	std::error_code ec;
	k << "ves-cache-1\n";
	k << "input=" << fs::path(input).filename().string() << '|' << fs::file_size(input, ec) << '|' << fs::last_write_time(input, ec).time_since_epoch().count() << '\n';
	k << "encoder=" << encoder << '/' << codec << '\n';
	k << "environment=" << encoder_environment(encoder, options) << '\n';
	k << "options=";
	for (auto& [name, value] : options) k << name << '=' << value << ';';
	char buf[128];
	snprintf(buf, sizeof buf, "\nbitrate=%lld kf=%.6g shape=%dx%d@%d/%d\n", bitrate_kbps, kf, shape.w, shape.h, shape.fps.num, shape.fps.den);
	k << buf;
	k << "scaling=" << scale.upscaler << '/' << scale.downscaler << ' ' << scaler_library_version() << '\n';
	// the scorer: library version, model, and the model file's contents when it is a path
	k << "vmaf=" << vmaf_library_version() << ' ' << vmaf.model;
	if (vmaf.model.rfind("path=", 0) == 0) {
		std::ifstream f(vmaf.model.substr(5), std::ios::binary);
		std::stringstream ss; ss << f.rdbuf();
		k << ' ' << hex64(fnv1a(ss.str()));
	}
	k << '\n';
	return k.str();
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

static fs::path cache_file(const std::string& dir, const std::string& key) { return fs::path(dir) / (hex64(fnv1a(key)) + ".json"); }

bool cache_load(const std::string& dir, const std::string& key, CachedRun& out) {
	std::ifstream f(cache_file(dir, key));
	if (!f) return false;
	try {
		std::stringstream ss; ss << f.rdbuf();
		Json j = Json::parse(ss.str());
		auto* k = j.find("key");
		auto* kbps = j.find("kbps");
		auto* scores = j.find("scores");
		if (!k || !kbps || !scores || k->text() != key) return false; // a hash collision or a damaged file counts as a miss
		out.kbps = atof(kbps->text().c_str());
		out.scores.clear();
		for (auto& s : scores->a) out.scores.push_back(atof(s.text().c_str()));
		return !out.scores.empty();
	} catch (const std::exception&) { return false; }
}

void cache_store(const std::string& dir, const std::string& key, const CachedRun& run) {
	std::error_code ec;
	fs::create_directories(dir, ec);
	fs::path file = cache_file(dir, key), tmp = file;
	tmp += ".tmp";
	{
		std::ofstream f(tmp, std::ios::binary);
		char b[32];
		snprintf(b, sizeof b, "%.17g", run.kbps);
		f << "{\"key\":\"" << json_escape(key) << "\",\"kbps\":" << b << ",\"scores\":[";
		for (size_t i = 0; i < run.scores.size(); i++) { snprintf(b, sizeof b, "%.17g", run.scores[i]); f << (i ? "," : "") << b; }
		f << "]}\n";
		if (!f) { fs::remove(tmp, ec); return; }
	}
	fs::rename(tmp, file, ec); // atomic, so an interrupted run never leaves a half-written entry
}
