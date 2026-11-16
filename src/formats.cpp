#include "formats.h"

#include "encoder.h"

#include <map>
#include <mutex>
#include "encoder_process.h"

extern "C" {
#include <x265.h>
}

bool encoder_supports(const std::string& encoder, const std::string& codec, int chroma, int depth, int gpu) {
	if (!valid(planar_fmt(chroma, depth))) return false;
	if (bool answer; isolated_supports(encoder, codec, chroma, depth, gpu, answer)) return answer;
	if (encoder == "x264") return depth <= 10;
	if (encoder == "nvenc") return nvenc_supports(codec, chroma, depth);
	if (encoder == "x265") return x265_api_get(depth) != nullptr; // the multi-bit-depth build links 8, 10 and 12 bit encoders; 4:2:0/4:2:2/4:4:4 all work
	if (encoder == "svtav1") return chroma == 420 && depth <= 10; // this SVT-AV1 refuses anything but 4:2:0 ("Only support 420 now")
	if (encoder == "amf") return amf_supports(codec, chroma, depth, gpu);
	if (encoder == "qsv") return qsv_supports(codec, chroma, depth);
	return false; // unknown encoder
}

std::string encoder_caps_json(const std::string& encoder, const std::string& codec, int gpu) {
	if (std::string answer; isolated_caps(encoder, codec, gpu, answer)) return answer;
	return encoder == "nvenc" ? nvenc_caps_json(codec, gpu) : encoder == "amf" ? amf_caps_json(codec, gpu) : "null";
}

namespace {

using Settings = std::vector<std::pair<std::string, std::string>>;

std::string lines_of(const std::vector<std::pair<std::string, std::string>>& kv) {
	std::string out;
	for (auto& [k, v] : kv) out += k + "=" + v + "\n";
	return out;
}
Settings from_lines(const std::string& text) {
	Settings out;
	for (size_t at = 0; at < text.size();) {
		size_t end = text.find('\n', at);
		if (end == std::string::npos) end = text.size();
		const size_t eq = text.find('=', at);
		if (eq != std::string::npos && eq < end) out.push_back({text.substr(at, eq - at), text.substr(eq + 1, end - eq - 1)});
		at = end + 1;
	}
	return out;
}

Settings open_and_read(const std::string& encoder, const std::string& codec, int gpu, const Settings& base, std::string& why) {
	auto e = make_encoder(encoder);
	if (!e) { why = "unknown encoder '" + encoder + "'"; return {}; }
	Params p;
	p.width = 1920; p.height = 1080; p.framerate = {30, 1}; p.bitrate = 6000000; p.gop = 60;
	p.codec = codec;
	int chroma = 420, depth = 8;
	for (auto& [k, v] : base) {
		if (k == "chroma") chroma = atoi(v.c_str());
		else if (k == "bitdepth") depth = atoi(v.c_str());
		else if (k != "range") p.options[k] = v;
	}
	p.pix = planar_fmt(chroma, depth);
	if (is_gpu_encoder(encoder)) p.options["gpu"] = std::to_string(gpu);
	if (!e->open(p, nullptr)) { why = encoder + "/" + codec + " could not be opened with " + (base.empty() ? std::string("no options") : lines_of(base)); return {}; }
	Settings s = e->settings();
	if (s.empty()) why = encoder + " does not tell its settings";
	if (encoder == "amf") e.release(); // AMD's runtime is not trusted to wind down (see amf_probe)
	return s;
}

} // namespace

std::string encoder_defaults_text(const std::string& encoder, const std::string& codec, int gpu, const std::string& base) {
	std::string why;
	const Settings s = open_and_read(encoder, codec, gpu, from_lines(base), why);
	return s.empty() ? "!" + why : lines_of(s);
}

std::vector<std::pair<std::string, std::string>> encoder_defaults(const std::string& encoder, const std::string& codec, int gpu, const Options& base, std::string* why) {
	static std::mutex m;
	static std::map<std::string, std::pair<Settings, std::string>> known;
	const std::string text = lines_of(Settings(base.begin(), base.end()));
	const std::string key = encoder + "|" + codec + "|" + std::to_string(gpu) + "|" + text;
	{
		std::lock_guard lock(m);
		if (auto it = known.find(key); it != known.end()) { if (why) *why = it->second.second; return it->second.first; }
	}
	Settings s;
	std::string reason;
	if (std::string answer; isolated_defaults(encoder, codec, gpu, text, answer)) {
		if (!answer.empty() && answer[0] == '!') reason = answer.substr(1);
		else s = from_lines(answer);
	} else s = open_and_read(encoder, codec, gpu, Settings(base.begin(), base.end()), reason);
	std::lock_guard lock(m);
	known[key] = {s, reason};
	if (why) *why = reason;
	return s;
}
