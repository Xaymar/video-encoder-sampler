#include "options.h"

#include <algorithm>
#include <cctype>
#include <cstdlib>
#include <map>
#include <cstdio>
#include <set>

std::string snake_case(const std::string& s) {
	std::string out;
	for (size_t i = 0; i < s.size(); i++) {
		const unsigned char c = (unsigned char)s[i];
		if (c == '-' || c == ' ') { out += '_'; continue; }
		if (isupper(c)) {
			const bool after_lower = i > 0 && (islower((unsigned char)s[i - 1]) || isdigit((unsigned char)s[i - 1]));
			const bool acronym_end = i > 0 && isupper((unsigned char)s[i - 1]) && i + 1 < s.size() && islower((unsigned char)s[i + 1]);
			if (after_lower || acronym_end) out += '_';
		}
		out += (char)tolower(c);
	}
	return out;
}

std::string kebab_case(const std::string& s) {
	std::string out = s;
	std::replace(out.begin(), out.end(), '_', '-');
	return out;
}

namespace {

// The common options (see options.h), with the values tried to find out whether an encoder has one: each that translates is listed.
struct Common {
	const char* name;
	const char* values; // shown as it is; nullptr lists the probe values that work
	const char* about;
	std::vector<const char*> probe;
};
const std::vector<Common>& commons() {
	static const std::vector<Common> t = {
		{"max_b_frames", "0..", "most B-frames in a row (0 = none)", {"1"}},
		{"lookahead_frames", "0..", "frames of lookahead for rate control (0 = off)", {"8"}},
		{"max_ref_frames", "1..", "most reference frames", {"2"}},
		{"b_reference_mode", nullptr, "B-frames as references", {"none", "middle", "each"}},
		{"rate_control", nullptr, "rate control mode at the target bitrate", {"cbr", "vbr"}},
	};
	return t;
}

} // namespace

bool is_common_option(const std::string& key) {
	for (auto& c : commons()) if (key == c.name) return true;
	return false;
}

namespace {

// One common option for one encoder: the native options it stands for, or an error.
bool translate(const std::string& enc, const std::string& codec, const std::string& key, const std::string& value, NativeOptions& out, std::string& error) {
	auto no = [&] { error = "option '" + key + "' is not available for " + enc + (enc == "amf" || enc == "qsv" ? "/" + codec : std::string()); return false; };
	auto bad = [&] { error = "bad value '" + value + "' for '" + key + "' (" + enc + ")"; return false; };
	const int n = atoi(value.c_str());
	const bool cpu_x = enc == "x264" || enc == "x265";

	if (key == "max_b_frames") {
		if (cpu_x) out.push_back({"bframes", value});
		else if (enc == "qsv") out.push_back({"gop_ref_dist", std::to_string(n + 1)});
		else if (enc == "amf" && codec == "h264") out.push_back({"b_pictures_pattern", value});
		else return no();
	} else if (key == "lookahead_frames") {
		if (cpu_x) out.push_back({"rc-lookahead", value});
		else if (enc == "svtav1") out.push_back({"lookahead", value});
		else if (enc == "qsv") out.push_back({"look_ahead_depth", value});
		else return no();
	} else if (key == "max_ref_frames") {
		if (cpu_x) out.push_back({"ref", value});
		else if (enc == "qsv") out.push_back({"num_ref_frame", value});
		else if (enc == "amf") out.push_back({"max_num_ref_frames", value});
		else return no();
	} else if (key == "b_reference_mode") {
		if (enc == "x264") {
			if (value == "none") out.push_back({"b-pyramid", "none"});
			else if (value == "middle") out.push_back({"b-pyramid", "normal"});
			else return value == "each" ? no() : bad();
		} else if (enc == "x265") {
			if (value != "none" && value != "middle") return value == "each" ? no() : bad();
			out.push_back({"b-pyramid", value == "none" ? "0" : "1"});
		} else if (enc == "qsv") {
			if (value == "none") out.push_back({"b_ref_type", "1"});
			else if (value == "middle") out.push_back({"b_ref_type", "2"});
			else return value == "each" ? no() : bad();
		} else if (enc == "amf" && codec == "h264") {
			if (value != "none" && value != "middle" && value != "each") return bad();
			out.push_back({"b_reference_enable", value == "none" ? "0" : "1"});
		} else return no();
	} else if (key == "rate_control") {
		if (value != "cbr" && value != "vbr") return bad();
		if (enc == "svtav1") { // CBR needs the low-delay prediction structure and VBR the random-access one
			out.push_back({"rc", value == "cbr" ? "2" : "1"});
			out.push_back({"pred-struct", value == "cbr" ? "1" : "2"});
		} else if (enc == "qsv") out.push_back({"rate_control_method", value});
		else return no();
	}
	return true;
}

} // namespace

namespace {

using Names = std::map<std::string, std::string>;

// Values by "encoder/codec". An encoder that is missing from a profile table for a codec does not offer that profile.
const std::map<std::string, Names>& profiles() {
	static const std::map<std::string, Names> t = {
		{"x264/h264", {{"baseline", "baseline"}, {"main", "main"}, {"high", "high"}, {"high10", "high10"}, {"high422", "high422"}, {"high444", "high444"}}},
		{"amf/h264", {{"baseline", "66"}, {"constrained_baseline", "256"}, {"main", "77"}, {"high", "100"}, {"constrained_high", "257"}}},
		{"qsv/h264", {{"baseline", "66"}, {"constrained_baseline", "578"}, {"main", "77"}, {"high", "100"}, {"constrained_high", "12388"}, {"high10", "110"}, {"high422", "122"}}},
		{"amf/hevc", {{"main", "1"}, {"main10", "2"}}},
		{"qsv/hevc", {{"main", "1"}, {"main10", "2"}, {"rext", "4"}}},
		{"svtav1/av1", {{"main", "0"}, {"high", "1"}, {"professional", "2"}}},
		{"amf/av1", {{"main", "1"}}},
		{"qsv/av1", {{"main", "1"}, {"high", "2"}, {"professional", "3"}}},
	};
	return t;
}

// "tier" is main or high everywhere; AMF takes 0 and 1.
const std::map<std::string, Names>& tiers() {
	static const std::map<std::string, Names> t = {{"amf/hevc", {{"main", "0"}, {"high", "1"}}}};
	return t;
}

const std::map<std::string, Names>& presets() {
	static const std::map<std::string, Names> t = {
		{"amf/h264", {{"balanced", "0"}, {"speed", "1"}, {"quality", "2"}, {"high_quality", "3"}}},
		{"amf/hevc", {{"quality", "0"}, {"balanced", "5"}, {"speed", "10"}, {"high_quality", "15"}}},
		{"amf/av1", {{"high_quality", "0"}, {"quality", "30"}, {"balanced", "70"}, {"speed", "100"}}},
	};
	return t;
}

// The levels of H.264 and HEVC, as 10 x level.
const std::set<int>& levels(const std::string& codec) {
	static const std::set<int> h264 = {10, 11, 12, 13, 20, 21, 22, 30, 31, 32, 40, 41, 42, 50, 51, 52, 60, 61, 62};
	static const std::set<int> hevc = {10, 20, 21, 30, 31, 40, 41, 50, 51, 52, 60, 61, 62};
	return codec == "hevc" ? hevc : h264;
}

// Levels are written as the standards write them ("4.1"). The libraries number them differently: x264 and x265 read the name, the others take
// 10 x level (30 x level for HEVC in NVENC and AMF), AV1 in NVENC and AMF an index counted from 2.0.
bool translate_level(const std::string& enc, const std::string& codec, std::string& value, std::string& error) {
	if (!value.empty() && value.find_first_not_of("0123456789") == std::string::npos) return true; // the encoder's own number
	const bool av1 = codec == "av1";
	if (value == "auto") {
		if (enc == "x264" || enc == "x265" || enc == "amf") value.clear();      // their default already is automatic
		else value = enc == "nvenc" && av1 ? "24" : "0";                        // NVENC's AV1 auto-select value follows level 7.3
		return true;
	}
	auto bad = [&] { error = "level '" + value + "' is not available for " + enc + "/" + codec; return false; };
	if (value == "1b") { // H.264 only
		if (codec != "h264") return bad();
		if (enc == "x264") return true;
		if (enc == "svtav1" || enc == "amf") return bad(); // neither has a level 1b
		value = "9";
		return true;
	}
	int major = 0, minor = 0;
	char extra = 0;
	if (sscanf(value.c_str(), "%d.%d%c", &major, &minor, &extra) < 2 && (sscanf(value.c_str(), "%d%c", &major, &extra) != 1)) return bad();
	if (extra) return bad();
	if (value.find('.') == std::string::npos) minor = 0;
	const int n = major * 10 + minor;
	if (av1) {
		if (major < 2 || major > 7 || minor < 0 || minor > 3) return bad();
	} else if (!levels(codec).count(n)) return bad();
	if (enc == "x264" || enc == "x265") { value = std::to_string(major) + "." + std::to_string(minor); if (minor == 0) value = std::to_string(major); return true; }
	if (av1 && (enc == "nvenc" || enc == "amf")) value = std::to_string((major - 2) * 4 + minor);
	else if (codec == "hevc" && (enc == "nvenc" || enc == "amf")) value = std::to_string(n * 3);
	else value = std::to_string(n);
	return true;
}

} // namespace

bool translate_named_value(const std::string& enc, const std::string& codec, const std::string& key, std::string& value, std::string& error) {
	if (key == "level") return translate_level(enc, codec, value, error);
	if (key == "profile" && enc == "x265") { value = kebab_case(value); return true; } // main444_10 -> main444-10; x265 knows its own list
	if (key != "preset" && key != "profile" && key != "tier") return true;
	const auto& table = key == "preset" ? presets() : key == "tier" ? tiers() : profiles();
	auto t = table.find(enc + "/" + codec);
	if (t == table.end()) return true; // this encoder takes its own spelling (or does not use the standard names at all)
	auto v = t->second.find(value);
	if (v != t->second.end()) { value = v->second; return true; }
	// a number is the encoder's own value, anything else is a name this encoder does not have
	if (!value.empty() && value.find_first_not_of("0123456789") == std::string::npos) return true;
	if (enc == "qsv" && key == "preset") return true; // QuickSync reads best_quality / balanced / best_speed itself
	error = key + " '" + value + "' is not available for " + enc + "/" + codec;
	return false;
}

bool resolve_options(const std::string& encoder, const std::string& codec, const Options& options,
                     const std::function<std::string(const std::string&)>& native_name, NativeOptions& out, std::string& error) {
	for (auto& [k, v] : options) {
		if (is_common_option(k)) { if (!translate(encoder, codec, k, v, out, error)) return false; }
		else if (k == "tier" && encoder == "x265") { // x265 cannot force the high tier: "high-tier" only allows it, and it is used when the bitrate needs it
			if (v == "high") { error = "x265 cannot force tier 'high': it uses it by itself when the bitrate needs it (tier=auto), or never with tier=main"; return false; }
			if (v != "main" && v != "auto") { error = "tier '" + v + "' is not available for x265/" + codec; return false; }
			out.push_back({"high-tier", v == "auto" ? "1" : "0"});
		} else {
			std::string value = v;
			if (!translate_named_value(encoder, codec, k, value, error)) return false;
			if (value.empty()) continue; // "auto" where the encoder's default already is
			out.push_back({encoder == "x265" && k == "level" ? "level-idc" : native_name(k), value});
		}
	}
	return true;
}

namespace {

std::string names_of(const std::map<std::string, Names>& table, const std::string& at) {
	auto t = table.find(at);
	if (t == table.end()) return {};
	std::string out;
	for (auto& [name, native] : t->second) out += (out.empty() ? "" : " | ") + name;
	return out;
}

} // namespace

OptionList ves_options(const std::string& enc, const std::string& codec) {
	OptionList out;
	const std::string at = enc + "/" + codec;
	if (is_gpu_encoder(enc)) out.push_back({"gpu", "n | all", "which GPU of this vendor encodes (counted from 0); all sweeps every one"});
	if (enc != "nvenc") { // NVENC reads the common options itself (its own list has them)
		for (auto& c : commons()) {
			std::string works;
			for (const char* v : c.probe) {
				NativeOptions native;
				std::string error;
				if (translate(enc, codec, c.name, v, native, error)) works += (works.empty() ? "" : " | ") + std::string(v);
			}
			if (!works.empty()) out.push_back({c.name, c.values ? c.values : works, c.about});
		}
	}
	if (auto v = names_of(presets(), at); !v.empty()) out.push_back({"preset", v, "speed against quality"});
	if (auto v = names_of(profiles(), at); !v.empty()) out.push_back({"profile", v, "left out, the encoder picks one that fits the chroma layout and bit depth"});
	if (enc == "x265") out.push_back({"tier", "main | auto", "auto lets x265 use the high tier when the bitrate needs it; it cannot force it"});
	else if (auto v = names_of(tiers(), at); !v.empty()) out.push_back({"tier", v, ""});
	std::string lv = "auto";
	if (codec == "av1") lv += " | 2.0 .. 7.3 (x.0 to x.3)";
	else {
		if (codec == "h264" && enc != "amf" && enc != "svtav1") lv += " | 1b";
		for (int n : levels(codec)) lv += " | " + std::to_string(n / 10) + (n % 10 ? "." + std::to_string(n % 10) : "");
	}
	out.push_back({"level", lv, "as the standards write it; auto leaves it to the encoder"});
	return out;
}

void merge_options(OptionList& list, const OptionList& more) {
	for (auto& o : more) {
		bool have = false;
		for (auto& l : list) if (l.name == o.name) { have = true; if (l.about.empty()) l.about = o.about; break; }
		if (!have) list.push_back(o);
	}
}

std::vector<std::pair<std::string, std::string>> settings_from_sei(const uint8_t* data, size_t size) {
	std::vector<std::pair<std::string, std::string>> out;
	const std::string all((const char*)data, size);
	size_t at = all.find("options: ");
	if (at == std::string::npos) return out;
	at += 9;
	size_t end = at;
	while (end < all.size() && (unsigned char)all[end] >= 32 && (unsigned char)all[end] < 127) end++;
	const std::string text = all.substr(at, end - at);
	for (size_t i = 0; i < text.size();) {
		size_t j = text.find(' ', i);
		if (j == std::string::npos) j = text.size();
		const std::string tok = text.substr(i, j - i);
		i = j + 1;
		if (tok.empty()) continue;
		const size_t eq = tok.find('=');
		if (eq != std::string::npos) out.push_back({snake_case(tok.substr(0, eq)), tok.substr(eq + 1)});
		else if (tok.compare(0, 3, "no-") == 0) out.push_back({snake_case(tok.substr(3)), "0"});
		else out.push_back({snake_case(tok), "1"});
	}
	return out;
}

namespace {

const std::string* lookup(const std::vector<std::pair<std::string, std::string>>& d, const std::string& key) {
	for (auto& [k, v] : d) if (k == key) return &v;
	return nullptr;
}

bool same_value(const std::string& a, const std::string& b) {
	auto norm = [](const std::string& s) { return s == "true" || s == "on" ? std::string("1") : s == "false" || s == "off" ? std::string("0") : s; };
	const std::string x = norm(a), y = norm(b);
	if (x == y) return true;
	char *ex = nullptr, *ey = nullptr;
	const double dx = strtod(x.c_str(), &ex), dy = strtod(y.c_str(), &ey);
	return !x.empty() && !y.empty() && !*ex && !*ey && dx == dy;
}

} // namespace

bool default_of(const std::string& enc, const std::string& codec, const std::string& key, const std::vector<std::pair<std::string, std::string>>& d, std::string& value) {
	if (const std::string* v = lookup(d, key)) { value = *v; return true; }
	if ((key != "max_b_frames" && key != "lookahead_frames" && key != "max_ref_frames") || enc == "nvenc") return false;
	// the library's own name for it (the numbers; the named values are spelled differently by the library): what the common option translates to, if it is one option
	NativeOptions native;
	std::string error;
	const char* probe = key == "b_reference_mode" ? "none" : key == "rate_control" ? "cbr" : "1";
	if (!translate(enc, codec, key, probe, native, error) || native.size() != 1) return false;
	const std::string* v = lookup(d, snake_case(native[0].first));
	if (!v) return false;
	value = *v;
	return true;
}

bool is_default(const std::string& enc, const std::string& codec, const std::string& key, const std::string& value, const std::vector<std::pair<std::string, std::string>>& d) {
	if (const std::string* v = lookup(d, key)) return same_value(value, *v);
	if (!is_common_option(key) || enc == "nvenc") return false;
	NativeOptions native;
	std::string error;
	if (!translate(enc, codec, key, value, native, error) || native.size() != 1) return false;
	const std::string* v = lookup(d, snake_case(native[0].first));
	return v && same_value(native[0].second, *v);
}
