#include "options.h"

#include <algorithm>
#include <cctype>
#include <map>
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

bool is_common_option(const std::string& key) {
	static const std::set<std::string> common = {"max_b_frames", "lookahead_frames", "max_ref_frames", "b_reference_mode", "rate_control"};
	return common.count(key) > 0;
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

} // namespace

bool translate_named_value(const std::string& enc, const std::string& codec, const std::string& key, std::string& value, std::string& error) {
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
		else if (k == "tier" && encoder == "x265") { // x265 has a switch for the high tier instead of a tier option
			if (v != "main" && v != "high") { error = "tier '" + v + "' is not available for x265/" + codec; return false; }
			out.push_back({"high-tier", v == "high" ? "1" : "0"});
		} else {
			std::string value = v;
			if (!translate_named_value(encoder, codec, k, value, error)) return false;
			out.push_back({native_name(k), value});
		}
	}
	return true;
}
