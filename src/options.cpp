#include "options.h"

#include <algorithm>
#include <cctype>
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

bool resolve_options(const std::string& encoder, const std::string& codec, const Options& options,
                     const std::function<std::string(const std::string&)>& native_name, NativeOptions& out, std::string& error) {
	for (auto& [k, v] : options) {
		if (is_common_option(k)) { if (!translate(encoder, codec, k, v, out, error)) return false; }
		else out.push_back({native_name(k), v});
	}
	return true;
}
