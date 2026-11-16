// `ves --list-options`: every encoder option, from the tables the backends apply them with (see options.h).
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <utility>
#include <vector>

#include "encoder.h"
#include "encoder_process.h"
#include "options.h"
#include "remote.h"

// `ves --list-options [encoder[/codec]] [gpu] [key=value ...]`: the options of each encoder and codec, as a config writes them, with what they take, and what each
// is when left out: the encoder is opened with the key=value options (preset, tune, ...) and asked what it ended up with.
int list_options_main(int argc, char** argv) {
	static const std::vector<std::pair<std::string, std::vector<std::string>>> all = {
		{"nvenc", {"h264", "hevc", "av1"}}, {"amf", {"h264", "hevc", "av1"}}, {"qsv", {"h264", "hevc", "av1"}},
		{"x264", {"h264"}}, {"x265", {"hevc"}}, {"svtav1", {"av1"}}};
	std::string want_enc, want_codec;
	int gpu = 0;
	Options base;
	set_isolated_encoders(false); // nothing but this list happens here
	for (int i = 2; i < argc; i++) {
		const std::string a = argv[i];
		if (const size_t eq = a.find('='); eq != std::string::npos) { base[a.substr(0, eq)] = a.substr(eq + 1); continue; }
		if (!a.empty() && a.find_first_not_of("0123456789") == std::string::npos) { gpu = atoi(a.c_str()); continue; }
		const size_t slash = a.find('/');
		want_enc = a.substr(0, slash);
		if (slash != std::string::npos) want_codec = a.substr(slash + 1);
	}
	bool found = false, used_amf = false;
	for (auto& [enc, codecs] : all) {
		if (!want_enc.empty() && enc != want_enc) continue;
		for (auto& codec : codecs) {
			if (!want_codec.empty() && codec != want_codec) continue;
			found = true;
			OptionList list = ves_options(enc, codec), own;
			std::string note;
			if (enc == "nvenc") own = nvenc_options(codec);
			else if (enc == "qsv") { own = qsv_options(codec); note = "Values are numbers, or on | off | adaptive and the names shown. Fields as in oneVPL's mfxstructures.h."; }
			else if (enc == "amf") {
				std::string why;
				if (!used_amf) amf_dedicate_process(gpu);
				used_amf = true;
				if (amf_options(codec, gpu, own, why)) note = "From the AMF runtime of AMD GPU " + std::to_string(gpu) + (amf_runtime().empty() ? "" : " (" + amf_runtime() + ")") + "; other GPUs can differ.";
				else note = "AMF's own properties come from the AMD GPU's runtime and could not be listed: " + why + ".";
			} else if (enc == "x264") { own = x264_options(); note = "Every other x264 option works too, by its command line name with _ for - (x264 --fullhelp)."; }
			else if (enc == "x265") { own = x265_options(); note = "Every other x265 option works too, by its command line name with _ for - (x265 --fullhelp)."; }
			else { own = svtav1_options(); note = "Every other SVT-AV1 option works too, by its command line name with _ for - (SvtAv1EncApp --help)."; }
			merge_options(list, own);
			std::string why;
			const auto defaults = encoder_defaults(enc, codec, gpu, base, &why);
			for (auto& o : list) {
				default_of(enc, codec, o.name, defaults, o.current);
				if (o.current.empty() && !defaults.empty() && base.count(o.name)) o.current = base.at(o.name); // asked for: preset, tune
			}
			if (enc == "x264" || enc == "x265") // everything else the library says about itself, by its own names (most of them work as options)
				for (auto& [k, v] : defaults) {
					bool have = false;
					std::string scratch;
					for (auto& o : list) if (o.name == k || (is_common_option(o.name) && default_of(enc, codec, o.name, {{k, v}}, scratch))) { have = true; break; }
					if (!have) list.push_back({k, "", enc + "'s setting", v});
				}
			std::string base_text;
			for (auto& [k, v] : base) base_text += " " + k + "=" + v;
			const std::string defaults_note = defaults.empty() ? "No defaults: " + why + "." : "Defaults (left out) with" + (base_text.empty() ? std::string(" no options") : base_text) +
				", at 1920x1080 8-bit 4:2:0, 30 fps, 6000 kbps" + (is_gpu_encoder(enc) ? ", GPU " + std::to_string(gpu) : std::string()) + ".";
			{ // the ones every encoder has first, in the same order everywhere; the backend's own after them as it lists them
				static const std::vector<std::string> first = {"gpu", "preset", "tune", "profile", "tier", "level", "rate_control", "max_b_frames",
				                                               "lookahead_frames", "max_ref_frames", "b_reference_mode"};
				auto rank = [&](const OptionInfo& o) { return (size_t)(std::find(first.begin(), first.end(), o.name) - first.begin()); };
				std::stable_sort(list.begin(), list.end(), [&](const OptionInfo& a, const OptionInfo& b) { return rank(a) < rank(b); });
			}
			size_t w = 0, vw = 0, dw = 0;
			for (auto& o : list) { w = std::max(w, o.name.size()); vw = std::max(vw, std::min<size_t>(o.values.size(), 40)); dw = std::max(dw, std::min<size_t>(o.current.size(), 12)); }
			printf("%s/%s\n", enc.c_str(), codec.c_str());
			if (dw) printf("  %-*s  %-*s  %s\n", (int)w, "option", (int)dw, "default", "values");
			for (auto& o : list) {
				std::string line = "  " + o.name + std::string(w - o.name.size() + 2, ' ');
				if (dw) line += (o.current.empty() ? std::string("-") : o.current) + std::string(o.current.size() < dw ? dw - std::max<size_t>(o.current.size(), 1) + 2 : 2, ' ');
				line += o.values;
				if (!o.about.empty()) line += std::string(o.values.size() < vw ? vw - o.values.size() + 2 : 2, ' ') + "// " + o.about;
				printf("%s\n", line.c_str());
			}
			printf("  (%s)\n", defaults_note.c_str());
			if (!note.empty()) printf("  (%s)\n", note.c_str());
			printf("\n");
		}
	}
	if (!found) {
		fprintf(stderr, "error: no encoder '%s'%s; the encoders are nvenc, amf, qsv (h264, hevc, av1), x264 (h264), x265 (hevc) and svtav1 (av1)\n", want_enc.c_str(),
		        want_codec.empty() ? "" : (" with codec '" + want_codec + "'").c_str());
		return 1;
	}
	if (used_amf) exit_now(0); // AMD's runtime is not trusted to wind down (see --probe-amf)
	return 0;
}
