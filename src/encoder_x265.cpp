// Native x265 backend: drives libx265 directly, no libavcodec in between.
// Options are x265's own long option names (bframes, rc-lookahead, ctu, ...) applied with
// param_parse, so everything the x265 CLI accepts can be swept. "preset", "tune" and "profile"
// are handled up front. Rate control is ABR at the target bitrate unless vbv-maxrate/vbv-bufsize are given.
#include "encoder.h"
#include "options.h"

#include <cstdio>

#include "formats.h"

extern "C" {
#include <x265.h>
}

namespace {

class X265 : public Encoder {
	const x265_api* api = nullptr;
	x265_param* par = nullptr;
	x265_encoder* h = nullptr;
	x265_picture *in = nullptr, *out = nullptr;
	Sink sink;
	Fmt pix;
	int csp = X265_CSP_I420, depth = 8;

	void emit(x265_nal* nal, uint32_t n) {
		size_t bytes = 0;
		for (uint32_t i = 0; i < n; i++) bytes += nal[i].sizeBytes;
		stats.packets++; stats.bytes += bytes;
		// NAL payloads are contiguous, so the first one points at the whole access unit.
		if (sink) sink(nal[0].payload, bytes, out->pts, IS_X265_TYPE_I(out->sliceType), out->dts);
	}

public:
	~X265() override {
		if (h) api->encoder_close(h);
		if (in) api->picture_free(in);
		if (out) api->picture_free(out);
		if (par) api->param_free(par);
	}

	bool open(const Params& p, Sink s) override {
		sink = std::move(s);
		const Fmt f = p.pix;
		pix = p.pix; depth = f.depth;
		csp = f.chroma == 444 ? X265_CSP_I444 : f.chroma == 422 ? X265_CSP_I422 : X265_CSP_I420;
		api = x265_api_get(depth);
		if (!api) { fprintf(stderr, "x265: no %d-bit encoder available\n", depth); return false; }
		Options o = p.options;
		auto take = [&](const char* k, std::string& v) {
			auto it = o.find(k);
			if (it == o.end()) return false;
			v = it->second; o.erase(it); return true;
		};
		std::string preset = "medium", tune, profile;
		take("preset", preset); take("tune", tune); take("profile", profile);

		if (!profile.empty()) {
			std::string why;
			translate_named_value("x265", p.codec, "profile", profile, why);
		}

		par = api->param_alloc();
		if (api->param_default_preset(par, preset.c_str(), tune.empty() ? nullptr : tune.c_str()) < 0) {
			fprintf(stderr, "x265: bad preset '%s' or tune '%s'\n", preset.c_str(), tune.c_str());
			return false;
		}
		par->sourceWidth = p.width; par->sourceHeight = p.height;
		par->internalCsp = csp;
		par->fpsNum = p.framerate.num; par->fpsDenom = p.framerate.den;
		par->keyframeMax = p.gop;
		par->bRepeatHeaders = 1; // parameter sets in-band so the stream decodes on its own
		par->rc.rateControlMode = X265_RC_ABR;
		par->rc.bitrate = (int)(p.bitrate / 1000);
		par->logLevel = X265_LOG_WARNING;
		if (color_full(p.color) || color_described(p.color)) {
			par->vui.bEnableVideoSignalTypePresentFlag = 1;
			par->vui.videoFormat = 5; // unspecified
			par->vui.bEnableVideoFullRangeFlag = color_full(p.color);
			if (color_described(p.color)) {
				par->vui.bEnableColorDescriptionPresentFlag = 1;
				par->vui.colorPrimaries = p.color.primaries; par->vui.transferCharacteristics = p.color.trc; par->vui.matrixCoeffs = p.color.space;
			}
		}
		NativeOptions native;
		std::string why;
		if (!resolve_options("x265", p.codec, o, kebab_case, native, why)) { fprintf(stderr, "x265: %s\n", why.c_str()); return false; }
		for (auto& [k, v] : native) {
			if (api->param_parse(par, k.c_str(), v.c_str()) != 0) { fprintf(stderr, "x265: bad option %s=%s\n", k.c_str(), v.c_str()); return false; }
		}
		if (!profile.empty() && api->param_apply_profile(par, profile.c_str()) < 0) { fprintf(stderr, "x265: bad profile '%s'\n", profile.c_str()); return false; }
		if (!(h = api->encoder_open(par))) { fprintf(stderr, "x265: encoder open failed\n"); return false; }
		in = api->picture_alloc(); out = api->picture_alloc();
		api->picture_init(par, in); api->picture_init(par, out);
		return true;
	}

	bool encode(const Picture* fr) override {
		x265_nal* nal;
		uint32_t n = 0;
		if (fr) {
			if (fr->fmt != pix) { fprintf(stderr, "x265: input must be %d-bit %d\n", pix.depth, pix.chroma); return false; }
			in->colorSpace = csp;
			in->bitDepth = depth;
			for (int i = 0; i < 3; i++) { in->planes[i] = (void*)fr->data[i]; in->stride[i] = fr->stride[i]; }
			in->pts = fr->pts;
			int r = api->encoder_encode(h, &nal, &n, in, out);
			if (r < 0) return false;
			if (r > 0) emit(nal, n);
			return true;
		}
		for (;;) { // flush
			int r = api->encoder_encode(h, &nal, &n, nullptr, out);
			if (r < 0) return false;
			if (r == 0) return true;
			emit(nal, n);
		}
	}
};

} // namespace

std::unique_ptr<Encoder> make_encoder_x265() { return std::make_unique<X265>(); }
