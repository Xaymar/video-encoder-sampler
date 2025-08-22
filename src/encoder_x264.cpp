// Native x264 backend: drives libx264 directly, no libavcodec in between.
// Options are x264's own long option names (bframes, ref, rc-lookahead, aq-mode, ...) applied with
// x264_param_parse, so everything the x264 CLI accepts can be swept. "preset", "tune" and "profile"
// are handled up front. Rate control is ABR at the target bitrate unless vbv-maxrate/vbv-bufsize are given.
#include "encoder.h"

#include <cstdio>
#include <cstring>
#include <vector>

#include "formats.h"

extern "C" {
#include <x264.h>
}

namespace {

class X264 : public Encoder {
	x264_t* h = nullptr;
	Sink sink;
	Fmt pix;
	int csp = X264_CSP_I420;

	bool emit(x264_nal_t* nal, int n, int bytes, const x264_picture_t& out) {
		if (bytes <= 0) return true;
		stats.packets++; stats.bytes += bytes;
		// NAL payloads are contiguous, so the first one points at the whole access unit.
		if (sink) sink(nal[0].p_payload, (size_t)bytes, out.i_pts, out.b_keyframe != 0, out.i_dts);
		return true;
	}

public:
	~X264() override { if (h) x264_encoder_close(h); }

	bool open(const Params& p, Sink s) override {
		sink = std::move(s);
		Options o = p.options;
		auto take = [&](const char* k) -> const char* {
			auto it = o.find(k);
			if (it == o.end()) return nullptr;
			static thread_local std::string v;
			v = it->second; o.erase(it);
			return v.c_str();
		};
		std::string preset = "medium", tune, profile;
		if (auto* v = take("preset")) preset = v;
		if (auto* v = take("tune")) tune = v;
		if (auto* v = take("profile")) profile = v;

		x264_param_t par;
		if (x264_param_default_preset(&par, preset.c_str(), tune.empty() ? nullptr : tune.c_str()) < 0) {
			fprintf(stderr, "x264: bad preset '%s' or tune '%s'\n", preset.c_str(), tune.c_str());
			return false;
		}
		par.i_width = p.width; par.i_height = p.height;
		const Fmt f = p.pix;
		pix = p.pix;
		csp = (f.chroma == 444 ? X264_CSP_I444 : f.chroma == 422 ? X264_CSP_I422 : X264_CSP_I420) | (f.depth > 8 ? X264_CSP_HIGH_DEPTH : 0);
		par.i_csp = csp;
		par.i_bitdepth = f.depth;
		par.i_fps_num = p.framerate.num; par.i_fps_den = p.framerate.den;
		par.i_timebase_num = p.framerate.den; par.i_timebase_den = p.framerate.num; // pts is the frame index
		par.b_vfr_input = 0;
		par.i_keyint_max = p.gop;
		par.rc.i_rc_method = X264_RC_ABR;
		par.rc.i_bitrate = (int)(p.bitrate / 1000);
		par.i_log_level = X264_LOG_WARNING;
		for (auto& [k, v] : o) {
			if (x264_param_parse(&par, k.c_str(), v.c_str()) < 0) { fprintf(stderr, "x264: bad option %s=%s\n", k.c_str(), v.c_str()); return false; }
		}
		if (!profile.empty() && x264_param_apply_profile(&par, profile.c_str()) < 0) { fprintf(stderr, "x264: bad profile '%s'\n", profile.c_str()); return false; }
		if (!(h = x264_encoder_open(&par))) { fprintf(stderr, "x264: encoder open failed\n"); return false; }
		return true;
	}

	bool encode(const Picture* fr) override {
		x264_picture_t in, out;
		x264_nal_t* nal;
		int n;
		if (fr) {
			if (fr->fmt != pix) { fprintf(stderr, "x264: input must be %d-bit %d\n", pix.depth, pix.chroma); return false; }
			x264_picture_init(&in);
			in.img.i_csp = csp;
			in.img.i_plane = 3;
			for (int i = 0; i < 3; i++) { in.img.plane[i] = (uint8_t*)fr->data[i]; in.img.i_stride[i] = fr->stride[i]; }
			in.i_pts = fr->pts;
			int bytes = x264_encoder_encode(h, &nal, &n, &in, &out);
			return bytes >= 0 && emit(nal, n, bytes, out);
		}
		while (x264_encoder_delayed_frames(h) > 0) { // flush
			int bytes = x264_encoder_encode(h, &nal, &n, nullptr, &out);
			if (bytes < 0 || !emit(nal, n, bytes, out)) return false;
		}
		return true;
	}
};

} // namespace

std::unique_ptr<Encoder> make_encoder_x264() { return std::make_unique<X264>(); }
