// Native SVT-AV1 backend: drives libSvtAv1Enc directly, no libavcodec in between.
// Options are SVT-AV1's own long option names (preset, lookahead, enable-tf, film-grain, ...) applied with
// svt_av1_enc_parse_parameter, so everything the SvtAv1EncApp CLI accepts can be swept.
// Rate control is VBR at the target bitrate unless "rc" says otherwise.
#include "encoder.h"

#include <cstdarg>
#include <cstdio>

#include "formats.h"

extern "C" {
#include <svt-av1/EbSvtAv1.h>
#include <svt-av1/EbSvtAv1Enc.h>
}

namespace {

// The library logs every stage to stderr; keep only warnings and errors.
void log_cb(void*, SvtAv1LogLevel level, const char* tag, const char* fmt, va_list args) {
	if (level > SVT_AV1_LOG_WARN) return;
	fprintf(stderr, "svtav1 [%s] ", tag ? tag : "");
	vfprintf(stderr, fmt, args);
}

class SvtAv1 : public Encoder {
	EbComponentType* h = nullptr;
	bool initialized = false;
	Sink sink;
	Fmt pix;
	int chroma = 420, depth = 8;

	// Hand every finished packet to the sink; returns true once the end-of-stream packet arrived.
	bool collect(bool block, bool& eos, bool& ok) {
		for (;;) {
			EbBufferHeaderType* pkt = nullptr;
			EbErrorType r = svt_av1_enc_get_packet(h, &pkt, block ? 1 : 0);
			if (r == EB_NoErrorEmptyQueue) return true;
			if (r != EB_ErrorNone || !pkt) { fprintf(stderr, "svtav1: get_packet failed (%d)\n", (int)r); ok = false; return false; }
			if (pkt->n_filled_len > 0) {
				stats.packets++; stats.bytes += pkt->n_filled_len;
				if (sink) sink(pkt->p_buffer, pkt->n_filled_len, pkt->pts, pkt->pic_type == EB_AV1_KEY_PICTURE);
			}
			eos = pkt->flags & EB_BUFFERFLAG_EOS;
			svt_av1_enc_release_out_buffer(&pkt);
			if (eos) return true;
		}
	}

public:
	~SvtAv1() override {
		if (h) {
			if (initialized) svt_av1_enc_deinit(h);
			svt_av1_enc_deinit_handle(h);
		}
	}

	bool open(const Params& p, Sink s) override {
		sink = std::move(s);
		svt_av1_set_log_callback(log_cb, nullptr);
		EbSvtAv1EncConfiguration cfg{};
		if (svt_av1_enc_init_handle(&h, &cfg) != EB_ErrorNone) { fprintf(stderr, "svtav1: init_handle failed\n"); return false; }
		cfg.source_width = p.width; cfg.source_height = p.height;
		cfg.frame_rate_numerator = p.framerate.num; cfg.frame_rate_denominator = p.framerate.den;
		const Fmt f = p.pix;
		pix = p.pix; chroma = f.chroma; depth = f.depth;
		cfg.encoder_bit_depth = depth;
		cfg.encoder_color_format = chroma == 444 ? EB_YUV444 : chroma == 422 ? EB_YUV422 : EB_YUV420;
		cfg.profile = chroma == 420 && depth <= 10 ? MAIN_PROFILE : chroma == 444 && depth <= 10 ? HIGH_PROFILE : PROFESSIONAL_PROFILE; // an explicit "profile" option overrides
		cfg.intra_period_length = p.gop - 1; // SVT counts the frames after the key frame
		cfg.rate_control_mode = SVT_AV1_RC_MODE_VBR;
		cfg.target_bit_rate = (uint32_t)p.bitrate;
		for (auto& [k, v] : p.options) {
			if (svt_av1_enc_parse_parameter(&cfg, k.c_str(), v.c_str()) != EB_ErrorNone) { fprintf(stderr, "svtav1: bad option %s=%s\n", k.c_str(), v.c_str()); return false; }
		}
		if (svt_av1_enc_set_parameter(h, &cfg) != EB_ErrorNone) { fprintf(stderr, "svtav1: set_parameter rejected the configuration\n"); return false; }
		if (svt_av1_enc_init(h) != EB_ErrorNone) { fprintf(stderr, "svtav1: init failed\n"); return false; }
		initialized = true;
		return true;
	}

	bool encode(const Picture* fr) override {
		bool eos = false, ok = true;
		EbBufferHeaderType in{};
		in.size = sizeof in;
		in.pic_type = EB_AV1_INVALID_PICTURE;
		if (fr) {
			if (fr->fmt != pix) { fprintf(stderr, "svtav1: input must be %d-bit %d\n", pix.depth, pix.chroma); return false; }
			EbSvtIOFormat io{};
			io.luma = (uint8_t*)fr->data[0]; io.cb = (uint8_t*)fr->data[1]; io.cr = (uint8_t*)fr->data[2];
			// strides count samples, not bytes; 10-bit samples are 16 bits, low bits used
			const int bytes = depth > 8 ? 2 : 1;
			io.y_stride = fr->stride[0] / bytes; io.cb_stride = fr->stride[1] / bytes; io.cr_stride = fr->stride[2] / bytes;
			in.p_buffer = (uint8_t*)&io;
			const long long luma = (long long)fr->width * fr->height;
			in.n_filled_len = in.n_alloc_len = (uint32_t)((chroma == 444 ? luma * 3 : chroma == 422 ? luma * 2 : luma * 3 / 2) * bytes);
			in.pts = fr->pts;
			if (svt_av1_enc_send_picture(h, &in) != EB_ErrorNone) { fprintf(stderr, "svtav1: send_picture failed\n"); return false; }
			collect(false, eos, ok);
			return ok;
		}
		in.flags = EB_BUFFERFLAG_EOS;
		if (svt_av1_enc_send_picture(h, &in) != EB_ErrorNone) { fprintf(stderr, "svtav1: sending end of stream failed\n"); return false; }
		collect(true, eos, ok);
		return ok && eos;
	}
};

} // namespace

std::unique_ptr<Encoder> make_encoder_svtav1() { return std::make_unique<SvtAv1>(); }
