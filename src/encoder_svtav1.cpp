// Native SVT-AV1 backend: drives libSvtAv1Enc directly, no libavcodec in between.
// Options are SVT-AV1's own long option names (preset, lookahead, enable-tf, film-grain, ...) applied with
// svt_av1_enc_parse_parameter, so everything the SvtAv1EncApp CLI accepts can be swept.
// Rate control is CBR at the target bitrate unless "rc" says otherwise.
#include "encoder.h"

#include <atomic>
#include <chrono>
#include <cstdarg>
#include <cstdio>
#include <thread>

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

	// Packets are collected on a thread of their own: in low-delay mode get_packet blocks until a packet is ready, and
	// the encoder may want more pictures before it produces one, so collecting from the thread that sends would deadlock.
	std::thread rx;
	std::atomic<bool> sent_all{false}, rx_ok{true}, got_eos{false};
	bool eos_sent = false;

	void receive() {
		for (;;) {
			EbBufferHeaderType* pkt = nullptr;
			EbErrorType r = svt_av1_enc_get_packet(h, &pkt, sent_all ? 1 : 0);
			if (r == EB_NoErrorEmptyQueue) { std::this_thread::sleep_for(std::chrono::milliseconds(1)); continue; }
			if (r != EB_ErrorNone || !pkt) { fprintf(stderr, "svtav1: get_packet failed (%d)\n", (int)r); rx_ok = false; return; }
			if (pkt->n_filled_len > 0) {
				stats.packets++; stats.bytes += pkt->n_filled_len;
				if (sink) sink(pkt->p_buffer, pkt->n_filled_len, pkt->pts, pkt->pic_type == EB_AV1_KEY_PICTURE);
			}
			const bool eos = pkt->flags & EB_BUFFERFLAG_EOS;
			svt_av1_enc_release_out_buffer(&pkt);
			if (eos) { got_eos = true; return; }
		}
	}

	bool send_eos() {
		EbBufferHeaderType in{};
		in.size = sizeof in;
		in.pic_type = EB_AV1_INVALID_PICTURE;
		in.flags = EB_BUFFERFLAG_EOS;
		eos_sent = true;
		const bool ok = svt_av1_enc_send_picture(h, &in) == EB_ErrorNone;
		sent_all = true;
		return ok;
	}

public:
	~SvtAv1() override {
		if (rx.joinable()) { if (!eos_sent) send_eos(); rx.join(); } // an encode that failed halfway still has the thread waiting
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
		cfg.rate_control_mode = SVT_AV1_RC_MODE_CBR; // streaming target, like the other encoders; "rc" overrides
		cfg.pred_structure = LOW_DELAY; // CBR requires it; "pred-struct" overrides
		cfg.target_bit_rate = (uint32_t)p.bitrate;
		for (auto& [k, v] : p.options) {
			if (svt_av1_enc_parse_parameter(&cfg, k.c_str(), v.c_str()) != EB_ErrorNone) { fprintf(stderr, "svtav1: bad option %s=%s\n", k.c_str(), v.c_str()); return false; }
		}
		if (svt_av1_enc_set_parameter(h, &cfg) != EB_ErrorNone) { fprintf(stderr, "svtav1: set_parameter rejected the configuration\n"); return false; }
		if (svt_av1_enc_init(h) != EB_ErrorNone) { fprintf(stderr, "svtav1: init failed\n"); return false; }
		initialized = true;
		rx = std::thread([this] { receive(); });
		return true;
	}

	bool encode(const Picture* fr) override {
		if (!fr) { // flush
			if (!send_eos()) { fprintf(stderr, "svtav1: sending end of stream failed\n"); return false; }
			rx.join();
			return rx_ok && got_eos;
		}
		if (fr->fmt != pix) { fprintf(stderr, "svtav1: input must be %d-bit %d\n", pix.depth, pix.chroma); return false; }
		EbBufferHeaderType in{};
		in.size = sizeof in;
		in.pic_type = EB_AV1_INVALID_PICTURE;
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
		return rx_ok;
	}
};

} // namespace

std::unique_ptr<Encoder> make_encoder_svtav1() { return std::make_unique<SvtAv1>(); }
