// Reproduces a crash (access violation reading address 0x28, in a worker thread of libSvtAv1Enc) when the encoder is started with "lp" = 1 through the public C API.
// The same stream with lp = 0 (the default) or lp >= 2 encodes normally.
//
// The frames are made up (a moving gradient), so no video file is needed. The settings are the ones a streaming sweep uses: CBR with the low-delay prediction
// structure, preset 10, 8-bit 4:2:0, a key frame every 60 frames.
//
// Build (Windows, clang; SVT-AV1's include folder and import library are the only dependencies):
//   clang++ -std=c++17 -I <svt-av1>/include svtav1_lp1.cpp <svt-av1>/lib/libSvtAv1Enc.dll.a -o svtav1_lp1.exe
// Run (libSvtAv1Enc-4.dll next to it or on the PATH):
//   svtav1_lp1.exe [lp] [preset] [width] [height] [frames]
//   svtav1_lp1.exe 1             crashes
//   svtav1_lp1.exe 2             encodes
//   svtav1_lp1.exe 0             encodes
//
// Observed (SVT-AV1 v4.2.0, MinGW GCC 16.1.0 64 bit build, Windows 11):
//   lp 1, any preset from 4 to 13, any size from 320x180 up, even a single frame, CBR + low delay  -> access violation in a thread of the encoder, exit code 0xC0000005
//   lp 0, 2..6 (and values above 6, which the library accepts)                                      -> encodes; the stream is identical for all of them
// The command line encoder (SvtAv1EncApp -i raw.yuv -w 1920 -h 1080 --fps-num 30 --fps-denom 1 --preset 10 --rc 2 --pred-struct 1 --tbr 3000 --keyint 59
// --lp 1) did not crash here, so something the application sets or does differs; this program sets only the fields above.
// Expected: lp = 1 encodes, or svt_av1_enc_set_parameter / svt_av1_enc_init return an error.
#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <thread>
#include <vector>

extern "C" {
#include <svt-av1/EbSvtAv1.h>
#include <svt-av1/EbSvtAv1Enc.h>
}

int main(int argc, char** argv) {
	const char* lp = argc > 1 ? argv[1] : "1";
	const char* preset = argc > 2 ? argv[2] : "10";
	const int width = argc > 3 ? atoi(argv[3]) : 1920, height = argc > 4 ? atoi(argv[4]) : 1080, frames = argc > 5 ? atoi(argv[5]) : 150;

	EbComponentType* enc = nullptr;
	EbSvtAv1EncConfiguration cfg{};
	if (svt_av1_enc_init_handle(&enc, &cfg) != EB_ErrorNone) { fprintf(stderr, "init_handle failed\n"); return 1; }
	cfg.source_width = (uint32_t)width;
	cfg.source_height = (uint32_t)height;
	cfg.frame_rate_numerator = 30;
	cfg.frame_rate_denominator = 1;
	cfg.encoder_bit_depth = 8;
	cfg.encoder_color_format = EB_YUV420;
	cfg.profile = MAIN_PROFILE;
	cfg.intra_period_length = 59;
	cfg.rate_control_mode = SVT_AV1_RC_MODE_CBR;
	cfg.pred_structure = LOW_DELAY;
	cfg.target_bit_rate = 3000 * 1000;
	if (svt_av1_enc_parse_parameter(&cfg, "preset", preset) != EB_ErrorNone) { fprintf(stderr, "bad preset\n"); return 1; }
	if (svt_av1_enc_parse_parameter(&cfg, "lp", lp) != EB_ErrorNone) { fprintf(stderr, "bad lp\n"); return 1; }
	if (svt_av1_enc_set_parameter(enc, &cfg) != EB_ErrorNone) { fprintf(stderr, "set_parameter rejected the configuration\n"); return 1; }
	if (svt_av1_enc_init(enc) != EB_ErrorNone) { fprintf(stderr, "init failed\n"); return 1; }
	fprintf(stderr, "lp=%s preset=%s %dx%d, %d frames\n", lp, preset, width, height, frames);

	// Packets are collected on a thread of their own (get_packet blocks in low-delay mode).
	std::atomic<bool> sent_all{false};
	std::atomic<long long> packets{0}, bytes{0};
	std::thread rx([&] {
		for (;;) {
			EbBufferHeaderType* pkt = nullptr;
			const EbErrorType r = svt_av1_enc_get_packet(enc, &pkt, sent_all ? 1 : 0);
			if (r == EB_NoErrorEmptyQueue) { std::this_thread::sleep_for(std::chrono::milliseconds(1)); continue; }
			if (r != EB_ErrorNone || !pkt) { fprintf(stderr, "get_packet failed (%d)\n", (int)r); return; }
			if (pkt->n_filled_len > 0) { packets++; bytes += pkt->n_filled_len; }
			const bool eos = pkt->flags & EB_BUFFERFLAG_EOS;
			svt_av1_enc_release_out_buffer(&pkt);
			if (eos) return;
		}
	});

	const size_t luma = (size_t)width * height;
	std::vector<std::vector<uint8_t>> keep; // the frames stay alive until the end
	for (int i = 0; i < frames; i++) {
		keep.emplace_back(luma * 3 / 2, 128);
		uint8_t* y = keep.back().data();
		for (int row = 0; row < height; row++)
			for (int x = 0; x < width; x++) y[(size_t)row * width + x] = (uint8_t)((x * 3 + row * 2 + i * 5) & 255);
		EbSvtIOFormat io{};
		io.luma = y; io.cb = y + luma; io.cr = y + luma + luma / 4;
		io.y_stride = (uint32_t)width; io.cb_stride = io.cr_stride = (uint32_t)width / 2;
		EbBufferHeaderType in{};
		in.size = sizeof in;
		in.pic_type = EB_AV1_INVALID_PICTURE;
		in.p_buffer = (uint8_t*)&io;
		in.n_filled_len = in.n_alloc_len = (uint32_t)(luma * 3 / 2);
		in.pts = i;
		if (svt_av1_enc_send_picture(enc, &in) != EB_ErrorNone) { fprintf(stderr, "send_picture failed\n"); return 1; }
	}
	EbBufferHeaderType eos{};
	eos.size = sizeof eos;
	eos.pic_type = EB_AV1_INVALID_PICTURE;
	eos.flags = EB_BUFFERFLAG_EOS;
	svt_av1_enc_send_picture(enc, &eos);
	sent_all = true;
	rx.join();
	fprintf(stderr, "done: %lld packets, %lld bytes\n", packets.load(), bytes.load());
	svt_av1_enc_deinit(enc);
	svt_av1_enc_deinit_handle(enc);
	return 0;
}
