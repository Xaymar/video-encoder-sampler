// ves-convert-test: runs the converter (vk_convert.cpp) on raw frames and writes what comes out, for checking it against a reference.
//   ves-convert-test <in.raw> <w> <h> <in pix_fmt> <range 1|2> <matrix> <trc> <out.raw> <ow> <oh> <out pix_fmt> [out range 0|1|2] [out trc] [upscaler] [downscaler] [frames] [gpu]
// pix_fmt by FFmpeg's names; the input file holds the frames one after the other as FFmpeg's rawvideo would. "gpu": the frames are first uploaded
// into FFmpeg Vulkan frames (yuv420p/yuv420p10 only, as NV12/P010 with "gpu=nv12"), to check that the GPU path gives the same.
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <string>
#include <vector>

#include "convert.h"
#include "decode.h"
#include "source.h"
#include "ff.h"
extern "C" {
#include <libavutil/imgutils.h>
}

int main(int argc, char** argv) {
	if (argc < 12) { fprintf(stderr, "usage: see the source\n"); return 2; }
	const char* in_path = argv[1];
	const int w = atoi(argv[2]), h = atoi(argv[3]);
	const AVPixelFormat ifmt = av_get_pix_fmt(argv[4]);
	const int range = atoi(argv[5]), matrix = atoi(argv[6]), trc = atoi(argv[7]);
	const char* out_path = argv[8];
	const int ow = atoi(argv[9]), oh = atoi(argv[10]);
	const AVPixelFormat ofmt = av_get_pix_fmt(argv[11]);
	const int orange = argc > 12 ? atoi(argv[12]) : 0, otrc = argc > 13 ? atoi(argv[13]) : 0;
	ScaleOptions so;
	if (argc > 14) so.upscaler = argv[14];
	if (argc > 15) so.downscaler = argv[15];
	const int frames = argc > 16 ? atoi(argv[16]) : 1;
	const std::string gpu = argc > 17 ? argv[17] : "";
	if (const char* c = getenv("BT709_CURVE")) so.bt709_curve = bt709_curve_of(c);
	so.trc = otrc;
	Fmt fi, fo;
	if (ifmt == AV_PIX_FMT_NONE || ofmt == AV_PIX_FMT_NONE || !describe_native(ifmt, fi) || !describe_native(ofmt, fo)) { fprintf(stderr, "bad pixel format\n"); return 2; }
	const int isize = av_image_get_buffer_size(ifmt, w, h, 1);
	FILE* fin = fopen(in_path, "rb");
	FILE* fout = fopen(out_path, "wb");
	if (!fin || !fout) { fprintf(stderr, "cannot open files\n"); return 2; }
	auto conv = make_converter();
	AVBufferRef* frames_ctx = nullptr;
	int vk_index = -1;
	if (!gpu.empty()) {
		set_hw_decode("vulkan"); // the converter then shares FFmpeg's Vulkan device, as in a real run
		for (auto& d : vulkan_devices_fastest_first()) { vk_index = d.index; break; }
		conv->set_output_on_gpu(getenv("OUT_GPU") != nullptr);
	}
	bool opened = false;
	int got_frames = 0;
	auto drain = [&] {
		Picture o;
		while (conv->pop(o) == 1) {
			if (o.gpu && !picture_to_memory(o)) { fprintf(stderr, "readback failed\n"); exit(1); }
			const Layout L = layout_of(ofmt);
			const int bps = fo.depth > 8 ? 2 : 1;
			for (int p = 0; p < 3; p++) {
				const int pw = p ? AV_CEIL_RSHIFT(ow, L.log2_cw) : ow, ph = p ? AV_CEIL_RSHIFT(oh, L.log2_ch) : oh;
				for (int y = 0; y < ph; y++) fwrite(o.data[p] + (size_t)y * o.stride[p], 1, (size_t)pw * bps, fout);
			}
			got_frames++;
		}
	};
	std::vector<uint8_t> buf(isize);
	for (int n = 0; n < frames && fread(buf.data(), 1, isize, fin) == (size_t)isize; n++) {
		AVFrame* f = av_frame_alloc();
		f->format = ifmt; f->width = w; f->height = h;
		av_frame_get_buffer(f, 0);
		const uint8_t* src[4];
		int ls[4];
		av_image_fill_arrays((uint8_t**)src, ls, buf.data(), ifmt, w, h, 1);
		av_image_copy(f->data, f->linesize, src, ls, ifmt, w, h);
		f->color_range = (AVColorRange)range; f->colorspace = (AVColorSpace)matrix; f->color_trc = (AVColorTransferCharacteristic)trc; f->color_primaries = AVCOL_PRI_BT709;
		f->pts = n;
		Picture p;
		if (!gpu.empty()) {
			AVFrame* g = vulkan_upload(f, vk_index, gpu == "nv12", frames_ctx);
			if (!g) { fprintf(stderr, "upload failed\n"); return 1; }
			g->color_range = f->color_range; g->colorspace = f->colorspace; g->color_trc = f->color_trc; g->pts = n;
			if (!vulkan_picture(g, vk_index, false, p)) { fprintf(stderr, "vulkan_picture failed\n"); return 1; }
			av_frame_free(&g);
		} else {
			p = picture_from_frame(f, fi);
		}
		p.duration = 1;
		av_frame_free(&f);
		if (!opened) { if (!conv->open(p, Rational{1, 60}, ow, oh, fo, so, orange)) return 1; opened = true; }
		if (!conv->push(&p)) return 1;
		drain();
	}
	if (!conv->push(nullptr)) return 1;
	drain();
	fclose(fout);
	fprintf(stderr, "%d frames\n", got_frames);
	av_buffer_unref(&frames_ctx);
	return 0;
}
