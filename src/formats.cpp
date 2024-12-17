#include "formats.h"

extern "C" {
#include <libavcodec/avcodec.h>
#include <libavutil/pixdesc.h>
}

bool describe_format(AVPixelFormat f, Fmt& out) {
	const AVPixFmtDescriptor* d = av_pix_fmt_desc_get(f);
	if (!d || d->nb_components < 3 || (d->flags & (AV_PIX_FMT_FLAG_HWACCEL | AV_PIX_FMT_FLAG_BITSTREAM | AV_PIX_FMT_FLAG_PAL | AV_PIX_FMT_FLAG_BAYER))) return false;
	out.rgb = d->flags & AV_PIX_FMT_FLAG_RGB;
	out.depth = d->comp[0].depth;
	if (out.depth > 12) return false;
	// shortcut: layouts libvmaf and the encoders lack (4:1:0, 4:1:1, 4:4:0) are treated as 4:2:0
	out.chroma = out.rgb || (!d->log2_chroma_w && !d->log2_chroma_h) ? 444 : d->log2_chroma_w == 1 && !d->log2_chroma_h ? 422 : 420;
	return true;
}

AVPixelFormat planar_format(int chroma, int depth) {
	static const AVPixelFormat t[3][3] = {
		{AV_PIX_FMT_YUV420P, AV_PIX_FMT_YUV420P10LE, AV_PIX_FMT_YUV420P12LE},
		{AV_PIX_FMT_YUV422P, AV_PIX_FMT_YUV422P10LE, AV_PIX_FMT_YUV422P12LE},
		{AV_PIX_FMT_YUV444P, AV_PIX_FMT_YUV444P10LE, AV_PIX_FMT_YUV444P12LE},
	};
	int c = chroma == 420 ? 0 : chroma == 422 ? 1 : chroma == 444 ? 2 : -1;
	int d = depth == 8 ? 0 : depth == 10 ? 1 : depth == 12 ? 2 : -1;
	return c < 0 || d < 0 ? AV_PIX_FMT_NONE : t[c][d];
}

bool encoder_supports(const std::string& encoder, const std::string& codec, int chroma, int depth) {
	if (planar_format(chroma, depth) == AV_PIX_FMT_NONE) return false;
	if (encoder == "x264") return depth <= 10;
	if (encoder == "x265" || encoder == "svtav1" || encoder == "nvenc" || encoder == "amf") return chroma == 420 && depth == 8; // shortcut: widened as each backend learns the formats
	const AVCodec* c = avcodec_find_encoder_by_name(encoder.c_str());
	if (!c) return false;
	const AVPixelFormat* fmts = nullptr;
	int n = 0;
	if (avcodec_get_supported_config(nullptr, c, AV_CODEC_CONFIG_PIX_FORMAT, 0, (const void**)&fmts, &n) < 0 || !fmts) return true;
	for (int i = 0; i < n; i++) if (fmts[i] == planar_format(chroma, depth)) return true;
	return false;
}
