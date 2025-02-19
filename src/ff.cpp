#include "ff.h"

#include <cstring>

bool describe_native(AVPixelFormat f, Fmt& out) {
	const AVPixFmtDescriptor* d = av_pix_fmt_desc_get(f);
	if (!d || d->nb_components < 3 || (d->flags & (AV_PIX_FMT_FLAG_HWACCEL | AV_PIX_FMT_FLAG_BITSTREAM | AV_PIX_FMT_FLAG_PAL | AV_PIX_FMT_FLAG_BAYER))) return false;
	out.rgb = d->flags & AV_PIX_FMT_FLAG_RGB;
	out.depth = d->comp[0].depth;
	if (out.depth > 12) return false;
	// shortcut: layouts libvmaf and the encoders lack (4:1:0, 4:1:1, 4:4:0) are treated as 4:2:0
	out.chroma = out.rgb || (!d->log2_chroma_w && !d->log2_chroma_h) ? 444 : d->log2_chroma_w == 1 && !d->log2_chroma_h ? 422 : 420;
	out.planar_yuv = (d->flags & AV_PIX_FMT_FLAG_PLANAR) && !out.rgb && d->nb_components == 3 && !(d->flags & AV_PIX_FMT_FLAG_BE) &&
	                 (d->log2_chroma_w + d->log2_chroma_h == (out.chroma == 420 ? 2 : out.chroma == 422 ? 1 : 0));
	return true;
}

AVPixelFormat av_planar_format(int chroma, int depth) {
	static const AVPixelFormat t[3][3] = {
		{AV_PIX_FMT_YUV420P, AV_PIX_FMT_YUV420P10LE, AV_PIX_FMT_YUV420P12LE},
		{AV_PIX_FMT_YUV422P, AV_PIX_FMT_YUV422P10LE, AV_PIX_FMT_YUV422P12LE},
		{AV_PIX_FMT_YUV444P, AV_PIX_FMT_YUV444P10LE, AV_PIX_FMT_YUV444P12LE},
	};
	int c = chroma == 420 ? 0 : chroma == 422 ? 1 : chroma == 444 ? 2 : -1;
	int d = depth == 8 ? 0 : depth == 10 ? 1 : depth == 12 ? 2 : -1;
	return c < 0 || d < 0 ? AV_PIX_FMT_NONE : t[c][d];
}

Picture picture_from_frame(AVFrame* frame, const Fmt& fmt) {
	AVFrame* own = av_frame_alloc();
	av_frame_move_ref(own, frame);
	Picture p;
	p.width = own->width; p.height = own->height;
	p.fmt = fmt;
	p.native = own->format;
	for (int i = 0; i < 3; i++) { p.data[i] = own->data[i]; p.stride[i] = own->linesize[i]; }
	p.pts = own->pts;
	p.duration = own->duration;
	p.color = {(int)own->color_range, (int)own->colorspace, (int)own->color_primaries, (int)own->color_trc};
	p.owner = std::shared_ptr<void>(own, [](void* f) { AVFrame* a = (AVFrame*)f; av_frame_free(&a); });
	return p;
}

void frame_view(const Picture& p, AVFrame& f) {
	memset(&f, 0, sizeof f);
	f.width = p.width; f.height = p.height;
	f.format = p.native;
	for (int i = 0; i < 3; i++) { f.data[i] = (uint8_t*)p.data[i]; f.linesize[i] = p.stride[i]; }
	f.pts = p.pts;
	f.duration = p.duration;
	f.color_range = (AVColorRange)p.color.range;
	f.colorspace = (AVColorSpace)p.color.space;
	f.color_primaries = (AVColorPrimaries)p.color.primaries;
	f.color_trc = (AVColorTransferCharacteristic)p.color.trc;
}
