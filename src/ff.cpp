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
	out.planar = (d->flags & AV_PIX_FMT_FLAG_PLANAR) && d->nb_components == 3 && !(d->flags & AV_PIX_FMT_FLAG_BE) &&
	                 d->comp[0].plane == 0 && d->comp[1].plane == 1 && d->comp[2].plane == 2 && // not NV12 and friends
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

Layout layout_of(AVPixelFormat f) {
	Layout l;
	const AVPixFmtDescriptor* d = av_pix_fmt_desc_get(f);
	if (!d) return l;
	for (int i = 0; i < d->nb_components && i < 4; i++) {
		const AVComponentDescriptor& c = d->comp[i];
		l.comp[i] = {c.plane, c.offset, c.shift, c.depth};
		l.step[c.plane] = c.step;
		if (c.plane + 1 > l.planes) l.planes = c.plane + 1;
	}
	l.log2_cw = d->log2_chroma_w; l.log2_ch = d->log2_chroma_h;
	l.big_endian = d->flags & AV_PIX_FMT_FLAG_BE;
	return l;
}

Picture picture_from_frame(AVFrame* frame, const Fmt& fmt) {
	AVFrame* own = av_frame_alloc();
	av_frame_move_ref(own, frame);
	Picture p;
	p.width = own->width; p.height = own->height;
	p.fmt = fmt;
	p.native = own->format;
	p.layout = layout_of((AVPixelFormat)own->format);
	for (int i = 0; i < 4; i++) { p.data[i] = own->data[i]; p.stride[i] = own->linesize[i]; }
	p.pts = own->pts;
	p.duration = own->duration;
	p.color = {(int)own->color_range, (int)own->colorspace, (int)own->color_primaries, (int)own->color_trc};
	p.owner = std::shared_ptr<void>(own, [](void* f) { AVFrame* a = (AVFrame*)f; av_frame_free(&a); });
	return p;
}

const AVCodec* software_decoder(AVCodecID id) {
	void* it = nullptr;
	while (const AVCodec* c = av_codec_iterate(&it))
		if (c->id == id && av_codec_is_decoder(c) && !(c->capabilities & (AV_CODEC_CAP_HARDWARE | AV_CODEC_CAP_HYBRID))) return c;
	return nullptr;
}
