#include "scaler.h"

#include <cstdio>

extern "C" {
#include <libavfilter/buffersink.h>
#include <libavfilter/buffersrc.h>
#include <libavutil/opt.h>
#include <libavutil/pixdesc.h>
#include <libavutil/pixfmt.h>
}

std::string buffer_args(const AVFrame* f, AVRational tb) {
	AVRational sar = f->sample_aspect_ratio.num ? f->sample_aspect_ratio : AVRational{1, 1};
	char b[256];
	snprintf(b, sizeof b, "video_size=%dx%d:pix_fmt=%d:time_base=%d/%d:pixel_aspect=%d/%d:colorspace=%d:range=%d",
	         f->width, f->height, f->format, tb.num, tb.den, sar.num, sar.den, (int)f->colorspace, (int)f->color_range);
	return b;
}

bool Scaler::open(const AVFrame* tmpl, AVRational tb, int w, int h, AVPixelFormat out_fmt, const ScaleOptions& opt) {
	g = avfilter_graph_alloc();
	bool up = w > tmpl->width || h > tmpl->height;
	const AVFilter* lp = avfilter_get_by_name("libplacebo");
	if (!lp) { fprintf(stderr, "this FFmpeg build has no libplacebo filter\n"); return false; }
	AVFilterContext* f = avfilter_graph_alloc_filter(g, lp, "scale");
	av_opt_set(f, "w", std::to_string(w).c_str(), AV_OPT_SEARCH_CHILDREN);
	av_opt_set(f, "h", std::to_string(h).c_str(), AV_OPT_SEARCH_CHILDREN);
	av_opt_set(f, "format", av_get_pix_fmt_name(out_fmt), AV_OPT_SEARCH_CHILDREN);
	const AVPixFmtDescriptor* in = av_pix_fmt_desc_get((AVPixelFormat)tmpl->format);
	const AVPixFmtDescriptor* out = av_pix_fmt_desc_get(out_fmt);
	if (in && out && (in->flags & AV_PIX_FMT_FLAG_RGB) && !(out->flags & AV_PIX_FMT_FLAG_RGB)) {
		av_opt_set(f, "colorspace", "bt709", AV_OPT_SEARCH_CHILDREN);
		av_opt_set(f, "range", "tv", AV_OPT_SEARCH_CHILDREN);
	}
	av_opt_set(f, up ? "upscaler" : "downscaler", (up ? opt.upscaler : opt.downscaler).c_str(), AV_OPT_SEARCH_CHILDREN);
	if (avfilter_init_str(f, nullptr) < 0) { fprintf(stderr, "libplacebo init failed (is a Vulkan GPU available?)\n"); return false; }
	return avfilter_graph_create_filter(&src, avfilter_get_by_name("buffer"), "in", buffer_args(tmpl, tb).c_str(), nullptr, g) >= 0
		&& avfilter_graph_create_filter(&sink, avfilter_get_by_name("buffersink"), "out", nullptr, nullptr, g) >= 0
		&& avfilter_link(src, 0, f, 0) >= 0 && avfilter_link(f, 0, sink, 0) >= 0
		&& avfilter_graph_config(g, nullptr) >= 0;
}

bool Scaler::push(AVFrame* f) { return src && av_buffersrc_add_frame(src, f) >= 0; }

int Scaler::pop(AVFrame* out) {
	if (!sink) return 0; // not opened yet
	int r = av_buffersink_get_frame(sink, out);
	if (r >= 0) return 1;
	return r == AVERROR(EAGAIN) || r == AVERROR_EOF ? 0 : r;
}
