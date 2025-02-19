// Converter on top of libavfilter's libplacebo filter (GPU, via Vulkan).
#include "convert.h"

#include <cstdio>

#include "ff.h"

extern "C" {
#include <libavfilter/avfilter.h>
#include <libavfilter/buffersink.h>
#include <libavfilter/buffersrc.h>
#include <libavutil/opt.h>
}

namespace {

// Arguments for a "buffer" source that matches the frame, so the graph does not see its properties change.
std::string buffer_args(const AVFrame& f, AVRational tb) {
	AVRational sar = f.sample_aspect_ratio.num ? f.sample_aspect_ratio : AVRational{1, 1};
	char b[256];
	snprintf(b, sizeof b, "video_size=%dx%d:pix_fmt=%d:time_base=%d/%d:pixel_aspect=%d/%d:colorspace=%d:range=%d",
	         f.width, f.height, f.format, tb.num, tb.den, sar.num, sar.den, (int)f.colorspace, (int)f.color_range);
	return b;
}

class FfConverter : public Converter {
	AVFilterGraph* g = nullptr;
	AVFilterContext *src = nullptr, *sink = nullptr;
	Fmt out_fmt;

public:
	~FfConverter() override { avfilter_graph_free(&g); }

	bool open(const Picture& first, Rational time_base, int w, int h, Fmt out, const ScaleOptions& opt) override {
		out_fmt = out;
		AVPixelFormat out_pix = av_planar_format(out.chroma, out.depth);
		AVFrame tmpl;
		frame_view(first, tmpl);
		g = avfilter_graph_alloc();
		bool up = w > first.width || h > first.height;
		const AVFilter* lp = avfilter_get_by_name("libplacebo");
		if (!lp) { fprintf(stderr, "this FFmpeg build has no libplacebo filter\n"); return false; }
		AVFilterContext* f = avfilter_graph_alloc_filter(g, lp, "scale");
		av_opt_set(f, "w", std::to_string(w).c_str(), AV_OPT_SEARCH_CHILDREN);
		av_opt_set(f, "h", std::to_string(h).c_str(), AV_OPT_SEARCH_CHILDREN);
		av_opt_set(f, "format", av_get_pix_fmt_name(out_pix), AV_OPT_SEARCH_CHILDREN);
		if (first.fmt.rgb && !out.rgb) { // RGB has no matrix of its own, YUV needs one
			av_opt_set(f, "colorspace", "bt709", AV_OPT_SEARCH_CHILDREN);
			av_opt_set(f, "range", "tv", AV_OPT_SEARCH_CHILDREN);
		}
		av_opt_set(f, up ? "upscaler" : "downscaler", (up ? opt.upscaler : opt.downscaler).c_str(), AV_OPT_SEARCH_CHILDREN);
		if (avfilter_init_str(f, nullptr) < 0) { fprintf(stderr, "libplacebo init failed (is a Vulkan GPU available?)\n"); return false; }
		return avfilter_graph_create_filter(&src, avfilter_get_by_name("buffer"), "in", buffer_args(tmpl, to_av(time_base)).c_str(), nullptr, g) >= 0
			&& avfilter_graph_create_filter(&sink, avfilter_get_by_name("buffersink"), "out", nullptr, nullptr, g) >= 0
			&& avfilter_link(src, 0, f, 0) >= 0 && avfilter_link(f, 0, sink, 0) >= 0
			&& avfilter_graph_config(g, nullptr) >= 0;
	}

	bool push(const Picture* p) override {
		if (!src) return false;
		if (!p) return av_buffersrc_add_frame(src, nullptr) >= 0;
		AVFrame view;
		frame_view(*p, view);
		return av_buffersrc_add_frame(src, &view) >= 0; // the frame is not reference counted, so the filter copies the data
	}

	int pop(Picture& out) override {
		if (!sink) return 0; // not opened yet
		AVFrame* f = av_frame_alloc();
		int r = av_buffersink_get_frame(sink, f);
		if (r >= 0) { out = picture_from_frame(f, out_fmt); av_frame_free(&f); return 1; }
		av_frame_free(&f);
		return r == AVERROR(EAGAIN) || r == AVERROR_EOF ? 0 : r;
	}
};

} // namespace

std::unique_ptr<Converter> make_converter() { return std::make_unique<FfConverter>(); }
