#pragma once
#include <string>

extern "C" {
#include <libavfilter/avfilter.h>
#include <libavutil/frame.h>
#include <libavutil/rational.h>
}

// Output geometry of an encode: resolution and frame rate.
struct Shape {
	int w = 0, h = 0;
	AVRational fps{0, 1};
	AVPixelFormat pix = AV_PIX_FMT_NONE; // pixel format; none means "as the source"
};

struct ScaleOptions {
	std::string upscaler = "ewa_lanczos";   // libplacebo scaler names, see its documentation
	std::string downscaler = "ewa_lanczos";
};

// Arguments for a libavfilter "buffer" source that matches `f`, so the graph does not see its properties change.
std::string buffer_args(const AVFrame* f, AVRational time_base);

// Scales and converts frames to `out_fmt` with libplacebo (GPU, via Vulkan). One frame in, one frame out, pts preserved.
// RGB input converted to YUV uses BT.709 limited range, the only choice that needs one; everything else keeps its tags.
class Scaler {
	AVFilterGraph* g = nullptr;
	AVFilterContext *src = nullptr, *sink = nullptr;

public:
	~Scaler() { avfilter_graph_free(&g); }
	bool open(const AVFrame* tmpl, AVRational time_base, int w, int h, AVPixelFormat out_fmt, const ScaleOptions& opt);
	bool push(AVFrame* f); // nullptr flushes
	int pop(AVFrame* out); // 1: frame produced, 0: nothing ready (or end of stream), <0: error
};
