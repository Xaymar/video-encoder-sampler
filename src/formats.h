#pragma once
#include <string>

extern "C" {
#include <libavutil/pixfmt.h>
}

// A pixel format reduced to what matters for choosing encoder input and the scoring format.
struct Fmt {
	int chroma = 420; // 420, 422 or 444 (RGB counts as 444)
	int depth = 8;    // bits per component
	bool rgb = false;
};

// Describes a source pixel format; false for formats that cannot be used (gray, palettes, hardware, > 12 bit).
bool describe_format(AVPixelFormat f, Fmt& out);

// Planar YUV format for a chroma layout and bit depth, AV_PIX_FMT_NONE if there is none.
AVPixelFormat planar_format(int chroma, int depth);

// Whether an encoder (as named in the config) can take this chroma layout at this bit depth.
bool encoder_supports(const std::string& encoder, const std::string& codec, int chroma, int depth);
