#pragma once
// The tool's own basic types. Nothing in here depends on FFmpeg; only the ff_*.cpp files and the libavcodec
// encoder translate between these and FFmpeg's types.
#include <cstdint>
#include <memory>

struct Rational {
	int num = 0, den = 1;
	double value() const { return den ? (double)num / den : 0.0; }
	Rational inverse() const { return {den, num}; }
};
inline int compare(Rational a, Rational b) { int64_t l = (int64_t)a.num * b.den, r = (int64_t)b.num * a.den; return l < r ? -1 : l > r; }

// Pixel format: chroma layout, bit depth, and how the samples are arranged.
struct Fmt {
	int chroma = 420;       // 420, 422 or 444 (RGB counts as 444)
	int depth = 8;          // bits per component
	bool rgb = false;
	bool planar = true;     // three separate planes (Y, U, V or G, B, R) of 8-bit or 16-bit little-endian samples; false: anything else
	bool operator==(const Fmt& o) const { return chroma == o.chroma && depth == o.depth && rgb == o.rgb && planar == o.planar; }
	bool operator!=(const Fmt& o) const { return !(*this == o); }
};

// Colour description as tagged in a file, numbered as in ITU-T H.273 (and FFmpeg); range is 0 unspecified, 1 limited, 2 full.
struct Color {
	int range = 0, space = 2, primaries = 2, trc = 2;
};

// One video frame. The memory belongs to `owner`, so a Picture can be copied and kept around cheaply.
struct Picture {
	int width = 0, height = 0;
	Fmt fmt;
	int native = -1;                // the FFmpeg pixel format code behind fmt, meaningful to the ff_*.cpp files only
	const uint8_t* data[3]{};
	int stride[3]{};                // bytes per row
	int64_t pts = 0;                // frame number in the stream it belongs to
	int64_t duration = 1;           // in the same units
	Color color;
	std::shared_ptr<void> owner;
};

// Output geometry of an encode; zero fields mean "as the source".
struct Shape {
	int w = 0, h = 0;
	Rational fps{0, 1};
	Fmt pix;
	bool has_pix = false;
};

enum class Codec { None, H264, HEVC, AV1 };
