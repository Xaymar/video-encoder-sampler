#pragma once
#include <memory>
#include <string>

#include "picture.h"

struct ScaleOptions {
	std::string upscaler = "ewa_lanczos";   // libplacebo scaler names, see its documentation
	std::string downscaler = "ewa_lanczos";
	std::string device;                     // Vulkan GPU for scaling, by its exact name; empty: libplacebo picks
};

// Scales and converts planar frames (YUV or GBR) to a planar YUV format. One frame in, one frame out in order; a converter
// may hold frames back, so feed it and collect with pop(). RGB input converted to YUV uses
// BT.709 limited range, the only choice that needs one; everything else keeps its tags.
class Converter {
public:
	virtual ~Converter() = default;
	virtual bool open(const Picture& first, Rational time_base, int w, int h, Fmt out, const ScaleOptions& opt) = 0;
	virtual bool push(const Picture* p) = 0; // nullptr flushes
	virtual int pop(Picture& out) = 0;       // 1: frame produced, 0: nothing ready (or end of stream), <0: error
};

std::unique_ptr<Converter> make_converter(); // libplacebo, on the GPU through Vulkan
