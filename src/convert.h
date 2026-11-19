#pragma once
#include <memory>
#include <string>

#include "picture.h"

struct ScaleOptions {
	std::string upscaler = "bilinear";      // scaler names as libplacebo has them (vk_convert.cpp lists them)
	std::string downscaler = "ewa_lanczos";
	std::string device;                     // Vulkan GPU for scaling, by its exact name; empty: the home GPU, else the fastest
	int trc = 0;                            // the output's transfer curve (H.273 number) when it differs from the source's; 0: as the source
	int bt709_curve = 1;                    // how the BT.709 family of transfer tags (1, 6, 11, 12, 14, 15) is turned into linear light and back:
	                                        // 1 the exact BT.709 camera curve ("oetf"), 1886 display gamma 2.4 ("bt1886"); see bt709_curve_of()
};

// Scales and converts planar frames (YUV or GBR) to a planar YUV format. One frame in, one frame out in order; a converter
// may hold frames back, so feed it and collect with pop(). RGB input converted to YUV uses
// BT.709 limited range, the only choice that needs one; everything else keeps its tags. `range` (0 keep, 1 limited, 2 full) converts the sample
// values to that range, which is a real change of the values and not a tag.
class Converter {
public:
	virtual ~Converter() = default;
	virtual bool open(const Picture& first, Rational time_base, int w, int h, Fmt out, const ScaleOptions& opt, int range = 0) = 0;
	virtual bool push(const Picture* p) = 0; // nullptr flushes
	virtual int pop(Picture& out) = 0;       // 1: frame produced, 0: nothing ready (or end of stream), <0: error
	// Before open(): the converted frames may stay on the GPU (Picture::gpu, FFmpeg Vulkan frames) where the converter works on FFmpeg's device, for a
	// consumer that takes them there (the GPU scorer) or reads them back itself (picture_to_memory()). VES_CONVERT_TO_GPU=0: never.
	virtual void set_output_on_gpu(bool) {}
};

std::unique_ptr<Converter> make_converter(); // our own, on the GPU through Vulkan (vk_convert.cpp)
// Converts a small known picture on the device `opt` names and checks what comes back: some drivers hand back nothing but zeros (an AMD Vega iGPU gave an all-green
// video) without reporting an error. False, with what was wrong in `why`, when the device cannot be trusted with frames.
bool converter_works(const ScaleOptions& opt, std::string& why);
// Converters run on FFmpeg's Vulkan device of their GPU (the one its decoders use) instead of one of their own, so that frames decoded there are
// converted where they are (Picture::gpu): set when hardware decoding may use Vulkan, before the first converter opens (VES_SHARED_VULKAN=0: never).
void set_converter_ffmpeg_vulkan(bool on);
bool converter_shares_ffmpeg_vulkan();

// ScaleOptions::bt709_curve of a name ("oetf" or "bt709", "bt1886" or "display"; empty: the default, oetf); 0 for another name. And the name of one.
int bt709_curve_of(const std::string& name);
std::string bt709_curve_name(int curve);
// A transfer characteristic (the "trc" option) by its FFmpeg name ("bt709", "iec61966-2-1", "smpte2084", ...; "srgb" and "pq", "hlg" too) or
// H.273 number; 0 for "source" or empty, -1 for an unknown name.
int trc_of(const std::string& name);
// Whether a scaler of this name exists.
bool scaler_known(const std::string& name);
