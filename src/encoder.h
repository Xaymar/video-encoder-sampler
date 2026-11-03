#pragma once
#include <cstdint>
#include <functional>
#include <map>
#include <memory>
#include <string>

#include "picture.h"

using Options = std::map<std::string, std::string>;

struct Params {
	int width = 0, height = 0;
	Fmt pix;                     // planar YUV format of the pictures handed to encode()
	Rational framerate{30, 1};   // what rate control assumes; pictures are numbered in order (pts)
	int64_t bitrate = 0;         // bits per second
	int gop = 0;                 // frames
	std::string codec = "h264";  // used by native backends
	Color color;                 // how the pictures are to be read; written into the stream (VUI, sequence header), where the encoder also decides by it
	Options options;             // backend specific, applied verbatim; unknown keys are an error
};

struct Stats {
	long long packets = 0, bytes = 0;
};

class Encoder {
public:
	// `dts` is the decode time the encoder reports in the same units as `pts`, or NO_DTS if it does not report one.
	static constexpr int64_t NO_DTS = INT64_MIN;
	using Sink = std::function<void(const uint8_t* data, size_t size, int64_t pts, bool key, int64_t dts)>;

	virtual ~Encoder() = default;
	virtual bool open(const Params&, Sink) = 0;
	virtual bool encode(const Picture*) = 0; // nullptr flushes
	// How many frames the display order can lag behind the coding order (B-frame depth), valid after open(); only needed from encoders
	// that report no DTS, so that a muxer can offset the display times.
	virtual int reorder_delay() const { return 0; }
	Stats stats;
};

// The native backends, selected by the names "nvenc", "amf", "qsv", "x264", "x265" and "svtav1".
std::unique_ptr<Encoder> make_encoder_nvenc();
std::unique_ptr<Encoder> make_encoder_amf();
std::unique_ptr<Encoder> make_encoder_qsv();
std::unique_ptr<Encoder> make_encoder_x264();
std::unique_ptr<Encoder> make_encoder_x265();
std::unique_ptr<Encoder> make_encoder_svtav1();

bool nvenc_supports(const std::string& codec, int chroma, int depth);

// What the encoder block of the n-th GPU reports about itself for a codec, as a JSON object (NVENC: all of its capability values), or "null"
// if the encoder cannot tell. Together with the PCI ids of the GPU this identifies the encoder generation: the CUDA generation of a GPU says
// nothing about it, since a newer GPU can carry an older encoder block.
std::string nvenc_caps_json(const std::string& codec, int gpu);
std::string amf_caps_json(const std::string& codec, int gpu = 0); // where the encoder runs (hardware, GPU compute, software) and the SDK's capability values of AMD GPU `gpu`
inline std::string encoder_caps_json(const std::string& encoder, const std::string& codec, int gpu) { return encoder == "nvenc" ? nvenc_caps_json(codec, gpu) : encoder == "amf" ? amf_caps_json(codec, gpu) : "null"; }
bool amf_supports(const std::string& codec, int chroma, int depth, int gpu = 0);
bool qsv_supports(const std::string& codec, int chroma, int depth);

inline bool is_gpu_encoder(const std::string& name) { return name == "nvenc" || name == "amf" || name == "qsv"; }
inline bool is_native_encoder(const std::string& name) { return is_gpu_encoder(name) || name == "x264" || name == "x265" || name == "svtav1"; }

inline std::unique_ptr<Encoder> make_encoder(const std::string& name) {
	if (name == "nvenc") return make_encoder_nvenc();
	if (name == "amf") return make_encoder_amf();
	if (name == "qsv") return make_encoder_qsv();
	if (name == "x264") return make_encoder_x264();
	if (name == "x265") return make_encoder_x265();
	if (name == "svtav1") return make_encoder_svtav1();
	return nullptr; // unknown name
}
