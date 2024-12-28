#pragma once
#include <cstdint>
#include <functional>
#include <map>
#include <memory>
#include <string>

extern "C" {
#include <libavutil/frame.h>
#include <libavutil/pixdesc.h>
#include <libavutil/pixfmt.h>
#include <libavutil/rational.h>
}

using Options = std::map<std::string, std::string>;

struct Params {
	int width = 0, height = 0;
	AVPixelFormat pix = AV_PIX_FMT_YUV420P; // planar YUV format of the frames handed to encode()
	AVRational framerate{30, 1}; // shortcut: constant frame rate, frame pts == frame index; revisit for VFR sources
	int64_t bitrate = 0;         // bits per second
	int gop = 0;                 // frames
	std::string codec = "h264";  // used by native backends
	Options options;             // backend specific, applied verbatim; unknown keys are an error
};

struct Stats {
	long long packets = 0, bytes = 0;
};

class Encoder {
public:
	using Sink = std::function<void(const uint8_t* data, size_t size, int64_t pts, bool key)>;

	virtual ~Encoder() = default;
	virtual bool open(const Params&, Sink) = 0;
	virtual bool encode(const AVFrame*) = 0; // nullptr flushes
	Stats stats;
};

// "nvenc", "amf", "x264", "x265" and "svtav1" select the native backends, anything else is a libavcodec encoder name.
std::unique_ptr<Encoder> make_encoder_nvenc();
std::unique_ptr<Encoder> make_encoder_amf();
std::unique_ptr<Encoder> make_encoder_x264();
std::unique_ptr<Encoder> make_encoder_x265();
std::unique_ptr<Encoder> make_encoder_svtav1();
std::unique_ptr<Encoder> make_encoder_libav(const std::string& name);

bool nvenc_supports(const std::string& codec, int chroma, int depth);

inline bool is_gpu_encoder(const std::string& name) { return name == "nvenc" || name == "amf"; }
inline bool is_native_encoder(const std::string& name) { return is_gpu_encoder(name) || name == "x264" || name == "x265" || name == "svtav1"; }

inline std::unique_ptr<Encoder> make_encoder(const std::string& name) {
	if (name == "nvenc") return make_encoder_nvenc();
	if (name == "amf") return make_encoder_amf();
	if (name == "x264") return make_encoder_x264();
	if (name == "x265") return make_encoder_x265();
	if (name == "svtav1") return make_encoder_svtav1();
	return make_encoder_libav(name);
}
