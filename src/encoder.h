#pragma once
#include <cstdint>
#include <functional>
#include <map>
#include <memory>
#include <string>

extern "C" {
#include <libavutil/frame.h>
#include <libavutil/rational.h>
}

using Options = std::map<std::string, std::string>;

struct Params {
	int width = 0, height = 0;
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

// "nvenc" selects the native NVENC backend, anything else is a libavcodec encoder name.
std::unique_ptr<Encoder> make_encoder(const std::string& name);
std::unique_ptr<Encoder> make_encoder_nvenc();
std::unique_ptr<Encoder> make_encoder_libav(const std::string& name);

inline std::unique_ptr<Encoder> make_encoder(const std::string& name) {
	return name == "nvenc" ? make_encoder_nvenc() : make_encoder_libav(name);
}
