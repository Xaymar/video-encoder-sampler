// Library versions for the result cache. x265.h must come first: its enum constants (DEFAULT, ...) clash with macros
// from the other headers.
extern "C" {
#include <x265.h>
}

#include "cache.h"

extern "C" {
#include <libavutil/avutil.h>
#include <libplacebo/config.h>
#include <libvmaf/libvmaf.h>
#include <svt-av1/EbSvtAv1Enc.h>
#include <x264.h>
}

std::string encoder_library_version(const std::string& encoder) {
	if (encoder == "x264") return "x264 build " + std::to_string(X264_BUILD) + " " X264_VERSION;
	if (encoder == "x265") return std::string("x265 ") + x265_version_str + " " + x265_build_info_str;
	if (encoder == "svtav1") return std::string("svt-av1 ") + svt_av1_get_version();
	return encoder; // nvenc and amf: the GPU and driver matter there, see gpu_description()
}

std::string decoder_library_version() { return std::string("ffmpeg ") + av_version_info(); }

std::string scaler_library_version() { return std::string("libplacebo ") + pl_version(); }

std::string vmaf_library_version() { return vmaf_version(); }
