#include "formats.h"

#include "encoder.h"

extern "C" {
#include <x265.h>
}

bool encoder_supports(const std::string& encoder, const std::string& codec, int chroma, int depth, int gpu) {
	if (!valid(planar_fmt(chroma, depth))) return false;
	if (encoder == "x264") return depth <= 10;
	if (encoder == "nvenc") return nvenc_supports(codec, chroma, depth);
	if (encoder == "x265") return x265_api_get(depth) != nullptr; // the multi-bit-depth build links 8, 10 and 12 bit encoders; 4:2:0/4:2:2/4:4:4 all work
	if (encoder == "svtav1") return chroma == 420 && depth <= 10; // this SVT-AV1 refuses anything but 4:2:0 ("Only support 420 now")
	if (encoder == "amf") return amf_supports(codec, chroma, depth, gpu);
	if (encoder == "qsv") return qsv_supports(codec, chroma, depth);
	return false; // unknown encoder
}
