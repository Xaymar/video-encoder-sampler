#pragma once
#include <string>

#include "picture.h"

// A planar YUV format of this chroma layout and bit depth, or a format with chroma 0 if there is no such thing.
// 4:2:2 is left out: NVENC only takes it through a proprietary CUDA surface, and almost nothing else wants it.
inline Fmt planar_fmt(int chroma, int depth) {
	Fmt f;
	bool ok = (chroma == 420 || chroma == 444) &&(depth == 8 || depth == 10 || depth == 12);
	f.chroma = ok ? chroma : 0;
	f.depth = depth;
	return f;
}
inline bool valid(const Fmt& f) { return f.chroma != 0; }

// Whether an encoder (as named in the config) can take this chroma layout at this bit depth.
bool encoder_supports(const std::string& encoder, const std::string& codec, int chroma, int depth);
