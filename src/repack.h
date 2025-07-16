#pragma once
// Copies from the planar frames the pipeline produces into the layouts the GPU encoders want. The destination is memory
// the driver handed out, often write-combined, where writing a sample at a time is very slow: rows are built in a scratch
// buffer first and written out in one piece. 10-bit samples go in the top bits of 16, so they are shifted left by 6.
#include <cstdint>
#include <cstring>
#include <vector>

#include "picture.h"

inline void pack_plane(const Picture& fr, int plane, int w, int h, int depth, uint8_t* dst, size_t pitch) {
	static thread_local std::vector<uint16_t> row;
	for (int y = 0; y < h; y++) {
		const uint8_t* s = fr.data[plane] + (size_t)y * fr.stride[plane];
		uint8_t* d = dst + (size_t)y * pitch;
		if (depth <= 8) { memcpy(d, s, (size_t)w); continue; }
		row.resize((size_t)w);
		const uint16_t* s16 = (const uint16_t*)s;
		for (int x = 0; x < w; x++) row[x] = (uint16_t)(s16[x] << 6);
		memcpy(d, row.data(), (size_t)w * 2);
	}
}

// The two chroma planes of a 4:2:0 frame interleaved (NV12, P010).
inline void pack_uv(const Picture& fr, int w, int h, int depth, uint8_t* dst, size_t pitch) {
	static thread_local std::vector<uint16_t> row16;
	static thread_local std::vector<uint8_t> row8;
	const int cw = (w + 1) / 2, ch = (h + 1) / 2;
	for (int y = 0; y < ch; y++) {
		const uint8_t *u = fr.data[1] + (size_t)y * fr.stride[1], *v = fr.data[2] + (size_t)y * fr.stride[2];
		uint8_t* d = dst + (size_t)y * pitch;
		if (depth <= 8) {
			row8.resize((size_t)cw * 2);
			for (int x = 0; x < cw; x++) { row8[2 * x] = u[x]; row8[2 * x + 1] = v[x]; }
			memcpy(d, row8.data(), row8.size());
		} else {
			row16.resize((size_t)cw * 2);
			const uint16_t *u16 = (const uint16_t*)u, *v16 = (const uint16_t*)v;
			for (int x = 0; x < cw; x++) { row16[2 * x] = (uint16_t)(u16[x] << 6); row16[2 * x + 1] = (uint16_t)(v16[x] << 6); }
			memcpy(d, row16.data(), row16.size() * 2);
		}
	}
}
