// Checks speed_common.glsl's fdiv(), dsqrt_f() and rsqrt_f() (and the double way) against the CPU's float results on 2M inputs, many near midpoints.
// Build: glslangValidator -V --target-env vulkan1.2 -I../../shaders -o speed_fntest.spv speed_fntest.comp; then compile this with
// src/vk_compute.cpp src/gpu.cpp (-I src). Run in this folder (it reads speed_fntest.spv); prints the mismatches (all 0 expected).
#include <cmath>
#include <cstdlib>
#include <cstdio>
#include <cstring>
#include <random>
#include <vector>
#include <fstream>
#include <iterator>
#include "vk_compute.h"
static std::vector<unsigned char> spv;
const unsigned char* spv_find(const char* name, size_t& size) { size = spv.size(); return spv.data(); }
static float nextf(float x, int d) { return std::nextafter(x, d > 0 ? INFINITY : 0.0f); }
int main(int argc, char** argv) {
	std::ifstream f("speed_fntest.spv", std::ios::binary); spv.assign(std::istreambuf_iterator<char>(f), {});
	const int n = 1 << 21;
	std::vector<float> in(2 * n);
	std::mt19937_64 rng(7);
	std::uniform_real_distribution<double> u(0, 1);
	for (int i = 0; i < n; i++) {
		const int kind = i % 4;
		float a, b;
		auto rnd = [&](int lo, int hi) { return (float)(std::ldexp(1 + u(rng), lo + (int)(u(rng) * (hi - lo))) * (u(rng) < 0.5 ? -1 : 1)); };
		if (kind == 0) { a = rnd(-30, 30); b = rnd(-30, 30); }
		else { // near midpoints
			float q = std::fabs(rnd(-20, 20)); double m = ((double)q + (double)nextf(q, 1)) / 2;
			b = rnd(-20, 20);
			if (kind == 1) a = (float)((double)b * m);              // a/b near the midpoint m
			else if (kind == 2) { a = (float)(m * m); }              // sqrt(a) near m
			else { a = rnd(-5, 5); b = (float)(1.0 / (m * m)); }     // 1/sqrt(b) near m
			if (u(rng) < 0.3) a = nextf(a, u(rng) < 0.5 ? 1 : -1);
		}
		in[2 * i] = a; in[2 * i + 1] = b;
	}
	Vk vk(argc > 1 ? atoi(argv[1]) : 0, 1);
	Buf bi = vk.buffer(in.size() * 4 + 16, 2), bo = vk.buffer((size_t)n * 6 * 4 + 16, 2), d = vk.buffer(64, 2);
	memcpy(bi.map, in.data(), in.size() * 4);
	std::vector<Buf*> bs{&bi, &bo, &d, &d, &d, &d, &d, &d, &d};
	Pipe p = vk.pipeline("speed_fntest", bs);
	struct { int DW, DH, fw, subw, subh, nbh, nbv, nb, mode; float s, nf, be, one; int pad; double l; } pc{};
	pc.DW = n; pc.one = 1.0f;
	vk.begin(0); vk.bind(p); vk.push(&pc, sizeof pc); vk.dispatch((n + 63) / 64); vk.submit(0); vk.wait(0);
	const float* o = (const float*)bo.map;
	long bad[6] = {};
	for (int i = 0; i < n; i++) {
		const float a = in[2 * i], b = in[2 * i + 1];
		const float ref[3] = {(float)((double)a / (double)b), (float)std::sqrt((double)std::fabs(a)), (float)(1.0 / std::sqrt((double)std::fabs(b)))};
		for (int k = 0; k < 6; k++) if (memcmp(&o[6 * i + k], &ref[k % 3], 4)) { if (bad[k]++ < 3) std::printf("k%d a=%a b=%a gpu=%a cpu=%a\n", k, a, b, o[6*i+k], ref[k%3]); }
	}
	std::printf("mismatches fdiv %ld dsqrt_f %ld rsqrt_f %ld | double way %ld %ld %ld of %d\n", bad[0], bad[1], bad[2], bad[3], bad[4], bad[5], n);
}
