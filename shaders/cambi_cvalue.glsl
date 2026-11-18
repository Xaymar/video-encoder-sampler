// cambi_cvalue.comp and cambi_cvalue_cols.comp (CAMBI_COLS: with the column counts, whose shared memory costs occupancy, so it is a
// pipeline of its own)
#include "cambi_common.glsl"
// The banding value (c value) of every pixel of one scale (calculate_c_values in cambi.c): from how many flat pixels of the window around it have
// its value and values a few steps above and below it. Each workgroup also writes how many of its values are not 0, and their sum.
// With pc.OW < 0 the c values are already there (cambi_cslide made them), and only the sums are made.
layout(std430, binding = 0) readonly buffer Img { uint img[]; };
layout(std430, binding = 1) readonly buffer Mask { uint mask[]; };
layout(std430, binding = 2) buffer CVal { float cval[]; };
layout(std430, binding = 3) readonly buffer Tables { int tvi[32]; int weight[32]; float recip[]; } tb;
layout(std430, binding = 4) writeonly buffer Part { double part[]; }; // per workgroup: count of values that are not 0, their sum

const int MAXD = 8; // nd <= 8 (max_log_contrast <= 3)
const int TILE = 16;
const int MAXPAD = 16;
shared int tile[(TILE + 2 * MAXPAD) * (TILE + 2 * MAXPAD)];
#ifdef CAMBI_COLS
// Column counts (tiles whose centres' values lie close together, the usual case): for each centre row of the tile and each column of the region,
// how many of the window's rows there hold each value base .. base + 31, in 8 bit lanes; laid out [word][row][column] so a warp reads along a row.
const int CW = TILE + 2 * MAXPAD, RMAX = 32;
shared uint colc[RMAX / 4 * TILE * CW];
shared int s_lo, s_hi;
#endif
shared double ssum[256];
shared uint scnt[256];

// A sample as the histograms see it: its value if it is flat and within the band, else NOT_COUNTED (so far below any value that q - v + nd is
// negative whatever v is).
const int NOT_COUNTED = -65536;
int counted(int r, int c) {
	if (r < 0 || c < 0 || r >= pc.H || c >= pc.W) return NOT_COUNTED;
	const int o = r * pc.W + c;
	if (mask[o] == 0u) return NOT_COUNTED;
	const int v = int(img[o]);
	return (v - pc.band_base) >= 0 && (v - pc.band_base) < pc.band_size ? v : NOT_COUNTED;
}

// The 2 nd + 1 <= 17 counts of the window, 16 bits each (a window has fewer than 4226 samples), four to a 64-bit word: kept in registers (an array
// indexed by the sample's value would live in slow local memory), each sample adding to its word without a branch (via Row, below).
struct Counts { uint64_t w0, w1, w2, w3, w4; };
// A row of the window (at most 65 samples) counted in 8 bit lanes of 32 bit words first, which is cheaper than 64 bit adds per sample, then
// spread into the 16 bit lanes of the same word of Counts. `small` (nd <= 4, so k <= 8: three words) is a literal at each call.
struct Row { uint r0, r1, r2, r3, r4; };
void count_row(inout Row r, uint k, const bool small) {
	const uint inc = 1u << (8u * (k & 3u));
	const uint wi = k >> 2u;
	r.r0 += wi == 0u ? inc : 0u;
	r.r1 += wi == 1u ? inc : 0u;
	r.r2 += wi == 2u ? inc : 0u;
	if (!small) {
		r.r3 += wi == 3u ? inc : 0u;
		r.r4 += wi == 4u ? inc : 0u;
	}
}
uint64_t spread(uint b) {
	const uint64_t x = uint64_t(b);
	return (x & 0xFFul) | ((x & 0xFF00ul) << 8) | ((x & 0xFF0000ul) << 16) | ((x & 0xFF000000ul) << 24);
}
void add_row(inout Counts c, Row r, const bool small) {
	c.w0 += spread(r.r0);
	c.w1 += spread(r.r1);
	c.w2 += spread(r.r2);
	if (!small) {
		c.w3 += spread(r.r3);
		c.w4 += spread(r.r4);
	}
}
void window_tiled(inout Counts cnt, int base, int tw, int pad, uint off, uint span, const bool small) {
	for (int dr = 0; dr <= 2 * pad; dr++) {
		const int row = base + dr * tw;
		Row r = Row(0u, 0u, 0u, 0u, 0u);
		for (int dc = 0; dc <= 2 * pad; dc++) {
			const uint k = uint(tile[row + dc]) + off;
			if (k <= span) count_row(r, k, small);
		}
		add_row(cnt, r, small);
	}
}
void window_direct(inout Counts cnt, int i, int j, int pad, uint off, uint span, const bool small) {
	for (int dr = -pad; dr <= pad; dr++) {
		Row r = Row(0u, 0u, 0u, 0u, 0u);
		for (int dc = -pad; dc <= pad; dc++) {
			const uint k = uint(counted(i + dr, j + dc)) + off;
			if (k <= span) count_row(r, k, small);
		}
		add_row(cnt, r, small);
	}
}

void count(inout Counts c, uint k, const bool small) { // one sample straight into the 16 bit counts (pc.lanes == 0)
	const uint64_t inc = 1ul << (16u * (k & 3u));
	const uint wi = k >> 2u;
	c.w0 += wi == 0u ? inc : 0ul;
	c.w1 += wi == 1u ? inc : 0ul;
	c.w2 += wi == 2u ? inc : 0ul;
	if (!small) {
		c.w3 += wi == 3u ? inc : 0ul;
		c.w4 += wi == 4u ? inc : 0ul;
	}
}
void window_tiled_plain(inout Counts cnt, int base, int tw, int pad, uint off, uint span, const bool small) {
	for (int dr = 0; dr <= 2 * pad; dr++) {
		const int row = base + dr * tw;
		for (int dc = 0; dc <= 2 * pad; dc++) {
			const uint k = uint(tile[row + dc]) + off;
			if (k <= span) count(cnt, k, small);
		}
	}
}
void window_direct_plain(inout Counts cnt, int i, int j, int pad, uint off, uint span, const bool small) {
	for (int dr = -pad; dr <= pad; dr++)
		for (int dc = -pad; dc <= pad; dc++) {
			const uint k = uint(counted(i + dr, j + dc)) + off;
			if (k <= span) count(cnt, k, small);
		}
}

#ifdef CAMBI_COLS
// The window's counts from the column counts: the 33 or fewer columns' words for the values a .. a + 2 nd (a = v - nd - base), summed in 8 bit
// lanes over up to 7 columns at a time (7 x 33 < 256), then into the 16 bit lanes; get_count() then takes k + (a & 3).
void window_cols(inout Counts cnt, int ly, int lx, int pad, int a, const bool small) {
	const int w0 = a >> 2, plane = TILE * CW;
	int base = w0 * plane + ly * CW + lx;
	Row r = Row(0u, 0u, 0u, 0u, 0u);
	int n = 0;
	for (int dc = 0; dc <= 2 * pad; dc++) {
		const int o = base + dc;
		r.r0 += colc[o];
		r.r1 += colc[o + plane];
		r.r2 += colc[o + 2 * plane];
		if (!small) { r.r3 += colc[o + 3 * plane]; r.r4 += colc[o + 4 * plane]; }
		if (++n == 7) { add_row(cnt, r, small); r = Row(0u, 0u, 0u, 0u, 0u); n = 0; }
	}
	add_row(cnt, r, small);
}

#endif

int get_count(Counts c, int k) {
	const int wi = k >> 2;
	const uint64_t w = wi == 0 ? c.w0 : wi == 1 ? c.w1 : wi == 2 ? c.w2 : wi == 3 ? c.w3 : c.w4;
	return int((w >> uint64_t(16 * (k & 3))) & 65535ul);
}

void main() {
	const int j = int(gl_GlobalInvocationID.x), i = int(gl_GlobalInvocationID.y);
	const int pad = pc.pad, nd = pc.nd;
	const bool sums_only = pc.OW < 0;
	const bool tiled = pad <= MAXPAD && !sums_only;
	const int tw = TILE + 2 * pad;
	if (tiled) {
		const int r0 = int(gl_WorkGroupID.y) * TILE - pad, c0 = int(gl_WorkGroupID.x) * TILE - pad;
		for (int k = int(gl_LocalInvocationIndex); k < tw * tw; k += 256) tile[k] = counted(r0 + k / tw, c0 + k % tw);
		barrier();
	}
	bool cols = false;
	int cbase = 0;
#ifdef CAMBI_COLS
	// Column counts when the centres' values span little: then each window is 2 pad + 1 column sums instead of (2 pad + 1)^2 samples.
	if (tiled) {
		if (gl_LocalInvocationIndex == 0u) { s_lo = 1 << 30; s_hi = -(1 << 30); }
		barrier();
		const int cv0 = tile[(int(gl_LocalInvocationID.y) + pad) * tw + int(gl_LocalInvocationID.x) + pad];
		if (j < pc.W && i < pc.H && cv0 != NOT_COUNTED) { atomicMin(s_lo, cv0); atomicMax(s_hi, cv0); }
		barrier();
		const int lo = s_lo, hi = s_hi;
		cbase = lo - nd;
		const int R = hi - lo + 2 * nd + 1;
		cols = lo <= hi && R <= RMAX;
		if (cols) {
			// one invocation per column and run of 4 centre rows: the first row's window counted, then slid down a row at a time
			for (int e = int(gl_LocalInvocationIndex); e < 4 * tw; e += 256) {
				const int x = e % tw, ly0 = (e / tw) * 4;
				uint w[8] = uint[8](0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u);
				for (int dr = 0; dr <= 2 * pad; dr++) {
					const uint k = uint(tile[(ly0 + dr) * tw + x] - cbase);
					if (k < uint(R)) {
						const uint inc = 1u << (8u * (k & 3u)), wi = k >> 2u;
						for (uint q = 0u; q < 8u; q++) w[q] += wi == q ? inc : 0u;
					}
				}
				for (int q = 0; q < 8; q++) colc[q * TILE * CW + ly0 * CW + x] = w[q];
				for (int ly = ly0 + 1; ly < ly0 + 4; ly++) {
					const uint ko = uint(tile[(ly - 1) * tw + x] - cbase), ki = uint(tile[(ly + 2 * pad) * tw + x] - cbase);
					if (ko < uint(R)) {
						const uint dec = 1u << (8u * (ko & 3u)), wi = ko >> 2u;
						for (uint q = 0u; q < 8u; q++) w[q] -= wi == q ? dec : 0u;
					}
					if (ki < uint(R)) {
						const uint inc = 1u << (8u * (ki & 3u)), wi = ki >> 2u;
						for (uint q = 0u; q < 8u; q++) w[q] += wi == q ? inc : 0u;
					}
					for (int q = 0; q < 8; q++) colc[q * TILE * CW + ly * CW + x] = w[q];
				}
			}
			barrier();
		}
	}
#endif
	float c_value = 0.0;
	if (sums_only) {
		if (j < pc.W && i < pc.H) c_value = cval[i * pc.W + j];
	} else if (j < pc.W && i < pc.H && mask[i * pc.W + j] != 0u) {
		const int v = int(img[i * pc.W + j]);
		const int compact = v - pc.band_base;
		if (compact >= 0 && compact < pc.band_size) {
			Counts cnt = Counts(0ul, 0ul, 0ul, 0ul, 0ul);
			const uint span = uint(2 * nd), off = uint(nd - v);
			int sh = 0; // where count k is: k + sh
#ifdef CAMBI_COLS
			if (cols) {
				const int a = v - nd - cbase;
				sh = a & 3;
				if (nd <= 4) window_cols(cnt, int(gl_LocalInvocationID.y), int(gl_LocalInvocationID.x), pad, a, true);
				else window_cols(cnt, int(gl_LocalInvocationID.y), int(gl_LocalInvocationID.x), pad, a, false);
			} else
#endif
			if (tiled) {
				const int base = int(gl_LocalInvocationID.y) * tw + int(gl_LocalInvocationID.x);
				if ((pc.lanes & 1) != 0) {
					if (nd <= 4) window_tiled(cnt, base, tw, pad, off, span, true);
					else window_tiled(cnt, base, tw, pad, off, span, false);
				} else {
					if (nd <= 4) window_tiled_plain(cnt, base, tw, pad, off, span, true);
					else window_tiled_plain(cnt, base, tw, pad, off, span, false);
				}
			} else if ((pc.lanes & 1) != 0) {
				if (nd <= 4) window_direct(cnt, i, j, pad, off, span, true);
				else window_direct(cnt, i, j, pad, off, span, false);
			} else {
				if (nd <= 4) window_direct_plain(cnt, i, j, pad, off, span, true);
				else window_direct_plain(cnt, i, j, pad, off, span, false);
			}
			const int p0 = get_count(cnt, nd + sh);
			const int value = v + nd; // as cambi.c has it: tvi[] is in the same offset space
			for (int d = 0; d < nd; d++) {
				if (value <= tb.tvi[d] && value + d + 1 > pc.vlt_luma) {
					const int p1 = get_count(cnt, nd + d + 1 + sh);
					const int p2 = compact - d - 1 >= 0 ? get_count(cnt, nd - d - 1 + sh) : 0;
					precise float val;
					if (p1 > p2) val = float(tb.weight[d] * p0 * p1) * tb.recip[p1 + p0];
					else val = float(tb.weight[d] * p0 * p2) * tb.recip[p2 + p0];
					if (val > c_value) c_value = val;
				}
			}
		}
	}
	if (j < pc.W && i < pc.H && !sums_only) cval[i * pc.W + j] = c_value;
	const uint lid = gl_LocalInvocationIndex;
	ssum[lid] = double(c_value);
	scnt[lid] = c_value != 0.0 ? 1u : 0u;
	barrier();
	for (uint s = 128u; s > 0u; s >>= 1) {
		if (lid < s) { ssum[lid] += ssum[lid + s]; scnt[lid] += scnt[lid + s]; }
		barrier();
	}
	if (lid == 0u) {
		const uint g = gl_WorkGroupID.y * gl_NumWorkGroups.x + gl_WorkGroupID.x;
		part[2 * g] = double(scnt[0]);
		part[2 * g + 1] = ssum[0];
	}
}
