// SpEED's statistics on the GPU (speed_blur_v/h, speed_mean, speed_center, speed_cov, speed_small, speed_blocks, speed_score): libvmaf 3.2.1 speed.c's est_params() and
// get_speed_score() after the decimation by 16 (speed_dec), for the 4 decimated planes of one SpEED unit (0, 1: the reference's U and V; 2, 3: the
// decoded U and V). They redo the C code's float and double operations one by one, in its order: `precise` keeps the compiler from fusing a multiply
// and an add (that rounds once instead of twice), division and square root go through double (Vulkan does not round float division correctly; a
// float quotient of floats rounded from a double one is the correctly rounded one), and log2 in double is computed here (GLSL has none for double).
layout(std430, binding = 0) readonly buffer Dec { float dec[]; };         // speed_dec's output: plane p at p * DW * DH
layout(std430, binding = 1) buffer Tmp { float tmpv[]; };                 // the 3x3 blur's vertical pass
layout(std430, binding = 2) buffer Prep { float prep[]; };                // plane minus its blur: what the statistics are taken of
layout(std430, binding = 3) readonly buffer Filt { float filt[]; };       // the blur's taps
layout(std430, binding = 4) buffer Cov { float covm[]; };                 // per plane: the 25x25 covariance matrix; after the 4 planes', their 25 means
layout(std430, binding = 5) buffer Small { float small[]; };              // per plane (SMALL floats): Q (625), R (625), eigenvalues (25), cannot invert (1)
layout(std430, binding = 6) buffer Blocks { float blk[]; };               // per plane: entropies (nb), then variances (nb)
layout(std430, binding = 7) buffer Final { float fin[]; };                // per chroma plane (U, V): score, failed (1: a matrix could not be inverted)
layout(std430, binding = 8) buffer Cen { double cen[]; };                 // per plane, per element of a block: the submatrix's samples minus their mean
layout(push_constant) uniform PC {
	int DW, DH;          // the decimated plane
	int fw;              // blur taps
	int subw, subh;      // the submatrix the covariances are taken over (truncated size - 4)
	int nbh, nbv, nb;    // 5x5 blocks across, down, in all
	int mode;            // speed_weight_var_mode
	float sigma_nn, nn_floor, base_entropy;
	float one;           // 1 (see rf())
	double log2_2pie;    // log2(2 pi e), as the CPU works it out
} pc;

const int N = 25;     // elements in a 5x5 block
const int SMALL = 1280;

// is_matrix_regular() and solve_triangular_system()'s test, from speed_small's eigenvalues and R of the plane at `base` in `small`: whether its
// covariance matrix cannot be inverted (an eigenvalue or a diagonal element of R below 1e-6).
bool cannot_invert(int base) {
	for (int i = 0; i < N; i++) if (double(small[base + 2 * N * N + i]) < 1e-6LF) return true;
	for (int i = N - 1; i >= 0; i--) if (double(abs(small[base + N * N + i * N + i])) < 1e-6LF) return true;
	return false;
}

// Vulkan only asks double division and square root to be about as exact as float ones (llvmpipe's are off by 1e-8): these are correctly rounded,
// from adds and multiplies (which are), as the CPU's. Exact products by Dekker's splitting; an approximation refined twice by Newton's method on
// exact residuals, then of it and its neighbours the one closest to the true value.
void two_prod(double a, double b, out double p, out double e) { // a * b = p + e exactly
	precise double pp = a * b;
	precise double ca = 134217729.0LF * a;
	precise double ahi = ca - (ca - a);
	precise double alo = a - ahi;
	precise double cb = 134217729.0LF * b;
	precise double bhi = cb - (cb - b);
	precise double blo = b - bhi;
	precise double ee = ((ahi * bhi - pp) + ahi * blo + alo * bhi) + alo * blo;
	p = pp;
	e = ee;
}
double nudge(double x, bool up) { // the next double above (up) or below x; x finite, not 0
	uvec2 b = unpackDouble2x32(x); // .x low word, .y high word (sign, exponent, high mantissa)
	const bool grow = (x > 0.0LF) == up; // away from 0
	if (grow) { b.x += 1u; if (b.x == 0u) b.y += 1u; }
	else { if (b.x == 0u) b.y -= 1u; b.x -= 1u; }
	return packDouble2x32(b);
}
double div_residual(double a, double b, double q) { // a - q b
	double p, e;
	two_prod(q, b, p, e);
	precise double r = (a - p) - e;
	return r;
}
// |a - q b| only grows away from the true quotient, so a candidate whose neighbours have larger residuals is the closest: when the GPU's own
// quotient (or root) already is, nothing more is needed (most GPUs round these correctly; llvmpipe does not).
double ddiv(double a, double b) {
	if (a == 0.0LF || b == 0.0LF || isinf(a) || isinf(b) || isnan(a) || isnan(b)) return a / b;
	precise double q = a / b;
	for (int pass = 0; pass < 2; pass++) {
		const double bd = abs(div_residual(a, b, q));
		const double lo = nudge(q, false), hi = nudge(q, true);
		const double dl = abs(div_residual(a, b, lo)), dh = abs(div_residual(a, b, hi));
		if (bd <= dl && bd <= dh) return q;
		if (pass == 1) return dl < dh ? lo : hi;
		for (int i = 0; i < 2; i++) { precise double c = q + div_residual(a, b, q) / b; q = c; }
	}
	return q;
}
double sqrt_residual(double x, double s) { // x - s s
	double p, e;
	two_prod(s, s, p, e);
	precise double r = (x - p) - e;
	return r;
}
double dsqrt(double x) {
	if (!(x > 0.0LF) || isinf(x)) return sqrt(x);
	precise double s = sqrt(x);
	// x - c^2 = (sqrt x - c)(sqrt x + c): the neighbour with the smallest residual is the closest (no square root of a double lies on a midpoint)
	for (int pass = 0; pass < 2; pass++) {
		const double bd = abs(sqrt_residual(x, s));
		const double lo = nudge(s, false), hi = nudge(s, true);
		const double dl = abs(sqrt_residual(x, lo)), dh = abs(sqrt_residual(x, hi));
		if (bd <= dl && bd <= dh) return s;
		if (pass == 1) return dl < dh ? lo : hi;
		for (int i = 0; i < 2; i++) { precise double c = s + sqrt_residual(x, s) / (2.0LF * s); s = c; }
	}
	return s;
}
// A double rounded to float, as C's (float) does. Not just float(d): compilers (Mesa's NIR, for one) take double(float(d)) for d, as if it had not
// been rounded, and a float that came from a double is often taken back to double here; a multiply by 1 they cannot see through keeps it rounded.
float rf(double d) { precise float f = float(d) * pc.one; return f; }
// C's float division (the double quotient of two floats rounds to the correctly rounded float) and (float)sqrt((double)x): the correctly rounded
// float. Fast way: the GPU's float result, moved to the float whose rounding interval holds the true value. The midpoints between floats have 25
// bits, so b times one (or one squared) is exact in double and the comparisons are exact; a true value never lies on a midpoint. Only for normal
// numbers (a GPU may flush the others): else the double way.
const float FLT_NORMAL = 1.17549435e-38, FLT_BIG = 1.0e38;
bool normal_f(float x) { const float m = abs(x); return m >= FLT_NORMAL && m <= FLT_BIG; }
// far enough from the ends of the normal floats that the few steps to a neighbour stay normal (split_f needs that)
bool inner_f(float x) { const float m = abs(x); return m >= 1.0e-37 && m <= 1.0e37; }
float up_f(float x) { return uintBitsToFloat(floatBitsToUint(x) + 1u); }   // the next float above x > 0
float down_f(float x) { return uintBitsToFloat(floatBitsToUint(x) - 1u); } // the next float below x > 0
// The comparisons below are exact, as with doubles (the products have at most 50 bits), but in integers: consumer GPUs run doubles at 1/64 the
// float rate, and these sit on the serial path of speed_small's eigenvalue iterations. x = m 2^e for a normal float x > 0 (m: 24 bits).
void split_f(float x, out uint m, out int e) { const uint b = floatBitsToUint(x); m = (b & 0x7fffffu) | 0x800000u; e = int((b >> 23) & 255u) - 150; }
// The midpoint of neighbouring normal floats x, y > 0 as S 2^E (S: at most 26 bits).
void mid_f(float x, float y, out uint S, out int E) {
	uint mx, my; int ex, ey;
	split_f(x, mx, ex); split_f(y, my, ey);
	const int e = min(ex, ey);
	S = (mx << uint(ex - e)) + (my << uint(ey - e));
	E = e - 1;
}
int msb64(uint64_t v) { const uint hi = uint(v >> 32); return hi != 0u ? 32 + findMSB(hi) : findMSB(uint(v)); }
// sign of X 2^x - Y 2^y, for X, Y > 0
int cmp_x2(uint64_t X, int x, uint64_t Y, int y) {
	const int sx = 63 - msb64(X), sy = 63 - msb64(Y);
	const int ex = x - sx, ey = y - sy;
	if (ex != ey) return ex > ey ? 1 : -1;
	const uint64_t nx = X << uint(sx), ny = Y << uint(sy);
	return nx == ny ? 0 : (nx > ny ? 1 : -1);
}
// sign of a - b m for positive normal floats a, b and the midpoint m = S 2^E
int cmp_div_mid(uint Ma, int ea, uint Mb, int eb, float q0, float q1) {
	uint S; int E;
	mid_f(q0, q1, S, E);
	return cmp_x2(uint64_t(Ma), ea, uint64_t(Mb) * uint64_t(S), eb + E);
}
float fdiv(float a, float b) {
	const float aa = abs(a), ab = abs(b);
	float q = aa / ab;
	if (!normal_f(a) || !normal_f(b) || !inner_f(q)) return rf(ddiv(double(a), double(b)));
	uint Ma, Mb; int ea, eb;
	split_f(aa, Ma, ea); split_f(ab, Mb, eb);
	int i = 0;
	for (; i < 8; i++) { // q too small while the true quotient is at or above the midpoint to the next float
		if (cmp_div_mid(Ma, ea, Mb, eb, q, up_f(q)) >= 0) q = up_f(q); else break;
	}
	int j = 0;
	for (; j < 8; j++) { // q too big while it is below the midpoint to the float before
		if (cmp_div_mid(Ma, ea, Mb, eb, q, down_f(q)) < 0) q = down_f(q); else break;
	}
	if (i == 8 || j == 8 || !normal_f(q)) return rf(ddiv(double(a), double(b))); // the GPU's quotient was far off
	return (a < 0.0) != (b < 0.0) ? -q : q;
}
// sign of x - m^2 for a positive normal float x and the midpoint m = S 2^E
int cmp_sq_mid(uint Mx, int ex, float s0, float s1) {
	uint S; int E;
	mid_f(s0, s1, S, E);
	return cmp_x2(uint64_t(Mx), ex, uint64_t(S) * uint64_t(S), 2 * E);
}
float dsqrt_f(float x) {
	float s = sqrt(x);
	if (!normal_f(x) || x < 0.0 || !inner_f(s)) return rf(dsqrt(double(x)));
	uint Mx; int ex;
	split_f(x, Mx, ex);
	int i = 0;
	for (; i < 8; i++) {
		if (cmp_sq_mid(Mx, ex, s, up_f(s)) >= 0) s = up_f(s); else break;
	}
	int j = 0;
	for (; j < 8; j++) {
		if (cmp_sq_mid(Mx, ex, s, down_f(s)) < 0) s = down_f(s); else break;
	}
	if (i == 8 || j == 8 || !normal_f(s)) return rf(dsqrt(double(x)));
	return s;
}
// (float)(1.0 / sqrt((double)q)), as the CPU's Givens rotation has it, without the double division and square root: the GPU's inverse square root,
// moved to the float whose rounding interval holds the true value 1/sqrt(q). For a midpoint m between two floats, 1/sqrt(q) >= m exactly when
// m^2 q <= 1; m has at most 26 bits and q 24, so m^2 q is an integer of at most 76 bits times a power of two, compared with 1 exactly. The CPU
// rounds twice (the root, then the quotient, to double) before rounding to float, which can only matter when the true value is within a few double
// units of a midpoint: there (|m^2 q - 1| below 2^-46), and for numbers that are not normal, the double way.
// Returns the sign of m^2 q - 1, or 2 when it is within 2^-46 of 0 (or out of the range handled here).
int rsqrt_side(uint Mq, int eq, float y0, float y1) {
	uint S; int E;
	mid_f(y0, y1, S, E);
	const uint64_t s2 = uint64_t(S) * uint64_t(S);                 // < 2^52
	const uint64_t lo = (s2 & 0xffffffffUL) * uint64_t(Mq);      // < 2^56
	const uint64_t hi = (s2 >> 32) * uint64_t(Mq) + (lo >> 32);  // < 2^45: m^2 q = (hi 2^32 + lo32) 2^K
	const uint lo32 = uint(lo);
	const int T = -(2 * E + eq);                                  // 1 = 2^T 2^K
	if (T < 32 || T > 94) return 2;
	const int64_t dh = int64_t(hi) - int64_t(1UL << uint(T - 32)); // m^2 q - 1 = (dh 2^32 + lo32) 2^K
	if (dh >= 2147483648L || dh < -2147483648L) return dh > 0 ? 1 : -1; // |D| >= 2^63 2^K, far from 2^-46 (T - 46 < 63)
	const int64_t D = dh * 4294967296L + int64_t(lo32);
	if (D == 0L) return 2;
	const uint64_t ad = uint64_t(D < 0L ? -D : D);
	if (T - 46 >= 0 && ad < (1UL << uint(T - 46))) return 2;
	return D > 0L ? 1 : -1;
}
float rsqrt_f(float q) {
	float y = inversesqrt(q);
	if (!normal_f(q) || q < 0.0 || !inner_f(y)) return rf(ddiv(1.0LF, dsqrt(double(q))));
	uint Mq; int eq;
	split_f(q, Mq, eq);
	int i = 0;
	for (; i < 8; i++) { // y too small while the true value is at or above the midpoint to the next float
		const int r = rsqrt_side(Mq, eq, y, up_f(y));
		if (r == 2) return rf(ddiv(1.0LF, dsqrt(double(q))));
		if (r <= 0) y = up_f(y); else break;
	}
	int j = 0;
	for (; j < 8; j++) { // y too big while the true value is below the midpoint to the float before
		const int r = rsqrt_side(Mq, eq, y, down_f(y));
		if (r == 2) return rf(ddiv(1.0LF, dsqrt(double(q))));
		if (r > 0) y = down_f(y); else break;
	}
	if (i == 8 || j == 8 || !normal_f(y)) return rf(ddiv(1.0LF, dsqrt(double(q))));
	return y;
}
int sgn(float x) { return x >= 0.0 ? 1 : -1; }
float pyth(float x, float y) { precise float s = x * x + y * y; return dsqrt_f(s); }

// log2 of a double to within a few units in the last place: x = m 2^e with m in [sqrt(1/2), sqrt(2)), ln m = 2 atanh((m - 1) / (m + 1)).
double log2d(double x) {
	if (isnan(x) || x < 0.0LF) return packDouble2x32(uvec2(0u, 0x7ff80000u));
	if (x == 0.0LF) return packDouble2x32(uvec2(0u, 0xfff00000u));
	if (isinf(x)) return x;
	int e;
	precise double m = frexp(x, e);
	if (m < 0.70710678118654757LF) { m = m * 2.0LF; e = e - 1; }
	precise double f = m - 1.0LF;
	precise double s = ddiv(f, 2.0LF + f);
	precise double s2 = s * s;
	precise double p = 0.043478260869565216LF;
	p = p * s2 + 0.047619047619047616LF;
	p = p * s2 + 0.052631578947368418LF;
	p = p * s2 + 0.058823529411764705LF;
	p = p * s2 + 0.066666666666666666LF;
	p = p * s2 + 0.076923076923076927LF;
	p = p * s2 + 0.090909090909090912LF;
	p = p * s2 + 0.1111111111111111LF;
	p = p * s2 + 0.14285714285714285LF;
	p = p * s2 + 0.2LF;
	p = p * s2 + 0.33333333333333331LF;
	precise double ln = 2.0LF * s + 2.0LF * s * (s2 * p);
	precise double r = double(e) + ln * 1.4426950408889634LF;
	return r;
}
