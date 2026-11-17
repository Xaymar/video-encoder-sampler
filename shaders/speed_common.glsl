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
double ddiv(double a, double b) {
	if (a == 0.0LF || b == 0.0LF || isinf(a) || isinf(b) || isnan(a) || isnan(b)) return a / b;
	precise double q = a / b;
	for (int i = 0; i < 2; i++) { precise double c = q + div_residual(a, b, q) / b; q = c; }
	double best = q, bd = abs(div_residual(a, b, q));
	const double lo = nudge(q, false), hi = nudge(q, true);
	const double dl = abs(div_residual(a, b, lo)), dh = abs(div_residual(a, b, hi));
	if (dl < bd) { best = lo; bd = dl; }
	if (dh < bd) best = hi;
	return best;
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
	for (int i = 0; i < 2; i++) { precise double c = s + sqrt_residual(x, s) / (2.0LF * s); s = c; }
	// x - c^2 = (sqrt x - c)(sqrt x + c): the neighbour with the smallest residual is the closest (no square root of a double lies on a midpoint)
	double best = s, bd = abs(sqrt_residual(x, s));
	const double lo = nudge(s, false), hi = nudge(s, true);
	const double dl = abs(sqrt_residual(x, lo)), dh = abs(sqrt_residual(x, hi));
	if (dl < bd) { best = lo; bd = dl; }
	if (dh < bd) best = hi;
	return best;
}
// A double rounded to float, as C's (float) does. Not just float(d): compilers (Mesa's NIR, for one) take double(float(d)) for d, as if it had not
// been rounded, and a float that came from a double is often taken back to double here; a multiply by 1 they cannot see through keeps it rounded.
float rf(double d) { precise float f = float(d) * pc.one; return f; }
// C's float division (the double quotient of two floats rounds to the correctly rounded float) and (float)sqrt((double)x)
float fdiv(float a, float b) { return rf(ddiv(double(a), double(b))); }
float dsqrt_f(float x) { return rf(dsqrt(double(x))); }
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
