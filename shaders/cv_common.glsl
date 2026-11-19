// Shared by the converter's shaders (cv_*.comp, vk_convert.cpp): the transfer curves, numbered as in ITU-T H.273, between the coded value V
// (0..1, as stored) and linear light L (0..1 relative to the curve's nominal peak; PQ: 1 = 10000 cd/m2). Negative values (ringing of a scaler)
// are mirrored, so a curve never turns them into NaN.
// `t` is the H.273 number, after vk_convert.cpp's choices: unknown is never passed in, and 1/6/14/15 are passed as 1 (exact BT.709 camera curve)
// or as 1886 (display gamma 2.4, BT.1886 with a black level of 0).
const float BT709_A = 1.09929682680944, BT709_B = 0.018053968510807;

float to_linear_pos(int t, float v) {
	if (t == 1) return v < 4.5 * BT709_B ? v / 4.5 : pow((v + (BT709_A - 1.0)) / BT709_A, 1.0 / 0.45);
	if (t == 1886) return pow(v, 2.4);
	if (t == 4) return pow(v, 2.2);
	if (t == 5) return pow(v, 2.8);
	if (t == 7) return v < 4.0 * 0.0228 ? v / 4.0 : pow((v + 0.1115) / 1.1115, 1.0 / 0.45);
	if (t == 9) return v <= 0.0 ? 0.0 : pow(10.0, 2.0 * (v - 1.0));
	if (t == 10) return v <= 0.0 ? 0.0 : pow(10.0, 2.5 * (v - 1.0));
	if (t == 13) return v <= 0.04045 ? v / 12.92 : pow((v + 0.055) / 1.055, 2.4);
	if (t == 16) {
		const float m1 = 2610.0 / 16384.0, m2 = 2523.0 / 4096.0 * 128.0, c1 = 3424.0 / 4096.0, c2 = 2413.0 / 4096.0 * 32.0, c3 = 2392.0 / 4096.0 * 32.0;
		const float p = pow(v, 1.0 / m2);
		return pow(max(p - c1, 0.0) / (c2 - c3 * p), 1.0 / m1);
	}
	if (t == 17) return pow(v, 2.6) * (52.37 / 48.0);
	if (t == 18) {
		const float a = 0.17883277, b = 1.0 - 4.0 * 0.17883277, c = 0.5 - 0.17883277 * log(4.0 * 0.17883277);
		return v <= 0.5 ? v * v / 3.0 : (exp((v - c) / a) + b) / 12.0;
	}
	return v; // 8: linear
}

float from_linear_pos(int t, float l) {
	if (t == 1) return l < BT709_B ? 4.5 * l : BT709_A * pow(l, 0.45) - (BT709_A - 1.0);
	if (t == 1886) return pow(l, 1.0 / 2.4);
	if (t == 4) return pow(l, 1.0 / 2.2);
	if (t == 5) return pow(l, 1.0 / 2.8);
	if (t == 7) return l < 0.0228 ? 4.0 * l : 1.1115 * pow(l, 0.45) - 0.1115;
	if (t == 9) return l < 0.01 ? 0.0 : 1.0 + log(l) / (2.0 * log(10.0));
	if (t == 10) return l < sqrt(10.0) / 1000.0 ? 0.0 : 1.0 + log(l) / (2.5 * log(10.0));
	if (t == 13) return l <= 0.0031308 ? 12.92 * l : 1.055 * pow(l, 1.0 / 2.4) - 0.055;
	if (t == 16) {
		const float m1 = 2610.0 / 16384.0, m2 = 2523.0 / 4096.0 * 128.0, c1 = 3424.0 / 4096.0, c2 = 2413.0 / 4096.0 * 32.0, c3 = 2392.0 / 4096.0 * 32.0;
		const float p = pow(l, m1);
		return pow((c1 + c2 * p) / (1.0 + c3 * p), m2);
	}
	if (t == 17) return pow(l * (48.0 / 52.37), 1.0 / 2.6);
	if (t == 18) {
		const float a = 0.17883277, b = 1.0 - 4.0 * 0.17883277, c = 0.5 - 0.17883277 * log(4.0 * 0.17883277);
		return l <= 1.0 / 12.0 ? sqrt(3.0 * l) : a * log(12.0 * l - b) + c;
	}
	return l;
}

float to_linear(int t, float v) { return v < 0.0 ? -to_linear_pos(t, -v) : to_linear_pos(t, v); }
float from_linear(int t, float l) { return l < 0.0 ? -from_linear_pos(t, -l) : from_linear_pos(t, l); }
