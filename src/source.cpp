#include "source.h"

double Source::duration() const {
	const double unit = tb.value();
	if (stamps.size() < 2) return read / framerate.value();
	// the last frame lasts as long as the average one
	return (stamps.back() + (double)stamps.back() / (stamps.size() - 1)) * unit;
}

int64_t Source::slot(long long index, Rational rate) const {
	if (index >= (long long)stamps.size()) return index * rate.num * framerate.den / ((int64_t)rate.den * framerate.num);
	// Timestamps are rounded to whole ticks, which can leave a frame just below its slot boundary (33 ms for 33.3 ms at
	// 30 fps); one tick of slack puts such frames in the slot they belong to.
	return (stamps[index] + 1) * tb.num * rate.num / ((int64_t)tb.den * rate.den);
}
