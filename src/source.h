#pragma once
#include <memory>
#include <string>
#include <vector>

#include "picture.h"

// Sequential video input. Frames come out in display order with pts == frame number.
// The real presentation times come from a scan of the file's packet timestamps, so variable frame rate
// sources keep their timing: `framerate` is the measured average and `duration()` the real length.
class Source {
protected:
	std::vector<int64_t> stamps; // presentation time of every frame in display order, in `tb` ticks from the first frame
	Rational tb{1, 1000};
	long long read = 0;          // frames handed out so far

public:
	int width = 0, height = 0;
	Fmt fmt;                     // as delivered by next()
	Rational framerate{30, 1};   // constant frame rate files: the file's; variable frame rate files: the measured average
	bool vfr = false;            // the timestamps do not follow `framerate`
	Color color;                 // as tagged in the file

	virtual ~Source() = default;
	virtual bool open(const std::string& path) = 0;
	virtual int next(Picture& out) = 0; // 1: frame produced, 0: end of stream, <0: error

	long long frame_count() const { return (long long)stamps.size(); } // 0 if the file carries no usable timestamps
	double duration() const;                                            // seconds, including the last frame's own length
	// Which slot of a constant `rate` output timeline frame `index` falls in: floor(time * rate).
	int64_t slot(long long index, Rational rate) const;
};

std::unique_ptr<Source> open_source(const std::string& path); // nullptr if it cannot be opened

// Threads per decoder: 0 for as many as there are cores. Each decodes a frame of its own, which is what makes high bitrate sources fast.
void set_decode_threads(int threads);
