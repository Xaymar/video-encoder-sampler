#pragma once
#include <cstdint>
#include <functional>
#include <string>
#include <vector>

#include "convert.h"
#include "ui.h"
#include "picture.h"
#include "source.h"

struct VmafOptions {
	std::string model = "path=models/vmaf_4k_v0.6.1.json"; // "path=<json model>", relative to the working directory, or "version=<builtin>" if libvmaf has it
	int threads = 0; // 0: all cores, shared between the scorers that run at the same time
};

// What scoring needs to know about the original video; its frames come from a feed.
struct RefInfo {
	int width = 0, height = 0;
	long long frames = 0; // expected number of original frames, for the progress display only
	Rational framerate;
	Color color;
};
using FrameFeed = std::function<int(Picture&)>; // 1: a frame, 0: no more, <0: error
// shown(k, n): the encoded frame that is on screen at original frame k; waits until that is known, false once k is past the last frame.
using ShownFeed = std::function<bool(size_t k, uint32_t& n)>;
using PacketFeed = std::function<bool(std::vector<uint8_t>&)>; // the next encoded packet in coding order; false at the end

// Decodes the encoded packets, scales them back to the original resolution, holds frames up to the original
// frame rate and scores them against the original frames (delivered by `next_ref`, already in `score_fmt`) with libvmaf, in-process.
// `enc` is the geometry and pixel format the packets were encoded at; `shown(k)` is the number of the encoded frame
// that is on screen at original frame k (dropped frames are held). Decoded frames are converted to `score_fmt` as well.
// `scores` receives one VMAF value per original frame.
bool vmaf_score(const RefInfo& ref, const FrameFeed& next_ref, Codec codec, const PacketFeed& next_packet, const std::vector<uint8_t>& extradata, const Shape& enc,
                const ShownFeed& shown, Fmt score_fmt, const VmafOptions& opt, const ScaleOptions& scale, std::vector<double>& scores, RunProgress* rp = nullptr);
