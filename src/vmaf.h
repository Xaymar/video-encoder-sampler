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
	std::string model = "path=models/vmaf_v1.0.16_1d5h_2160.json"; // "path=<json model>", relative to the working directory, or "version=<builtin>" if libvmaf has it
	int threads = 0; // 0: all cores, shared between the scorers that run at the same time
	std::string device = "cpu"; // "cpu": libvmaf; "gpu": Vulkan compute (8 bit input, an error where it cannot be used); "auto": the GPU where it can be used, else the CPU
	std::string gpu;            // which Vulkan device scores (GpuVmaf::pick_device): "" or "auto" for the GPU that encodes, else the first discrete one; a Vulkan index; a name
	unsigned encoder_vendor = 0; // set per run: the vendor (gpu.h) and GPU index of the encoder, for gpu "auto"; 0 for a CPU encoder
	int encoder_gpu = 0;
};

// Whether libvmaf can score with this model: it loads and every feature it uses is there (VMAF v1's chroma feature needs libvmaf 3.2.1, or 3.2.0
// built with enable_float). `why` says what is missing.
bool vmaf_check_model(const std::string& model, std::string& why);
// The lowest bit depth runs are scored at with this model: 10 for VMAF v1 (its banding feature, CAMBI, works on 10 bit values, and Netflix
// scores v1 at 10 bit even for 8 bit encodes), else 8 (whatever the source and the encode have).
int vmaf_model_min_depth(const std::string& model);
// The model's name as results carry it ("vmaf_v1.0.16_1d5h_2160", the file name without .json) and its generation ("v0", "v1"; "" if unknown).
// v0 and v1 scores are on different scales.
std::string vmaf_model_display_name(const std::string& model);
std::string vmaf_model_generation(const std::string& model);

// What scoring needs to know about the original video; its frames come from a feed.
struct RefInfo {
	int width = 0, height = 0;
	long long frames = 0; // expected number of original frames, for the progress display only
	Rational framerate;
	Color color;
	int score_range = 1; // the range of the reference frames (1 limited, 2 full); the decoded stream is converted to it
};
using FrameFeed = std::function<int(Picture&)>; // 1: a frame, 0: no more, <0: error
// shown(k, n): the encoded frame that is on screen at original frame k; waits until that is known, false once k is past the last frame.
using ShownFeed = std::function<bool(size_t k, uint32_t& n)>;
using PacketFeed = std::function<bool(std::vector<uint8_t>&)>; // the next encoded packet in coding order; false at the end

// Decodes the encoded packets, scales them back to the original resolution, holds frames up to the original
// frame rate and scores them against the original frames (delivered by `next_ref`, already in `score_fmt`) with libvmaf, in-process.
// `enc` is the geometry and pixel format the packets were encoded at; `shown(k)` is the number of the encoded frame
// that is on screen at original frame k; `decoded_color` is the colour the encoder was given (the stream may not say) (dropped frames are held). Decoded frames are converted to `score_fmt` as well.
// `scores` receives one VMAF value per original frame.
bool vmaf_score(const RefInfo& ref, const FrameFeed& next_ref, Codec codec, const PacketFeed& next_packet, const std::vector<uint8_t>& extradata, const Shape& enc,
                const Color& decoded_color, const ShownFeed& shown, Fmt score_fmt, const VmafOptions& opt, const ScaleOptions& scale, std::vector<double>& scores, RunProgress* rp = nullptr);
