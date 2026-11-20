#pragma once
// Translation between the tool's own types and FFmpeg's. Only the ff_*.cpp files and the libavcodec encoder use this.
extern "C" {
#include <libavcodec/avcodec.h>
#include <libavutil/frame.h>
#include <libavutil/pixdesc.h>
#include <libavutil/pixfmt.h>
#include <libavutil/rational.h>
}

#include <string>
#include <vector>

#include "picture.h"

inline AVRational to_av(Rational r) { return {r.num, r.den}; }
inline Rational from_av(AVRational r) { return {r.num, r.den}; }

// Describes an FFmpeg pixel format; false for formats that cannot be used (gray, palettes, hardware, > 12 bit).
bool describe_native(AVPixelFormat f, Fmt& out);

// The planar YUV format for a chroma layout and bit depth, AV_PIX_FMT_NONE if there is none.
AVPixelFormat av_planar_format(int chroma, int depth);

// The sample layout of an FFmpeg pixel format.
Layout layout_of(AVPixelFormat f);

// A software decoder for the codec, never a hardware or hybrid one (those are less exact); nullptr if there is none.
const AVCodec* software_decoder(AVCodecID id);

// The decoder that reconstructs on a hardware device of `type` ("cuda", "d3d11va", ...) for the codec: FFmpeg's own one, which for AV1 is not the
// software decoder (libdav1d has no hardware support); nullptr if there is none.
const AVCodec* hwaccel_decoder(AVCodecID id, const std::string& type);

// Threads for a decoder (see set_decode_threads()).
int decode_threads();

// Moves the frame's contents into a Picture that owns them.
Picture picture_from_frame(AVFrame* frame, const Fmt& fmt);

// Semi-planar formats (NV12, P010, ...) keep U and V interleaved in one plane; their planar twin, or nullptr for other formats.
struct SemiPlanar { AVPixelFormat from, to; int sx, sy; bool high; };
const SemiPlanar* semi_planar(AVPixelFormat f);
// Returns `f` itself for planar formats, else `planar` filled with the samples unpacked to planar YUV, exactly.
AVFrame* unpack_semi_planar(AVFrame* f, AVFrame* planar);

// Whether two pictures hold the same samples, whatever the memory layout (NV12 against planar, say).
bool same_pictures(const Picture& a, const Picture& b);
// Says on stderr how two pictures that are not the same differ (`a` is the software decode, `b` the hardware one); VES_DUMP_FRAME=<folder> also writes them out.
void describe_difference(const Picture& a, const Picture& b, int frame);

// Hardware decoding for streams that come as packets (the encoded streams that are scored): the device types to try for the codec, in order, with the
// device in that type's terms, as --hw-decode and the decode GPU (set_hw_decode(), set_hw_decode_gpu()) say. Empty: software only.
// The home GPU's choices come first (rank 0, Vulkan first among them), then those of the other GPUs, fastest first (rank 1, 2, ...): a kind of stream
// the home GPU decodes wrongly is decoded on the next GPU that gets it right, and in software only when none does. `gpu`: that GPU with its driver
// (as hw_decode_gpu_identity() says it), for the verdict keys; empty for a type's default device.
struct HwChoice { std::string type, device, gpu; int rank = 0; };
std::vector<HwChoice> hw_decode_choices(AVCodecID id);
// The decode (home) GPU with its driver ("<name> (driver <version>)"), for keys of what was found out about it.
std::string hw_decode_gpu_identity();
// The hardware device of a type, made once per run and shared by every decoder; nullptr if it cannot be made. The caller gets a reference of its own.
AVBufferRef* shared_hw_device(const std::string& type, const std::string& device);

#ifndef VES_NO_VULKAN
#include "vk_compute.h"
// An FFmpeg Vulkan frame, decoded on Vulkan device `device` (FFmpeg's device that the scorer shares), described for reading it in place; false if it
// cannot be (another device, a queue family the scorer does not use, a layout it does not know).
bool vulkan_frame(const AVFrame* f, int device, VkDecodedFrame& out);
// A Vulkan frame of FFmpeg's device `device` as a Picture that stays on the GPU (Picture::gpu, which holds the frame): its size, format (the planar
// twin of the frame's own) and color, no samples. for_scorer: false unless the scorer can read it in place (vulkan_frame()); otherwise it can still be
// converted in place (pl_convert.cpp) or read back (picture_to_memory()).
bool vulkan_picture(const AVFrame* f, int device, bool for_scorer, Picture& out);
// The FFmpeg frame behind a picture that is on the GPU, nullptr for one in memory.
const AVFrame* gpu_avframe(const Picture& p);
// FFmpeg's Vulkan device `index` (as Vk::devices() numbers them) if it is the GPU with these PCI ids, else nullptr; the caller gets a reference of its own.
AVBufferRef* ffmpeg_vulkan_device(int index, uint32_t vendor_id, uint32_t device_id);
// For tests without a GPU decoder: a copy of software frame `f` (yuv420p or yuv420p10) in a new Vulkan frame on FFmpeg's device `device`, interleaved
// to NV12/P010 if `nv12`. `frames` is the frames context made on the first call (the caller unrefs it). nullptr if it cannot be made.
AVFrame* vulkan_upload(const AVFrame* f, int device, bool nv12, AVBufferRef*& frames);
#endif
