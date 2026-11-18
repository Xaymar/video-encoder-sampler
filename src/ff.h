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
struct HwChoice { std::string type, device; };
std::vector<HwChoice> hw_decode_choices(AVCodecID id);
// The hardware device of a type, made once per run and shared by every decoder; nullptr if it cannot be made. The caller gets a reference of its own.
AVBufferRef* shared_hw_device(const std::string& type, const std::string& device);
