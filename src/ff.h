#pragma once
// Translation between the tool's own types and FFmpeg's. Only the ff_*.cpp files and the libavcodec encoder use this.
extern "C" {
#include <libavcodec/avcodec.h>
#include <libavutil/frame.h>
#include <libavutil/pixdesc.h>
#include <libavutil/pixfmt.h>
#include <libavutil/rational.h>
}

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

// Moves the frame's contents into a Picture that owns them.
Picture picture_from_frame(AVFrame* frame, const Fmt& fmt);

