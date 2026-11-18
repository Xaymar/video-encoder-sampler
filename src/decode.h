#pragma once
#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>

#include "picture.h"

// Decodes the elementary streams our encoders produce. Frames come out in display order.
class Decoder {
public:
	virtual ~Decoder() = default;
	virtual bool open(Codec codec, const std::vector<uint8_t>& extradata = {}, const Shape* enc = nullptr) = 0; // extradata: avcC, hvcC or av1C when the packets come from an MP4;
	// enc: what was encoded; with it the stream may be decoded on the GPU (see ff_decode.cpp), without it always in software
	virtual bool send(const uint8_t* data, size_t size) = 0; // one encoded access unit
	virtual bool flush() = 0;                                // no more input
	virtual int receive(Picture& out) = 0;                   // 1: frame produced, 0: none ready, <0: error
};

std::unique_ptr<Decoder> make_decoder(); // libavcodec
