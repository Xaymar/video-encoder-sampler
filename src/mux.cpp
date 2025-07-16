#include "mux.h"

#include <algorithm>
#include <cstdio>
#include <cstring>

#include "ff.h"
#include "spool.h"

extern "C" {
#include <libavformat/avformat.h>
#include <libavutil/mem.h>
}

namespace {

// Annex B stream: the NAL units of the types in `want` (with their start codes), in order of appearance, for the codec's header.
std::vector<uint8_t> annexb_headers(const std::vector<uint8_t>& d, Codec codec) {
	std::vector<uint8_t> out;
	auto start_at = [&](size_t i, size_t& len) { // start code at i? 3 or 4 bytes
		if (i + 3 <= d.size() && d[i] == 0 && d[i + 1] == 0 && d[i + 2] == 1) { len = 3; return true; }
		if (i + 4 <= d.size() && d[i] == 0 && d[i + 1] == 0 && d[i + 2] == 0 && d[i + 3] == 1) { len = 4; return true; }
		return false;
	};
	std::vector<std::pair<size_t, size_t>> nals; // payload begin, next start code
	size_t len = 0;
	for (size_t i = 0; i < d.size(); i++) {
		if (!start_at(i, len)) continue;
		if (!nals.empty()) nals.back().second = i;
		nals.push_back({i + len, d.size()});
		i += len - 1;
	}
	for (auto& [b, e] : nals) {
		if (b >= d.size()) continue;
		const int type = codec == Codec::H264 ? (d[b] & 0x1F) : ((d[b] >> 1) & 0x3F);
		const bool header = codec == Codec::H264 ? (type == 7 || type == 8) : (type == 32 || type == 33 || type == 34);
		if (!header) continue;
		const uint8_t sc[4] = {0, 0, 0, 1};
		out.insert(out.end(), sc, sc + 4);
		out.insert(out.end(), d.begin() + b, d.begin() + e);
	}
	return out;
}

// The sequence header OBU of an AV1 temporal unit (low-overhead format), as it goes into the av1C box.
std::vector<uint8_t> av1_sequence_header(const std::vector<uint8_t>& d) {
	size_t i = 0;
	while (i < d.size()) {
		const size_t start = i;
		const uint8_t h = d[i++];
		const int type = (h >> 3) & 0xF;
		if (h & 4) i++; // extension byte
		uint64_t size = 0;
		if (h & 2) { // has_size_field, leb128
			for (int k = 0; k < 8 && i < d.size(); k++) {
				const uint8_t b = d[i++];
				size |= (uint64_t)(b & 0x7F) << (7 * k);
				if (!(b & 0x80)) break;
			}
		} else size = d.size() - i;
		if (i + size > d.size()) break;
		if (type == 1) return std::vector<uint8_t>(d.begin() + start, d.begin() + i + size);
		i += size;
	}
	return {};
}

} // namespace

bool write_mp4(const std::string& path, Codec codec, const Shape& enc, const Color& color, const std::string& spool_path) {
	// First pass: the display numbers and the first packet. The delay B-frames put between coding and display order has to be known
	// before the first packet is written, so it takes a look at the whole stream; the payloads are skipped.
	std::vector<int64_t> pts;
	std::vector<uint8_t> first;
	int64_t lead = 0;
	{
		SpoolReader scan;
		if (!scan.open(spool_path)) return false;
		int64_t p;
		for (bool more = scan.next(pts.empty() ? &first : nullptr, &p); more; more = scan.next(nullptr, &p)) {
			lead = std::max<int64_t>(lead, (int64_t)pts.size() - p);
			pts.push_back(p);
		}
	}
	if (pts.empty()) return false;
	SpoolReader packets;
	if (!packets.open(spool_path)) return false;
	const AVCodecID id = codec == Codec::H264 ? AV_CODEC_ID_H264 : codec == Codec::HEVC ? AV_CODEC_ID_HEVC : codec == Codec::AV1 ? AV_CODEC_ID_AV1 : AV_CODEC_ID_NONE;
	if (id == AV_CODEC_ID_NONE) return false;

	AVFormatContext* oc = nullptr;
	if (avformat_alloc_output_context2(&oc, nullptr, "mp4", path.c_str()) < 0 || !oc) return false;
	bool ok = false;
	AVDictionary* opts = nullptr;
	do {
		AVStream* st = avformat_new_stream(oc, nullptr);
		if (!st) break;
		AVCodecParameters* cp = st->codecpar;
		cp->codec_type = AVMEDIA_TYPE_VIDEO;
		cp->codec_id = id;
		cp->width = enc.w; cp->height = enc.h;
		cp->format = av_planar_format(enc.pix.chroma, enc.pix.depth);
		cp->color_range = (AVColorRange)color.range;
		cp->color_space = (AVColorSpace)color.space;
		cp->color_primaries = (AVColorPrimaries)color.primaries;
		cp->color_trc = (AVColorTransferCharacteristic)color.trc;
		// The header boxes (avcC, hvcC, av1C) are built from the parameter sets, which the first packet carries.
		const std::vector<uint8_t> extra = id == AV_CODEC_ID_AV1 ? av1_sequence_header(first) : annexb_headers(first, codec);
		if (extra.empty()) { fprintf(stderr, "mp4: no parameter sets in the first packet\n"); break; }
		cp->extradata = (uint8_t*)av_mallocz(extra.size() + AV_INPUT_BUFFER_PADDING_SIZE);
		if (!cp->extradata) break;
		memcpy(cp->extradata, extra.data(), extra.size());
		cp->extradata_size = (int)extra.size();
		const AVRational tb{enc.fps.den, enc.fps.num}; // one tick per frame
		st->time_base = tb;
		st->avg_frame_rate = AVRational{enc.fps.num, enc.fps.den};
		if (avio_open(&oc->pb, path.c_str(), AVIO_FLAG_WRITE) < 0) break;
		av_dict_set(&opts, "movflags", "frag_keyframe+empty_moov+default_base_moof", 0);
		if (avformat_write_header(oc, &opts) < 0) break;

		// B-frames put the display time behind the coding time; shifting both by the largest lead keeps every decode time at or before
		// its display time without negative values.
		ok = true;
		std::vector<uint8_t> data;
		bool is_key = false;
		for (size_t i = 0; ok && i < pts.size(); i++) {
			int64_t unused;
			if (!packets.next(&data, &unused, &is_key)) { ok = false; break; }
			AVPacket* pkt = av_packet_alloc();
			if (!pkt || av_new_packet(pkt, (int)data.size()) < 0) { av_packet_free(&pkt); ok = false; break; }
			memcpy(pkt->data, data.data(), data.size());
			pkt->stream_index = 0;
			pkt->dts = (int64_t)i;
			pkt->pts = pts[i] + lead;
			pkt->duration = 1;
			if (is_key) pkt->flags |= AV_PKT_FLAG_KEY;
			av_packet_rescale_ts(pkt, tb, st->time_base);
			ok = av_interleaved_write_frame(oc, pkt) >= 0;
			av_packet_free(&pkt);
		}
		ok = ok && av_write_trailer(oc) >= 0;
	} while (false);
	av_dict_free(&opts);
	if (oc->pb) avio_closep(&oc->pb);
	avformat_free_context(oc);
	return ok;
}
