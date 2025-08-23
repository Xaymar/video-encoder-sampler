#include "mux.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <map>
#include <memory>

#define NOMINMAX
#include <windows.h>

#include "ff.h"

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

// File access for libavformat through Win32, with big buffers: the default 32 KB buffer turns a stream into thousands of small writes.
// Scratch files get FILE_ATTRIBUTE_TEMPORARY, which tells the cache manager not to write them out unless memory runs short, so
// they live in memory and only spill to disk under pressure.
namespace {

constexpr int IO_BUFFER = 4 << 20;

struct Win32File {
	HANDLE h = INVALID_HANDLE_VALUE;
	~Win32File() { if (h != INVALID_HANDLE_VALUE) CloseHandle(h); }
};

std::wstring wide(const std::string& s) {
	const int n = MultiByteToWideChar(CP_UTF8, 0, s.c_str(), -1, nullptr, 0);
	std::wstring w((size_t)std::max(n, 1), L'\0');
	MultiByteToWideChar(CP_UTF8, 0, s.c_str(), -1, w.data(), n);
	return w;
}

int io_write(void* o, const uint8_t* b, int n) {
	DWORD w = 0;
	return WriteFile(((Win32File*)o)->h, b, (DWORD)n, &w, nullptr) && (int)w == n ? n : AVERROR(EIO);
}

int io_read(void* o, uint8_t* b, int n) {
	DWORD r = 0;
	if (!ReadFile(((Win32File*)o)->h, b, (DWORD)n, &r, nullptr)) return AVERROR(EIO);
	return r ? (int)r : AVERROR_EOF;
}

int64_t io_seek(void* o, int64_t off, int whence) {
	HANDLE h = ((Win32File*)o)->h;
	if (whence & AVSEEK_SIZE) {
		LARGE_INTEGER size;
		return GetFileSizeEx(h, &size) ? size.QuadPart : AVERROR(EIO);
	}
	LARGE_INTEGER to, now;
	to.QuadPart = off;
	const DWORD from = (whence & 3) == SEEK_SET ? FILE_BEGIN : (whence & 3) == SEEK_CUR ? FILE_CURRENT : FILE_END;
	return SetFilePointerEx(h, to, &now, from) ? now.QuadPart : AVERROR(EIO);
}

// An AVIOContext over a file; the file and buffer go with it.
struct Io {
	std::unique_ptr<Win32File> file = std::make_unique<Win32File>();
	AVIOContext* ctx = nullptr;

	bool open(const std::string& path, bool write, bool scratch) {
		file->h = CreateFileW(wide(path).c_str(), write ? GENERIC_WRITE : GENERIC_READ, FILE_SHARE_READ | FILE_SHARE_WRITE | FILE_SHARE_DELETE, nullptr,
		                      write ? CREATE_ALWAYS : OPEN_EXISTING, (scratch ? FILE_ATTRIBUTE_TEMPORARY : FILE_ATTRIBUTE_NORMAL) | FILE_FLAG_SEQUENTIAL_SCAN, nullptr);
		if (file->h == INVALID_HANDLE_VALUE) return false;
		uint8_t* buf = (uint8_t*)av_malloc(IO_BUFFER);
		if (!buf) return false;
		ctx = avio_alloc_context(buf, IO_BUFFER, write, file.get(), write ? nullptr : io_read, write ? io_write : nullptr, io_seek);
		if (!ctx) { av_free(buf); return false; }
		return true;
	}
	~Io() {
		if (!ctx) return;
		av_freep(&ctx->buffer);
		avio_context_free(&ctx);
	}
};

} // namespace

struct Mp4Writer::Impl {
	std::unique_ptr<Io> io;
};

Mp4Writer::Mp4Writer() : impl(new Impl) {}

Mp4Writer::~Mp4Writer() {
	if (th.joinable()) { { std::lock_guard lock(m); closing = true; } cv.notify_all(); th.join(); }
	if (oc) {
		oc->pb = nullptr; // the Io owns it
		avformat_free_context(oc);
	}
	delete impl;
}

bool Mp4Writer::open(const std::string& path, Codec codec, const Shape& enc, const Color& color, int delay_hint, bool scratch) {
	path_ = path; codec_ = codec; enc_ = enc; color_ = color; hint_ = delay_hint; scratch_ = scratch;
	if (codec != Codec::H264 && codec != Codec::HEVC && codec != Codec::AV1) return false;
	th = std::thread([this] { run(); });
	return true;
}

// The header boxes (avcC, hvcC, av1C) are built from the parameter sets, which the first packet carries.
bool Mp4Writer::start(const uint8_t* first, size_t size) {
	const AVCodecID id = codec_ == Codec::H264 ? AV_CODEC_ID_H264 : codec_ == Codec::HEVC ? AV_CODEC_ID_HEVC : AV_CODEC_ID_AV1;
	if (avformat_alloc_output_context2(&oc, nullptr, "mp4", path_.c_str()) < 0 || !oc) return false;
	st = avformat_new_stream(oc, nullptr);
	if (!st) return false;
	AVCodecParameters* cp = st->codecpar;
	cp->codec_type = AVMEDIA_TYPE_VIDEO;
	cp->codec_id = id;
	cp->width = enc_.w; cp->height = enc_.h;
	cp->format = av_planar_format(enc_.pix.chroma, enc_.pix.depth);
	cp->color_range = (AVColorRange)color_.range;
	cp->color_space = (AVColorSpace)color_.space;
	cp->color_primaries = (AVColorPrimaries)color_.primaries;
	cp->color_trc = (AVColorTransferCharacteristic)color_.trc;
	const std::vector<uint8_t> data(first, first + size);
	const std::vector<uint8_t> extra = id == AV_CODEC_ID_AV1 ? av1_sequence_header(data) : annexb_headers(data, codec_);
	if (extra.empty()) { fprintf(stderr, "mp4: no parameter sets in the first packet\n"); return false; }
	cp->extradata = (uint8_t*)av_mallocz(extra.size() + AV_INPUT_BUFFER_PADDING_SIZE);
	if (!cp->extradata) return false;
	memcpy(cp->extradata, extra.data(), extra.size());
	cp->extradata_size = (int)extra.size();
	st->time_base = AVRational{enc_.fps.den, enc_.fps.num}; // one tick per frame
	st->avg_frame_rate = AVRational{enc_.fps.num, enc_.fps.den};
	impl->io = std::make_unique<Io>();
	if (!impl->io->open(path_, true, scratch_)) return false;
	oc->pb = impl->io->ctx;
	oc->flags |= AVFMT_FLAG_CUSTOM_IO;
	AVDictionary* opts = nullptr;
	av_dict_set(&opts, "movflags", "frag_keyframe+empty_moov+default_base_moof", 0);
	const int r = avformat_write_header(oc, &opts);
	av_dict_free(&opts);
	return r >= 0;
}

bool Mp4Writer::write(const Item& it) {
	if (!started) {
		// Times are shifted so that no decode time is negative: by what the encoder's first DTS says, or by the B-frame depth.
		reported = it.dts != INT64_MIN;
		shift = reported ? std::max<int64_t>(0, -it.dts) : hint_;
		if (!(started = start(it.data.data(), it.data.size()))) return false;
	}
	int64_t d = reported ? it.dts + shift : index;
	if (d <= last_dts) d = last_dts + 1; // decode times must increase
	int64_t p = it.pts + shift;
	if (p < d) { // the encoder reordered more than it said
		if (!warned) { fprintf(stderr, "mp4: display time before decode time, the reorder depth was underestimated\n"); warned = true; }
		p = d;
	}
	last_dts = d;
	index++;
	AVPacket* pkt = av_packet_alloc();
	if (!pkt || av_new_packet(pkt, (int)it.data.size()) < 0) { av_packet_free(&pkt); return false; }
	memcpy(pkt->data, it.data.data(), it.data.size());
	pkt->stream_index = 0;
	pkt->dts = d;
	pkt->pts = p;
	pkt->duration = 1;
	if (it.key) pkt->flags |= AV_PKT_FLAG_KEY;
	av_packet_rescale_ts(pkt, AVRational{enc_.fps.den, enc_.fps.num}, st->time_base);
	const bool ok = av_interleaved_write_frame(oc, pkt) >= 0;
	av_packet_free(&pkt);
	return ok;
}

// The thread that does the muxing, so the encoder never waits for the disk: packets queue up to MAX_QUEUED bytes.
void Mp4Writer::run() {
	for (;;) {
		Item it;
		{
			std::unique_lock lock(m);
			cv.wait(lock, [&] { return !queue.empty() || closing; });
			if (queue.empty()) return;
			it = std::move(queue.front());
			queue.pop_front();
			queued -= it.data.size();
			cv.notify_all();
		}
		if (!failed && !write(it)) failed = true;
	}
}

bool Mp4Writer::add(const uint8_t* data, size_t size, int64_t pts, bool key, int64_t dts) {
	if (failed) return false;
	std::unique_lock lock(m);
	cv.wait(lock, [&] { return queued < MAX_QUEUED || failed; });
	queue.push_back(Item{std::vector<uint8_t>(data, data + size), pts, dts, key});
	queued += size;
	cv.notify_all();
	return true;
}

bool Mp4Writer::close() {
	if (!th.joinable()) return false;
	{ std::lock_guard lock(m); closing = true; }
	cv.notify_all();
	th.join();
	bool ok = !failed && started && oc && av_write_trailer(oc) >= 0;
	if (oc) {
		oc->pb = nullptr;
		avformat_free_context(oc);
		oc = nullptr;
	}
	impl->io.reset(); // flushes nothing more: the trailer went through avio already; closes the file
	return ok;
}

struct Mp4Reader::Impl {
	std::unique_ptr<Io> io;
};

Mp4Reader::Mp4Reader() : impl(new Impl) {}

Mp4Reader::~Mp4Reader() {
	if (ic) {
		ic->pb = nullptr;
		avformat_close_input(&ic);
	}
	delete impl;
}

bool Mp4Reader::open(const std::string& path) {
	impl->io = std::make_unique<Io>();
	if (!impl->io->open(path, false, false)) return false;
	ic = avformat_alloc_context();
	if (!ic) return false;
	ic->pb = impl->io->ctx;
	ic->flags |= AVFMT_FLAG_CUSTOM_IO;
	if (avformat_open_input(&ic, path.c_str(), nullptr, nullptr) < 0) { ic = nullptr; return false; }
	if (avformat_find_stream_info(ic, nullptr) < 0 || ic->nb_streams < 1) return false;
	const AVCodecParameters* cp = ic->streams[0]->codecpar;
	if (cp->extradata && cp->extradata_size > 0) extra_.assign(cp->extradata, cp->extradata + cp->extradata_size);
	return true;
}

bool Mp4Reader::next(std::vector<uint8_t>& out) {
	AVPacket* pkt = av_packet_alloc();
	bool ok = false;
	while (pkt && av_read_frame(ic, pkt) >= 0) {
		if (pkt->stream_index == 0) { out.assign(pkt->data, pkt->data + pkt->size); ok = true; av_packet_unref(pkt); break; }
		av_packet_unref(pkt);
	}
	av_packet_free(&pkt);
	return ok;
}
