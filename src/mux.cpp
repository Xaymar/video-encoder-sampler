#include "mux.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <memory>

#ifdef _WIN32
#define NOMINMAX
#include <windows.h>
#else
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#endif

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

// File access for libavformat through memory-mapped files. The writer maps the file in windows and copies packets straight into
// them, extending the file as it grows; the reader maps the finished file. The system's memory manager then decides what stays in
// memory and what goes to storage, orders the writes, and reads ahead, instead of every packet going through a system call.
namespace {

constexpr int IO_BUFFER = 1 << 20;
constexpr uint64_t CHUNK = 64ull << 20; // size of a mapped window, a multiple of the allocation granularity; the file grows by this much

#ifdef _WIN32
std::wstring wide(const std::string& s) {
	const int n = MultiByteToWideChar(CP_UTF8, 0, s.c_str(), -1, nullptr, 0);
	std::wstring w((size_t)std::max(n, 1), L'\0');
	MultiByteToWideChar(CP_UTF8, 0, s.c_str(), -1, w.data(), n);
	return w;
}

class Mapped {
	HANDLE file = INVALID_HANDLE_VALUE, map = nullptr;
	uint8_t* view = nullptr;
	uint64_t view_off = 0, view_len = 0;
	uint64_t pos = 0, size = 0, capacity = 0;
	bool writing = false;

	void unmap_view() {
		if (view) UnmapViewOfFile(view);
		view = nullptr;
		view_len = 0;
	}

	// Maps the window that holds `at`.
	bool window(uint64_t at) {
		if (view && at >= view_off && at < view_off + view_len) return true;
		unmap_view();
		view_off = at / CHUNK * CHUNK;
		view_len = std::min<uint64_t>(CHUNK, capacity - view_off);
		view = (uint8_t*)MapViewOfFile(map, writing ? FILE_MAP_WRITE : FILE_MAP_READ, (DWORD)(view_off >> 32), (DWORD)view_off, (SIZE_T)view_len);
		if (!view) { view_len = 0; fprintf(stderr, "mux: cannot map %llu bytes of the file (Windows error %lu)\n", (unsigned long long)CHUNK, GetLastError()); }
		return view != nullptr;
	}

	bool grow(uint64_t need) {
		const uint64_t cap = (need + CHUNK - 1) / CHUNK * CHUNK;
		unmap_view();
		if (map) { CloseHandle(map); map = nullptr; }
		LARGE_INTEGER to;
		to.QuadPart = (LONGLONG)cap;
		if (!SetFilePointerEx(file, to, nullptr, FILE_BEGIN) || !SetEndOfFile(file)) { fprintf(stderr, "mux: cannot extend the file to %llu bytes (Windows error %lu)\n", (unsigned long long)cap, GetLastError()); return false; }
		capacity = cap;
		map = CreateFileMappingW(file, nullptr, PAGE_READWRITE, 0, 0, nullptr);
		if (!map) fprintf(stderr, "mux: cannot map the file (Windows error %lu)\n", GetLastError());
		return map != nullptr;
	}

public:
	~Mapped() { close(); }

	bool open(const std::string& path, bool write) {
		writing = write;
		file = CreateFileW(wide(path).c_str(), write ? GENERIC_READ | GENERIC_WRITE : GENERIC_READ, FILE_SHARE_READ | FILE_SHARE_WRITE | FILE_SHARE_DELETE, nullptr,
		                   write ? CREATE_ALWAYS : OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, nullptr);
		if (file == INVALID_HANDLE_VALUE) return false;
		if (write) return true; // mapped when the first bytes arrive
		LARGE_INTEGER sz;
		if (!GetFileSizeEx(file, &sz) || sz.QuadPart == 0) return false;
		size = capacity = (uint64_t)sz.QuadPart;
		map = CreateFileMappingW(file, nullptr, PAGE_READONLY, 0, 0, nullptr);
		return map != nullptr;
	}

	int write_bytes(const uint8_t* b, int n) {
		if (pos + (uint64_t)n > capacity && !grow(pos + (uint64_t)n)) return AVERROR(EIO);
		int left = n;
		while (left > 0) {
			if (!window(pos)) return AVERROR(EIO);
			const uint64_t here = std::min<uint64_t>((uint64_t)left, view_off + view_len - pos);
			memcpy(view + (pos - view_off), b, (size_t)here);
			b += here; pos += here; left -= (int)here;
		}
		size = std::max(size, pos);
		return n;
	}

	int read_bytes(uint8_t* b, int n) {
		if (pos >= size) return AVERROR_EOF;
		const int total = (int)std::min<uint64_t>((uint64_t)n, size - pos);
		int left = total;
		while (left > 0) {
			if (!window(pos)) return AVERROR(EIO);
			const uint64_t here = std::min<uint64_t>((uint64_t)left, view_off + view_len - pos);
			memcpy(b, view + (pos - view_off), (size_t)here);
			b += here; pos += here; left -= (int)here;
		}
		return total;
	}

	int64_t seek(int64_t off, int whence) {
		if (whence & AVSEEK_SIZE) return (int64_t)size;
		switch (whence & 3) {
		case SEEK_SET: pos = (uint64_t)off; break;
		case SEEK_CUR: pos = (uint64_t)((int64_t)pos + off); break;
		case SEEK_END: pos = (uint64_t)((int64_t)size + off); break;
		default: return AVERROR(EINVAL);
		}
		return (int64_t)pos;
	}

	void close() {
		unmap_view();
		if (map) CloseHandle(map);
		map = nullptr;
		if (file != INVALID_HANDLE_VALUE) {
			if (writing) { // the file was extended in whole windows; cut it back to what was written
				LARGE_INTEGER to;
				to.QuadPart = (LONGLONG)size;
				SetFilePointerEx(file, to, nullptr, FILE_BEGIN);
				SetEndOfFile(file);
			}
			CloseHandle(file);
		}
		file = INVALID_HANDLE_VALUE;
	}
};

#else
class Mapped {
	int file = -1;
	uint8_t* view = nullptr;
	uint64_t view_off = 0, view_len = 0;
	uint64_t pos = 0, size = 0, capacity = 0;
	bool writing = false;

	void unmap_view() {
		if (view) munmap(view, (size_t)view_len);
		view = nullptr;
		view_len = 0;
	}

	bool window(uint64_t at) {
		if (view && at >= view_off && at < view_off + view_len) return true;
		unmap_view();
		view_off = at / CHUNK * CHUNK;
		view_len = std::min<uint64_t>(CHUNK, capacity - view_off);
		void* v = mmap(nullptr, (size_t)view_len, writing ? PROT_READ | PROT_WRITE : PROT_READ, MAP_SHARED, file, (off_t)view_off);
		if (v == MAP_FAILED) { view_len = 0; return false; }
		view = (uint8_t*)v;
		return true;
	}

	bool grow(uint64_t need) {
		const uint64_t cap = (need + CHUNK - 1) / CHUNK * CHUNK;
		unmap_view();
		if (ftruncate(file, (off_t)cap) != 0) return false;
		capacity = cap;
		return true;
	}

public:
	~Mapped() { close(); }

	bool open(const std::string& path, bool write) {
		writing = write;
		file = ::open(path.c_str(), write ? O_RDWR | O_CREAT | O_TRUNC : O_RDONLY, 0644);
		if (file < 0) return false;
		if (write) return true; // mapped when the first bytes arrive
		struct stat st;
		if (fstat(file, &st) != 0 || st.st_size == 0) return false;
		size = capacity = (uint64_t)st.st_size;
		return true;
	}

	int write_bytes(const uint8_t* b, int n) {
		if (pos + (uint64_t)n > capacity && !grow(pos + (uint64_t)n)) return AVERROR(EIO);
		int left = n;
		while (left > 0) {
			if (!window(pos)) return AVERROR(EIO);
			const uint64_t here = std::min<uint64_t>((uint64_t)left, view_off + view_len - pos);
			memcpy(view + (pos - view_off), b, (size_t)here);
			b += here; pos += here; left -= (int)here;
		}
		size = std::max(size, pos);
		return n;
	}

	int read_bytes(uint8_t* b, int n) {
		if (pos >= size) return AVERROR_EOF;
		const int total = (int)std::min<uint64_t>((uint64_t)n, size - pos);
		int left = total;
		while (left > 0) {
			if (!window(pos)) return AVERROR(EIO);
			const uint64_t here = std::min<uint64_t>((uint64_t)left, view_off + view_len - pos);
			memcpy(b, view + (pos - view_off), (size_t)here);
			b += here; pos += here; left -= (int)here;
		}
		return total;
	}

	int64_t seek(int64_t off, int whence) {
		if (whence & AVSEEK_SIZE) return (int64_t)size;
		switch (whence & 3) {
		case SEEK_SET: pos = (uint64_t)off; break;
		case SEEK_CUR: pos = (uint64_t)((int64_t)pos + off); break;
		case SEEK_END: pos = (uint64_t)((int64_t)size + off); break;
		default: return AVERROR(EINVAL);
		}
		return (int64_t)pos;
	}

	void close() {
		unmap_view();
		if (file >= 0) {
			if (writing && ftruncate(file, (off_t)size) != 0) {} // cut back to what was written
			::close(file);
		}
		file = -1;
	}
};
#endif

int io_write(void* o, const uint8_t* b, int n) { return ((Mapped*)o)->write_bytes(b, n); }
int io_read(void* o, uint8_t* b, int n) { return ((Mapped*)o)->read_bytes(b, n); }
int64_t io_seek(void* o, int64_t off, int whence) { return ((Mapped*)o)->seek(off, whence); }

// An AVIOContext over a mapped file.
struct Io {
	Mapped file;
	AVIOContext* ctx = nullptr;

	bool open(const std::string& path, bool write) {
		if (!file.open(path, write)) return false;
		uint8_t* buf = (uint8_t*)av_malloc(IO_BUFFER);
		if (!buf) return false;
		ctx = avio_alloc_context(buf, IO_BUFFER, write, &file, write ? nullptr : io_read, write ? io_write : nullptr, io_seek);
		if (!ctx) { av_free(buf); return false; }
		return true;
	}
	~Io() {
		if (!ctx) return;
		avio_flush(ctx);
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

bool Mp4Writer::open(const std::string& path, Codec codec, const Shape& enc, const Color& color, int delay_hint) {
	path_ = path; codec_ = codec; enc_ = enc; color_ = color; hint_ = delay_hint;
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
	if (!impl->io->open(path_, true)) { fprintf(stderr, "mp4: cannot open %s for writing\n", path_.c_str()); return false; }
	oc->pb = impl->io->ctx;
	oc->flags |= AVFMT_FLAG_CUSTOM_IO;
	AVDictionary* opts = nullptr;
	av_dict_set(&opts, "movflags", "frag_keyframe+empty_moov+default_base_moof", 0);
	const int r = avformat_write_header(oc, &opts);
	av_dict_free(&opts);
	if (r < 0) { char e[128]; av_strerror(r, e, sizeof e); fprintf(stderr, "mp4: cannot write the header: %s\n", e); }
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
	const int wr = av_interleaved_write_frame(oc, pkt);
	if (wr < 0) { char e[128]; av_strerror(wr, e, sizeof e); fprintf(stderr, "mp4: cannot write a packet: %s\n", e); }
	av_packet_free(&pkt);
	return wr >= 0;
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
	if (!ok) fprintf(stderr, "mp4: closing failed (%s)\n", failed ? "an earlier write failed" : !started ? "no packet was ever written" : "the trailer could not be written");
	if (oc) {
		oc->pb = nullptr;
		avformat_free_context(oc);
		oc = nullptr;
	}
	impl->io.reset(); // flushes nothing more: the trailer went through avio already; closes the file
	return ok;
}
