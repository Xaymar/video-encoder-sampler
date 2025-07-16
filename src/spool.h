#pragma once
// An encoder's packets, kept on disk as they arrive so a run does not hold its whole output in memory. A record is
// [size u32][pts i64][key u8][payload]. The spool is read back for scoring and for writing the MP4, then deleted.
#include <cstdint>
#include <cstdio>
#include <filesystem>
#include <string>
#include <vector>

struct SpoolInfo {
	long long count = 0, bytes = 0;
	uint64_t hash = 1469598103934665603ull; // FNV-1a over every packet and its size: equal hashes, equal streams
};

class SpoolWriter {
	FILE* f = nullptr;
	SpoolInfo info_;

	void mix(uint8_t b) { info_.hash = (info_.hash ^ b) * 1099511628211ull; }

public:
	~SpoolWriter() { close(); }
	// An empty path keeps nothing and only counts and hashes.
	bool open(const std::string& path) {
		if (path.empty()) return true;
		f = fopen(path.c_str(), "wb");
		if (f) setvbuf(f, nullptr, _IOFBF, 1 << 20);
		return f != nullptr;
	}
	bool add(const uint8_t* d, size_t n, int64_t pts, bool key) {
		for (int i = 0; i < 8; i++) mix((uint8_t)((uint64_t)n >> (8 * i)));
		for (size_t i = 0; i < n; i++) mix(d[i]);
		info_.count++;
		info_.bytes += (long long)n;
		if (!f) return true;
		const uint32_t size = (uint32_t)n;
		const uint8_t k = key ? 1 : 0;
		return fwrite(&size, 4, 1, f) == 1 && fwrite(&pts, 8, 1, f) == 1 && fwrite(&k, 1, 1, f) == 1 && fwrite(d, 1, n, f) == n;
	}
	bool close() {
		if (!f) return true;
		const bool ok = fclose(f) == 0;
		f = nullptr;
		return ok;
	}
	const SpoolInfo& info() const { return info_; }
};

class SpoolReader {
	FILE* f = nullptr;

public:
	~SpoolReader() { if (f) fclose(f); }
	bool open(const std::string& path) {
		f = fopen(path.c_str(), "rb");
		if (f) setvbuf(f, nullptr, _IOFBF, 1 << 20);
		return f != nullptr;
	}
	// The next record; with `payload` null the packet bytes are skipped.
	bool next(std::vector<uint8_t>* payload, int64_t* pts = nullptr, bool* key = nullptr) {
		uint32_t size;
		int64_t p;
		uint8_t k;
		if (!f || fread(&size, 4, 1, f) != 1 || fread(&p, 8, 1, f) != 1 || fread(&k, 1, 1, f) != 1) return false;
		if (pts) *pts = p;
		if (key) *key = k != 0;
		if (payload) {
			payload->resize(size);
			return fread(payload->data(), 1, size, f) == size;
		}
		return fseek(f, (long)size, SEEK_CUR) == 0;
	}
};
