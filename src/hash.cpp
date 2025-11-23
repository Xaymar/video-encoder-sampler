#include "hash.h"

#include <cstdio>
#include <vector>

extern "C" {
#include <libavutil/mem.h>
#include <libavutil/sha.h>
}

std::string sha256_file(const std::string& path) {
	FILE* f = fopen(path.c_str(), "rb");
	if (!f) return {};
	AVSHA* sha = av_sha_alloc();
	if (!sha) { fclose(f); return {}; }
	av_sha_init(sha, 256);
	std::vector<uint8_t> buf(1 << 20);
	size_t n;
	while ((n = fread(buf.data(), 1, buf.size(), f)) > 0) av_sha_update(sha, buf.data(), (uint64_t)n);
	const bool ok = !ferror(f);
	fclose(f);
	uint8_t digest[32];
	av_sha_final(sha, digest);
	av_free(sha);
	if (!ok) return {};
	std::string hex;
	for (uint8_t b : digest) { char h[3]; snprintf(h, sizeof h, "%02x", b); hex += h; }
	return hex;
}
