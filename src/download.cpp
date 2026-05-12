#include "download.h"

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <stdexcept>

#include <curl/curl.h>

#include "ui.h"

namespace fs = std::filesystem;

namespace {

// What the server said about the file, kept next to the download to tell later whether it changed.
struct Meta {
	std::string url, etag, last_modified;
	uint64_t size = 0;
};

Meta read_meta(const fs::path& p) {
	Meta m;
	std::ifstream f(p);
	std::string line;
	while (std::getline(f, line)) {
		const size_t eq = line.find('=');
		if (eq == std::string::npos) continue;
		const std::string k = line.substr(0, eq), v = line.substr(eq + 1);
		if (k == "url") m.url = v;
		else if (k == "etag") m.etag = v;
		else if (k == "last_modified") m.last_modified = v;
		else if (k == "size") m.size = strtoull(v.c_str(), nullptr, 10);
	}
	return m;
}

void write_meta(const fs::path& p, const Meta& m) {
	std::ofstream f(p, std::ios::trunc);
	f << "url=" << m.url << "\netag=" << m.etag << "\nlast_modified=" << m.last_modified << "\nsize=" << m.size << "\n";
}

std::string file_name_of(const std::string& url) {
	std::string path = url.substr(url.find("://") + 3);
	path = path.substr(path.find('/') == std::string::npos ? path.size() : path.find('/'));
	path = path.substr(0, path.find_first_of("?#"));
	std::string name = path.substr(path.rfind('/') == std::string::npos ? 0 : path.rfind('/') + 1);
	for (char& c : name) if (!isalnum((unsigned char)c) && c != '.' && c != '-' && c != '_') c = '_';
	return name.empty() ? "video" : name;
}

std::string size_text(uint64_t b) {
	char s[32];
	if (b >= 1ull << 30) snprintf(s, sizeof s, "%.2f GB", b / double(1ull << 30));
	else snprintf(s, sizeof s, "%.1f MB", b / double(1 << 20));
	return s;
}

std::string lower(std::string s) {
	std::transform(s.begin(), s.end(), s.begin(), [](unsigned char c) { return (char)tolower(c); });
	return s;
}

// One request and what it left behind.
struct Transfer {
	// in
	fs::path part;
	uint64_t resume_at = 0;
	bool have = false;
	Meta old;
	std::string name;
	// the response, as the headers come in (every redirect hop starts over)
	int status = 0;
	std::string etag, modified;
	uint64_t length = 0, total = 0;
	// progress
	std::ofstream out;
	uint64_t done = 0;
	RunProgress* rp = nullptr; // when set, the live display shows the progress and nothing is printed
	bool unchanged = false, bad_status = false, write_failed = false;
	std::chrono::steady_clock::time_point last = std::chrono::steady_clock::now();
	uint64_t at_last = 0;
};

size_t on_header(char* buf, size_t size, size_t n, void* user) {
	Transfer& t = *(Transfer*)user;
	std::string line(buf, size * n);
	while (!line.empty() && (line.back() == '\r' || line.back() == '\n')) line.pop_back();
	if (line.rfind("HTTP/", 0) == 0) { // a new response: the first, or the one a redirect led to
		t.status = atoi(line.c_str() + line.find(' ') + 1);
		t.etag.clear(); t.modified.clear();
		t.length = t.total = 0;
		return size * n;
	}
	const size_t colon = line.find(':');
	if (colon == std::string::npos) return size * n;
	const std::string key = lower(line.substr(0, colon));
	std::string value = line.substr(colon + 1);
	value.erase(0, value.find_first_not_of(" \t"));
	if (key == "etag") t.etag = value;
	else if (key == "last-modified") t.modified = value;
	else if (key == "content-length") t.length = strtoull(value.c_str(), nullptr, 10);
	else if (key == "content-range") {
		const size_t slash = value.find('/');
		if (slash != std::string::npos) t.total = strtoull(value.c_str() + slash + 1, nullptr, 10);
	}
	return size * n;
}

size_t on_body(char* buf, size_t size, size_t n, void* user) {
	Transfer& t = *(Transfer*)user;
	if (!t.out.is_open()) { // the first bytes: time to decide what this response is
		if (t.status != 200 && t.status != 206) { t.bad_status = true; return 0; }
		if (t.status == 200 && t.have && t.length == t.old.size &&
		    ((!t.etag.empty() && t.etag == t.old.etag) || (t.etag.empty() && !t.modified.empty() && t.modified == t.old.last_modified))) {
			t.unchanged = true; // the server sends the file we already have: no need for it
			return 0;
		}
		if (t.status == 206) {
			if (!t.total) t.total = t.resume_at + t.length;
			t.done = t.resume_at;
			t.out.open(t.part, std::ios::binary | std::ios::app);
		} else {
			if (!t.total) t.total = t.length;
			t.done = 0;
			t.out.open(t.part, std::ios::binary | std::ios::trunc);
		}
		if (!t.out) { t.write_failed = true; return 0; }
		std::string what = t.status == 206 ? "continuing the download at " + size_text(t.done) + (t.total ? " of " + size_text(t.total) : "") : "downloading" + (t.total ? " " + size_text(t.total) : std::string());
		if (t.rp) { t.rp->total = (long long)t.total; t.rp->done = (long long)t.done; t.rp->since_ms = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now().time_since_epoch()).count(); }
		else { printf("Input %s: %s\n", t.name.c_str(), what.c_str()); fflush(stdout); }
	}
	t.out.write(buf, (std::streamsize)(size * n));
	if (!t.out) { t.write_failed = true; return 0; }
	t.done += size * n;
	if (t.rp) t.rp->done = (long long)t.done;
	return size * n;
}

int on_progress(void* user, curl_off_t, curl_off_t, curl_off_t, curl_off_t) {
	Transfer& t = *(Transfer*)user;
	const auto now = std::chrono::steady_clock::now();
	const double secs = std::chrono::duration<double>(now - t.last).count();
	if (!t.rp && t.out.is_open() && secs >= 2.0) {
		printf("  %s%s  %.1f MB/s\n", size_text(t.done).c_str(), t.total ? (" of " + size_text(t.total) + " (" + std::to_string(t.done * 100 / t.total) + "%)").c_str() : "",
		       (t.done - t.at_last) / secs / (1 << 20));
		fflush(stdout);
		t.last = now;
		t.at_last = t.done;
	}
	return 0;
}

} // namespace

std::string url_file_name(const std::string& url) { return file_name_of(url); }

bool is_url(const std::string& s) { return s.rfind("http://", 0) == 0 || s.rfind("https://", 0) == 0; }

std::string fetch_url(const std::string& url, const std::string& dir, RunProgress* progress) {
	uint64_t h = 1469598103934665603ull;
	for (unsigned char c : url) h = (h ^ c) * 1099511628211ull;
	char folder[20];
	snprintf(folder, sizeof folder, "%016llx", (unsigned long long)h); // one folder per URL, so that the file keeps its own name
	std::error_code ec;
	fs::create_directories(fs::path(dir) / folder, ec);
	const fs::path file = fs::path(dir) / folder / file_name_of(url);
	const fs::path part = file.string() + ".part", meta_file = file.string() + ".meta";

	Transfer t;
	t.name = file_name_of(url);
	t.rp = progress;
	t.part = part;
	t.have = fs::exists(file) && fs::exists(meta_file);
	if (t.have) t.old = read_meta(meta_file);

	static const bool curl_ready = curl_global_init(CURL_GLOBAL_DEFAULT) == CURLE_OK;
	if (!curl_ready) throw std::runtime_error("cannot start libcurl");

	// Ask for the file; if there is an earlier copy, only if it changed, and if there is the start of one, only the rest.
	// A server that cannot do the second sends the whole file, and a second try starts over.
	for (int attempt = 0; attempt < 2; attempt++) {
		t.status = 0;
		t.out.close();
		t.unchanged = t.bad_status = t.write_failed = false;
		t.resume_at = (!t.have && attempt == 0 && fs::exists(part)) ? fs::file_size(part) : 0;
		if (t.resume_at == 0) fs::remove(part, ec);

		CURL* c = curl_easy_init();
		if (!c) throw std::runtime_error("cannot start libcurl");
		curl_slist* headers = nullptr;
		if (t.have) {
			if (!t.old.etag.empty()) headers = curl_slist_append(headers, ("If-None-Match: " + t.old.etag).c_str());
			else if (!t.old.last_modified.empty()) headers = curl_slist_append(headers, ("If-Modified-Since: " + t.old.last_modified).c_str());
		}
		curl_easy_setopt(c, CURLOPT_URL, url.c_str());
		curl_easy_setopt(c, CURLOPT_USERAGENT, "ves/0.2");
		curl_easy_setopt(c, CURLOPT_FOLLOWLOCATION, 1L);
		curl_easy_setopt(c, CURLOPT_MAXREDIRS, 10L);
		curl_easy_setopt(c, CURLOPT_CONNECTTIMEOUT, 30L);
		curl_easy_setopt(c, CURLOPT_LOW_SPEED_LIMIT, 1L); // give up when nothing arrives for a minute
		curl_easy_setopt(c, CURLOPT_LOW_SPEED_TIME, 60L);
		curl_easy_setopt(c, CURLOPT_SSL_OPTIONS, (long)CURLSSLOPT_NATIVE_CA); // the system's certificates
		if (headers) curl_easy_setopt(c, CURLOPT_HTTPHEADER, headers);
		if (t.resume_at) curl_easy_setopt(c, CURLOPT_RESUME_FROM_LARGE, (curl_off_t)t.resume_at);
		curl_easy_setopt(c, CURLOPT_HEADERFUNCTION, on_header);
		curl_easy_setopt(c, CURLOPT_HEADERDATA, &t);
		curl_easy_setopt(c, CURLOPT_WRITEFUNCTION, on_body);
		curl_easy_setopt(c, CURLOPT_WRITEDATA, &t);
		curl_easy_setopt(c, CURLOPT_NOPROGRESS, 0L);
		curl_easy_setopt(c, CURLOPT_XFERINFOFUNCTION, on_progress);
		curl_easy_setopt(c, CURLOPT_XFERINFODATA, &t);
		const CURLcode res = curl_easy_perform(c);
		long code = 0;
		curl_easy_getinfo(c, CURLINFO_RESPONSE_CODE, &code);
		curl_slist_free_all(headers);
		curl_easy_cleanup(c);
		t.out.close();

		if (res == CURLE_OK && code == 304) t.unchanged = true;
		if (t.unchanged) {
			if (progress) ui::log("Input " + t.name + ": already downloaded (" + size_text(t.old.size) + ")");
			else { printf("Input %s: already downloaded (%s)\n", t.name.c_str(), size_text(t.old.size).c_str()); fflush(stdout); }
			return file.string();
		}
		if (res == CURLE_RANGE_ERROR && t.resume_at) continue; // the server would not continue where we stopped
		if (t.bad_status || (res == CURLE_OK && code != 200 && code != 206)) {
			if (t.have) { fprintf(stderr, "could not check %s (the server answered %ld): using the copy from earlier\n", url.c_str(), t.status ? (long)t.status : code); return file.string(); }
			throw std::runtime_error("the server answered " + std::to_string(t.status ? t.status : (int)code) + " for " + url);
		}
		if (t.write_failed) throw std::runtime_error("cannot write " + part.string());
		if (res != CURLE_OK) {
			if (t.have) { fprintf(stderr, "could not check %s (%s): using the copy from earlier\n", url.c_str(), curl_easy_strerror(res)); return file.string(); }
			throw std::runtime_error(std::string("the download of ") + url + " failed: " + curl_easy_strerror(res) + (t.done ? " (run again to continue at " + size_text(t.done) + ")" : ""));
		}
		if (t.total && t.done != t.total) throw std::runtime_error("the download is incomplete (" + size_text(t.done) + " of " + size_text(t.total) + "); run again to continue");

		fs::remove(file, ec);
		fs::rename(part, file, ec);
		if (ec) throw std::runtime_error("cannot move the download into place: " + ec.message());
		Meta m;
		m.url = url; m.etag = t.etag; m.last_modified = t.modified; m.size = t.done;
		write_meta(meta_file, m);
		if (progress) ui::log("Input " + t.name + ": downloaded " + size_text(t.done));
		else { printf("  done: %s\n", size_text(t.done).c_str()); fflush(stdout); }
		return file.string();
	}
	throw std::runtime_error("the server would not continue the download of " + url);
}
