#include "upload.h"

#include <algorithm>
#include <cstdint>
#include <atomic>
#include <condition_variable>
#include <cstdio>
#include <deque>
#include <filesystem>
#include <fstream>
#include <memory>
#include <mutex>
#include <random>
#include <sstream>
#include <thread>

#include <curl/curl.h>

#include "hash.h"
#include "ui.h"
#include "json.h"

namespace fs = std::filesystem;

namespace {

struct Reply {
	long status = 0;   // 0: no answer at all
	std::string body, error;
	Json json;         // the body, when it is JSON
};

size_t on_body(char* buf, size_t size, size_t n, void* user) {
	((std::string*)user)->append(buf, size * n);
	return size * n;
}

std::string escape(const std::string& s) {
	CURL* c = curl_easy_init();
	char* e = curl_easy_escape(c, s.c_str(), (int)s.size());
	std::string out = e ? e : "";
	curl_free(e);
	curl_easy_cleanup(c);
	return out;
}

std::string random_id() {
	std::random_device rd;
	std::string id;
	for (int i = 0; i < 16; i++) { char h[3]; snprintf(h, sizeof h, "%02x", rd() & 0xFF); id += h; }
	return id;
}

std::string text_of(const Json& j, const char* key) {
	const Json* v = j.find(key);
	return v && (v->type == Json::Str || v->type == Json::Num) ? v->s : std::string();
}

bool flag_of(const Json& j, const char* key) {
	const Json* v = j.find(key);
	return v && v->type == Json::Bool && v->s == "1";
}

// What libcurl tells while a chunk goes out, for the progress of the file: bytes before this chunk plus what has been sent of it.
struct Transfer {
	RunProgress* rp;
	long long base;
};

int on_transfer(void* user, curl_off_t, curl_off_t, curl_off_t, curl_off_t sent) {
	auto* t = (Transfer*)user;
	t->rp->done = t->base + (long long)sent;
	return 0;
}

class Site {
	std::string base, key;

public:
	Site(const UploadOptions& o) : base(o.url), key(o.key) {
		while (!base.empty() && base.back() == '/') base.pop_back();
	}

	// One request. `body` is sent as a POST body when `post` is set. The timeout is generous because the last chunk of a video is answered only
	// after the server has hashed the whole file.
	Reply request(bool post, const std::string& path, const char* content_type, const char* data, size_t size, long timeout, Transfer* progress = nullptr) const {
		Reply r;
		CURL* c = curl_easy_init();
		if (!c) { r.error = "cannot start libcurl"; return r; }
		curl_slist* headers = curl_slist_append(nullptr, ("Authorization: Bearer " + key).c_str());
		if (content_type) headers = curl_slist_append(headers, (std::string("Content-Type: ") + content_type).c_str());
		curl_easy_setopt(c, CURLOPT_URL, (base + path).c_str());
		curl_easy_setopt(c, CURLOPT_USERAGENT, "ves/0.2");
		curl_easy_setopt(c, CURLOPT_HTTPHEADER, headers);
		curl_easy_setopt(c, CURLOPT_CONNECTTIMEOUT, 30L);
		curl_easy_setopt(c, CURLOPT_TIMEOUT, timeout);
		curl_easy_setopt(c, CURLOPT_SSL_OPTIONS, (long)CURLSSLOPT_NATIVE_CA);
		curl_easy_setopt(c, CURLOPT_WRITEFUNCTION, on_body);
		curl_easy_setopt(c, CURLOPT_WRITEDATA, &r.body);
		if (progress) {
			curl_easy_setopt(c, CURLOPT_NOPROGRESS, 0L);
			curl_easy_setopt(c, CURLOPT_XFERINFOFUNCTION, on_transfer);
			curl_easy_setopt(c, CURLOPT_XFERINFODATA, progress);
		}
		if (post) {
			curl_easy_setopt(c, CURLOPT_POST, 1L);
			curl_easy_setopt(c, CURLOPT_POSTFIELDS, data);
			curl_easy_setopt(c, CURLOPT_POSTFIELDSIZE_LARGE, (curl_off_t)size);
		}
		const CURLcode res = curl_easy_perform(c);
		if (res != CURLE_OK) r.error = curl_easy_strerror(res);
		else curl_easy_getinfo(c, CURLINFO_RESPONSE_CODE, &r.status);
		curl_slist_free_all(headers);
		curl_easy_cleanup(c);
		try { r.json = Json::parse(r.body); } catch (const std::exception&) {}
		return r;
	}

	static std::string why(const Reply& r) {
		if (r.status == 0) return r.error;
		const std::string e = text_of(r.json, "error");
		return "HTTP " + std::to_string(r.status) + (e.empty() ? "" : ": " + e);
	}

	// Sends a file in chunks; returns the final reply (status 200 or 201 with "done"), or a reply that says what went wrong.
	Reply upload_file(const std::string& kind, const std::string& file, const std::string& extra_query, uint64_t chunk, RunProgress* rp, long long base) const {
		Reply last;
		std::error_code ec;
		const uint64_t total = fs::file_size(file, ec);
		std::ifstream in(file, std::ios::binary);
		if (ec || !in || total == 0) { last.error = "cannot read " + file; return last; }
		const std::string id = random_id();
		std::vector<char> buf((size_t)std::min<uint64_t>(chunk, total));
		uint64_t offset = 0;
		int failures = 0;
		while (offset < total) {
			const size_t n = (size_t)std::min<uint64_t>(chunk, total - offset);
			in.clear();
			in.seekg((std::streamoff)offset);
			in.read(buf.data(), (std::streamsize)n);
			if ((size_t)in.gcount() != n) { last.error = "cannot read " + file; last.status = 0; return last; }
			const std::string path = "/api/v1/upload?kind=" + kind + "&upload=" + id + "&offset=" + std::to_string(offset) + "&total=" + std::to_string(total) + extra_query;
			Transfer progress{rp, base + (long long)offset};
			last = request(true, path, "application/octet-stream", buf.data(), n, offset + n == total ? 3600 : 300, rp ? &progress : nullptr);
			if (last.status == 409) { // the server holds another amount: continue where it says
				const uint64_t held = (uint64_t)strtoull(text_of(last.json, "size").c_str(), nullptr, 10);
				if (held == offset || held > total || ++failures > 3) return last;
				offset = held;
				continue;
			}
			if (last.status == 0 || last.status >= 500) { // a network error or a server hiccup: try the same chunk again
				if (++failures > 3) return last;
				continue;
			}
			if (last.status != 200 && last.status != 201) return last;
			failures = 0;
			if (flag_of(last.json, "done")) return last;
			offset += n;
			if (rp) rp->done = base + (long long)offset;
		}
		last.error = "the server did not confirm the upload";
		last.status = 0;
		return last;
	}
};

std::string read_text(const std::string& path) {
	std::ifstream f(path, std::ios::binary);
	std::stringstream ss;
	ss << f.rdbuf();
	return ss.str();
}

bool is_local_host(const std::string& url) {
	const size_t s = url.find("://");
	const std::string host = url.substr(s == std::string::npos ? 0 : s + 3);
	for (const char* h : {"localhost", "127.0.0.1", "[::1]"}) if (host.rfind(h, 0) == 0) return true;
	return false;
}

} // namespace

struct Uploader::Impl {
	UploadOptions options;
	std::function<void(const std::string&)> log;
	std::mutex m;
	std::condition_variable cv;
	std::deque<UploadItem> queue;
	std::thread worker;
	bool closing = false;
	std::atomic<bool> disabled{false};                 // the key was refused, the site is out of reach or the account may not upload: the rest is skipped
	bool announced = false;
	int scores_new = 0, scores_known = 0, videos_sent = 0, problems = 0, skipped = 0;
	std::atomic<int> added{0}, handled{0}; // for the count in the live display

	// What /me said, once it has been asked.
	bool ready = false, may_scores = false, may_videos = false;
	uint64_t chunk = 0;
	std::unique_ptr<Site> site;

	// Asks the site who we are, once. False when nothing can be uploaded.
	bool connect() {
		if (ready) return true;
		if (options.key.empty()) { log("Upload: no API key (settings.upload.key, --upload-key or the VES_API_KEY environment variable)"); return false; }
		if (options.url.rfind("https://", 0) != 0 && !(options.url.rfind("http://", 0) == 0 && is_local_host(options.url))) {
			log("Upload: the address must be https:// (the key is sent with every request)");
			return false;
		}
		static const bool curl_ready = curl_global_init(CURL_GLOBAL_DEFAULT) == CURLE_OK;
		if (!curl_ready) { log("Upload: cannot start libcurl"); return false; }
		site = std::make_unique<Site>(options);
		const Reply me = site->request(false, "/api/v1/me", nullptr, nullptr, 0, 60);
		if (me.status == 401) { log("Upload: the site refused the API key (it was replaced or mistyped)"); return false; }
		if (me.status != 200) { log("Upload: cannot reach the site (" + Site::why(me) + ")"); return false; }
		const Json* can = me.json.find("can");
		may_scores = can && flag_of(*can, "upload_scores");
		may_videos = can && flag_of(*can, "upload_videos");
		chunk = std::max<uint64_t>(64 << 10, (uint64_t)strtoull(text_of(me.json, "chunk_bytes").c_str(), nullptr, 10));
		if (!may_scores && !(may_videos && options.videos)) { log("Upload: this account may not upload anything"); return false; }
		log("Upload: sending results to " + options.url + (may_videos && options.videos ? " (with videos)" : ""));
		return ready = true;
	}

	void process(const UploadItem& item) {
		const std::string who = "Upload " + item.label + ": ";
		try {
			const Json result = Json::parse(read_text(item.result_file));
			const Json* k = result.find("key");
			if (!k || k->type != Json::Str) throw std::runtime_error("the result file has no key");
			std::string video_hash = text_of(result, "video_sha256"), reference_hash = text_of(result, "reference_sha256");
			uint64_t video_size = 0;
			std::error_code ec;
			const bool with_video = options.videos && may_videos && !item.video_file.empty() && !video_hash.empty() && fs::exists(item.video_file, ec);
			if (with_video) video_size = fs::file_size(item.video_file, ec);

			std::string body = "{\"key_sha256\":\"" + sha256_bytes(k->s) + "\"";
			if (with_video) {
				body += ",\"video_sha256\":\"" + video_hash + "\"";
				if (!reference_hash.empty()) body += ",\"reference_sha256\":\"" + reference_hash + "\"";
				body += ",\"video_size\":" + std::to_string(video_size);
			}
			body += "}";
			const Reply check = site->request(true, "/api/v1/check", "application/json", body.data(), body.size(), 120);
			if (check.status != 200) { log(who + "check failed (" + Site::why(check) + ")"); problems++; return; }
			const Json* score = check.json.find("score");
			const Json* video = check.json.find("video");
			std::string run = video ? text_of(*video, "run") : std::string();
			std::string done;

			const bool send_score = score && flag_of(*score, "can_upload") && may_scores;
			const bool send_video = with_video && video && flag_of(*video, "can_upload") && (send_score || !run.empty());
			const long long score_bytes = send_score ? (long long)fs::file_size(item.result_file, ec) : 0;
			RunProgress rp; // shown in the live display while the files go out
			int handle = 0;
			if (score_bytes + (send_video ? (long long)video_size : 0) > 0) {
				rp.begin(4, score_bytes + (send_video ? (long long)video_size : 0));
				handle = ui::attach(&rp, item.label, "server");
			}
			struct Detach { RunProgress& rp; int handle; ~Detach() { if (handle) { rp.finish(); ui::detach(handle); } } } detach{rp, handle};

			if (send_score) {
				const Reply up = site->upload_file("score", item.result_file, "", chunk, &rp, 0);
				if (up.status == 200 || up.status == 201) {
					run = text_of(up.json, "uuid");
					(flag_of(up.json, "new") ? scores_new : scores_known)++;
					done = "score stored";
				} else { log(who + "the score was refused (" + Site::why(up) + ")"); problems++; return; }
			} else if (score && text_of(*score, "status") == "exists") {
				scores_known++;
				done = "score already there";
			}

			if (video && with_video) {
				const std::string status = text_of(*video, "status");
				if (flag_of(*video, "can_upload") && !run.empty()) {
					const Reply up = site->upload_file("video", item.video_file, "&run=" + run + "&name=" + escape(fs::path(item.video_file).filename().string()), chunk, &rp, score_bytes);
					if (up.status == 200 || up.status == 201) { videos_sent++; done += ", video stored"; }
					else { log(who + "the video was refused (" + Site::why(up) + ")"); problems++; }
				} else if (status == "exists") done += ", video already there";
				else {
					const std::string reason = text_of(*video, "reason");
					if (!reason.empty()) done += ", no video: " + reason;
				}
			}
			if (!done.empty() && !ui::live()) log(who + done); // the live display shows what is going out; plain output says what happened to each
		} catch (const std::exception& e) {
			log(who + e.what());
			problems++;
		}
	}

	void run() {
		for (;;) {
			UploadItem item;
			{
				std::unique_lock lock(m);
				cv.wait(lock, [&] { return !queue.empty() || closing; });
				if (queue.empty()) return;
				item = std::move(queue.front());
				queue.pop_front();
			}
			if (disabled) skipped++;
			else if (!connect()) { disabled = true; skipped++; }
			else process(item);
			ui::uploads(++handled, added);
		}
	}
};

Uploader::Uploader(const UploadOptions& options, std::function<void(const std::string&)> log) : impl(new Impl) {
	impl->options = options;
	impl->log = std::move(log);
	impl->worker = std::thread([this] { impl->run(); });
}

Uploader::~Uploader() { finish(); }

void Uploader::add(UploadItem item) {
	ui::uploads(impl->handled, ++impl->added);
	{ std::lock_guard lock(impl->m); impl->queue.push_back(std::move(item)); }
	impl->cv.notify_all();
}

void Uploader::finish() {
	if (!impl->worker.joinable()) return;
	size_t waiting;
	{ std::lock_guard lock(impl->m); waiting = impl->queue.size(); impl->closing = true; }
	if (waiting && !impl->disabled) impl->log("Upload: " + std::to_string(waiting) + " still to send");
	impl->cv.notify_all();
	impl->worker.join();
	if (!impl->ready) return; // said why in connect(), or nothing was ever queued
	impl->log("Upload finished: " + std::to_string(impl->scores_new) + " new scores, " + std::to_string(impl->scores_known) + " already known, " + std::to_string(impl->videos_sent) + " videos" +
	          (impl->problems ? ", " + std::to_string(impl->problems) + " problems (they are tried again next time)" : ""));
}
