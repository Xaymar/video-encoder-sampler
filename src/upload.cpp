#include "upload.h"

#include <algorithm>
#include <cstdint>
#include <atomic>
#include <condition_variable>
#include <cstdio>
#include <deque>
#include <filesystem>
#include <fstream>
#include <map>
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
	Reply request(bool post, const std::string& path, const char* content_type, const char* data, size_t size, long timeout, Transfer* progress = nullptr, const char* method = nullptr) const {
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
		if (method) curl_easy_setopt(c, CURLOPT_CUSTOMREQUEST, method);
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
		if (r.status == 0) return r.error.empty() ? std::string("no answer") : r.error;
		const std::string e = text_of(r.json, "error");
		return "HTTP " + std::to_string(r.status) + (e.empty() ? (r.body.empty() ? "" : ", body: " + snippet(r.body)) : ": " + e);
	}

	// The start of a reply body, on one line, for messages about answers that are not what was expected.
	static std::string snippet(const std::string& body) {
		std::string s;
		for (char c : body.substr(0, 300)) s += (c == '\n' || c == '\r' || c == '\t') ? ' ' : c;
		return body.size() > 300 ? s + "..." : s;
	}

	// Sends a file in chunks; returns the final reply (status 200 or 201 with "done"), or a reply that says what went wrong.
	Reply upload_file(const std::string& kind, const std::string& file, const std::string& extra_query, uint64_t chunk, RunProgress* rp, long long base,
	                  const std::function<void(const std::string&)>& log = nullptr) const {
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
				if (log) log(kind + " chunk at " + std::to_string(offset) + ": the server holds " + std::to_string(held) + " bytes, continuing there");
				offset = held;
				continue;
			}
			if (last.status == 0 || last.status >= 500) { // a network error or a server hiccup: try the same chunk again
				if (++failures > 3) return last;
				if (log) log(kind + " chunk at " + std::to_string(offset) + " failed (" + why(last) + "), trying again (" + std::to_string(failures) + "/3)");
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
	std::string disabled_why;
	int scores_new = 0, scores_known = 0, scores_topped = 0, videos_sent = 0, videos_removed = 0, problems = 0, skipped = 0;
	std::atomic<int> added{0}, handled{0}; // for the count in the live display

	// What /me said, once it has been asked.
	bool ready = false, may_scores = false, may_videos = false;
	uint64_t chunk = 0;
	std::unique_ptr<Site> site;
	std::map<std::string, int> declined; // what the site said no to, and how often, for the summary at the end

	void decline(const std::string& who, const std::string& what, const std::string& reason) {
		declined[what + ": " + (reason.empty() ? "(the site gave no reason)" : reason)]++;
		log(who + what + " not sent: " + (reason.empty() ? "the site gave no reason" : reason));
	}

	bool give_up(const std::string& why) {
		log("Upload: " + why + "; nothing will be uploaded in this sweep");
		disabled_why = why;
		return false;
	}

	// Asks the site who we are, once. False when nothing can be uploaded.
	bool connect() {
		if (ready) return true;
		log("Upload: connecting to " + options.url + " ...");
		if (options.key.empty()) return give_up("no API key (upload.key in the config, --upload-key or the VES_API_KEY environment variable)");
		if (options.url.rfind("https://", 0) != 0 && !(options.url.rfind("http://", 0) == 0 && is_local_host(options.url)))
			return give_up("the address must be https:// (the key is sent with every request)");
		static const bool curl_ready = curl_global_init(CURL_GLOBAL_DEFAULT) == CURLE_OK;
		if (!curl_ready) return give_up("cannot start libcurl");
		site = std::make_unique<Site>(options);
		const Reply me = site->request(false, "/api/v1/me", nullptr, nullptr, 0, 60);
		if (me.status == 401) return give_up("the site refused the API key (it was replaced or mistyped)");
		if (me.status != 200) return give_up("cannot reach the site (" + Site::why(me) + ")");
		const Json* can = me.json.find("can");
		if (!can) return give_up("the site's answer to /api/v1/me is not what this version expects (" + Site::snippet(me.body) + ")");
		may_scores = flag_of(*can, "upload_scores");
		may_videos = flag_of(*can, "upload_videos");
		chunk = std::max<uint64_t>(64 << 10, (uint64_t)strtoull(text_of(me.json, "chunk_bytes").c_str(), nullptr, 10));
		const std::string user = text_of(me.json, "nickname");
		log("Upload: signed in" + (user.empty() ? std::string() : " as " + user) + "; this account may upload scores: " + (may_scores ? "yes" : "no") +
		    ", videos: " + (may_videos ? "yes" : "no") + (options.videos ? "" : " (videos are off in the config)"));
		if (!may_scores && !(may_videos && options.videos)) return give_up("this account may not upload anything");
		log("Upload: sending results to " + options.url + (may_videos && options.videos ? " (with videos)" : "") + " as the runs finish");
		return ready = true;
	}

	void process(const UploadItem& item) {
		const std::string who = "Upload " + item.label + ": ";
		try {
			const std::string text = read_text(item.result_file);
			if (text.empty()) throw std::runtime_error("cannot read the result file " + item.result_file);
			const Json result = Json::parse(text);
			const Json* k = result.find("key");
			if (!k || k->type != Json::Str) throw std::runtime_error("the result file has no key (" + item.result_file + ")");
			std::string video_hash = text_of(result, "video_sha256"), reference_hash = text_of(result, "reference_sha256");
			if (item.drop_video) { // a stream this tool does not keep any more: if the site has it, it goes
				if (!options.videos) { log(who + "superseded video left alone (videos are off)"); return; }
				if (video_hash.empty()) { log(who + "superseded video left alone (the result has no video_sha256)"); return; }
				const std::string b = "{\"key_sha256\":\"" + sha256_bytes(k->s) + "\",\"video_sha256\":\"" + video_hash + "\"}";
				const Reply check = site->request(true, "/api/v1/check", "application/json", b.data(), b.size(), 120);
				if (check.status != 200) { log(who + "check for a superseded video failed (" + Site::why(check) + ")"); problems++; return; }
				const Json* v = check.json.find("video");
				const std::string run = v && text_of(*v, "status") == "exists" ? text_of(*v, "run") : std::string();
				if (run.empty()) { log(who + "superseded video is not on the site, nothing to remove"); return; }
				const Reply del = site->request(false, "/api/v1/run/" + run + "/video", nullptr, nullptr, 0, 120, nullptr, "DELETE");
				if (del.status == 200) { videos_removed++; log(who + "video removed from the site, a better one was found"); }
				else { log(who + "its video could not be removed from the site (" + Site::why(del) + ")"); problems++; }
				return;
			}
			uint64_t video_size = 0;
			std::error_code ec;
			const bool with_video = options.videos && may_videos && !item.video_file.empty() && !video_hash.empty() && fs::exists(item.video_file, ec);
			if (with_video) video_size = fs::file_size(item.video_file, ec);
			if (reference_hash.empty()) log(who + "the result file has no reference_sha256 (its input could not be hashed); the site will not take a new score without it");

			std::string body = "{\"key_sha256\":\"" + sha256_bytes(k->s) + "\"";
			if (!reference_hash.empty()) body += ",\"reference_sha256\":\"" + reference_hash + "\""; // the site wants it for the score as well
			if (with_video) body += ",\"video_sha256\":\"" + video_hash + "\",\"video_size\":" + std::to_string(video_size);
			const Json* vm = result.find("vmaf");
			if (const Json* c = vm ? vm->find("distorted") : nullptr; c && c->type == Json::Arr && !c->a.empty()) body += ",\"distorted\":true"; // a run stored without it can get it now
			body += "}";
			const Reply check = site->request(true, "/api/v1/check", "application/json", body.data(), body.size(), 120);
			if (check.status != 200) { log(who + "check failed (" + Site::why(check) + ")"); problems++; return; }
			const Json* score = check.json.find("score");
			const Json* video = check.json.find("video");
			if (!score) { log(who + "the site's answer to the check is not what this version expects (" + Site::snippet(check.body) + ")"); problems++; return; }
			std::string run = video ? text_of(*video, "run") : std::string();
			std::string done;

			const bool send_score = flag_of(*score, "can_upload") && may_scores;
			const bool send_video = with_video && video && flag_of(*video, "can_upload") && (send_score || !run.empty());
			const long long score_bytes = send_score ? (long long)fs::file_size(item.result_file, ec) : 0;
			RunProgress rp; // shown in the live display while the files go out
			int handle = 0;
			if (score_bytes + (send_video ? (long long)video_size : 0) > 0) {
				rp.begin(4, score_bytes + (send_video ? (long long)video_size : 0));
				handle = ui::attach(&rp, item.label, "server");
			}
			struct Detach { RunProgress& rp; int handle; ~Detach() { if (handle) { rp.finish(); ui::detach(handle); } } } detach{rp, handle};
			auto chunk_log = [&](const std::string& line) { log(who + line); };

			if (send_score) {
				const Reply up = site->upload_file("score", item.result_file, "", chunk, &rp, 0, chunk_log);
				if (up.status == 200 || up.status == 201) {
					run = text_of(up.json, "uuid");
					const std::string note = text_of(up.json, "note"); // newer sites say what they did with it
					if (flag_of(up.json, "new")) { scores_new++; done = "score stored (new)"; }
					else if (flag_of(up.json, "topped_up") || flag_of(up.json, "ceiling_added")) { scores_topped++; done = "stored score topped up (distorted score or more models)"; }
					else { scores_known++; done = "score sent, but the site kept what it had"; }
					if (!note.empty()) done += ": " + note;
				} else { log(who + "the score was refused (" + Site::why(up) + ")"); problems++; return; }
			} else if (text_of(*score, "status") == "exists") {
				if (!item.late) scores_known++;
				const std::string reason = text_of(*score, "reason");
				done = "score already on the site" + (reason.empty() ? std::string() : " (" + reason + ")");
			} else if (!may_scores) decline(who, "score", "this account may not upload scores");
			else decline(who, "score", text_of(*score, "reason"));

			if (with_video) {
				const std::string status = video ? text_of(*video, "status") : std::string();
				if (send_video) {
					const Reply up = site->upload_file("video", item.video_file, "&run=" + run + "&name=" + escape(fs::path(item.video_file).filename().string()), chunk, &rp, score_bytes, chunk_log);
					if (up.status == 200 || up.status == 201) { videos_sent++; done += ", video stored"; }
					else { log(who + "the video was refused (" + Site::why(up) + ")"); problems++; }
				} else if (status == "exists") done += ", video already there";
				else if (video && flag_of(*video, "can_upload")) done += ", no video: the score is not on the site";
				else {
					const std::string reason = video ? text_of(*video, "reason") : std::string("the site said nothing about the video");
					done += ", no video";
					decline(who, "video", reason);
				}
			} else if (options.videos && may_videos && !item.video_file.empty()) done += video_hash.empty() ? ", no video: the result has no video_sha256" : ", no video: " + item.video_file + " is gone";
			if (!done.empty()) log(who + done);
		} catch (const std::exception& e) {
			log(who + "failed: " + e.what());
			problems++;
		}
	}

	void run() {
		// Signs in right away, so a wrong key or an unreachable site shows when the sweep starts, not when its first run is through.
		if (!connect()) disabled = true;
		for (;;) {
			UploadItem item;
			{
				std::unique_lock lock(m);
				cv.wait(lock, [&] { return !queue.empty() || closing; });
				if (queue.empty()) return;
				item = std::move(queue.front());
				queue.pop_front();
			}
			if (disabled) { if (skipped++ == 0) log("Upload " + item.label + ": skipped, and so is every later one (" + disabled_why + ")"); }
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
	if (!impl->ready) {
		impl->log("Upload finished: nothing sent (" + (impl->disabled_why.empty() ? std::string("not signed in") : impl->disabled_why) + "), " + std::to_string(impl->skipped) + " results skipped");
		return;
	}
	if (impl->added == 0) { impl->log("Upload finished: no results were handed to the uploader (no run of this sweep finished with a result)"); return; }
	impl->log("Upload finished: " + std::to_string(impl->added) + " handed over, " + std::to_string(impl->scores_new) + " new scores, " + std::to_string(impl->scores_topped) + " topped up, " +
	          std::to_string(impl->scores_known) + " already known, " + std::to_string(impl->videos_sent) + " videos" + (impl->videos_removed ? ", " + std::to_string(impl->videos_removed) + " superseded videos removed" : "") +
	          (impl->problems ? ", " + std::to_string(impl->problems) + " problems (they are tried again next time)" : ""));
	for (auto& [what, n] : impl->declined) impl->log("Upload: " + std::to_string(n) + " not sent, " + what);
}
