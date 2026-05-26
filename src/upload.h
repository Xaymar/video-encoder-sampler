#pragma once
// Uploads results (and their encoded videos) to the VES website, see docs/api.md of that project: GET /api/v1/me, POST /api/v1/check and
// POST /api/v1/upload in chunks. Nothing here stops a sweep: whatever cannot be uploaded is reported and left for the next run.
#include <functional>
#include <memory>
#include <string>
#include <vector>

struct UploadOptions {
	std::string url;     // the site's address, e.g. https://ves.example.com
	std::string key;     // the API key (ves_...), from the account page
	bool videos = true;  // also upload the encoded videos where the site allows it
};

struct UploadItem {
	std::string result_file; // the run's result .json
	std::string video_file;  // its encoded video; empty if none was saved
	bool drop_video = false; // not an upload: delete this run's video from the site, if it has one (a better stream was found)
	bool late = false;      // the score went out with an earlier item; this one brings the video (not counted as another score)
	std::string label;       // for the messages
};

// Sends results to the site on a thread of its own, one after the other, as they are added: a run goes out right after it was scored, while the
// next ones are still being encoded. `log` receives one line at a time (from that thread). Nothing here fails a sweep: what cannot be sent is
// reported and left for the next run, and once the key is refused or the site cannot be reached the rest is skipped.
class Uploader {
public:
	Uploader(const UploadOptions& options, std::function<void(const std::string&)> log);
	~Uploader(); // waits like finish()
	void add(UploadItem item);
	void finish(); // waits until everything added is sent or given up on, and says how it went

private:
	struct Impl;
	std::unique_ptr<Impl> impl;
};
