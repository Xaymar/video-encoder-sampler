#pragma once
// Uploads results (and their encoded videos) to the VES website, see docs/api.md of that project: GET /api/v1/me, POST /api/v1/check and
// POST /api/v1/upload in chunks. Nothing here stops a sweep: whatever cannot be uploaded is reported and left for the next run.
#include <functional>
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
	std::string label;       // for the messages
};

// `log` receives one line at a time.
void upload_results(const UploadOptions& options, const std::vector<UploadItem>& items, const std::function<void(const std::string&)>& log);
