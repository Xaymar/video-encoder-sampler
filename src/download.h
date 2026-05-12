#pragma once
// Inputs given as http(s) URLs are fetched once into a local folder and used from there, so that large test videos do not have to be
// kept on the machine by hand. A later run asks the server whether the file changed (a conditional request) and only downloads it
// again if it did; an interrupted download is resumed where it stopped.
#include <string>

bool is_url(const std::string& s);

// The name the file gets (and shows under), from the URL: its last path segment with anything but letters, digits and . - _ replaced.
std::string url_file_name(const std::string& url);

// Returns the local path of the file behind `url`, downloading it into `dir` if needed. With `progress` the live display shows it (stage 5, bytes);
// without, progress is printed to stdout.
// Throws std::runtime_error when the file cannot be had (no server and no earlier copy, HTTP errors, a short download).
struct RunProgress;
std::string fetch_url(const std::string& url, const std::string& dir, RunProgress* progress = nullptr);
