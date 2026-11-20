#pragma once
// Helpers for files that several ves processes (or threads) on one machine can touch at the same time.
#include <filesystem>
#include <string>

// A name for a temporary file next to `target` that no other process or thread uses: `target`.<process id>.<counter>.<suffix>.
// Write there, then rename over `target`, so that nobody ever sees (or mixes up) a half-written file.
std::filesystem::path unique_temp_path(const std::filesystem::path& target, const char* suffix = "tmp");

// An exclusive lock on a file, held for as long as the object lives. The operating system lets go of it when the process ends,
// so a crash never leaves it locked. Waits for the holder (polling, up to `wait_seconds`; <0 = forever); `locked()` tells if it got it.
class FileLock {
public:
	FileLock(const std::filesystem::path& lock_file, int wait_seconds = -1);
	~FileLock();
	FileLock(const FileLock&) = delete;
	FileLock& operator=(const FileLock&) = delete;
	bool locked() const { return held; }
private:
	bool held = false;
	long long handle = -1; // fd, or HANDLE on Windows
};
