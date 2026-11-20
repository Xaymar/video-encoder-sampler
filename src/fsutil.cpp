#include "fsutil.h"

#include <atomic>
#include <chrono>
#include <thread>

#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#else
#include <fcntl.h>
#include <sys/file.h>
#include <unistd.h>
#endif

namespace fs = std::filesystem;

fs::path unique_temp_path(const fs::path& target, const char* suffix) {
	static std::atomic<unsigned> counter{0};
#ifdef _WIN32
	const unsigned long pid = GetCurrentProcessId();
#else
	const unsigned long pid = (unsigned long)getpid();
#endif
	fs::path p = target;
	p += "." + std::to_string(pid) + "." + std::to_string(counter++) + "." + suffix;
	return p;
}

FileLock::FileLock(const fs::path& lock_file, int wait_seconds) {
	std::error_code ec;
	fs::create_directories(lock_file.parent_path(), ec);
	const auto start = std::chrono::steady_clock::now();
	auto expired = [&] { return wait_seconds >= 0 && std::chrono::steady_clock::now() - start > std::chrono::seconds(wait_seconds); };
#ifdef _WIN32
	HANDLE h = CreateFileW(lock_file.c_str(), GENERIC_READ | GENERIC_WRITE, FILE_SHARE_READ | FILE_SHARE_WRITE | FILE_SHARE_DELETE, nullptr, OPEN_ALWAYS, FILE_ATTRIBUTE_NORMAL, nullptr);
	if (h == INVALID_HANDLE_VALUE) return;
	for (;;) {
		OVERLAPPED ov{};
		if (LockFileEx(h, LOCKFILE_EXCLUSIVE_LOCK | LOCKFILE_FAIL_IMMEDIATELY, 0, 1, 0, &ov)) { held = true; break; }
		if (expired()) break;
		std::this_thread::sleep_for(std::chrono::milliseconds(200));
	}
	if (held) handle = (long long)(intptr_t)h; else CloseHandle(h);
#else
	const int fd = open(lock_file.c_str(), O_RDWR | O_CREAT | O_CLOEXEC, 0644);
	if (fd < 0) return;
	for (;;) {
		if (flock(fd, LOCK_EX | LOCK_NB) == 0) { held = true; break; }
		if (expired()) break;
		std::this_thread::sleep_for(std::chrono::milliseconds(200));
	}
	if (held) handle = fd; else close(fd);
#endif
}

FileLock::~FileLock() {
	if (!held) return;
#ifdef _WIN32
	CloseHandle((HANDLE)(intptr_t)handle); // closing releases the lock
#else
	close((int)handle);
#endif
}
