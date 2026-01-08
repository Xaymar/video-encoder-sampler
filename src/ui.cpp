#include "ui.h"

#define NOMINMAX

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <deque>
#include <fcntl.h>
#include <io.h>
#include <map>
#include <mutex>
#include <thread>

#include <windows.h>
#include <psapi.h>

namespace {

using Clock = std::chrono::steady_clock;

int64_t now_ms() { return std::chrono::duration_cast<std::chrono::milliseconds>(Clock::now().time_since_epoch()).count(); }

struct Entry {
	RunProgress* p;
	std::string label, pool;
};

std::mutex m;
std::map<int, Entry> entries;
std::deque<std::string> pending; // permanent lines not printed yet
int next_handle = 1, live_lines = 0, runs_done = 0, runs_total = 0;
bool is_live = false, running = false;
Clock::time_point t0;
std::thread renderer, stderr_reader;
int saved_stderr = -1, pipe_read = -1;

std::string span(double sec) {
	char b[32];
	if (sec < 90) snprintf(b, sizeof b, "%.0fs", sec);
	else if (sec < 5400) snprintf(b, sizeof b, "%.0fm%02.0fs", (double)(int)(sec / 60), (double)((int)sec % 60));
	else snprintf(b, sizeof b, "%.0fh%02.0fm", (double)(int)(sec / 3600), (double)((int)(sec / 60) % 60));
	return b;
}

int console_width() {
	CONSOLE_SCREEN_BUFFER_INFO info;
	if (GetConsoleScreenBufferInfo(GetStdHandle(STD_OUTPUT_HANDLE), &info)) return std::max(40, info.srWindow.Right - info.srWindow.Left + 1);
	return 100;
}

int console_height() {
	CONSOLE_SCREEN_BUFFER_INFO info;
	if (GetConsoleScreenBufferInfo(GetStdHandle(STD_OUTPUT_HANDLE), &info)) return std::max(8, info.srWindow.Bottom - info.srWindow.Top + 1);
	return 30;
}

std::string fit(std::string s, int width) {
	if ((int)s.size() > width) s.resize(width);
	return s;
}

// One line per run in flight.
std::string run_line(const Entry& e, int width) {
	static const char* names[] = {"wait ", "encode", "score ", "decode"};
	const int st = e.p->stage;
	const long long done = e.p->done, total = e.p->total;
	const double secs = std::max(0.001, (now_ms() - e.p->since_ms) / 1000.0);
	char head[96];
	const int pct = total > 0 ? (int)std::min<long long>(100, done * 100 / total) : 0;
	std::string bar(20, '-');
	for (int i = 0; i < pct / 5; i++) bar[i] = '#';
	static const char* waits[] = {"", "starved", "backed up"};
	const int ws = e.p->state;
	snprintf(head, sizeof head, "%-9.9s %-6s [%s] %3d%% %5lld/%-5lld %6.1f fps %-9s ", e.pool.c_str(), names[st >= 0 && st < 4 ? st : 0], bar.c_str(), pct, done, total, done / secs, waits[ws >= 0 && ws < 3 ? ws : 0]);
	return fit(head + e.label, width);
}

void render() {
	std::string out;
	if (live_lines) out += "\x1b[" + std::to_string(live_lines) + "A";
	out += "\r\x1b[J";
	for (auto& l : pending) out += l + "\n";
	pending.clear();
	const int width = console_width() - 1;
	const double elapsed = std::chrono::duration<double>(Clock::now() - t0).count();
	char head[160];
	if (runs_total > 0) {
		snprintf(head, sizeof head, "ves  %d/%d runs finished  %s elapsed  ~%s left", runs_done, runs_total, span(elapsed).c_str(),
		         runs_done ? span(elapsed / runs_done * (runs_total - runs_done)).c_str() : "?");
	} else snprintf(head, sizeof head, "ves  preparing  %s elapsed", span(elapsed).c_str());
	out += fit(head, width) + "\x1b[K\n";
	int lines = 1;
	const int room = console_height() - 3;
	for (auto& [h, e] : entries) {
		if (lines > room) break;
		out += run_line(e, width) + "\x1b[K\n";
		lines++;
	}
	live_lines = lines;
	fwrite(out.data(), 1, out.size(), stdout);
	fflush(stdout);
}

// Without a console to draw on: one line per run in flight now and then.
void plain_status() {
	static int64_t last = now_ms();
	if (now_ms() - last < 15000) return;
	last = now_ms();
	for (auto& [h, e] : entries) {
		const long long total = e.p->total, done = e.p->done;
		if (e.p->stage == 0 || total <= 0) continue;
		static const char* waits[] = {"", " [starved: waiting for frames]", " [backed up: waiting for room downstream]"};
		const int ws = e.p->state;
		printf("  %s %s: %lld%% (%lld/%lld frames)%s  %s\n", e.p->stage == 1 ? "encoding" : e.p->stage == 3 ? "decoding" : "scoring", e.pool.c_str(), done * 100 / total, done, total, waits[ws >= 0 && ws < 3 ? ws : 0], e.label.c_str());
	}
	fflush(stdout);
}

void render_loop() {
	for (;;) {
		std::this_thread::sleep_for(std::chrono::milliseconds(200));
		std::lock_guard lock(m);
		if (!running) return;
		if (is_live) render(); else plain_status();
	}
}

// Everything the libraries write to stderr would tear the live area, so it goes through the permanent lines instead.
void capture_stderr() {
	int fds[2];
	if (_pipe(fds, 1 << 16, _O_BINARY) != 0) return;
	saved_stderr = _dup(2);
	_dup2(fds[1], 2);
	_close(fds[1]);
	pipe_read = fds[0];
	stderr_reader = std::thread([] {
		std::string line;
		char buf[512];
		int n;
		while ((n = _read(pipe_read, buf, sizeof buf)) > 0) {
			for (int i = 0; i < n; i++) {
				if (buf[i] == '\n') { if (!line.empty()) ui::log(line); line.clear(); }
				else if (buf[i] != '\r') line += buf[i];
			}
		}
		if (!line.empty()) ui::log(line);
	});
}

} // namespace

void RunProgress::begin(int st, long long tot) {
	busy_ms = starved_ms = blocked_ms = 0;
	state = 0;
	done = 0;
	total = tot;
	since_ms = now_ms();
	stage = st;
}

namespace ui {

void start(bool plain) {
	std::lock_guard lock(m);
	t0 = Clock::now();
	running = true;
	HANDLE out = GetStdHandle(STD_OUTPUT_HANDLE);
	DWORD mode = 0;
	if (!plain && GetConsoleMode(out, &mode) && SetConsoleMode(out, mode | ENABLE_VIRTUAL_TERMINAL_PROCESSING)) {
		is_live = true;
		capture_stderr();
	}
	renderer = std::thread(render_loop);
}

void stop() {
	{
		std::lock_guard lock(m);
		if (!running) return;
		running = false;
	}
	renderer.join();
	std::lock_guard lock(m);
	if (is_live) {
		if (live_lines) { std::string up = "\x1b[" + std::to_string(live_lines) + "A\r\x1b[J"; fwrite(up.data(), 1, up.size(), stdout); live_lines = 0; }
		for (auto& l : pending) printf("%s\n", l.c_str());
		pending.clear();
		fflush(stdout);
		if (saved_stderr >= 0) {
			_dup2(saved_stderr, 2); // closes the pipe's write end, which ends the reader
			_close(saved_stderr);
			saved_stderr = -1;
			m.unlock();
			stderr_reader.join();
			m.lock();
		}
		is_live = false;
	}
}

bool live() { return is_live; }

int attach(RunProgress* p, const std::string& label, const std::string& pool) {
	std::lock_guard lock(m);
	entries[next_handle] = {p, label, pool};
	return next_handle++;
}

void detach(int handle) {
	std::lock_guard lock(m);
	entries.erase(handle);
}

void log(const std::string& line) {
	std::lock_guard lock(m);
	if (is_live) pending.push_back(line);
	else { printf("%s\n", line.c_str()); fflush(stdout); }
}

void log_plain(const std::string& line) {
	std::lock_guard lock(m);
	if (!is_live) { printf("%s\n", line.c_str()); fflush(stdout); }
}

long long peak_memory_mb() {
	PROCESS_MEMORY_COUNTERS pmc{};
	return K32GetProcessMemoryInfo(GetCurrentProcess(), &pmc, sizeof pmc) ? (long long)(pmc.PeakWorkingSetSize >> 20) : 0;
}

void overall(int done, int total) {
	std::lock_guard lock(m);
	runs_done = done;
	runs_total = total;
}

} // namespace ui
