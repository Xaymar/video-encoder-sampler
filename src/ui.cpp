#include "ui.h"

#include "json.h"

#define NOMINMAX

#include <algorithm>
#include <chrono>
#include <condition_variable>
#include <ctime>
#include <cstdio>
#include <csignal>
#include <deque>
#include <fcntl.h>
#include <map>
#include <mutex>
#include <thread>
#include <vector>

#ifdef _WIN32
#include <io.h>
#include <windows.h>
#include <psapi.h>
#define pipe_open(fds) _pipe(fds, 1 << 16, _O_BINARY)
#else
#include <sys/ioctl.h>
#include <sys/resource.h>
#include <unistd.h>
#define pipe_open(fds) pipe(fds)
#define _dup dup
#define _dup2 dup2
#define _close close
#define _read read
#endif

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
int next_handle = 1, live_lines = 0, runs_done = 0, runs_total = 0, uploads_done = 0, uploads_total = 0;
bool is_live = false, running = false;
Clock::time_point t0;
std::string custom_title;
bool progress_stdout = false;
int64_t last_snapshot = 0;
std::map<std::string, std::pair<std::string, std::vector<ui::Progress>>> remote_rows; // by worker and job: the prefix of their labels, and the runs
std::thread renderer, stderr_reader;
int saved_stderr = -1, pipe_read = -1;
#ifdef _WIN32
unsigned saved_code_page = 0;
#endif

// Everything for stdout goes through one writer thread, so that a console that does not take output (on Windows a click into it starts a selection, which holds
// every write until it ends) holds only that thread, not the sweep: frames are skipped meanwhile and log lines wait.
std::mutex out_m;
std::condition_variable out_cv;
std::string out_q;
bool out_busy = false, out_quit = false, out_running = false;
std::thread out_thread;
void emit(const std::string& text) {
	{
		std::lock_guard lock(out_m);
		if (out_running) { out_q += text; out_cv.notify_all(); return; }
	}
	fwrite(text.data(), 1, text.size(), stdout); // no writer (an encoder process, or before start()): as it comes
	fflush(stdout);
}
bool output_behind() {
	std::lock_guard lock(out_m);
	return out_busy || !out_q.empty();
}
void out_loop() {
	std::unique_lock lock(out_m);
	for (;;) {
		out_cv.wait(lock, [] { return out_quit || !out_q.empty(); });
		if (out_q.empty()) return;
		std::string text;
		text.swap(out_q);
		out_busy = true;
		lock.unlock();
		fwrite(text.data(), 1, text.size(), stdout);
		fflush(stdout);
		lock.lock();
		out_busy = false;
	}
}
void start_output() {
	std::lock_guard lock(out_m);
	if (out_running) return;
	out_running = true;
	out_quit = false;
	out_thread = std::thread(out_loop);
}
void stop_output() { // writes what is left first
	{
		std::lock_guard lock(out_m);
		if (!out_running) return;
		out_quit = true;
	}
	out_cv.notify_all();
	out_thread.join();
	std::lock_guard lock(out_m);
	out_running = false;
}

// --log: every permanent line, with the date and time, in a file of its own.
std::mutex log_m;
FILE* log_file = nullptr;
void to_log_file(const std::string& line) {
	std::lock_guard lock(log_m);
	if (!log_file) return;
	const auto now = std::chrono::system_clock::now();
	const time_t t = std::chrono::system_clock::to_time_t(now);
	const int ms = (int)(std::chrono::duration_cast<std::chrono::milliseconds>(now.time_since_epoch()).count() % 1000);
	tm local{};
#ifdef _WIN32
	localtime_s(&local, &t);
#else
	localtime_r(&t, &local);
#endif
	char stamp[40];
	const size_t n = strftime(stamp, sizeof stamp, "%Y-%m-%d %H:%M:%S", &local);
	snprintf(stamp + n, sizeof stamp - n, ".%03d ", ms);
	fputs(stamp, log_file);
	fputs(line.c_str(), log_file);
	fputc('\n', log_file);
	fflush(log_file);
}

std::string span(double sec) {
	char b[32];
	if (sec < 90) snprintf(b, sizeof b, "%.0fs", sec);
	else if (sec < 5400) snprintf(b, sizeof b, "%.0fm%02.0fs", (double)(int)(sec / 60), (double)((int)sec % 60));
	else snprintf(b, sizeof b, "%.0fh%02.0fm", (double)(int)(sec / 3600), (double)((int)(sec / 60) % 60));
	return b;
}

#ifdef _WIN32
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
#else
int console_width() {
	winsize ws;
	return ioctl(STDOUT_FILENO, TIOCGWINSZ, &ws) == 0 && ws.ws_col ? std::max<int>(40, ws.ws_col) : 100;
}

int console_height() {
	winsize ws;
	return ioctl(STDOUT_FILENO, TIOCGWINSZ, &ws) == 0 && ws.ws_row ? std::max<int>(8, ws.ws_row) : 30;
}
#endif

// Text is UTF-8 and the frame is drawn with box characters; widths count code points (labels are ASCII apart from file names).
int cells(const std::string& s) {
	int n = 0;
	for (unsigned char c : s) n += (c & 0xC0) != 0x80;
	return n;
}

// The first `width` characters of s, ending in an ellipsis when something was cut.
std::string cut(const std::string& s, int width) {
	if (width <= 0) return "";
	if (cells(s) <= width) return s;
	int n = 0;
	size_t i = 0;
	for (; i < s.size(); i++) {
		if (((unsigned char)s[i] & 0xC0) != 0x80 && n++ == width - 1) break;
	}
	return s.substr(0, i) + "\xE2\x80\xA6";
}

std::string pad(const std::string& s, int width, bool right = false) {
	std::string t = cut(s, width);
	const std::string fill(std::max(0, width - cells(t)), ' ');
	return right ? fill + t : t + fill;
}

std::string repeat(const char* s, int n) {
	std::string o;
	for (int i = 0; i < n; i++) o += s;
	return o;
}

// A bar of `width` cells, filled in eighths of a cell.
std::string bar(double fraction, int width) {
	static const char* part[] = {"", "\xE2\x96\x8F", "\xE2\x96\x8E", "\xE2\x96\x8D", "\xE2\x96\x8C", "\xE2\x96\x8B", "\xE2\x96\x8A", "\xE2\x96\x89"};
	const int eighths = (int)(std::min(1.0, std::max(0.0, fraction)) * width * 8);
	const int full = eighths / 8;
	std::string o = repeat("\xE2\x96\x88", full);
	int used = full;
	if (eighths % 8 && used < width) { o += part[eighths % 8]; used++; }
	return o + repeat("\xC2\xB7", width - used);
}

// What a run in flight shows: its task, speed, status and how far it is.
struct Row {
	std::string task, speed, status;
	double fraction = 0;
};

Row row_of(const ui::Progress& p, const std::string& prefix) {
	static const char* names[] = {"wait", "encode", "score", "decode", "upload", "download"};
	static const char* waits[] = {"working", "starved", "backed up"};
	Row r;
	const std::string stage = names[p.stage >= 0 && p.stage < 6 ? p.stage : 0];
	r.task = prefix + (p.label.find(stage) != std::string::npos ? p.label : stage + " " + p.label); // some labels already begin with it (decode+convert ..)
	r.fraction = p.total > 0 ? (double)p.done / p.total : 0;
	char b[32];
	if (p.stage == 4 || p.stage == 5) snprintf(b, sizeof b, "%.1f MB/s", p.rate / 1048576.0);
	else snprintf(b, sizeof b, "%.1f fps", p.rate);
	r.speed = b;
	r.status = p.finished ? (p.failed ? "failed" : "done") : p.stage == 0 ? "waiting" : waits[p.state >= 0 && p.state < 3 ? p.state : 0];
	return r;
}

ui::Progress progress_of(const Entry& e) {
	ui::Progress p;
	const int64_t end = e.p->end_ms;
	const double secs = std::max(0.001, ((end ? end : now_ms()) - e.p->since_ms) / 1000.0);
	p.stage = e.p->stage;
	p.label = e.label;
	p.done = e.p->done;
	p.total = e.p->total;
	p.rate = p.done / secs;
	p.state = e.p->state;
	p.finished = end != 0;
	p.failed = e.p->failed;
	return p;
}

Row make_row(const Entry& e) { return row_of(progress_of(e), ""); }

// The live area: a frame with the overall progress in its top border and one row per run in flight. It takes the width of the terminal: the
// task column and the bar share what the fixed columns leave, and columns drop out when the terminal is too narrow for them.
void render() {
	if (output_behind()) return; // the last frame is not out yet: the lines wait for the next one
	const int term_w = console_width(), term_h = console_height();
	static int last_w = 0;
	std::string out;
	if (live_lines) {
		// Narrowing the window re-wraps the old frame into more rows than it had.
		int rows = live_lines;
		if (last_w && term_w < last_w) rows = std::min(term_h - 1, rows * ((last_w + term_w - 1) / term_w));
		out += "\x1b[" + std::to_string(rows) + "A";
	}
	last_w = term_w;
	out += "\r\x1b[J";
	for (auto& l : pending) out += l + "\n";
	pending.clear();
	const int width = term_w - 1;

	const double elapsed = std::chrono::duration<double>(Clock::now() - t0).count();
	char head[160];
	if (runs_total > 0 && runs_done >= runs_total) snprintf(head, sizeof head, "ves  %d/%d runs finished  %s elapsed", runs_done, runs_total, span(elapsed).c_str());
	else if (runs_total > 0) snprintf(head, sizeof head, "ves  %d/%d runs finished  %s elapsed  ~%s left", runs_done, runs_total, span(elapsed).c_str(), runs_done ? span(elapsed / runs_done * (runs_total - runs_done)).c_str() : "?");
	else snprintf(head, sizeof head, "ves  preparing  %s elapsed", span(elapsed).c_str());
	std::string title = head;
	if (uploads_total > 0) title += "  |  uploads " + std::to_string(uploads_done) + "/" + std::to_string(uploads_total);
	if (!custom_title.empty()) title = custom_title;

	std::vector<Row> rows;
	for (auto& [h, e] : entries) rows.push_back(make_row(e));
	for (auto& [key, group] : remote_rows) for (auto& p : group.second) rows.push_back(row_of(p, group.first));
	// Rows that fit: the frame has a top border, a header, a rule and a bottom border around them.
	const size_t room = (size_t)std::max(1, term_h - 6);
	size_t hidden = 0;
	if (rows.size() > room) { hidden = rows.size() - (room - 1); rows.resize(room - 1); }

	int speed_w = 8, status_w = 9;
	for (auto& r : rows) speed_w = std::max(speed_w, cells(r.speed));
	bool show_speed = true;
	// Each column costs its width, two spaces of padding and a border; the frame has one more border at the end.
	auto fixed = [&] { return 1 + (show_speed ? speed_w + 3 : 0) + status_w + 3 + 3 + 3; }; // + task and bar paddings and borders
	if (width - fixed() < 30) show_speed = false;
	const int rest = std::max(10, width - fixed());
	const int bar_w = std::max(4, std::min(40, rest * 2 / 5));
	const int task_w = std::max(4, rest - bar_w);

	struct Col { std::string name; int w; bool right; };
	std::vector<Col> cols;
	cols.push_back({"Task", task_w, false});
	if (show_speed) cols.push_back({"Speed", speed_w, true});
	cols.push_back({"Status", status_w, false});
	cols.push_back({"Progress", bar_w, false});

	// Top border with the title in it.
	int total_w = 1;
	for (auto& c : cols) total_w += c.w + 3;
	const std::string title_cut = cut(title, std::max(1, total_w - 6));
	std::string top = "\xE2\x94\x8C\xE2\x94\x80 " + title_cut + " " + repeat("\xE2\x94\x80", std::max(0, total_w - 5 - cells(title_cut))) + "\xE2\x94\x90";
	auto rule = [&](const char* l, const char* mid, const char* r) {
		std::string o = l;
		for (size_t i = 0; i < cols.size(); i++) o += repeat("\xE2\x94\x80", cols[i].w + 2) + (i + 1 < cols.size() ? mid : r);
		return o;
	};
	auto line = [&](const std::vector<std::string>& cells_) {
		std::string o = "\xE2\x94\x82";
		for (size_t i = 0; i < cols.size(); i++) o += " " + cells_[i] + " \xE2\x94\x82";
		return o;
	};
	out += top + "\x1b[K\n";
	{
		std::vector<std::string> h;
		for (auto& c : cols) h.push_back(pad(c.name, c.w, c.right));
		out += line(h) + "\x1b[K\n";
	}
	out += rule("\xE2\x94\x9C", "\xE2\x94\xBC", "\xE2\x94\xA4") + "\x1b[K\n";
	int lines = 3;
	for (auto& r : rows) {
		std::vector<std::string> c;
		c.push_back(pad(r.task, task_w));
		if (show_speed) c.push_back(pad(r.speed, speed_w, true));
		c.push_back(pad(r.status, status_w));
		c.push_back(bar(r.fraction, bar_w));
		out += line(c) + "\x1b[K\n";
		lines++;
	}
	if (hidden) {
		std::vector<std::string> c(cols.size());
		for (size_t i = 0; i < cols.size(); i++) c[i] = pad(cols[i].name == "Task" ? "+" + std::to_string(hidden) + " more" : "", cols[i].w);
		out += line(c) + "\x1b[K\n";
		lines++;
	}
	out += rule("\xE2\x94\x94", "\xE2\x94\xB4", "\xE2\x94\x98") + "\x1b[K\n";
	live_lines = lines + 1;
	emit(out);
}

// Without a console to draw on: one line per run in flight now and then.
void plain_status() {
	static int64_t last = now_ms();
	if (now_ms() - last < 15000) return;
	last = now_ms();
	std::string out;
	char b[1024];
	for (auto& [h, e] : entries) {
		const long long total = e.p->total, done = e.p->done;
		if (e.p->stage == 0 || total <= 0 || e.p->end_ms) continue; // not started, or already finished
		if (e.p->stage == 4 || e.p->stage == 5) { snprintf(b, sizeof b, "  %s: %lld%% (%.1f/%.1f MB)  %s\n", e.p->stage == 4 ? "uploading" : "downloading", done * 100 / total, done / 1048576.0, total / 1048576.0, e.label.c_str()); out += b; continue; }
		static const char* waits[] = {"", " [starved: waiting for frames]", " [backed up: waiting for room downstream]"};
		const int ws = e.p->state;
		snprintf(b, sizeof b, "  %s %s: %lld%% (%lld/%lld frames)%s  %s\n", e.p->stage == 1 ? "encoding" : e.p->stage == 3 ? "decoding" : "scoring", e.pool.c_str(), done * 100 / total, done, total, waits[ws >= 0 && ws < 3 ? ws : 0], e.label.c_str());
		out += b;
	}
	for (auto& [key, group] : remote_rows)
		for (auto& p : group.second) {
			const Row r = row_of(p, group.first);
			snprintf(b, sizeof b, "  %s: %.0f%% %s %s\n", r.status.c_str(), r.fraction * 100, r.speed.c_str(), r.task.c_str());
			out += b;
		}
	if (!out.empty()) emit(out);
}

std::string json_string(const std::string& s) {
	std::string o = "\"";
	for (unsigned char c : s) {
		if (c == '"' || c == '\\') { o += '\\'; o += (char)c; }
		else if (c < 0x20) { char b[8]; snprintf(b, sizeof b, "\\u%04x", c); o += b; }
		else o += (char)c;
	}
	return o + "\"";
}

// One line for the process that started this one: a record separator, then [stage, label, done, total, rate, state, finished] for every run in flight.
std::string snapshot() {
	std::string line = "[";
	bool first = true;
	auto add = [&](const ui::Progress& p, const std::string& prefix) {
		char numbers[160];
		snprintf(numbers, sizeof numbers, ",%lld,%lld,%.3f,%d,%d]", p.done, p.total, p.rate, p.state, p.finished ? (p.failed ? 2 : 1) : 0);
		line += std::string(first ? "" : ",") + "[" + std::to_string(p.stage) + "," + json_string(prefix + p.label) + numbers;
		first = false;
	};
	for (auto& [h, e] : entries) add(progress_of(e), "");
	// and the rows this process shows for others: the runs of its device processes, which a worker's host wants to see as much as the runs of this one
	for (auto& [key, group] : remote_rows) for (auto& p : group.second) add(p, group.first);
	return line + "]";
}

void write_snapshot() {
	emit(std::string(1, (char)30) + snapshot() + "\n");
}

void render_loop() {
	for (;;) {
		std::this_thread::sleep_for(std::chrono::milliseconds(250));
		std::lock_guard lock(m);
		if (!running) return;
		if (progress_stdout && now_ms() - last_snapshot >= 250) { last_snapshot = now_ms(); write_snapshot(); }
		if (is_live) render(); else plain_status();
	}
}

// Everything the libraries write to stderr would tear the live area, so it goes through the permanent lines instead.
void capture_stderr() {
	int fds[2];
	if (pipe_open(fds) != 0) return;
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

void RunProgress::finish() { end_ms = now_ms(); }

void RunProgress::begin(int st, long long tot) {
	end_ms = 0;
	failed = false;
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
	start_output();
#ifdef _WIN32
	HANDLE out = GetStdHandle(STD_OUTPUT_HANDLE);
	DWORD mode = 0;
	const bool terminal = GetConsoleMode(out, &mode) && SetConsoleMode(out, mode | ENABLE_VIRTUAL_TERMINAL_PROCESSING);
#else
	const bool terminal = isatty(STDOUT_FILENO) != 0;
#endif
	if (!plain && terminal) {
		is_live = true;
#ifdef _WIN32
		saved_code_page = GetConsoleOutputCP();
		SetConsoleOutputCP(CP_UTF8); // the frame is drawn with box characters
#endif
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
		std::string out;
		if (live_lines) { out = "\x1b[" + std::to_string(live_lines) + "A\r\x1b[J"; live_lines = 0; }
		for (auto& l : pending) out += l + "\n";
		pending.clear();
		emit(out);
		m.unlock();
		stop_output(); // what is left is written before stderr goes back and the live display ends
		m.lock();
		if (saved_stderr >= 0) {
			_dup2(saved_stderr, 2); // closes the pipe's write end, which ends the reader
			_close(saved_stderr);
			saved_stderr = -1;
			m.unlock();
			stderr_reader.join();
			m.lock();
		}
		is_live = false;
#ifdef _WIN32
		if (saved_code_page) SetConsoleOutputCP(saved_code_page);
#endif
	}
	m.unlock();
	stop_output();
	m.lock();
}

bool live() { return is_live; }

void set_title(const std::string& title) {
	std::lock_guard lock(m);
	custom_title = title;
}

#ifdef _WIN32
static BOOL WINAPI on_interrupt(DWORD type) {
	if (type != CTRL_C_EVENT && type != CTRL_BREAK_EVENT && type != CTRL_CLOSE_EVENT) return FALSE;
	TerminateProcess(GetCurrentProcess(), 130);
	return TRUE;
}
void end_on_interrupt() { SetConsoleCtrlHandler(on_interrupt, TRUE); }
#else
static void on_interrupt(int) { _exit(130); }
void end_on_interrupt() {
	signal(SIGINT, on_interrupt);
	signal(SIGTERM, on_interrupt);
}
#endif

void emit_record(char kind, const std::string& payload) {
	std::lock_guard lock(m);
	if (!progress_stdout) return;
	emit(std::string(1, kind) + payload + "\n");
}

std::string snapshot_json() {
	std::lock_guard lock(m);
	return snapshot();
}

void set_progress_stdout(bool on) {
	std::lock_guard lock(m);
	progress_stdout = on;
}

std::vector<Progress> parse_progress(const std::string& json) {
	std::vector<Progress> list;
	try {
		Json rows = Json::parse(json);
		for (auto& r : rows.a) {
			if (r.a.size() < 7) continue;
			Progress p;
			p.stage = atoi(r.a[0].text().c_str());
			p.label = r.a[1].text();
			p.done = atoll(r.a[2].text().c_str());
			p.total = atoll(r.a[3].text().c_str());
			p.rate = atof(r.a[4].text().c_str());
			p.state = atoi(r.a[5].text().c_str());
			p.finished = r.a[6].text() != "0";
			p.failed = r.a[6].text() == "2";
			list.push_back(std::move(p));
		}
	} catch (const std::exception&) {}
	return list;
}

void set_remote(const std::string& key, const std::string& prefix, const std::vector<Progress>& list) {
	std::lock_guard lock(m);
	if (list.empty()) remote_rows.erase(key);
	else remote_rows[key] = {prefix, list};
}

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
	to_log_file(line);
	std::lock_guard lock(m);
	if (is_live) pending.push_back(line);
	else emit(line + "\n");
}

void log_plain(const std::string& line) {
	to_log_file(line);
	std::lock_guard lock(m);
	if (!is_live) emit(line + "\n");
}

bool open_log(const std::string& path, const std::string& first_line, std::string& err) {
	FILE* f = fopen(path.c_str(), "ab"); // appended to: a restart keeps what came before
	if (!f) { err = "cannot open the log file '" + path + "'"; return false; }
	{
		std::lock_guard lock(log_m);
		if (log_file) fclose(log_file);
		log_file = f;
	}
	to_log_file(first_line);
	return true;
}

long long peak_memory_mb() {
#ifdef _WIN32
	PROCESS_MEMORY_COUNTERS pmc{};
	return K32GetProcessMemoryInfo(GetCurrentProcess(), &pmc, sizeof pmc) ? (long long)(pmc.PeakWorkingSetSize >> 20) : 0;
#else
	rusage ru{};
	return getrusage(RUSAGE_SELF, &ru) == 0 ? (long long)(ru.ru_maxrss >> 10) : 0; // kilobytes on Linux
#endif
}

void uploads(int sent, int queued) {
	std::lock_guard lock(m);
	uploads_done = sent;
	uploads_total = queued;
}

void overall(int done, int total) {
	std::lock_guard lock(m);
	runs_done = done;
	runs_total = total;
}

} // namespace ui
