#include "encoder_process.h"

#include <algorithm>
#include <atomic>
#include <cerrno>
#include <chrono>
#include <condition_variable>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <map>
#include <mutex>
#include <random>
#include <thread>
#include <vector>

#include "convert.h"
#include "formats.h"
#include "gpu.h"
#include "pipeline.h"
#include "source.h"
#include "vk_compute.h"
#include "priority.h"
#include "remote.h"
#include "ui.h"

#ifdef _WIN32
#define NOMINMAX
#include <windows.h>
#include <io.h>
#else
#include <fcntl.h>
#include <signal.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <sys/wait.h>
#include <unistd.h>
#endif

namespace {

std::atomic<bool> isolation{true};
std::mutex args_m;
std::vector<std::string> process_args;

// ---- messages ----
// Every message is a header (payload size, type, id) and the payload, the same way in both directions. Numbers are in the byte order of the machine: both
// ends are the same program on the same machine.
//   to the encoder process:   'W' a wave (input, source, scaling and VMAF options, the runs)   'S' supports (codec, chroma, depth)   'K' capabilities (codec)
//   from it:                  'P' the wave's progress rows (as ui::snapshot_json)   'w' the wave's results   's' supports (1/0)   'k' capabilities (JSON)
constexpr size_t HEADER = 9;

struct Writer {
	std::string s;
	template <class T> Writer& num(T v) { s.append((const char*)&v, sizeof v); return *this; }
	Writer& str(const std::string& v) { num((uint32_t)v.size()); s += v; return *this; }
	Writer& bytes(const void* d, size_t n) { s.append((const char*)d, n); return *this; }
};

struct Reader {
	const char *p, *e;
	bool ok = true;
	explicit Reader(const std::string& s) : p(s.data()), e(s.data() + s.size()) {}
	template <class T> T num() {
		T v{};
		if ((size_t)(e - p) < sizeof v) { ok = false; return v; }
		memcpy(&v, p, sizeof v);
		p += sizeof v;
		return v;
	}
	std::string str() {
		const uint32_t n = num<uint32_t>();
		if (!ok || (size_t)(e - p) < n) { ok = false; return {}; }
		std::string v(p, n);
		p += n;
		return v;
	}
	std::string rest() { std::string v(p, e); p = e; return v; }
};

std::string header(uint8_t type, uint32_t id, size_t size) {
	std::string h(HEADER, '\0');
	const uint32_t n = (uint32_t)size;
	memcpy(&h[0], &n, 4);
	h[4] = (char)type;
	memcpy(&h[5], &id, 4);
	return h;
}

template <class T> void put_list(Writer& w, const std::vector<T>& v) { w.num((uint32_t)v.size()); for (auto& x : v) w.num(x); }

void put(Writer& w, const Fmt& f) { w.num(f.chroma).num(f.depth).num((uint8_t)f.rgb).num((uint8_t)f.planar); }
void get(Reader& r, Fmt& f) { f.chroma = r.num<int>(); f.depth = r.num<int>(); f.rgb = r.num<uint8_t>() != 0; f.planar = r.num<uint8_t>() != 0; }
void put(Writer& w, const Color& c) { w.num(c.range).num(c.space).num(c.primaries).num(c.trc); }
void get(Reader& r, Color& c) { c.range = r.num<int>(); c.space = r.num<int>(); c.primaries = r.num<int>(); c.trc = r.num<int>(); }
void put(Writer& w, const Rational& q) { w.num(q.num).num(q.den); }
void get(Reader& r, Rational& q) { q.num = r.num<int>(); q.den = r.num<int>(); }
void put(Writer& w, const Shape& s) { w.num(s.w).num(s.h); put(w, s.fps); put(w, s.pix); w.num((uint8_t)s.has_pix).num((uint8_t)s.upscale); }
void get(Reader& r, Shape& s) { s.w = r.num<int>(); s.h = r.num<int>(); get(r, s.fps); get(r, s.pix); s.has_pix = r.num<uint8_t>() != 0; s.upscale = r.num<uint8_t>() != 0; }
void put(Writer& w, const Options& o) { w.num((uint32_t)o.size()); for (auto& [k, v] : o) w.str(k).str(v); }
void get(Reader& r, Options& o) {
	const uint32_t n = r.num<uint32_t>();
	for (uint32_t i = 0; i < n && r.ok; i++) { std::string k = r.str(); o[k] = r.str(); }
}
void put(Writer& w, const SourceInfo& s) { w.num(s.width).num(s.height); put(w, s.fmt); put(w, s.framerate); put(w, s.color); w.num(s.duration).str(s.decoder).num(s.frame_count); }
void get(Reader& r, SourceInfo& s) { s.width = r.num<int>(); s.height = r.num<int>(); get(r, s.fmt); get(r, s.framerate); get(r, s.color); s.duration = r.num<double>(); s.decoder = r.str(); s.frame_count = r.num<long long>(); }
void put(Writer& w, const Plan& p) { put(w, p.enc); put(w, p.score_fmt); w.num((uint8_t)p.convert_rate).num((uint8_t)p.rescale).num(p.range); put(w, p.color); put(w, p.options); }
void get(Reader& r, Plan& p) { get(r, p.enc); get(r, p.score_fmt); p.convert_rate = r.num<uint8_t>() != 0; p.rescale = r.num<uint8_t>() != 0; p.range = r.num<int>(); get(r, p.color); get(r, p.options); }
void put(Writer& w, const ScaleOptions& s) { w.str(s.upscaler).str(s.downscaler).str(s.device); }
void get(Reader& r, ScaleOptions& s) { s.upscaler = r.str(); s.downscaler = r.str(); s.device = r.str(); }
void put(Writer& w, const VmafOptions& v) { w.str(v.model).num(v.threads).str(v.device).str(v.gpu).num(v.encoder_vendor).num(v.encoder_gpu); }
void get(Reader& r, VmafOptions& v) { v.model = r.str(); v.threads = r.num<int>(); v.device = r.str(); v.gpu = r.str(); v.encoder_vendor = r.num<unsigned>(); v.encoder_gpu = r.num<int>(); }

// ---- the process ----
// A child with its stdin, stdout and stderr on pipes. Only those three handles are passed on: a handle of another child's pipe that a child kept open would hide
// that other child's end.
class Process {
public:
	~Process() { close_all(); }

	bool start(const std::string& exe, const std::vector<std::string>& args) {
#ifdef _WIN32
		auto quote = [](const std::string& a) {
			std::string o = "\"";
			for (size_t i = 0; i < a.size(); i++) {
				size_t slashes = 0;
				while (i < a.size() && a[i] == '\\') { slashes++; i++; }
				if (i == a.size()) { o.append(slashes * 2, '\\'); break; }
				if (a[i] == '"') { o.append(slashes * 2 + 1, '\\'); o += '"'; }
				else { o.append(slashes, '\\'); o += a[i]; }
			}
			return o + "\"";
		};
		std::string line = quote(exe);
		for (auto& a : args) line += " " + quote(a);
		std::wstring cmd(MultiByteToWideChar(CP_UTF8, 0, line.c_str(), -1, nullptr, 0), L'\0');
		MultiByteToWideChar(CP_UTF8, 0, line.c_str(), -1, cmd.data(), (int)cmd.size());
		SECURITY_ATTRIBUTES sa{sizeof sa, nullptr, TRUE};
		HANDLE in_r = nullptr, out_w = nullptr, err_w = nullptr;
		if (!CreatePipe(&in_r, &in_w, &sa, 1 << 16) || !CreatePipe(&out_r, &out_w, &sa, 1 << 20) || !CreatePipe(&err_r, &err_w, &sa, 1 << 16)) {
			for (HANDLE h : {in_r, out_w, err_w}) if (h) CloseHandle(h);
			return false;
		}
		for (HANDLE h : {in_w, out_r, err_r}) SetHandleInformation(h, HANDLE_FLAG_INHERIT, 0);
		HANDLE inherit[3] = {in_r, out_w, err_w};
		SIZE_T attr_size = 0;
		InitializeProcThreadAttributeList(nullptr, 1, 0, &attr_size);
		std::vector<char> attr_buf(attr_size);
		auto attrs = (LPPROC_THREAD_ATTRIBUTE_LIST)attr_buf.data();
		bool ok = InitializeProcThreadAttributeList(attrs, 1, 0, &attr_size) && UpdateProcThreadAttribute(attrs, 0, PROC_THREAD_ATTRIBUTE_HANDLE_LIST, inherit, sizeof inherit, nullptr, nullptr);
		STARTUPINFOEXW si{};
		si.StartupInfo.cb = sizeof si;
		si.StartupInfo.dwFlags = STARTF_USESTDHANDLES;
		si.StartupInfo.hStdInput = in_r;
		si.StartupInfo.hStdOutput = out_w;
		si.StartupInfo.hStdError = err_w;
		si.lpAttributeList = attrs;
		PROCESS_INFORMATION pi{};
		// No console of its own (Ctrl-C ends this process, and the job object then ends the child); suspended until it is in the job.
		ok = ok && CreateProcessW(nullptr, cmd.data(), nullptr, nullptr, TRUE, EXTENDED_STARTUPINFO_PRESENT | CREATE_NO_WINDOW | CREATE_SUSPENDED, nullptr, nullptr, &si.StartupInfo, &pi);
		DeleteProcThreadAttributeList(attrs);
		for (HANDLE h : {in_r, out_w, err_w}) CloseHandle(h);
		if (!ok) return false;
		// Every child is in one job that ends them all when this process ends, however it ends.
		static HANDLE job = [] {
			HANDLE j = CreateJobObjectW(nullptr, nullptr);
			JOBOBJECT_EXTENDED_LIMIT_INFORMATION li{};
			li.BasicLimitInformation.LimitFlags = JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE;
			if (j) SetInformationJobObject(j, JobObjectExtendedLimitInformation, &li, sizeof li);
			return j;
		}();
		if (job) AssignProcessToJobObject(job, pi.hProcess);
		ResumeThread(pi.hThread);
		CloseHandle(pi.hThread);
		proc = pi.hProcess;
		pid = (long)pi.dwProcessId;
		return true;
#else
		static std::once_flag sigpipe;
		std::call_once(sigpipe, [] { signal(SIGPIPE, SIG_IGN); }); // writing to a child that died is an error, not the end of this process
		// The child ends when its stdin does, which is when this process ends, however it ends. (PR_SET_PDEATHSIG would end it with the thread that started it.)
		int in[2], out[2], err[2];
		if (pipe2(in, O_CLOEXEC) != 0) return false;
		if (pipe2(out, O_CLOEXEC) != 0) { ::close(in[0]); ::close(in[1]); return false; }
		if (pipe2(err, O_CLOEXEC) != 0) { for (int f : {in[0], in[1], out[0], out[1]}) ::close(f); return false; }
		std::vector<char*> argv;
		argv.push_back(const_cast<char*>(exe.c_str()));
		for (auto& a : args) argv.push_back(const_cast<char*>(a.c_str()));
		argv.push_back(nullptr);
		const pid_t p = fork();
		if (p == 0) {
			dup2(in[0], 0);
			dup2(out[1], 1);
			dup2(err[1], 2);
			execv(exe.c_str(), argv.data());
			_exit(127);
		}
		::close(in[0]); ::close(out[1]); ::close(err[1]);
		in_w = in[1]; out_r = out[0]; err_r = err[0];
		if (p < 0) { close_all(); return false; }
		pid = (long)p;
		return true;
#endif
	}

	bool write(const std::string& a, const std::string& b) {
		std::lock_guard lock(wm);
		return write_all(a.data(), a.size()) && write_all(b.data(), b.size());
	}

	bool read(void* d, size_t n) {
		char* p = (char*)d;
		while (n) {
#ifdef _WIN32
			DWORD got = 0;
			if (!ReadFile(out_r, p, (DWORD)std::min<size_t>(n, 1 << 20), &got, nullptr) || !got) return false;
#else
			const ssize_t got = ::read(out_r, p, n);
			if (got < 0 && errno == EINTR) continue;
			if (got <= 0) return false;
#endif
			p += got;
			n -= (size_t)got;
		}
		return true;
	}

	long read_errors(char* d, size_t n) { // what the child prints to stderr; 0 at the end
#ifdef _WIN32
		DWORD got = 0;
		return ReadFile(err_r, d, (DWORD)n, &got, nullptr) ? (long)got : 0;
#else
		for (;;) {
			const ssize_t got = ::read(err_r, d, n);
			if (got < 0 && errno == EINTR) continue;
			return got > 0 ? (long)got : 0;
		}
#endif
	}

	void close_input() {
		std::lock_guard lock(wm);
#ifdef _WIN32
		if (in_w) CloseHandle(in_w);
		in_w = nullptr;
#else
		if (in_w >= 0) ::close(in_w);
		in_w = -1;
#endif
	}

	void kill() {
#ifdef _WIN32
		if (proc) TerminateProcess(proc, 1);
#else
		if (pid > 0) ::kill((pid_t)pid, SIGKILL);
#endif
	}

	// Waits for the end of the child; its exit code, the code of the exception that ended it (Windows), or minus the signal (elsewhere).
	long long wait() {
#ifdef _WIN32
		WaitForSingleObject(proc, INFINITE);
		DWORD code = 1;
		GetExitCodeProcess(proc, &code);
		return (long long)code;
#else
		int status = 0;
		while (waitpid((pid_t)pid, &status, 0) < 0 && errno == EINTR) {}
		return WIFEXITED(status) ? WEXITSTATUS(status) : WIFSIGNALED(status) ? -WTERMSIG(status) : -1;
#endif
	}

	long pid = 0;

private:
	std::mutex wm;
#ifdef _WIN32
	HANDLE proc = nullptr, in_w = nullptr, out_r = nullptr, err_r = nullptr;
	bool write_all(const char* d, size_t n) {
		while (n) {
			DWORD put = 0;
			if (!in_w || !WriteFile(in_w, d, (DWORD)std::min<size_t>(n, 1 << 20), &put, nullptr)) return false;
			d += put;
			n -= put;
		}
		return true;
	}
	void close_all() {
		for (HANDLE* h : {&proc, &in_w, &out_r, &err_r}) if (*h) { CloseHandle(*h); *h = nullptr; }
	}
#else
	int in_w = -1, out_r = -1, err_r = -1;
	bool write_all(const char* d, size_t n) {
		while (n) {
			const ssize_t put = in_w < 0 ? -1 : ::write(in_w, d, n);
			if (put < 0 && errno == EINTR) continue;
			if (put <= 0) return false;
			d += put;
			n -= (size_t)put;
		}
		return true;
	}
	void close_all() {
		for (int* f : {&in_w, &out_r, &err_r}) if (*f >= 0) { ::close(*f); *f = -1; }
	}
#endif
};

std::string describe_exit(long long code) {
	char b[64];
#ifdef _WIN32
	if (code >= 0x80000000LL) snprintf(b, sizeof b, "crashed (code 0x%08llx)", code);
	else snprintf(b, sizeof b, "ended with code %lld", code);
#else
	if (code < 0) snprintf(b, sizeof b, "crashed (signal %lld)", -code);
	else snprintf(b, sizeof b, "ended with code %lld", code);
#endif
	return b;
}

// Something in this process that waits for answers from an encoder process: an encoder, or a question. Called with the device's lock held.
struct Waiter {
	virtual ~Waiter() = default;
	virtual void on_message(uint8_t type, Reader& r) = 0;
	virtual void on_death() = 0;
	virtual bool counts() const { return false; } // runs that fail when the process dies
	virtual size_t count() const { return 0; }    // how many
};

// One encoder process, and what waits for it.
struct Device {
	std::string encoder, name; // name: for messages ("nvenc GPU 0", "x264")
	int gpu = 0;
	Process proc;
	std::mutex m;
	bool alive = false, stopping = false, ended = false;
	std::condition_variable ended_cv;
	std::map<uint32_t, Waiter*> waiters;
	uint32_t next_id = 1;
	std::thread reader, errors;
	// The last lines the process printed, numbered, so that a wave can tell which came while it ran: they are why its runs failed.
	std::mutex lines_m;
	std::deque<std::string> lines;
	uint64_t line_count = 0;
	uint64_t lines_so_far() {
		std::lock_guard lock(lines_m);
		return line_count;
	}
	std::string lines_since(uint64_t from) { // FFmpeg's own lines ("[h264_cuvid @ ...] ...") only when there is nothing else
		std::lock_guard lock(lines_m);
		const uint64_t first = line_count - lines.size();
		std::vector<std::string> own, all;
		for (uint64_t i = std::max(from, first); i < line_count; i++) {
			const std::string& l = lines[(size_t)(i - first)];
			all.push_back(l);
			if (l[0] != '[' && l.rfind("hardware decoding", 0) != 0) own.push_back(l);
		}
		const auto& pick = own.empty() ? all : own;
		std::string s;
		for (size_t i = pick.size() > 4 ? pick.size() - 4 : 0; i < pick.size(); i++) s += (s.empty() ? "" : "; ") + pick[i];
		return s;
	}

	bool send(uint8_t type, uint32_t id, const std::string& body) { return proc.write(header(type, id, body.size()), body); }

	bool start() {
		std::vector<std::string> args{"--encoder-process", encoder, std::to_string(gpu)};
		{
			std::lock_guard lock(args_m);
			args.insert(args.end(), process_args.begin(), process_args.end());
		}
		if (!proc.start(executable_path(), args)) return false;
		alive = true;
		ui::log_plain("Encoder process for " + name + " started (pid " + std::to_string(proc.pid) + ")");
		reader = std::thread([this] { read_loop(); });
		errors = std::thread([this] { error_loop(); });
		return true;
	}

	void read_loop() {
		std::string head(HEADER, '\0'), body;
		for (;;) {
			if (!proc.read(head.data(), HEADER)) break;
			uint32_t size, id;
			memcpy(&size, head.data(), 4);
			memcpy(&id, head.data() + 5, 4);
			body.resize(size);
			if (size && !proc.read(body.data(), size)) break;
			std::lock_guard lock(m);
			auto it = waiters.find(id);
			if (it == waiters.end()) continue; // for an encoder that was closed already
			Reader r(body);
			it->second->on_message((uint8_t)head[4], r);
		}
		const long long code = proc.wait();
		size_t runs = 0;
		bool expected;
		{
			std::lock_guard lock(m);
			alive = false;
			expected = stopping;
			for (auto& [id, w] : waiters) { if (w->counts()) runs += w->count(); w->on_death(); }
		}
		if (!expected) {
			std::string what = "The encoder process for " + name + " " + describe_exit(code);
			if (runs) what += ": the " + std::to_string(runs) + (runs == 1 ? " run" : " runs") + " in it failed";
			ui::log(what + "; the next run starts a new one");
		}
		{
			std::lock_guard lock(m);
			ended = true;
		}
		ended_cv.notify_all();
	}

	// The child's messages, as lines of this process's own, with the name of the process in front.
	void error_loop() {
		std::string pending;
		char buf[4096];
		for (long n; (n = proc.read_errors(buf, sizeof buf)) > 0;) {
			pending.append(buf, (size_t)n);
			for (size_t at; (at = pending.find('\n')) != std::string::npos;) {
				std::string line = pending.substr(0, at);
				pending.erase(0, at + 1);
				if (!line.empty() && line.back() == '\r') line.pop_back();
				if (!line.empty()) keep(line);
			}
		}
		if (!pending.empty()) keep(pending);
	}
	void keep(const std::string& line) {
		ui::log("[" + name + "] " + line);
		std::lock_guard lock(lines_m);
		lines.push_back(line);
		if (lines.size() > 64) lines.pop_front();
		line_count++;
	}

	void stop() {
		{
			std::lock_guard lock(m);
			stopping = true;
		}
		proc.close_input(); // the process ends when its input does
		{
			std::unique_lock lock(m);
			if (!ended_cv.wait_for(lock, std::chrono::seconds(5), [&] { return ended; })) proc.kill();
		}
		if (reader.joinable()) reader.join();
		if (errors.joinable()) errors.join();
	}

	// A process that ended is cleaned up before it is replaced.
	void reap() {
		if (reader.joinable()) reader.join();
		if (errors.joinable()) errors.join();
	}
};

std::mutex devices_m;
std::map<std::string, std::shared_ptr<Device>> devices;

// The process of a GPU (or CPU encoder), started when it is not running.
std::shared_ptr<Device> device_for(const std::string& encoder, int gpu) {
	const bool on_gpu = is_gpu_encoder(encoder);
	const std::string key = on_gpu ? encoder + ":" + std::to_string(gpu) : "cpu"; // the CPU encoders share one process, and with it one decode of the input
	std::lock_guard lock(devices_m);
	auto& d = devices[key];
	if (d) {
		std::lock_guard dl(d->m);
		if (d->alive) return d;
	}
	if (d) d->reap();
	auto fresh = std::make_shared<Device>();
	fresh->encoder = on_gpu ? encoder : "cpu";
	fresh->gpu = on_gpu ? gpu : 0;
	fresh->name = on_gpu ? encoder + " GPU " + std::to_string(gpu) : "CPU encoders";
	if (!fresh->start()) {
		ui::log("cannot start the encoder process for " + fresh->name);
		d = nullptr;
		return nullptr;
	}
	d = fresh;
	return d;
}

// ---- a wave in an encoder process (this side) ----
// What the process of one device sends back while it works on its part of a wave: its progress rows, and at the end every run's result.
struct WaveWait : Waiter {
	std::condition_variable cv;
	std::string key; // of its progress rows in the live display
	bool done = false, dead = false;
	std::string result;
	size_t runs = 0;
	void on_message(uint8_t type, Reader& r) override {
		if (type == 'P') { ui::set_remote(key, "", ui::parse_progress(r.rest())); return; }
		if (type == 'w') { result = r.rest(); done = true; cv.notify_all(); }
	}
	void on_death() override { dead = true; cv.notify_all(); }
	bool counts() const override { return true; }
	size_t count() const override { return runs; }
};

// The key of the process a run belongs to: its GPU's (encoder and GPU index), or the one of the CPU encoders.
std::pair<std::string, int> device_of(const WaveRun& run) {
	if (!is_gpu_encoder(run.enc.encoder)) return {"cpu", 0};
	const auto& o = run.enc.plan->options;
	const auto it = o.find("gpu");
	return {run.enc.encoder, it == o.end() ? 0 : atoi(it->second.c_str())};
}


// A question for an encoder process.
struct Question : Waiter {
	std::condition_variable cv;
	bool answered = false, dead = false;
	std::string answer;
	void on_message(uint8_t, Reader& r) override { answer = r.rest(); answered = true; cv.notify_all(); }
	void on_death() override { dead = true; cv.notify_all(); }
};

// Asks the process of a GPU; a process that dies on the question gets a second chance in a new process. False: no answer.
bool ask(const std::string& encoder, int gpu, uint8_t type, const std::string& body, std::string& answer) {
	for (int attempt = 0; attempt < 2; attempt++) {
		auto dev = device_for(encoder, gpu);
		if (!dev) return false;
		Question q;
		uint32_t id;
		{
			std::lock_guard lock(dev->m);
			if (!dev->alive) continue;
			id = dev->next_id++;
			dev->waiters[id] = &q;
		}
		dev->send(type, id, body);
		std::unique_lock lock(dev->m);
		q.cv.wait(lock, [&] { return q.answered || q.dead; });
		dev->waiters.erase(id);
		if (q.answered) { answer = std::move(q.answer); return true; }
	}
	return false;
}

// ---- the encoder process ----
#ifdef _WIN32
HANDLE channel_in = nullptr, channel_out = nullptr;
bool channel_read(void* d, size_t n) {
	char* p = (char*)d;
	while (n) {
		DWORD got = 0;
		if (!ReadFile(channel_in, p, (DWORD)std::min<size_t>(n, 1 << 20), &got, nullptr) || !got) return false;
		p += got;
		n -= got;
	}
	return true;
}
bool channel_write_all(const char* d, size_t n) {
	while (n) {
		DWORD put = 0;
		if (!WriteFile(channel_out, d, (DWORD)std::min<size_t>(n, 1 << 20), &put, nullptr)) return false;
		d += put;
		n -= put;
	}
	return true;
}
#else
int channel_in = -1, channel_out = -1;
bool channel_read(void* d, size_t n) {
	char* p = (char*)d;
	while (n) {
		const ssize_t got = ::read(channel_in, p, n);
		if (got < 0 && errno == EINTR) continue;
		if (got <= 0) return false;
		p += got;
		n -= (size_t)got;
	}
	return true;
}
bool channel_write_all(const char* d, size_t n) {
	while (n) {
		const ssize_t put = ::write(channel_out, d, n);
		if (put < 0 && errno == EINTR) continue;
		if (put <= 0) return false;
		d += put;
		n -= (size_t)put;
	}
	return true;
}
#endif
std::mutex channel_m;
void reply(uint8_t type, uint32_t id, const std::string& body) {
	std::lock_guard lock(channel_m);
	const std::string h = header(type, id, body.size());
	if (!channel_write_all(h.data(), h.size()) || !channel_write_all(body.data(), body.size())) exit_now(1); // nobody is listening any more
}

// An encoder of the process that started this one, on a thread of its own: it gets its messages in order.
// ---- a wave in this process (the encoder process) ----
unsigned process_vendor = 0; // the GPU this process serves; 0 for the process of the CPU encoders
int process_gpu = 0;

// The Vulkan device this process converts frames on: its GPU's, when a test picture comes back right from it; else the first other one that gets it right (discrete GPUs
// first). Some drivers convert to nothing but zeros without an error: the Vega iGPU of an AMD laptop gave all-green videos. Found once per process.
std::string scaling_device() {
	static std::once_flag once;
	static std::string name;
	std::call_once(once, [] {
		std::string why;
		auto all = Vk::devices(why);
		const int own = vulkan_device_of_gpu(process_vendor, process_gpu);
		std::vector<const VkDeviceInfo*> order;
		for (auto& d : all) if (d.index == own) order.push_back(&d);
		for (int discrete = 1; discrete >= 0; discrete--) for (auto& d : all) if (d.index != own && d.discrete == (discrete == 1)) order.push_back(&d);
		for (const VkDeviceInfo* d : order) {
			ScaleOptions test;
			test.device = d->name;
			std::string wrong;
			if (converter_works(test, wrong)) {
				if (d->index != own) fprintf(stderr, "frames of this GPU are converted on %s instead%s\n", d->name.c_str(), own < 0 ? " (it has no Vulkan device of its own)" : "");
				name = d->name;
				return;
			}
			fprintf(stderr, "converting frames on %s does not work: %s\n", d->name.c_str(), wrong.c_str());
		}
		fprintf(stderr, "no Vulkan device converts frames right; leaving the choice to libplacebo\n");
	});
	return name;
}

void run_wave_here(uint32_t id, const std::string& body) {
	Reader r(body);
	const std::string input = r.str();
	SourceInfo planned;
	get(r, planned);
	ScaleOptions so;
	get(r, so);
	VmafOptions vo;
	get(r, vo);
	const uint32_t n = r.num<uint32_t>();
	std::vector<Plan> plans(n);
	std::vector<WaveRun> runs(n);
	std::vector<std::unique_ptr<RunProgress>> progress;
	for (uint32_t i = 0; i < n && r.ok; i++) {
		EncodeJob& ej = runs[i].enc;
		ej.encoder = r.str();
		ej.codec = r.str();
		get(r, plans[i]);
		ej.bitrate = r.num<int64_t>();
		ej.kf = r.num<double>();
		ej.file = r.str();
		ej.id = (Codec)r.num<int>();
		get(r, ej.color);
		runs[i].codec = (Codec)r.num<int>();
		runs[i].label = r.str();
		runs[i].pool = r.str();
	}
	if (!r.ok) { fprintf(stderr, "a garbled wave\n"); exit_now(1); }
	// This process decodes the input itself, on its own GPU where it can: the frames can come in another layout than the planner's software decode gave
	// (NV12 instead of planar 4:2:0), and then they go through the converter.
	SourceInfo si;
	if (!source_info(input, si)) si = planned;
	const int source_range = si.fmt.rgb ? 1 : (si.color.range ? si.color.range : 1);
	for (auto& pl : plans) pl.rescale = pl.enc.w != si.width || pl.enc.h != si.height || pl.enc.pix != si.fmt || (pl.range && pl.range != source_range);
	if (so.device.empty() && process_vendor) so.device = scaling_device(); // scaling on this GPU too, where it gives the right pictures
	std::vector<WaveRun*> ptrs;
	std::vector<int> handles;
	for (uint32_t i = 0; i < n; i++) {
		WaveRun& run = runs[i];
		run.enc.plan = &plans[i];
		progress.push_back(std::make_unique<RunProgress>());
		run.enc.rp = progress.back().get();
		progress.push_back(std::make_unique<RunProgress>());
		run.score_rp = progress.back().get();
		handles.push_back(ui::attach(run.enc.rp, run.label, run.pool));
		handles.push_back(ui::attach(run.score_rp, run.label, run.pool));
		ptrs.push_back(&run);
	}
	RunProgress feed_rp;
	feed_rp.begin(3, 1);
	const size_t slash = input.find_last_of("/\\");
	handles.push_back(ui::attach(&feed_rp, "decode+convert " + input.substr(slash == std::string::npos ? 0 : slash + 1) + " to " + std::to_string(plans[0].enc.w) + "x" + std::to_string(plans[0].enc.h), "decode"));
	std::atomic<bool> finished{false};
	std::thread rows([&] { // the rows of the live display, for the process that started this one
		while (!finished) {
			reply('P', id, ui::snapshot_json());
			for (int i = 0; i < 5 && !finished; i++) std::this_thread::sleep_for(std::chrono::milliseconds(50));
		}
	});
	FeedResult fr;
	run_wave(input, si, ptrs, so, vo, &feed_rp, fr);
	finished = true;
	rows.join();
	for (int h : handles) ui::detach(h);
	Writer w;
	w.num((uint8_t)fr.ok).num(fr.frames).num((int64_t)feed_rp.blocked_ms).str(si.decoder);
	for (auto& run : runs) {
		const EncodeJob& ej = run.enc;
		w.num((uint8_t)ej.ok).num(ej.kbps).num(ej.output.count).num(ej.output.bytes).num(ej.output.hash);
		w.num((int64_t)ej.rp->busy_ms).num((int64_t)ej.rp->starved_ms).num((uint8_t)run.score_ok).num((int64_t)run.score_rp->starved_ms);
		put_list(w, run.scores);
	}
	reply('w', id, w.s);
}


} // namespace

void set_isolated_encoders(bool on) { isolation = on; }
bool isolated_encoders() { return isolation; }

void set_encoder_process_args(const std::vector<std::string>& args) {
	std::lock_guard lock(args_m);
	process_args = args;
}

bool run_wave_isolated(const std::string& input, const SourceInfo& si, std::vector<WaveRun*>& runs, const ScaleOptions& so, const VmafOptions& vo, RunProgress* feed_rp, FeedResult& fr,
                       std::string& decoders) {
	std::map<std::pair<std::string, int>, std::vector<WaveRun*>> parts;
	for (WaveRun* run : runs) parts[device_of(*run)].push_back(run);
	VmafOptions shared_vo = vo;
	if (shared_vo.threads <= 0) shared_vo.threads = std::max(2, (int)std::thread::hardware_concurrency() / (int)runs.size()); // shared by every scorer of the wave, in all processes
	static std::atomic<unsigned> wave_counter{0};
	std::mutex m;
	bool all_ok = true;
	long long frames = -1;
	int64_t blocked = 0;
	std::vector<std::string> decoder_list;
	std::vector<std::thread> threads;
	for (auto& [where, part] : parts) threads.emplace_back([&, where = where, part = part] {
		std::shared_ptr<Device> dev;
		uint64_t first_line = 0;
		auto why = [&](const std::string& fallback) {
			std::string s = dev ? dev->lines_since(first_line) : std::string();
			return s.empty() ? fallback : s;
		};
		auto fail_all = [&](bool crashed) {
			const std::string error = why(!dev ? "its encoder process did not start" : crashed ? "its encoder process ended" : "its encoder process sent a garbled answer");
			for (WaveRun* run : part) {
				run->enc.ok = false;
				run->enc.crashed = crashed;
				run->enc.error = error;
				run->score_ok = false;
				if (run->enc.rp) { run->enc.rp->failed = true; run->enc.rp->finish(); }
				if (run->score_rp) { run->score_rp->failed = true; run->score_rp->finish(); }
			}
			std::lock_guard lock(m);
			all_ok = false;
		};
		dev = device_for(where.first, where.second);
		if (!dev) { fail_all(false); return; }
		first_line = dev->lines_so_far();
		Writer w;
		w.str(input);
		put(w, si);
		put(w, so);
		put(w, shared_vo);
		w.num((uint32_t)part.size());
		for (WaveRun* run : part) {
			const EncodeJob& ej = run->enc;
			w.str(ej.encoder).str(ej.codec);
			put(w, *ej.plan);
			w.num((int64_t)ej.bitrate).num(ej.kf).str(ej.file).num((int)ej.id);
			put(w, ej.color);
			w.num((int)run->codec).str(run->label).str(run->pool);
		}
		WaveWait ww;
		ww.key = "process " + dev->name + " " + std::to_string(wave_counter++);
		ww.runs = part.size();
		uint32_t id;
		{
			std::lock_guard lock(dev->m);
			if (!dev->alive) { fail_all(true); return; }
			id = dev->next_id++;
			dev->waiters[id] = &ww;
		}
		dev->send('W', id, w.s);
		{
			std::unique_lock lock(dev->m);
			ww.cv.wait(lock, [&] { return ww.done || ww.dead; });
			dev->waiters.erase(id);
		}
		ui::set_remote(ww.key, "", {});
		if (!ww.done) { fail_all(true); return; }
		Reader r(ww.result);
		const bool feed_ok = r.num<uint8_t>() != 0;
		const long long part_frames = r.num<long long>();
		const int64_t part_blocked = r.num<int64_t>();
		const std::string decoder = r.str();
		for (WaveRun* run : part) {
			EncodeJob& ej = run->enc;
			ej.ok = r.num<uint8_t>() != 0;
			ej.kbps = r.num<double>();
			ej.output.count = r.num<long long>();
			ej.output.bytes = r.num<long long>();
			ej.output.hash = r.num<uint64_t>();
			const int64_t busy = r.num<int64_t>(), starved = r.num<int64_t>();
			run->score_ok = r.num<uint8_t>() != 0;
			const int64_t score_starved = r.num<int64_t>();
			const uint32_t count = r.num<uint32_t>();
			run->scores.clear();
			for (uint32_t i = 0; i < count && r.ok; i++) run->scores.push_back(r.num<double>());
			if (!ej.ok || !run->score_ok) ej.error = why(!ej.ok ? "the encoder failed and said nothing" : "scoring failed and said nothing");
			if (ej.rp) { ej.rp->busy_ms = busy; ej.rp->starved_ms = starved; ej.rp->failed = !ej.ok; ej.rp->finish(); }
			if (run->score_rp) { run->score_rp->starved_ms = score_starved; run->score_rp->failed = !run->score_ok; run->score_rp->finish(); }
		}
		if (!r.ok) { fail_all(false); return; }
		std::lock_guard lock(m);
		all_ok = all_ok && feed_ok;
		frames = std::max(frames, part_frames);
		blocked = std::max(blocked, part_blocked);
		if (std::find(decoder_list.begin(), decoder_list.end(), decoder) == decoder_list.end()) decoder_list.push_back(decoder);
	});
	for (auto& t : threads) t.join();
	fr.ok = all_ok;
	fr.frames = std::max(0LL, frames);
	if (feed_rp) feed_rp->blocked_ms = blocked;
	decoders.clear();
	for (auto& d : decoder_list) decoders += (decoders.empty() ? "" : ", ") + d;
	return fr.ok;
}

bool isolated_supports(const std::string& encoder, const std::string& codec, int chroma, int depth, int gpu, bool& answer) {
	if (!isolation || !is_gpu_encoder(encoder)) return false;
	static std::mutex m;
	static std::map<std::string, bool> known;
	const std::string key = encoder + "|" + codec + "|" + std::to_string(chroma) + "|" + std::to_string(depth) + "|" + std::to_string(gpu);
	{
		std::lock_guard lock(m);
		if (auto it = known.find(key); it != known.end()) { answer = it->second; return true; }
	}
	Writer w;
	w.str(codec).num(chroma).num(depth);
	std::string reply_text;
	answer = ask(encoder, gpu, 'S', w.s, reply_text) && reply_text == "1";
	std::lock_guard lock(m);
	known[key] = answer;
	return true;
}

bool isolated_caps(const std::string& encoder, const std::string& codec, int gpu, std::string& answer) {
	if (!isolation || !is_gpu_encoder(encoder)) return false;
	static std::mutex m;
	static std::map<std::string, std::string> known;
	const std::string key = encoder + "|" + codec + "|" + std::to_string(gpu);
	{
		std::lock_guard lock(m);
		if (auto it = known.find(key); it != known.end()) { answer = it->second; return true; }
	}
	Writer w;
	w.str(codec);
	if (!ask(encoder, gpu, 'K', w.s, answer) || answer.empty()) answer = "null";
	std::lock_guard lock(m);
	known[key] = answer;
	return true;
}

void stop_encoder_processes() {
	std::map<std::string, std::shared_ptr<Device>> all;
	{
		std::lock_guard lock(devices_m);
		all.swap(devices);
	}
	for (auto& [key, d] : all) if (d) d->stop();
}

int encoder_process_main(int argc, char** argv) {
	set_isolated_encoders(false); // the encoders of this process are the real ones
	if (argc < 4) { fprintf(stderr, "usage: ves --encoder-process <encoder> <gpu>\n"); return 1; }
	const std::string encoder = argv[2];
	const int gpu = atoi(argv[3]);
	for (int i = 4; i + 1 < argc; i++) {
		const std::string a = argv[i], v = argv[++i];
		if (a == "--priority" && v == "low") set_low_priority();
		else if (a == "--hw-decode") set_hw_decode(v);
		else if (a == "--decode-threads") set_decode_threads(atoi(v.c_str()));
	}
	process_vendor = encoder_vendor(encoder); // a GPU's process decodes, scales and scores on that GPU
	process_gpu = gpu;
	if (process_vendor) set_hw_decode_gpu(process_vendor, gpu);
	// The messages travel on stdin and stdout; whatever a library prints goes to stderr, where it cannot get in their way.
#ifdef _WIN32
	// Copies of their own: the C runtime's stdout owns the original handle and closes it below, and the next handle Windows makes (the copy of stderr) can get the
	// very same number, which would send the messages to stderr.
	DuplicateHandle(GetCurrentProcess(), GetStdHandle(STD_INPUT_HANDLE), GetCurrentProcess(), &channel_in, 0, FALSE, DUPLICATE_SAME_ACCESS);
	DuplicateHandle(GetCurrentProcess(), GetStdHandle(STD_OUTPUT_HANDLE), GetCurrentProcess(), &channel_out, 0, FALSE, DUPLICATE_SAME_ACCESS);
	SetStdHandle(STD_OUTPUT_HANDLE, GetStdHandle(STD_ERROR_HANDLE));
	_dup2(2, 1);
#else
	channel_in = dup(0);
	channel_out = dup(1);
	dup2(2, 1);
	const int null = ::open("/dev/null", O_RDONLY);
	if (null >= 0) { dup2(null, 0); ::close(null); }
#endif
	setvbuf(stderr, nullptr, _IONBF, 0);
	if (encoder == "amf") amf_dedicate_process(gpu);

	std::string head(HEADER, '\0'), body;
	for (;;) {
		if (!channel_read(head.data(), HEADER)) break;
		uint32_t size, id;
		memcpy(&size, head.data(), 4);
		memcpy(&id, head.data() + 5, 4);
		body.resize(size);
		if (size && !channel_read(body.data(), size)) break;
		const uint8_t type = (uint8_t)head[4];
		if (type == 'S' || type == 'K') {
			std::thread([encoder, gpu, type, id, body] {
				Reader r(body);
				const std::string codec = r.str();
				if (type == 'S') {
					const int chroma = r.num<int>(), depth = r.num<int>();
					reply('s', id, encoder_supports(encoder, codec, chroma, depth, gpu) ? "1" : "0");
				}
				else reply('k', id, encoder_caps_json(encoder, codec, gpu));
			}).detach();
			continue;
		}
		if (type == 'W') std::thread([id, body] { run_wave_here(id, body); }).detach(); // meanwhile questions still get answers, and the end of stdin is still seen
	}
	// The process that started this one is gone or done: whatever is still encoding is of no use to anyone. Nothing is wound down (see exit_now).
	exit_now(0);
}
