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

#include "formats.h"
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
// Every message is a header (payload size, type, session) and the payload, the same way in both directions. Numbers are in the byte order of the machine: both
// ends are the same program on the same machine.
//   to the encoder process:   'O' open (shared memory, slot size and count, Params)   'F' a picture is in a slot (slot, pts, duration)   'E' flush   'C' close
//                             'S' supports (codec, chroma, depth)   'K' capabilities (codec)
//   from it:                  'o' opened (ok, reorder delay)   'p' packet (pts, dts, key, data)   'f' slot free again (slot, ok)   'e' flushed (ok)
//                             's' supports (yes/no)   'k' capabilities (JSON)
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

void put_params(Writer& w, const Params& p) {
	w.num(p.width).num(p.height).num(p.pix.chroma).num(p.pix.depth).num((uint8_t)p.pix.rgb).num((uint8_t)p.pix.planar);
	w.num(p.framerate.num).num(p.framerate.den).num(p.bitrate).num(p.gop).str(p.codec);
	w.num(p.color.range).num(p.color.space).num(p.color.primaries).num(p.color.trc);
	w.num((uint32_t)p.options.size());
	for (auto& [k, v] : p.options) w.str(k).str(v);
}

Params get_params(Reader& r) {
	Params p;
	p.width = r.num<int>(); p.height = r.num<int>();
	p.pix.chroma = r.num<int>(); p.pix.depth = r.num<int>(); p.pix.rgb = r.num<uint8_t>() != 0; p.pix.planar = r.num<uint8_t>() != 0;
	p.framerate.num = r.num<int>(); p.framerate.den = r.num<int>();
	p.bitrate = r.num<int64_t>(); p.gop = r.num<int>(); p.codec = r.str();
	p.color.range = r.num<int>(); p.color.space = r.num<int>(); p.color.primaries = r.num<int>(); p.color.trc = r.num<int>();
	const uint32_t n = r.num<uint32_t>();
	for (uint32_t i = 0; i < n && r.ok; i++) {
		std::string k = r.str();
		p.options[k] = r.str();
	}
	return p;
}

// Where the planes of a picture go in a slot: one after the other, rows without padding.
struct SlotLayout {
	size_t offset[3]{}, stride[3]{};
	int rows[3]{};
	size_t size = 0;
	SlotLayout() = default;
	SlotLayout(const Params& p) {
		const size_t bps = p.pix.depth > 8 ? 2 : 1;
		const int cw = p.pix.chroma == 444 ? p.width : (p.width + 1) / 2, ch = p.pix.chroma == 420 ? (p.height + 1) / 2 : p.height;
		for (int i = 0; i < 3; i++) {
			offset[i] = size;
			stride[i] = (size_t)(i ? cw : p.width) * bps;
			rows[i] = i ? ch : p.height;
			size += stride[i] * (size_t)rows[i];
		}
		size = (size + 63) & ~(size_t)63;
	}
};

// ---- shared memory ----
class SharedMemory {
public:
	uint8_t* data = nullptr;
	size_t size = 0;

	~SharedMemory() { close(); }

	bool create(const std::string& name, size_t n) {
		size = n;
#ifdef _WIN32
		const std::wstring wname(name.begin(), name.end());
		h = CreateFileMappingW(INVALID_HANDLE_VALUE, nullptr, PAGE_READWRITE, (DWORD)((uint64_t)n >> 32), (DWORD)n, wname.c_str());
		if (!h || GetLastError() == ERROR_ALREADY_EXISTS) return false;
		data = (uint8_t*)MapViewOfFile(h, FILE_MAP_ALL_ACCESS, 0, 0, n);
#else
		this->name = name;
		fd = shm_open(name.c_str(), O_CREAT | O_EXCL | O_RDWR | O_CLOEXEC, 0600);
		if (fd < 0) return false;
		if (ftruncate(fd, (off_t)n) != 0) return false;
		void* m = mmap(nullptr, n, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
		data = m == MAP_FAILED ? nullptr : (uint8_t*)m;
#endif
		return data != nullptr;
	}

	bool open(const std::string& name, size_t n) {
		size = n;
#ifdef _WIN32
		const std::wstring wname(name.begin(), name.end());
		h = OpenFileMappingW(FILE_MAP_ALL_ACCESS, FALSE, wname.c_str());
		if (!h) return false;
		data = (uint8_t*)MapViewOfFile(h, FILE_MAP_ALL_ACCESS, 0, 0, n);
#else
		fd = shm_open(name.c_str(), O_RDWR | O_CLOEXEC, 0600);
		if (fd < 0) return false;
		void* m = mmap(nullptr, n, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
		data = m == MAP_FAILED ? nullptr : (uint8_t*)m;
#endif
		return data != nullptr;
	}

	void unlink() { // the name is not needed once both sides have the memory
#ifndef _WIN32
		if (!name.empty()) shm_unlink(name.c_str());
		name.clear();
#endif
	}

	void close() {
		unlink();
#ifdef _WIN32
		if (data) UnmapViewOfFile(data);
		if (h) CloseHandle(h);
		h = nullptr;
#else
		if (data) munmap(data, size);
		if (fd >= 0) ::close(fd);
		fd = -1;
#endif
		data = nullptr;
	}

private:
#ifdef _WIN32
	HANDLE h = nullptr;
#else
	int fd = -1;
	std::string name;
#endif
};

std::string memory_name() {
	static std::atomic<unsigned> counter{0};
	static const unsigned salt = std::random_device{}();
	char b[96];
#ifdef _WIN32
	snprintf(b, sizeof b, "Local\\ves-%lu-%08x-%u", (unsigned long)GetCurrentProcessId(), salt, counter++);
#else
	snprintf(b, sizeof b, "/ves-%ld-%08x-%u", (long)getpid(), salt, counter++);
#endif
	return b;
}

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
	virtual bool counts() const { return false; } // a run that fails when the process dies
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
			for (auto& [id, w] : waiters) { runs += w->counts(); w->on_death(); }
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
				if (!line.empty()) fprintf(stderr, "[%s] %s\n", name.c_str(), line.c_str());
			}
		}
		if (!pending.empty()) fprintf(stderr, "[%s] %s\n", name.c_str(), pending.c_str());
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
	const std::string key = on_gpu ? encoder + ":" + std::to_string(gpu) : encoder;
	std::lock_guard lock(devices_m);
	auto& d = devices[key];
	if (d) {
		std::lock_guard dl(d->m);
		if (d->alive) return d;
	}
	if (d) d->reap();
	auto fresh = std::make_shared<Device>();
	fresh->encoder = encoder;
	fresh->gpu = on_gpu ? gpu : 0;
	fresh->name = on_gpu ? encoder + " GPU " + std::to_string(gpu) : encoder;
	if (!fresh->start()) {
		ui::log("cannot start the encoder process for " + fresh->name);
		d = nullptr;
		return nullptr;
	}
	d = fresh;
	return d;
}

// ---- the encoder in this process ----
class Isolated : public Encoder, Waiter {
	static constexpr int SLOTS = 3; // pictures on their way: the next one is copied while the encoder works on the one before
	const std::string name;
	std::shared_ptr<Device> dev;
	uint32_t id = 0;
	Params params;
	SlotLayout layout;
	SharedMemory shm;
	Sink sink;
	// under dev->m
	struct Packet {
		std::string data;
		int64_t pts, dts;
		bool key;
	};
	std::condition_variable cv;
	std::deque<Packet> packets;
	bool busy[SLOTS]{};
	bool replied = false, open_ok = false, flushed = false, flush_ok = false, failed = false, dead = false;
	int reorder = 0;
	bool lost = false; // the process ended while this encoder was in it

	void on_message(uint8_t type, Reader& r) override {
		if (type == 'p') {
			Packet k;
			k.pts = r.num<int64_t>();
			k.dts = r.num<int64_t>();
			k.key = r.num<uint8_t>() != 0;
			k.data = r.rest();
			packets.push_back(std::move(k));
		}
		else if (type == 'f') {
			const uint32_t slot = r.num<uint32_t>();
			if (slot < SLOTS) busy[slot] = false;
			if (!r.num<uint8_t>()) failed = true;
		}
		else if (type == 'o') { open_ok = r.num<uint8_t>() != 0; reorder = r.num<int32_t>(); replied = true; }
		else if (type == 'e') { flush_ok = r.num<uint8_t>() != 0; flushed = true; }
		cv.notify_all();
	}
	void on_death() override { dead = true; cv.notify_all(); }
	bool counts() const override { return true; }

	// Hands the packets that came in to the sink, on the thread that encodes (as the encoders in this process do); the lock is let go meanwhile.
	void deliver(std::unique_lock<std::mutex>& lock) {
		while (!packets.empty()) {
			Packet k = std::move(packets.front());
			packets.pop_front();
			lock.unlock();
			stats.packets++;
			stats.bytes += (long long)k.data.size();
			if (sink) sink((const uint8_t*)k.data.data(), k.data.size(), k.pts, k.key, k.dts);
			lock.lock();
		}
	}

public:
	explicit Isolated(std::string n) : name(std::move(n)) {}

	~Isolated() override {
		if (!dev || !id) return;
		bool gone;
		{
			std::lock_guard lock(dev->m);
			dev->waiters.erase(id);
			gone = dead;
		}
		if (!gone) dev->send('C', id, {});
	}

	bool crashed() const override { return lost; }

	bool open(const Params& p, Sink s) override {
		sink = std::move(s);
		params = p;
		int gpu = 0;
		if (auto it = p.options.find("gpu"); it != p.options.end()) gpu = atoi(it->second.c_str());
		dev = device_for(name, gpu);
		if (!dev) return false;
		layout = SlotLayout(p);
		const std::string mem = memory_name();
		if (!shm.create(mem, layout.size * SLOTS)) { fprintf(stderr, "%s: cannot set up shared memory for the encoder process\n", name.c_str()); return false; }
		{
			std::lock_guard lock(dev->m);
			if (!dev->alive) { lost = true; return false; }
			id = dev->next_id++;
			dev->waiters[id] = this;
		}
		Writer w;
		w.str(mem).num((uint64_t)layout.size).num((uint32_t)SLOTS);
		put_params(w, p);
		dev->send('O', id, w.s);
		std::unique_lock lock(dev->m);
		cv.wait(lock, [&] { return replied || dead; });
		shm.unlink();
		if (!replied) { lost = true; return false; }
		return open_ok;
	}

	int reorder_delay() const override { return reorder; }

	bool encode(const Picture* pic) override {
		if (!pic) return flush();
		if (pic->fmt != params.pix || pic->width != params.width || pic->height != params.height) {
			fprintf(stderr, "%s: the picture is not the %dx%d %d-bit %d the encoder was opened for\n", name.c_str(), params.width, params.height, params.pix.depth, params.pix.chroma);
			return false;
		}
		int slot = -1;
		{
			std::unique_lock lock(dev->m);
			for (;;) {
				deliver(lock);
				if (dead) { lost = true; return false; }
				if (failed) return false;
				for (int i = 0; i < SLOTS && slot < 0; i++) if (!busy[i]) slot = i;
				if (slot >= 0) break;
				cv.wait(lock, [&] {
					for (bool b : busy) if (!b) return true;
					return !packets.empty() || dead || failed;
				});
			}
			busy[slot] = true;
		}
		uint8_t* base = shm.data + (size_t)slot * layout.size;
		for (int i = 0; i < 3; i++) {
			uint8_t* d = base + layout.offset[i];
			for (int y = 0; y < layout.rows[i]; y++) memcpy(d + (size_t)y * layout.stride[i], pic->data[i] + (size_t)y * pic->stride[i], layout.stride[i]);
		}
		Writer w;
		w.num((uint32_t)slot).num(pic->pts).num(pic->duration);
		dev->send('F', id, w.s); // a process that is gone is noticed by the reader
		return true;
	}

	bool flush() {
		dev->send('E', id, {});
		std::unique_lock lock(dev->m);
		for (;;) {
			deliver(lock);
			if (flushed) break;
			if (dead) { lost = true; return false; }
			cv.wait(lock, [&] { return !packets.empty() || flushed || dead; });
		}
		deliver(lock); // the packets come before the answer
		return flush_ok && !failed;
	}
};

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
class Session {
	const uint32_t id;
	const std::string encoder;
	std::mutex m;
	std::condition_variable cv;
	std::deque<std::pair<uint8_t, std::string>> inbox;
	std::thread worker;

	void run() {
		std::unique_ptr<Encoder> enc;
		SharedMemory shm;
		Params p;
		SlotLayout layout;
		uint32_t slots = 0;
		bool failed = false;
		for (;;) {
			std::pair<uint8_t, std::string> msg;
			{
				std::unique_lock lock(m);
				cv.wait(lock, [&] { return !inbox.empty(); });
				msg = std::move(inbox.front());
				inbox.pop_front();
			}
			Reader r(msg.second);
			switch (msg.first) {
			case 'O': {
				const std::string mem = r.str();
				const uint64_t slot_size = r.num<uint64_t>();
				slots = r.num<uint32_t>();
				p = get_params(r);
				layout = SlotLayout(p);
				bool ok = r.ok && layout.size == slot_size;
				if (!ok) fprintf(stderr, "%s: a garbled request to open an encoder\n", encoder.c_str());
				if (ok && !(ok = shm.open(mem, slot_size * slots))) fprintf(stderr, "%s: cannot open the shared memory of the encoder\n", encoder.c_str());
				if (ok) {
					enc = make_encoder(encoder);
					const uint32_t sid = id;
					ok = enc && enc->open(p, [sid](const uint8_t* d, size_t n, int64_t pts, bool key, int64_t dts) {
						Writer w;
						w.num(pts).num(dts).num((uint8_t)key).bytes(d, n);
						reply('p', sid, w.s);
					});
				}
				if (!ok) failed = true;
				Writer w;
				w.num((uint8_t)ok).num((int32_t)(ok ? enc->reorder_delay() : 0));
				reply('o', id, w.s);
				break;
			}
			case 'F': {
				const uint32_t slot = r.num<uint32_t>();
				Picture pic;
				pic.pts = r.num<int64_t>();
				pic.duration = r.num<int64_t>();
				bool ok = !failed && enc && slot < slots;
				if (ok) {
					pic.width = p.width; pic.height = p.height;
					pic.fmt = p.pix;
					pic.color = p.color;
					const uint8_t* base = shm.data + (size_t)slot * layout.size;
					for (int i = 0; i < 3; i++) { pic.data[i] = base + layout.offset[i]; pic.stride[i] = (int)layout.stride[i]; }
					ok = enc->encode(&pic);
					if (!ok) failed = true;
				}
				Writer w;
				w.num(slot).num((uint8_t)ok);
				reply('f', id, w.s);
				break;
			}
			case 'E': {
				const bool ok = !failed && enc && enc->encode(nullptr);
				Writer w;
				w.num((uint8_t)ok);
				reply('e', id, w.s);
				break;
			}
			case 'C':
				enc.reset();
				return;
			}
		}
	}

public:
	Session(uint32_t id, std::string encoder) : id(id), encoder(std::move(encoder)) { worker = std::thread([this] { run(); }); }
	~Session() { if (worker.joinable()) worker.join(); }
	void push(uint8_t type, std::string body) {
		{
			std::lock_guard lock(m);
			inbox.emplace_back(type, std::move(body));
		}
		cv.notify_all();
	}
};

} // namespace

void set_isolated_encoders(bool on) { isolation = on; }
bool isolated_encoders() { return isolation; }

void set_encoder_process_args(const std::vector<std::string>& args) {
	std::lock_guard lock(args_m);
	process_args = args;
}

std::unique_ptr<Encoder> make_isolated_encoder(const std::string& name) {
	if (!is_native_encoder(name)) return nullptr;
	return std::make_unique<Isolated>(name);
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
	for (int i = 4; i < argc; i++) {
		const std::string a = argv[i];
		if (a == "--priority" && i + 1 < argc && std::string(argv[++i]) == "low") set_low_priority();
	}
	// The messages travel on stdin and stdout; whatever a library prints goes to stderr, where it cannot get in their way.
#ifdef _WIN32
	channel_in = GetStdHandle(STD_INPUT_HANDLE);
	channel_out = GetStdHandle(STD_OUTPUT_HANDLE);
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

	std::map<uint32_t, std::unique_ptr<Session>> sessions;
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
		if (type == 'O') sessions[id] = std::make_unique<Session>(id, encoder);
		auto it = sessions.find(id);
		if (it == sessions.end()) continue;
		it->second->push(type, body);
		if (type == 'C') { // the encoder closes on a thread of its own, which can take a while; meanwhile the others get their messages
			std::thread([s = std::move(it->second)]() mutable { s.reset(); }).detach();
			sessions.erase(it);
		}
	}
	// The process that started this one is gone or done: whatever is still encoding is of no use to anyone. Nothing is wound down (see exit_now).
	exit_now(0);
}
