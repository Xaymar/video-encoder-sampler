#include "remote.h"

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdio>
#include <deque>
#include <filesystem>
#include <fstream>
#include <map>
#include <mutex>
#include <sstream>
#include <thread>

#ifdef _WIN32
#define NOMINMAX
#include <windows.h>
#else
#include <fcntl.h>
#include <sys/wait.h>
#include <unistd.h>
#endif

#include "hash.h"
#include "json.h"
#include "net.h"

namespace fs = std::filesystem;

namespace {

enum : uint8_t { M_HELLO = 1, M_JOB = 2, M_NEED = 3, M_FILE_DATA = 4, M_FILE_END = 5, M_RESULT = 6, M_DONE = 7 };
constexpr int kVersion = 1;
constexpr size_t kChunk = 1 << 20;
constexpr int kTries = 3;

std::string jq(const std::string& s) { // JSON string
	std::string o = "\"";
	for (unsigned char c : s) {
		if (c == '"' || c == '\\') { o += '\\'; o += (char)c; }
		else if (c < 0x20) { char b[8]; snprintf(b, sizeof b, "\\u%04x", c); o += b; }
		else o += (char)c;
	}
	return o + "\"";
}

bool flag_set(const std::string& v) { return v == "1" || v == "true"; } // JSON booleans read back as 1 or 0

bool is_url(const std::string& s) { return s.rfind("http://", 0) == 0 || s.rfind("https://", 0) == 0; }

// "name_1a2b3c4d.json" -> "name": result files of the same run are named alike on both sides except for the hash of the key, which depends on the machine.
std::string result_prefix(const std::string& file_name) {
	if (file_name.size() > 14 && file_name.compare(file_name.size() - 5, 5, ".json") == 0 && file_name[file_name.size() - 14] == '_') return file_name.substr(0, file_name.size() - 14);
	return file_name;
}

// The stored result starts {"key":"...",...}; the host's own key goes in place of the worker's.
bool replace_key(std::string& content, const std::string& key) {
	static const std::string head = "{\"key\":\"";
	if (content.compare(0, head.size(), head) != 0) return false;
	size_t i = head.size();
	while (i < content.size() && content[i] != '"') i += content[i] == '\\' ? 2 : 1;
	if (i >= content.size()) return false;
	content = "{\"key\":" + jq(key) + content.substr(i + 1);
	return true;
}

std::string read_file(const fs::path& p) {
	std::ifstream f(p, std::ios::binary);
	return std::string((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
}

// ---- host ----

struct HostState {
	const std::vector<RemoteJobSpec>& jobs;
	const RemoteHostOptions& opt;
	const std::function<void(const std::string&)>& log;
	std::mutex m;
	std::condition_variable cv;
	std::deque<size_t> queue;
	std::vector<int> tries;
	std::map<std::string, std::string> sha_of;   // input path -> SHA-256
	size_t left = 0, imported = 0;
	int workers = 0, total_workers = 0;
	HostState(const std::vector<RemoteJobSpec>& j, const RemoteHostOptions& o, const std::function<void(const std::string&)>& l) : jobs(j), opt(o), log(l) {}
};

std::string input_sha(HostState& s, const std::string& path) {
	{
		std::lock_guard lock(s.m);
		auto it = s.sha_of.find(path);
		if (it != s.sha_of.end()) return it->second;
	}
	const std::string sha = sha256_file(path);
	std::lock_guard lock(s.m);
	return s.sha_of[path] = sha;
}

// Stores a result file the worker sent. Returns whether it belonged to the job.
bool import_result(HostState& s, const RemoteJobSpec& job, const std::string& name, std::string content) {
	const std::string want = result_prefix(name);
	for (auto& r : job.runs) {
		if (result_prefix(fs::path(r.result_file).filename().string()) != want) continue;
		if (!replace_key(content, r.key)) return false;
		std::error_code ec;
		fs::create_directories(fs::path(r.result_file).parent_path(), ec);
		const fs::path tmp = r.result_file + ".tmp";
		{
			std::ofstream f(tmp, std::ios::binary);
			f << content;
			if (!f) return false;
		}
		fs::rename(tmp, r.result_file, ec);
		return !ec;
	}
	return false;
}

void serve(HostState& s, std::unique_ptr<net::Conn> conn) {
	uint8_t type;
	std::string payload;
	std::string who = conn->peer;
	try {
		if (!conn->recv_msg(type, payload) || type != M_HELLO) return;
		Json hello = Json::parse(payload);
		if (auto* t = hello.find("token"); (t ? t->text() : std::string()) != s.opt.token) { s.log("worker " + who + " refused: wrong token"); return; }
		if (auto* v = hello.find("version"); !v || atoi(v->text().c_str()) != kVersion) { s.log("worker " + who + " refused: other protocol version"); return; }
		if (auto* n = hello.find("name")) who = n->text() + " (" + conn->peer + ")";
		conn->send_msg(M_HELLO, "{}");
	} catch (const std::exception&) { return; }
	{
		std::lock_guard lock(s.m);
		s.workers++;
		s.total_workers++;
	}
	s.cv.notify_all();
	s.log("worker " + who + " connected");

	for (;;) {
		size_t id;
		{
			std::unique_lock lock(s.m);
			s.cv.wait(lock, [&] { return s.left == 0 || (!s.queue.empty() && s.total_workers >= s.opt.min_workers); });
			if (s.left == 0) break;
			id = s.queue.front();
			s.queue.pop_front();
			s.tries[id]++;
		}
		const RemoteJobSpec& job = s.jobs[id];
		bool ok = false;
		size_t stored = 0;
		try {
			std::string sha, size = "0";
			if (!is_url(job.input)) { sha = input_sha(s, job.input); size = std::to_string(fs::file_size(job.input)); }
			std::string msg = "{\"id\":" + std::to_string(id) + ",\"name\":" + jq(job.name) + ",\"config\":" + jq(job.config) + ",\"input\":{\"url\":" + (is_url(job.input) ? "true" : "false") +
			                  ",\"name\":" + jq(is_url(job.input) ? job.input : fs::path(job.input).filename().string()) + ",\"sha256\":" + jq(sha) + ",\"size\":" + size + "}}";
			s.log("job " + job.name + " -> " + who);
			if (!conn->send_msg(M_JOB, msg)) throw std::runtime_error("connection lost");
			std::string log_tail;
			for (;;) {
				if (!conn->recv_msg(type, payload)) throw std::runtime_error("connection lost");
				if (type == M_NEED) {
					std::ifstream f(job.input, std::ios::binary);
					std::string chunk(kChunk, '\0');
					while (f) {
						f.read(chunk.data(), (std::streamsize)kChunk);
						const size_t n = (size_t)f.gcount();
						if (n && !conn->send_msg(M_FILE_DATA, chunk.substr(0, n))) throw std::runtime_error("connection lost");
					}
					if (!conn->send_msg(M_FILE_END, "")) throw std::runtime_error("connection lost");
				} else if (type == M_RESULT) {
					if (payload.size() < 2) continue;
					const size_t nl = (uint8_t)payload[0] | ((uint8_t)payload[1] << 8);
					if (payload.size() < 2 + nl) continue;
					const std::string name = fs::path(payload.substr(2, nl)).filename().string();
					if (import_result(s, job, name, payload.substr(2 + nl))) stored++;
				} else if (type == M_DONE) {
					Json d = Json::parse(payload);
					ok = d.find("ok") && flag_set(d.find("ok")->text());
					if (auto* l = d.find("log")) log_tail = l->text();
					break;
				}
			}
			s.log("job " + job.name + " done on " + who + ": " + std::to_string(stored) + " of " + std::to_string(job.runs.size()) + " results" + (ok ? "" : " (the worker reported a failure)"));
			if (!log_tail.empty() && !ok) s.log(log_tail);
		} catch (const std::exception& e) {
			s.log("job " + job.name + " lost with " + who + ": " + e.what());
			std::lock_guard lock(s.m);
			if (s.tries[id] < kTries) s.queue.push_front(id);
			else { s.left--; s.log("job " + job.name + " given up after " + std::to_string(kTries) + " tries"); }
			s.workers--;
			s.cv.notify_all();
			return;
		}
		{
			std::lock_guard lock(s.m);
			s.imported += stored;
			s.left--;
		}
		s.cv.notify_all();
	}
	std::lock_guard lock(s.m);
	s.workers--;
}

// ---- worker ----

std::string self_exe() {
#ifdef _WIN32
	wchar_t buf[MAX_PATH * 4];
	const DWORD n = GetModuleFileNameW(nullptr, buf, (DWORD)(sizeof buf / sizeof *buf));
	return fs::path(std::wstring(buf, n)).string();
#else
	std::error_code ec;
	return fs::read_symlink("/proc/self/exe", ec).string();
#endif
}

// Runs a program with its output in `log_file`; returns its exit code, or -1 if it could not be started.
int run_process(const std::string& exe, const std::vector<std::string>& args, const std::string& log_file) {
#ifdef _WIN32
	std::wstring cmd;
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
	cmd = fs::path(line).wstring();
	SECURITY_ATTRIBUTES sa{sizeof sa, nullptr, TRUE};
	HANDLE log = CreateFileW(fs::path(log_file).wstring().c_str(), GENERIC_WRITE, FILE_SHARE_READ, &sa, CREATE_ALWAYS, FILE_ATTRIBUTE_NORMAL, nullptr);
	STARTUPINFOW si{};
	si.cb = sizeof si;
	si.dwFlags = STARTF_USESTDHANDLES;
	si.hStdOutput = si.hStdError = log;
	si.hStdInput = GetStdHandle(STD_INPUT_HANDLE);
	PROCESS_INFORMATION pi{};
	std::vector<wchar_t> mutable_cmd(cmd.begin(), cmd.end());
	mutable_cmd.push_back(0);
	const BOOL ok = CreateProcessW(nullptr, mutable_cmd.data(), nullptr, nullptr, TRUE, 0, nullptr, nullptr, &si, &pi);
	if (log != INVALID_HANDLE_VALUE) CloseHandle(log);
	if (!ok) return -1;
	WaitForSingleObject(pi.hProcess, INFINITE);
	DWORD code = 1;
	GetExitCodeProcess(pi.hProcess, &code);
	CloseHandle(pi.hProcess);
	CloseHandle(pi.hThread);
	return (int)code;
#else
	const pid_t pid = fork();
	if (pid < 0) return -1;
	if (pid == 0) {
		const int fd = open(log_file.c_str(), O_WRONLY | O_CREAT | O_TRUNC, 0644);
		if (fd >= 0) { dup2(fd, 1); dup2(fd, 2); close(fd); }
		std::vector<char*> argv;
		argv.push_back(const_cast<char*>(exe.c_str()));
		for (auto& a : args) argv.push_back(const_cast<char*>(a.c_str()));
		argv.push_back(nullptr);
		execv(exe.c_str(), argv.data());
		_exit(127);
	}
	int status = 0;
	waitpid(pid, &status, 0);
	return WIFEXITED(status) ? WEXITSTATUS(status) : -1;
#endif
}

std::string tail_of(const std::string& text, size_t lines) {
	std::vector<std::string> all;
	std::istringstream in(text);
	for (std::string l; std::getline(in, l);) if (!l.empty()) all.push_back(l);
	std::string out;
	for (size_t i = all.size() > lines ? all.size() - lines : 0; i < all.size(); i++) out += all[i] + "\n";
	return out;
}

// Fetches an input from the host (if it is not here already) and returns its path.
bool fetch_input(net::Conn& conn, const fs::path& dir, const std::string& name, const std::string& sha, fs::path& out, std::string& err) {
	std::string safe = fs::path(name).filename().string();
	out = dir / (sha.substr(0, 16) + "_" + safe);
	std::error_code ec;
	if (fs::exists(out, ec)) return true;
	if (!conn.send_msg(M_NEED, "{\"sha256\":" + jq(sha) + "}")) { err = "connection lost"; return false; }
	fs::create_directories(dir, ec);
	const fs::path tmp = out.string() + ".part";
	std::ofstream f(tmp, std::ios::binary);
	uint8_t type;
	std::string payload;
	for (;;) {
		if (!conn.recv_msg(type, payload)) { err = "connection lost"; return false; }
		if (type == M_FILE_END) break;
		if (type != M_FILE_DATA) { err = "unexpected message"; return false; }
		f.write(payload.data(), (std::streamsize)payload.size());
	}
	f.close();
	if (sha256_file(tmp.string()) != sha) { fs::remove(tmp, ec); err = "the input arrived damaged"; return false; }
	fs::rename(tmp, out, ec);
	return !ec;
}

} // namespace

size_t remote_run_jobs(const std::vector<RemoteJobSpec>& jobs, const RemoteHostOptions& options, const std::function<void(const std::string&)>& log) {
	if (jobs.empty()) return 0;
	HostState s(jobs, options, log);
	for (size_t i = 0; i < jobs.size(); i++) s.queue.push_back(i);
	s.tries.assign(jobs.size(), 0);
	s.left = jobs.size();
	net::Listener listener;
	std::string err;
	if (!listener.open(options.port, err)) { log("remote: " + err); return 0; }
	log("remote: " + std::to_string(jobs.size()) + " jobs wait for " + std::to_string(options.min_workers) + " or more workers on port " + std::to_string(options.port));
	std::vector<std::thread> threads;
	auto last_note = std::chrono::steady_clock::now();
	for (;;) {
		{
			std::lock_guard lock(s.m);
			if (s.left == 0) break;
		}
		if (auto c = listener.accept(200)) threads.emplace_back(serve, std::ref(s), std::move(c));
		if (std::chrono::steady_clock::now() - last_note > std::chrono::seconds(60)) {
			last_note = std::chrono::steady_clock::now();
			std::lock_guard lock(s.m);
			log("remote: " + std::to_string(s.left) + " jobs left, " + std::to_string(s.workers) + " workers connected");
		}
	}
	s.cv.notify_all();
	listener.close();
	for (auto& t : threads) t.join();
	return s.imported;
}

int remote_worker(const std::string& host, int port, const std::string& token, const std::string& workdir, const std::vector<std::string>& extra_args,
                  const std::function<void(const std::string&)>& log) {
	std::error_code ec;
	fs::create_directories(workdir, ec);
	const std::string exe = self_exe();
	std::string name = "worker";
#ifdef _WIN32
	if (const char* n = getenv("COMPUTERNAME")) name = n;
#else
	if (const char* n = getenv("HOSTNAME")) name = n;
#endif
	for (;;) {
		std::string err;
		auto conn = net::connect_to(host, port, err);
		if (!conn) {
			log(err + "; trying again in 5 s");
			std::this_thread::sleep_for(std::chrono::seconds(5));
			continue;
		}
		if (!conn->send_msg(M_HELLO, "{\"version\":" + std::to_string(kVersion) + ",\"token\":" + jq(token) + ",\"name\":" + jq(name) + "}")) continue;
		uint8_t type;
		std::string payload;
		if (!conn->recv_msg(type, payload) || type != M_HELLO) {
			log("the host refused this worker (token or version); trying again in 10 s");
			std::this_thread::sleep_for(std::chrono::seconds(10));
			continue;
		}
		log("connected to " + conn->peer);
		while (conn->recv_msg(type, payload)) {
			if (type != M_JOB) continue;
			Json job = Json::parse(payload);
			const std::string id = job.find("id")->text();
			const std::string jname = job.find("name")->text();
			const Json& in = *job.find("input");
			const fs::path dir = fs::path(workdir) / ("job-" + id);
			fs::remove_all(dir, ec);
			fs::create_directories(dir / "out", ec);
			log("job " + jname);
			std::string input, failure;
			if (flag_set(in.find("url")->text())) input = in.find("name")->text();
			else {
				fs::path p;
				if (fetch_input(*conn, fs::path(workdir) / "inputs", in.find("name")->text(), in.find("sha256")->text(), p, failure)) input = p.string();
			}
			bool ok = false;
			std::string log_text = failure;
			if (failure.empty()) {
				{
					std::ofstream f(dir / "config.jsonc", std::ios::binary);
					f << job.find("config")->text();
				}
				std::vector<std::string> args{"--plain", "--config", (dir / "config.jsonc").string(), "--input", input, "--output", (dir / "out").string(), "--no-videos",
				                               "--cache", (fs::path(workdir) / "cache").string(), "--upload", "off"};
				args.insert(args.end(), extra_args.begin(), extra_args.end());
				const int code = run_process(exe, args, (dir / "log.txt").string());
				log_text = tail_of(read_file(dir / "log.txt"), 12);
				ok = code == 0;
				if (!ok) log("job " + jname + " failed with code " + std::to_string(code));
				// The run results are the .json files in the input's folder; the sweep summary next to it is not one.
				for (auto& folder : fs::directory_iterator(dir / "out")) {
					if (!folder.is_directory()) continue;
					for (auto& f : fs::directory_iterator(folder.path())) {
						if (f.path().extension() != ".json") continue;
						const std::string n = f.path().filename().string();
						std::string msg;
						msg += (char)(n.size() & 255);
						msg += (char)(n.size() >> 8);
						msg += n + read_file(f.path());
						if (!conn->send_msg(M_RESULT, msg)) break;
					}
				}
			}
			fs::remove_all(dir, ec);
			if (!conn->send_msg(M_DONE, "{\"ok\":" + std::string(ok ? "true" : "false") + ",\"log\":" + jq(log_text) + "}")) break;
		}
		log("lost the connection; trying again in 5 s");
		std::this_thread::sleep_for(std::chrono::seconds(5));
	}
}
