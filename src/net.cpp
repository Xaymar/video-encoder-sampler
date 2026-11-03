#include "net.h"

#include <cstring>
#include <mutex>
#include <vector>

#ifdef _WIN32
#define NOMINMAX
#include <winsock2.h>
#include <ws2tcpip.h>
using sock_t = SOCKET;
static const sock_t kBad = INVALID_SOCKET;
static void sock_close(sock_t s) { closesocket(s); }
static int last_error() { return WSAGetLastError(); }
#else
#include <arpa/inet.h>
#include <netdb.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <poll.h>
#include <sys/socket.h>
#include <unistd.h>
#include <cerrno>
using sock_t = int;
static const sock_t kBad = -1;
static void sock_close(sock_t s) { ::close(s); }
static int last_error() { return errno; }
#endif

namespace net {

namespace {

constexpr uint32_t kMaxMessage = 64u << 20; // a message is at most 64 MB; files are sent in chunks

void startup() {
#ifdef _WIN32
	static std::once_flag once;
	std::call_once(once, [] { WSADATA d; WSAStartup(MAKEWORD(2, 2), &d); });
#endif
}

std::string peer_name(const sockaddr_storage& ss) {
	char host[64] = "?";
	int port = 0;
	if (ss.ss_family == AF_INET) {
		auto* a = (const sockaddr_in*)&ss;
		inet_ntop(AF_INET, &a->sin_addr, host, sizeof host);
		port = ntohs(a->sin_port);
	} else if (ss.ss_family == AF_INET6) {
		auto* a = (const sockaddr_in6*)&ss;
		inet_ntop(AF_INET6, &a->sin6_addr, host, sizeof host);
		port = ntohs(a->sin6_port);
	}
	return std::string(host) + ":" + std::to_string(port);
}

} // namespace

Conn::~Conn() { close(); }

void Conn::close() {
	if (fd != (intptr_t)kBad) { sock_close((sock_t)fd); fd = (intptr_t)kBad; }
}

bool Conn::send_all(const void* data, size_t size) {
	const char* p = (const char*)data;
	while (size) {
		const int n = (int)::send((sock_t)fd, p, (int)std::min<size_t>(size, 1 << 20), 0);
		if (n <= 0) return false;
		p += n;
		size -= (size_t)n;
	}
	return true;
}

bool Conn::recv_all(void* data, size_t size) {
	char* p = (char*)data;
	while (size) {
		const int n = (int)::recv((sock_t)fd, p, (int)std::min<size_t>(size, 1 << 20), 0);
		if (n <= 0) return false;
		p += n;
		size -= (size_t)n;
	}
	return true;
}

bool Conn::send_msg(uint8_t type, const std::string& payload) {
	if (payload.size() > kMaxMessage) return false;
	uint8_t head[5];
	const uint32_t len = htonl((uint32_t)payload.size());
	head[0] = type;
	memcpy(head + 1, &len, 4);
	return send_all(head, sizeof head) && send_all(payload.data(), payload.size());
}

bool Conn::recv_msg(uint8_t& type, std::string& payload) {
	uint8_t head[5];
	if (!recv_all(head, sizeof head)) return false;
	uint32_t len;
	memcpy(&len, head + 1, 4);
	len = ntohl(len);
	if (len > kMaxMessage) return false;
	type = head[0];
	payload.resize(len);
	return len == 0 || recv_all(payload.data(), len);
}

std::unique_ptr<Conn> connect_to(const std::string& host, int port, std::string& err) {
	startup();
	addrinfo hints{}, *res = nullptr;
	hints.ai_socktype = SOCK_STREAM;
	hints.ai_family = AF_UNSPEC;
	if (getaddrinfo(host.c_str(), std::to_string(port).c_str(), &hints, &res) != 0 || !res) { err = "cannot resolve " + host; return nullptr; }
	sock_t s = kBad;
	std::string peer;
	for (addrinfo* a = res; a; a = a->ai_next) {
		s = ::socket(a->ai_family, a->ai_socktype, a->ai_protocol);
		if (s == kBad) continue;
		if (::connect(s, a->ai_addr, (int)a->ai_addrlen) == 0) {
			sockaddr_storage ss{};
			memcpy(&ss, a->ai_addr, std::min<size_t>((size_t)a->ai_addrlen, sizeof ss));
			peer = peer_name(ss);
			break;
		}
		sock_close(s);
		s = kBad;
	}
	freeaddrinfo(res);
	if (s == kBad) { err = "cannot connect to " + host + ":" + std::to_string(port) + " (error " + std::to_string(last_error()) + ")"; return nullptr; }
	int one = 1;
	setsockopt(s, IPPROTO_TCP, TCP_NODELAY, (const char*)&one, sizeof one);
	// A host that vanishes without closing the connection (power off, cable out) is noticed after about 25 s of silence, not hours.
#ifdef _WIN32
	struct { ULONG onoff, time_ms, interval_ms; } keep{1, 10000, 5000};
	DWORD returned = 0;
	WSAIoctl(s, 0x98000004 /* SIO_KEEPALIVE_VALS */, &keep, sizeof keep, nullptr, 0, &returned, nullptr, nullptr);
#else
	setsockopt(s, SOL_SOCKET, SO_KEEPALIVE, &one, sizeof one);
	int idle = 10, interval = 5, count = 3;
	setsockopt(s, IPPROTO_TCP, TCP_KEEPIDLE, &idle, sizeof idle);
	setsockopt(s, IPPROTO_TCP, TCP_KEEPINTVL, &interval, sizeof interval);
	setsockopt(s, IPPROTO_TCP, TCP_KEEPCNT, &count, sizeof count);
#endif
	auto c = std::make_unique<Conn>((intptr_t)s);
	c->peer = peer;
	return c;
}

Listener::~Listener() { close(); }

void Listener::close() {
	if (fd != (intptr_t)kBad && fd != -1) sock_close((sock_t)fd);
	fd = -1;
}

bool Listener::open(int port, std::string& err) {
	startup();
	sock_t s = ::socket(AF_INET6, SOCK_STREAM, 0); // dual stack where the system has it
	bool v6 = s != kBad;
	if (v6) {
		int off = 0;
		setsockopt(s, IPPROTO_IPV6, IPV6_V6ONLY, (const char*)&off, sizeof off);
	} else {
		s = ::socket(AF_INET, SOCK_STREAM, 0);
		if (s == kBad) { err = "cannot create a socket"; return false; }
	}
	int one = 1;
#ifndef _WIN32
	setsockopt(s, SOL_SOCKET, SO_REUSEADDR, (const char*)&one, sizeof one);
#endif
	int rc;
	if (v6) {
		sockaddr_in6 a{};
		a.sin6_family = AF_INET6;
		a.sin6_addr = in6addr_any;
		a.sin6_port = htons((uint16_t)port);
		rc = ::bind(s, (sockaddr*)&a, sizeof a);
	} else {
		sockaddr_in a{};
		a.sin_family = AF_INET;
		a.sin_addr.s_addr = htonl(INADDR_ANY);
		a.sin_port = htons((uint16_t)port);
		rc = ::bind(s, (sockaddr*)&a, sizeof a);
	}
	if (rc != 0 || ::listen(s, 16) != 0) {
		err = "cannot listen on port " + std::to_string(port) + " (error " + std::to_string(last_error()) + ")";
		sock_close(s);
		return false;
	}
	fd = (intptr_t)s;
	return true;
}

std::unique_ptr<Conn> Listener::accept(int timeout_ms) {
	if (fd == -1) return nullptr;
	fd_set set;
	FD_ZERO(&set);
	FD_SET((sock_t)fd, &set);
	timeval tv{timeout_ms / 1000, (timeout_ms % 1000) * 1000};
	if (::select((int)fd + 1, &set, nullptr, nullptr, &tv) <= 0) return nullptr;
	sockaddr_storage ss{};
	socklen_t len = sizeof ss;
	sock_t c = ::accept((sock_t)fd, (sockaddr*)&ss, &len);
	if (c == kBad) return nullptr;
	int one = 1;
	setsockopt(c, IPPROTO_TCP, TCP_NODELAY, (const char*)&one, sizeof one);
	auto conn = std::make_unique<Conn>((intptr_t)c);
	conn->peer = peer_name(ss);
	return conn;
}

} // namespace net
