#pragma once
// Minimal JSON reader for config files; also takes comments (// and /* */) and trailing commas. Numbers keep their source text so "6000" and "2.0" survive round trips.
#include <cctype>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

struct Json {
	enum Type { Null, Bool, Num, Str, Arr, Obj } type = Null;
	std::string s; // Str: the text, Num: source text, Bool: "1" or "0"
	std::vector<Json> a;
	std::vector<std::pair<std::string, Json>> o;

	const Json* find(const std::string& k) const {
		for (auto& [key, v] : o) if (key == k) return &v;
		return nullptr;
	}
	// Scalars as the string an encoder option would take.
	std::string text() const {
		if (type == Arr || type == Obj || type == Null) throw std::runtime_error("expected a string, number or boolean");
		return s;
	}

	static Json parse(const std::string& src) {
		size_t i = 0;
		Json j = value(src, i);
		ws(src, i);
		if (i != src.size()) fail(i, "trailing data");
		return j;
	}

private:
	[[noreturn]] static void fail(size_t at, const char* what) { throw std::runtime_error(std::string("JSON error at offset ") + std::to_string(at) + ": " + what); }
	// Whitespace, and the comments JSONC allows between tokens: // to the end of the line and /* ... */.
	static void ws(const std::string& s, size_t& i) {
		for (;;) {
			while (i < s.size() && isspace((unsigned char)s[i])) i++;
			if (s.compare(i, 2, "//") == 0) { while (i < s.size() && s[i] != '\n') i++; }
			else if (s.compare(i, 2, "/*") == 0) {
				size_t e = s.find("*/", i + 2);
				if (e == std::string::npos) fail(i, "unterminated comment");
				i = e + 2;
			}
			else return;
		}
	}

	static std::string str(const std::string& s, size_t& i) {
		std::string out;
		for (i++; i < s.size() && s[i] != '"'; i++) {
			if (s[i] != '\\') { out += s[i]; continue; }
			if (++i >= s.size()) break;
			switch (s[i]) {
			case 'n': out += '\n'; break;
			case 't': out += '\t'; break;
			case 'r': out += '\r'; break;
			case 'b': out += '\b'; break;
			case 'f': out += '\f'; break;
			case 'u': { // shortcut: BMP only, no surrogate pairs
				if (i + 4 >= s.size()) fail(i, "bad \\u escape");
				unsigned c = (unsigned)std::stoul(s.substr(i + 1, 4), nullptr, 16);
				i += 4;
				if (c < 0x80) out += (char)c;
				else if (c < 0x800) { out += (char)(0xC0 | c >> 6); out += (char)(0x80 | (c & 0x3F)); }
				else { out += (char)(0xE0 | c >> 12); out += (char)(0x80 | ((c >> 6) & 0x3F)); out += (char)(0x80 | (c & 0x3F)); }
				break;
			}
			default: out += s[i]; // " \ /
			}
		}
		if (i >= s.size()) fail(i, "unterminated string");
		i++;
		return out;
	}

	static Json value(const std::string& s, size_t& i) {
		ws(s, i);
		if (i >= s.size()) fail(i, "unexpected end");
		Json j;
		char c = s[i];
		if (c == '{' || c == '[') {
			bool obj = c == '{';
			j.type = obj ? Obj : Arr;
			char close = obj ? '}' : ']';
			i++; ws(s, i);
			if (i < s.size() && s[i] == close) { i++; return j; }
			for (;;) {
				ws(s, i);
				if (i < s.size() && s[i] == close) { i++; return j; } // a trailing comma is fine
				std::string key;
				if (obj) {
					if (i >= s.size() || s[i] != '"') fail(i, "expected key");
					key = str(s, i); ws(s, i);
					if (i >= s.size() || s[i] != ':') fail(i, "expected ':'");
					i++;
				}
				Json v = value(s, i);
				if (obj) j.o.emplace_back(std::move(key), std::move(v)); else j.a.push_back(std::move(v));
				ws(s, i);
				if (i < s.size() && s[i] == ',') { i++; continue; }
				if (i < s.size() && s[i] == close) { i++; return j; }
				fail(i, "expected ',' or closing bracket");
			}
		}
		if (c == '"') { j.type = Str; j.s = str(s, i); return j; }
		size_t b = i;
		while (i < s.size() && (isalnum((unsigned char)s[i]) || s[i] == '-' || s[i] == '+' || s[i] == '.')) i++;
		std::string t = s.substr(b, i - b);
		if (t == "null") return j;
		if (t == "true" || t == "false") { j.type = Bool; j.s = t == "true" ? "1" : "0"; return j; }
		if (t.empty() || !(isdigit((unsigned char)t[0]) || t[0] == '-')) fail(b, "unexpected token");
		j.type = Num; j.s = t;
		return j;
	}
};
