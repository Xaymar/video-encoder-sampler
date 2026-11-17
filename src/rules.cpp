#include "rules.h"

#include <cctype>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <stdexcept>

#include "json.h"

namespace {

struct Value {
	enum Kind { Null, Num, Str } kind = Null; // Null: not known
	double n = 0;
	std::string s;
	static Value num(double d) { Value v; v.kind = Num; v.n = d; return v; }
	static Value str(const std::string& t) { Value v; v.kind = Str; v.s = t; return v; }
	static Value boolean(bool b) { return num(b ? 1 : 0); }
	// An option's value: a number when it reads as one, else a word.
	static Value option(const std::string& t) {
		char* end = nullptr;
		const double d = strtod(t.c_str(), &end);
		if (!t.empty() && *end == 0) return num(d);
		return str(t);
	}
};

// 1 true, 0 false, -1 not known. A word is true except the ones that mean off.
int truth(const Value& v) {
	if (v.kind == Value::Null) return -1;
	if (v.kind == Value::Num) return v.n != 0;
	return !(v.s.empty() || v.s == "false" || v.s == "none" || v.s == "off" || v.s == "disabled");
}

Value tri(int t) { return t < 0 ? Value() : Value::boolean(t == 1); }

} // namespace

struct RuleExpr {
	enum Op { Const, Name, Set, Neg, Not, And, Or, Eq, Ne, Lt, Le, Gt, Ge, In, NotIn, Add, Sub, Mul, Div, Mod, Min, Max, Abs, Floor, Ceil } op = Const;
	Value value;     // Const
	std::string name; // Name, Set
	std::vector<std::shared_ptr<const RuleExpr>> args;

	Value eval(const Rules::Lookup& look) const {
		auto arg = [&](size_t i) { return args[i]->eval(look); };
		switch (op) {
		case Const: return value;
		case Name: {
			bool set = false;
			const std::string* v = look(name, set);
			return v && !v->empty() ? Value::option(*v) : Value();
		}
		case Set: { bool set = false; look(name, set); return Value::boolean(set); }
		case Neg: { Value a = arg(0); return a.kind == Value::Num ? Value::num(-a.n) : Value(); }
		case Not: { const int t = truth(arg(0)); return tri(t < 0 ? -1 : !t); }
		case And: {
			const int a = truth(arg(0));
			if (a == 0) return Value::boolean(false);
			const int b = truth(arg(1));
			return tri(b == 0 ? 0 : a < 0 || b < 0 ? -1 : 1);
		}
		case Or: {
			const int a = truth(arg(0));
			if (a == 1) return Value::boolean(true);
			const int b = truth(arg(1));
			return tri(b == 1 ? 1 : a < 0 || b < 0 ? -1 : 0);
		}
		case Eq: case Ne: {
			const int e = equal(arg(0), arg(1));
			return tri(e < 0 ? -1 : (op == Eq) == (e == 1));
		}
		case Lt: case Le: case Gt: case Ge: {
			Value a = arg(0), b = arg(1);
			if (a.kind != Value::Num || b.kind != Value::Num) return Value(); // words have no order
			return Value::boolean(op == Lt ? a.n < b.n : op == Le ? a.n <= b.n : op == Gt ? a.n > b.n : a.n >= b.n);
		}
		case In: case NotIn: {
			Value a = arg(0);
			if (a.kind == Value::Null) return Value();
			bool unknown = false;
			for (size_t i = 1; i < args.size(); i++) {
				const int e = equal(a, arg(i));
				if (e == 1) return Value::boolean(op == In);
				unknown |= e < 0;
			}
			return unknown ? Value() : Value::boolean(op == NotIn);
		}
		case Add: case Sub: case Mul: case Div: case Mod: {
			Value a = arg(0), b = arg(1);
			if (a.kind != Value::Num || b.kind != Value::Num) return Value();
			if ((op == Div || op == Mod) && b.n == 0) return Value();
			return Value::num(op == Add ? a.n + b.n : op == Sub ? a.n - b.n : op == Mul ? a.n * b.n : op == Div ? a.n / b.n : std::fmod(a.n, b.n));
		}
		case Min: case Max: {
			Value best;
			for (size_t i = 0; i < args.size(); i++) {
				Value a = arg(i);
				if (a.kind != Value::Num) return Value();
				if (!i || (op == Min ? a.n < best.n : a.n > best.n)) best = a;
			}
			return best;
		}
		case Abs: case Floor: case Ceil: {
			Value a = arg(0);
			if (a.kind != Value::Num) return Value();
			return Value::num(op == Abs ? std::fabs(a.n) : op == Floor ? std::floor(a.n) : std::ceil(a.n));
		}
		}
		return Value();
	}

	// 1 equal, 0 not, -1 not known. Numbers compare as numbers ("4" == 4.0), words as text; a number is never a word.
	static int equal(const Value& a, const Value& b) {
		if (a.kind == Value::Null || b.kind == Value::Null) return -1;
		if (a.kind != b.kind) return 0;
		return a.kind == Value::Num ? a.n == b.n : a.s == b.s;
	}
};

namespace {

using Ptr = std::shared_ptr<const RuleExpr>;

class Parser {
public:
	Parser(const std::string& text, const std::string& where, std::set<std::string>& names) : t_(text), where_(where), names_(names) {}

	Ptr parse() {
		Ptr e = or_();
		skip();
		if (p_ < t_.size()) fail("unexpected '" + t_.substr(p_, 1) + "'");
		return e;
	}

private:
	const std::string& t_;
	std::string where_;
	std::set<std::string>& names_;
	size_t p_ = 0;

	[[noreturn]] void fail(const std::string& why) const {
		throw std::runtime_error(where_ + "rule '" + t_ + "', column " + std::to_string(p_ + 1) + ": " + why);
	}
	void skip() { while (p_ < t_.size() && isspace((unsigned char)t_[p_])) p_++; }
	bool ident_start(size_t i) const { return i < t_.size() && (isalpha((unsigned char)t_[i]) || t_[i] == '_'); }
	bool ident_char(size_t i) const {
		if (i >= t_.size()) return false;
		const unsigned char c = (unsigned char)t_[i];
		return isalnum(c) || c == '_' || ((c == '-' || c == '.') && ident_start(i + 1)); // "b-adapt"; "a-1" is a minus
	}
	// The next word without taking it ("" when there is none).
	std::string peek_word() {
		skip();
		if (!ident_start(p_)) return {};
		size_t e = p_;
		while (ident_char(e)) e++;
		return t_.substr(p_, e - p_);
	}
	bool word(const char* w) {
		if (peek_word() != w) return false;
		p_ += strlen(w);
		return true;
	}
	bool sym(const char* s) {
		skip();
		if (t_.compare(p_, strlen(s), s) != 0) return false;
		if (s[0] == '<' || s[0] == '>' || s[0] == '=' || s[0] == '!') // "<" is not the start of "<="
			if (!s[1] && p_ + 1 < t_.size() && t_[p_ + 1] == '=') return false;
		p_ += strlen(s);
		return true;
	}
	static Ptr node(RuleExpr::Op op, std::vector<Ptr> args) {
		auto e = std::make_shared<RuleExpr>();
		e->op = op;
		e->args = std::move(args);
		return e;
	}

	Ptr or_() {
		Ptr a = and_();
		while (word("or") || sym("||")) a = node(RuleExpr::Or, {a, and_()});
		return a;
	}
	Ptr and_() {
		Ptr a = not_();
		while (word("and") || sym("&&")) a = node(RuleExpr::And, {a, not_()});
		return a;
	}
	Ptr not_() {
		if (word("not") || (peek_char() == '!' && !next_is('=') && sym("!"))) return node(RuleExpr::Not, {not_()});
		return cmp();
	}
	char peek_char() { skip(); return p_ < t_.size() ? t_[p_] : 0; }
	bool next_is(char c) const { return p_ + 1 < t_.size() && t_[p_ + 1] == c; }
	Ptr cmp() {
		Ptr a = add();
		static const std::pair<const char*, RuleExpr::Op> ops[] = {{"==", RuleExpr::Eq}, {"!=", RuleExpr::Ne}, {"<=", RuleExpr::Le}, {">=", RuleExpr::Ge},
		                                                           {"<", RuleExpr::Lt}, {">", RuleExpr::Gt}, {"=", RuleExpr::Eq}};
		for (auto& [s, op] : ops) if (sym(s)) {
			Ptr b = add();
			if (peek_cmp()) fail("comparisons do not chain: write 'a < b and b < c'");
			return node(op, {a, b});
		}
		const size_t at = p_;
		bool negated = false;
		if (word("not")) {
			if (peek_word() != "in") { p_ = at; return a; }
			negated = true;
		}
		if (word("in")) {
			skip();
			if (!sym("[")) fail("'in' takes a list: x in [1, 2]");
			std::vector<Ptr> args{a};
			if (!sym("]")) {
				do args.push_back(add()); while (sym(","));
				if (!sym("]")) fail("expected ']' or ','");
			}
			return node(negated ? RuleExpr::NotIn : RuleExpr::In, std::move(args));
		}
		return a;
	}
	bool peek_cmp() {
		skip();
		const char c = p_ < t_.size() ? t_[p_] : 0;
		return c == '<' || c == '>' || c == '=' || (c == '!' && next_is('='));
	}
	Ptr add() {
		Ptr a = mul();
		for (;;) {
			if (sym("+")) a = node(RuleExpr::Add, {a, mul()});
			else if (sym("-")) a = node(RuleExpr::Sub, {a, mul()});
			else return a;
		}
	}
	Ptr mul() {
		Ptr a = unary();
		for (;;) {
			if (sym("*")) a = node(RuleExpr::Mul, {a, unary()});
			else if (sym("/")) a = node(RuleExpr::Div, {a, unary()});
			else if (sym("%")) a = node(RuleExpr::Mod, {a, unary()});
			else return a;
		}
	}
	Ptr unary() {
		if (sym("-")) return node(RuleExpr::Neg, {unary()});
		if (sym("+")) return unary();
		return primary();
	}
	Ptr primary() {
		skip();
		if (p_ >= t_.size()) fail("the rule ends where a value was expected");
		const char c = t_[p_];
		if (c == '(') {
			p_++;
			Ptr e = or_();
			if (!sym(")")) fail("expected ')'");
			return e;
		}
		if (c == '\'' || c == '"') {
			const size_t start = ++p_;
			while (p_ < t_.size() && t_[p_] != c) p_++;
			if (p_ >= t_.size()) { p_ = start - 1; fail("the string is not closed"); }
			auto e = std::make_shared<RuleExpr>();
			e->value = Value::str(t_.substr(start, p_ - start));
			p_++;
			return e;
		}
		if (isdigit((unsigned char)c) || (c == '.' && p_ + 1 < t_.size() && isdigit((unsigned char)t_[p_ + 1]))) {
			char* end = nullptr;
			const double d = strtod(t_.c_str() + p_, &end);
			p_ = (size_t)(end - t_.c_str());
			auto e = std::make_shared<RuleExpr>();
			e->value = Value::num(d);
			return e;
		}
		const std::string w = peek_word();
		if (w.empty()) fail("unexpected '" + std::string(1, c) + "'");
		p_ += w.size();
		if (w == "true" || w == "false") { auto e = std::make_shared<RuleExpr>(); e->value = Value::boolean(w == "true"); return e; }
		if (w == "and" || w == "or" || w == "not" || w == "in") { p_ -= w.size(); fail("'" + w + "' where a value was expected"); }
		if (sym("(")) { // a function
			if (w == "set") {
				const std::string n = peek_word();
				if (n.empty()) fail("set() takes an option name: set(max_b_frames)");
				p_ += n.size();
				if (!sym(")")) fail("expected ')'");
				names_.insert(n);
				auto e = std::make_shared<RuleExpr>();
				e->op = RuleExpr::Set;
				e->name = n;
				return e;
			}
			static const std::pair<const char*, RuleExpr::Op> fns[] = {{"min", RuleExpr::Min}, {"max", RuleExpr::Max}, {"abs", RuleExpr::Abs}, {"floor", RuleExpr::Floor}, {"ceil", RuleExpr::Ceil}};
			for (auto& [name, op] : fns) if (w == name) {
				std::vector<Ptr> args;
				if (!sym(")")) {
					do args.push_back(or_()); while (sym(","));
					if (!sym(")")) fail("expected ')' or ','");
				}
				const bool one = op == RuleExpr::Abs || op == RuleExpr::Floor || op == RuleExpr::Ceil;
				if (one ? args.size() != 1 : args.empty()) fail(w + "() takes " + (one ? "one value" : "one or more values"));
				return node(op, std::move(args));
			}
			p_ -= w.size() + 1;
			fail("'" + w + "' is not a function (set, min, max, abs, floor, ceil)");
		}
		names_.insert(w);
		auto e = std::make_shared<RuleExpr>();
		e->op = RuleExpr::Name;
		e->name = w;
		return e;
	}
};

std::string quote(const std::string& s) {
	std::string o = "\"";
	for (unsigned char c : s) {
		if (c == '"' || c == '\\') { o += '\\'; o += (char)c; }
		else if (c < 0x20) { char b[8]; snprintf(b, sizeof b, "\\u%04x", c); o += b; }
		else o += (char)c;
	}
	return o + "\"";
}

void parse_list(const Json& j, std::vector<Rules::Rule>& out, std::set<std::string>& names);

Rules::Rule parse_rule(const Json& j, std::set<std::string>& names) {
	Rules::Rule r;
	if (j.type == Json::Obj) {
		for (auto& [k, v] : j.o)
			if (k != "if" && k != "then" && k != "else") throw std::runtime_error(v.where() + "a rule has an unknown key '" + k + "' (it takes if, then and else)");
		const Json* c = j.find("if");
		const Json* t = j.find("then");
		const Json* e = j.find("else");
		if (!c) throw std::runtime_error(j.where() + "a rule object needs 'if' (or write the rule as a plain expression)");
		if (!t && !e) throw std::runtime_error(j.where() + "a rule with 'if' needs 'then' or 'else'");
		if (c->type != Json::Str) throw std::runtime_error(c->where() + "'if' takes an expression in a string");
		r.text = c->s;
		r.cond = Parser(r.text, c->where(), r.names).parse();
		if (t) parse_list(*t, r.then_, r.names);
		if (e) { parse_list(*e, r.else_, r.names); r.has_else = true; }
	} else if (j.type == Json::Str) {
		r.text = j.s;
		r.expr = Parser(r.text, j.where(), r.names).parse();
	} else throw std::runtime_error(j.where() + "a rule is an expression in a string or an object with if/then/else");
	names.insert(r.names.begin(), r.names.end());
	return r;
}

void parse_list(const Json& j, std::vector<Rules::Rule>& out, std::set<std::string>& names) {
	if (j.type == Json::Arr) { for (auto& x : j.a) out.push_back(parse_rule(x, names)); }
	else out.push_back(parse_rule(j, names));
}

bool rule_holds(const Rules::Rule& r, const Rules::Lookup& look) {
	if (r.expr) return truth(r.expr->eval(look)) != 0;
	const int c = truth(r.cond->eval(look));
	if (c < 0) return true; // undecided: nothing is left out
	for (auto& x : c ? r.then_ : r.else_) if (!rule_holds(x, look)) return false;
	return true;
}

std::string rule_json(const Rules::Rule& r) {
	if (r.expr) return quote(r.text);
	std::string o = "{\"if\":" + quote(r.text);
	for (auto* list : {&r.then_, &r.else_}) {
		if (list == &r.else_ && !r.has_else) continue;
		if (list == &r.then_ && r.then_.empty()) continue;
		o += list == &r.then_ ? ",\"then\":[" : ",\"else\":[";
		for (size_t i = 0; i < list->size(); i++) o += (i ? "," : "") + rule_json((*list)[i]);
		o += "]";
	}
	return o + "}";
}

} // namespace

void Rules::parse(const Json& j) {
	std::set<std::string> unused;
	parse_list(j, rules_, unused);
}

std::set<std::string> Rules::names() const {
	std::set<std::string> all;
	for (auto& r : rules_) all.insert(r.names.begin(), r.names.end());
	return all;
}

bool Rules::holds(size_t rule, const Lookup& look) const { return rule_holds(rules_[rule], look); }

std::string Rules::json() const {
	std::string o = "[";
	for (size_t i = 0; i < rules_.size(); i++) o += (i ? "," : "") + rule_json(rules_[i]);
	return o + "]";
}
