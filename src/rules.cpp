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

std::string quote(const std::string& s) {
	std::string o = "\"";
	for (unsigned char c : s) {
		if (c == '"' || c == '\\') { o += '\\'; o += (char)c; }
		else if (c < 0x20) { char b[8]; snprintf(b, sizeof b, "\\u%04x", c); o += b; }
		else o += (char)c;
	}
	return o + "\"";
}

// The JSON of a parsed value, compact: a worker gets the host's rules as they were written.
std::string dump(const Json& j) {
	switch (j.type) {
	case Json::Null: return "null";
	case Json::Bool: return j.s == "1" || j.s == "true" ? "true" : "false";
	case Json::Num: return j.s;
	case Json::Str: return quote(j.s);
	case Json::Arr: { std::string o = "["; for (size_t i = 0; i < j.a.size(); i++) o += (i ? "," : "") + dump(j.a[i]); return o + "]"; }
	case Json::Obj: { std::string o = "{"; for (size_t i = 0; i < j.o.size(); i++) o += (i ? "," : "") + quote(j.o[i].first) + ":" + dump(j.o[i].second); return o + "}"; }
	}
	return "null";
}

Ptr constant(Value v) { auto e = std::make_shared<RuleExpr>(); e->value = std::move(v); return e; }

// An expression: a number, true/false or a string (a value), {"option": name}, {"set": name}, or {"op": ..., ...}:
//   "==" "!=" "<" "<=" ">" ">=" "+" "-" "*" "/" "%"   with "left" and "right"
//   "and" "or"                                        with "args": [...] (or "left" and "right")
//   "in" "not in"                                     with "left" and "right": [values]
//   "not" "neg" "abs" "floor" "ceil"                  with "value"
//   "min" "max"                                       with "args": [...]
Ptr parse_expr(const Json& j, std::set<std::string>& names) {
	switch (j.type) {
	case Json::Num: return constant(Value::option(j.s));
	case Json::Bool: return constant(Value::boolean(j.s == "1" || j.s == "true"));
	case Json::Str: return constant(Value::option(j.s)); // "4" is the number 4, as an option's value is
	case Json::Null: throw std::runtime_error(j.where() + "null is not a value in a rule");
	case Json::Arr: throw std::runtime_error(j.where() + "a list is only a value as the right side of \"in\"");
	case Json::Obj: break;
	}
	auto name_of = [&](const char* key) -> std::string {
		const Json* n = j.find(key);
		if (j.o.size() != 1) throw std::runtime_error(j.where() + "{\"" + key + "\": name} takes nothing else");
		if (n->type != Json::Str || n->s.empty()) throw std::runtime_error(n->where() + "\"" + key + "\" takes an option name");
		names.insert(n->s);
		return n->s;
	};
	if (j.find("option") || j.find("set")) {
		auto e = std::make_shared<RuleExpr>();
		e->op = j.find("option") ? RuleExpr::Name : RuleExpr::Set;
		e->name = name_of(e->op == RuleExpr::Name ? "option" : "set");
		return e;
	}
	const Json* opj = j.find("op");
	if (!opj) throw std::runtime_error(j.where() + "an expression object needs \"op\" (or is {\"option\": name} or {\"set\": name})");
	if (opj->type != Json::Str) throw std::runtime_error(opj->where() + "\"op\" takes a string such as \"==\"");
	const std::string op = opj->s;
	static const std::pair<const char*, RuleExpr::Op> binary[] = {{"==", RuleExpr::Eq}, {"!=", RuleExpr::Ne}, {"<", RuleExpr::Lt}, {"<=", RuleExpr::Le}, {">", RuleExpr::Gt},
	                                                             {">=", RuleExpr::Ge}, {"+", RuleExpr::Add}, {"-", RuleExpr::Sub}, {"*", RuleExpr::Mul}, {"/", RuleExpr::Div},
	                                                             {"%", RuleExpr::Mod}};
	static const std::pair<const char*, RuleExpr::Op> unary[] = {{"not", RuleExpr::Not}, {"neg", RuleExpr::Neg}, {"abs", RuleExpr::Abs}, {"floor", RuleExpr::Floor}, {"ceil", RuleExpr::Ceil}};
	static const std::pair<const char*, RuleExpr::Op> many[] = {{"and", RuleExpr::And}, {"or", RuleExpr::Or}, {"min", RuleExpr::Min}, {"max", RuleExpr::Max}};
	auto only = [&](std::initializer_list<const char*> keys) {
		for (auto& [k, v] : j.o) {
			bool ok = k == "op";
			for (const char* x : keys) ok |= k == x;
			if (!ok) {
				std::string list;
				for (const char* x : keys) list += std::string(list.empty() ? "" : ", ") + "\"" + x + "\"";
				throw std::runtime_error(v.where() + "\"op\": \"" + op + "\" takes " + list + ", not \"" + k + "\"");
			}
		}
	};
	auto need = [&](const char* key) -> const Json& {
		const Json* v = j.find(key);
		if (!v) throw std::runtime_error(j.where() + "\"op\": \"" + op + "\" needs \"" + key + "\"");
		return *v;
	};
	auto node = [](RuleExpr::Op o, std::vector<Ptr> args) { auto e = std::make_shared<RuleExpr>(); e->op = o; e->args = std::move(args); return e; };
	for (auto& [s, o] : binary) if (op == s) { only({"left", "right"}); return node(o, {parse_expr(need("left"), names), parse_expr(need("right"), names)}); }
	for (auto& [s, o] : unary) if (op == s) { only({"value"}); return node(o, {parse_expr(need("value"), names)}); }
	for (auto& [s, o] : many) if (op == s) {
		std::vector<Ptr> args;
		if (j.find("args")) {
			only({"args"});
			const Json& a = need("args");
			if (a.type != Json::Arr || a.a.empty()) throw std::runtime_error(a.where() + "\"args\" takes a list of one or more expressions");
			for (auto& x : a.a) args.push_back(parse_expr(x, names));
		} else {
			only({"left", "right"});
			args = {parse_expr(need("left"), names), parse_expr(need("right"), names)};
		}
		if (o == RuleExpr::And || o == RuleExpr::Or) // as a chain of pairs
			while (args.size() > 1) { Ptr b = args.back(); args.pop_back(); Ptr a = args.back(); args.back() = node(o, {a, b}); }
		if (o == RuleExpr::And || o == RuleExpr::Or) return args[0];
		return node(o, std::move(args));
	}
	if (op == "in" || op == "not in") {
		only({"left", "right"});
		const Json& r = need("right");
		if (r.type != Json::Arr) throw std::runtime_error(r.where() + "\"right\" of \"" + op + "\" takes a list");
		std::vector<Ptr> args{parse_expr(need("left"), names)};
		for (auto& x : r.a) args.push_back(parse_expr(x, names));
		return node(op == "in" ? RuleExpr::In : RuleExpr::NotIn, std::move(args));
	}
	throw std::runtime_error(opj->where() + "unknown \"op\" \"" + op + "\" (== != < <= > >= + - * / % and or not in \"not in\" neg abs floor ceil min max)");
}

void parse_list(const Json& j, std::vector<Rules::Rule>& out, std::set<std::string>& names);

// A rule: an expression that must hold, or {"if": expression, "require": rules, "else_require": rules}, where rules are a rule or a list of them.
Rules::Rule parse_rule(const Json& j, std::set<std::string>& names) {
	Rules::Rule r;
	if (j.type != Json::Obj && j.type != Json::Bool) throw std::runtime_error(j.where() + "a rule is an expression object ({\"op\": ...}) or {\"if\": ..., \"require\": ..., \"else_require\": ...}");
	r.text = dump(j);
	if (j.find("if")) {
		for (auto& [k, v] : j.o)
			if (k != "if" && k != "require" && k != "else_require") throw std::runtime_error(v.where() + "a rule with \"if\" has an unknown key '" + k + "' (it takes if, require and else_require)");
		const Json* t = j.find("require");
		const Json* e = j.find("else_require");
		if (!t && !e) throw std::runtime_error(j.where() + "a rule with \"if\" needs \"require\" or \"else_require\"");
		r.cond = parse_expr(*j.find("if"), r.names);
		if (t) parse_list(*t, r.then_, r.names);
		if (e) { parse_list(*e, r.else_, r.names); r.has_else = true; }
	} else if (j.find("require") || j.find("else_require")) throw std::runtime_error(j.where() + "\"require\" and \"else_require\" need \"if\"");
	else r.expr = parse_expr(j, r.names);
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

std::string rule_json(const Rules::Rule& r) { return r.text; }
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
