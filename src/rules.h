#pragma once
// Rules between options of one encoder entry: combinations that cannot work (NVENC with B-frames as references and fewer than 3 reference frames) are never made.
//
//   "rules": [
//     { "if": { "op": "in", "left": { "option": "b_reference_mode" }, "right": ["middle", "each"] },
//       "require": { "op": ">=", "left": { "option": "max_ref_frames" }, "right": 3 } },
//     { "op": "<=", "left": { "option": "lookahead_frames" }, "right": 32 }
//   ]
//
// A rule is an expression that must hold, or { "if": ..., "require": ..., "else_require": ... } where "require" must hold when "if" does and "else_require" (optional) when it
// does not. "require" and "else_require" are a rule or a list of them, so rules nest. A combination is made only when every rule holds.
//
// Expressions are JSON: a number, true/false or a string (a value), { "option": name } (its value), { "set": name } (whether the combination sets it itself,
// not its default), or { "op": ..., ... }:
//   "==" "!=" "<" "<=" ">" ">=" "+" "-" "*" "/" "%"   with "left" and "right"
//   "and" "or"                                        with "args": [...] (or "left" and "right")
//   "in" "not in"                                     with "left" and "right": [values]
//   "not" "neg" "abs" "floor" "ceil"                  with "value"
//   "min" "max"                                       with "args": [...]
// Numbers compare as numbers ("4" == 4.0), words as text; a word is true unless it is "", false, none, off or disabled.
// An option the combination leaves unset ("", not listed, or its condition failed) reads as the encoder's default with the combination's preset, tune and
// profile, as requires/unless do. What is not known (an unset option without a known default, a word compared with < or used in arithmetic) leaves the rule
// undecided, and an undecided rule drops nothing: rules only leave out combinations they can show to be wrong.
#include <functional>
#include <memory>
#include <set>
#include <string>
#include <vector>

struct Json;
struct RuleExpr;

class Rules {
public:
	// The option's value and whether the combination sets it; nullptr (with set false) when it is unset and its default is not known.
	using Lookup = std::function<const std::string*(const std::string& key, bool& set)>;

	void parse(const Json& j); // the entry's "rules": a rule or a list of them; throws with the line of what is wrong
	bool empty() const { return rules_.empty(); }
	size_t size() const { return rules_.size(); }
	const std::set<std::string>& names(size_t rule) const { return rules_[rule].names; } // the options rule `rule` reads (nested rules too)
	std::set<std::string> names() const;
	bool holds(size_t rule, const Lookup& look) const; // false only when the rule is shown not to hold
	bool holds(const Lookup& look) const { for (size_t i = 0; i < rules_.size(); i++) if (!holds(i, look)) return false; return true; }
	std::string text(size_t rule) const { return rules_[rule].text; } // as JSON, for messages
	std::string json() const; // the list as the config writes it (a worker gets the host's rules)

	struct Rule {
		std::shared_ptr<const RuleExpr> expr, cond; // expr: a plain rule; cond: an if
		std::vector<Rule> then_, else_;
		bool has_else = false;
		std::string text;  // the rule as JSON
		std::set<std::string> names;
	};

private:
	std::vector<Rule> rules_;
};
