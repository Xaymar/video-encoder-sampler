#pragma once
// The option combinations of one encoder entry, as rows of small integers (one per option, the index of its value) instead of option maps with strings: a big config has
// millions of them for every input, bit rate and size, and only a few are ever run (see ComboTable::layer), so only the layers that are asked for are listed.
#include <cstdint>
#include <set>
#include <string>
#include <vector>

#include "encoder.h"

// A condition on another option: that it is set, set to one of some values, or compares so with a number (">=2").
struct Cond {
	std::string key;
	std::vector<std::string> any_of; // empty: any value
	std::string op;                  // "", or ">=", ">", "<=", "<", "==", "!=": the value is read as a number
	double num = 0;
	static bool number(const std::string& v, double& out);
	bool holds_value(const std::string* have) const; // have: the other option's value, null when it is not set
	bool holds(const Options& o) const;
	std::string json() const; // as the config writes it: true, a list of values, or a comparison
};

struct Axis { // one swept option. null is not a value: it would mean "whatever the preset gives", which is not known; leave the option out to run without it
	std::string key;
	std::vector<std::string> values;
	// Options that only matter in some combinations: where the condition does not hold the option is left unset.
	std::vector<Cond> requires_all; // it is only set when all of these hold
	std::vector<Cond> unless_any;   // it is only set when none of these holds
};

class ComboTable {
public:
	static constexpr uint8_t UNSET = 255;
	using Id = uint64_t; // a combination: its row as a number with one digit per option (the value's index, or one more for "not set")

	// `axes`: the swept options in the order of the config; `free_axes`: more options that take every value and are never pruned (the scenario's chroma, bit
	// depth and range). `prune`: whether options with more than two values can be pruned; `fixed`: options that never are (nor are gpu, chroma, bitdepth, range).
	// Nothing is listed here: a combination is a number, and a layer is listed when it is first asked for.
	ComboTable(const std::string& encoder, const std::vector<Axis>& axes, const std::vector<Axis>& free_axes, const std::set<std::string>& fixed, bool prune);

	Options options(Id combo) const;

	// Combinations by how many prunable options are neither at their lowest nor at their highest value. Layer 0 are the probes (and everything of options that
	// are not pruned); a combination of layer k is only worth running when the lines it is on in k options show an effect, which layer k-1 tells. Layers are in
	// the order the combinations are made in (options in config order, values in list order) and are kept once listed; one may be empty.
	size_t layers() const { return prunable_.size() + 1; }
	const std::vector<Id>& layer(size_t k) const;

	struct Line { int axis; int64_t lo, hi; }; // the combinations with the option at its lowest and at its highest value, all else the same; -1: there is none
	void mid_lines(Id combo, std::vector<Line>& out) const;
	const std::string& axis_key(int axis) const { return axes_[(size_t)axis].key; }
	size_t prunable() const { return prunable_.size(); }

private:
	struct Ax {
		std::string key;
		std::vector<std::string> values;
		std::vector<Cond> requires_all, unless_any;
		std::vector<std::pair<int, const Cond*>> conds; // the conditions with the place of the option they name (-1: none of the axes), requires first
		size_t requires_n = 0;                          // how many of conds are requires_all
		int lo = -1, hi = -1;                           // index of the lowest and highest value of a prunable option
		bool prunable = false;
		Id radix = 1, weight = 1;                       // values + 1 ("not set"); the digit's place value
	};
	std::vector<uint8_t> row(Id c) const;
	Id id(const std::vector<uint8_t>& r) const;
	bool applies(size_t k, const std::vector<uint8_t>& r) const; // whether option k is set, given the options before it
	bool valid(const std::vector<uint8_t>& r) const;              // whether the combinations are made with this row
	int64_t find(const std::vector<uint8_t>& r) const { return valid(r) ? (int64_t)id(r) : -1; }

	std::vector<Ax> axes_;          // in the order the combinations are made: an option after the ones its conditions name
	std::vector<int> prunable_;     // indices into axes_
	mutable std::vector<std::vector<Id>> layers_;
	mutable std::vector<bool> listed_;
};
