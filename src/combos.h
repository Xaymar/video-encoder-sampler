#pragma once
// The option combinations of one encoder entry, as rows of small integers (one per option, the index of its value) instead of option maps with strings: a big config has
// hundreds of thousands of them for every input, bit rate and size, and only a few are ever run (see ComboTable::layer).
#include <cstdint>
#include <set>
#include <string>
#include <unordered_map>
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

	// `axes`: the swept options in the order of the config; `free_axes`: more options that take every value and are never pruned (the scenario's chroma, bit
	// depth and range). `prune`: whether options with more than two values can be pruned; `fixed`: options that never are (nor are gpu, chroma, bitdepth, range).
	ComboTable(const std::string& encoder, const std::vector<Axis>& axes, const std::vector<Axis>& free_axes, const std::set<std::string>& fixed, bool prune);

	size_t size() const { return count_; }
	Options options(uint32_t combo) const;

	// Combinations by how many prunable options are neither at their lowest nor at their highest value. Layer 0 are the probes (and everything of options that
	// are not pruned); a combination of layer k is only worth running when the lines it is on in k options show an effect, which layer k-1 tells.
	size_t layers() const { return layers_.size(); }
	const std::vector<uint32_t>& layer(size_t k) const { return layers_[k]; }

	struct Line { int axis; int64_t lo, hi; }; // the combinations with the option at its lowest and at its highest value, all else the same; -1: there is none
	void mid_lines(uint32_t combo, std::vector<Line>& out) const;
	const std::string& axis_key(int axis) const { return axes_[(size_t)axis].key; }
	size_t prunable() const { return prunable_.size(); }

private:
	struct Ax {
		std::string key;
		std::vector<std::string> values;
		std::vector<Cond> requires_all, unless_any;
		int lo = -1, hi = -1; // index of the lowest and highest value of a prunable option
	};
	const uint8_t* row(uint32_t c) const { return rows_.data() + (size_t)c * axes_.size(); }
	int64_t find(const std::vector<uint8_t>& r) const;
	static uint64_t hash(const uint8_t* r, size_t n);

	std::vector<Ax> axes_;          // in the order the combinations are made: an option after the ones its conditions name
	std::vector<int> prunable_;     // indices into axes_
	size_t count_ = 0;
	std::vector<uint8_t> rows_;     // count_ rows of axes_.size() value indices (UNSET: the option is not set)
	std::vector<std::vector<uint32_t>> layers_;
	std::unordered_map<uint64_t, uint32_t> index_;
};
