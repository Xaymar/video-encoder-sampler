#include "combos.h"

#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <map>
#include <stdexcept>

#include "gpu.h"

bool Cond::number(const std::string& v, double& out) {
	char* end = nullptr;
	out = strtod(v.c_str(), &end);
	return !v.empty() && *end == 0;
}

bool Cond::holds_value(const std::string* have_value) const {
	if (!have_value) return false;
	double have = 0;
	if (!op.empty()) {
		if (!number(*have_value, have)) return false;
		return op == ">=" ? have >= num : op == ">" ? have > num : op == "<=" ? have <= num : op == "<" ? have < num : op == "==" ? have == num : have != num;
	}
	if (any_of.empty()) return true;
	double want = 0;
	for (auto& v : any_of) if (v == *have_value || (number(v, want) && number(*have_value, have) && want == have)) return true;
	return false;
}

bool Cond::holds(const Options& o) const {
	const auto it = o.find(key);
	return holds_value(it == o.end() ? nullptr : &it->second);
}

uint64_t ComboTable::hash(const uint8_t* r, size_t n) {
	uint64_t h = 1469598103934665603ull;
	for (size_t i = 0; i < n; i++) h = (h ^ r[i]) * 1099511628211ull;
	return h;
}

ComboTable::ComboTable(const std::string& encoder, const std::vector<Axis>& axes, const std::vector<Axis>& free_axes, const std::set<std::string>& fixed, bool prune) {
	// The values of every option, without repeats (gpu=all is every GPU of the encoder's vendor).
	std::vector<Ax> config;
	for (auto* list : {&axes, &free_axes}) for (auto& a : *list) {
		if (a.values.empty()) continue;
		Ax x;
		x.key = a.key;
		x.requires_all = a.requires_all;
		x.unless_any = a.unless_any;
		auto add = [&](const std::string& v) { if (std::find(x.values.begin(), x.values.end(), v) == x.values.end()) x.values.push_back(v); };
		for (auto& v : a.values) {
			if (a.key == "gpu" && v == "all") {
				const size_t n = std::max<size_t>(list_gpus(encoder_vendor(encoder)).size(), 1);
				for (size_t i = 0; i < n; i++) add(std::to_string(i));
			} else add(v);
		}
		if (x.values.size() > 254) throw std::runtime_error("option '" + x.key + "' has more than 254 values");
		config.push_back(std::move(x));
	}
	// An option goes after the options its conditions name, so the combinations that cannot be are never made. The rest keeps the order it was written in.
	std::vector<bool> placed(config.size());
	auto waits = [&](size_t k) {
		for (auto* list : {&config[k].requires_all, &config[k].unless_any}) for (auto& c : *list)
			for (size_t j = 0; j < config.size(); j++) if (!placed[j] && j != k && config[j].key == c.key) return true;
		return false;
	};
	while (axes_.size() < config.size()) {
		bool progress = false;
		for (size_t k = 0; k < config.size(); k++) if (!placed[k] && !waits(k)) { placed[k] = true; axes_.push_back(config[k]); progress = true; }
		if (!progress) for (size_t k = 0; k < config.size(); k++) if (!placed[k]) { placed[k] = true; axes_.push_back(config[k]); } // a circle
	}
	std::map<std::string, int> position;
	for (size_t k = 0; k < axes_.size(); k++) position[axes_[k].key] = (int)k;

	// Which options can be pruned: those with something between their lowest and highest value.
	for (size_t k = 0; k < axes_.size(); k++) {
		Ax& a = axes_[k];
		if (!prune || fixed.count(a.key) || a.key == "gpu" || a.key == "chroma" || a.key == "bitdepth" || a.key == "range" || a.values.size() < 3) continue;
		bool numeric = true;
		for (auto& v : a.values) { double d; numeric &= Cond::number(v, d); }
		if (numeric) {
			auto less = [&](int x, int y) { return atof(a.values[(size_t)x].c_str()) < atof(a.values[(size_t)y].c_str()); };
			int lo = 0, hi = 0;
			for (int i = 1; i < (int)a.values.size(); i++) { if (less(i, lo)) lo = i; if (less(hi, i)) hi = i; }
			a.lo = lo; a.hi = hi;
		} else { a.lo = 0; a.hi = (int)a.values.size() - 1; } // not numbers (presets): the order of the list
		prunable_.push_back((int)k);
	}

	// Every valid combination, depth first.
	const size_t n = axes_.size();
	std::vector<uint8_t> current(n, UNSET);
	auto value_of = [&](const std::string& key) -> const std::string* {
		auto it = position.find(key);
		if (it == position.end() || current[(size_t)it->second] == UNSET) return nullptr;
		return &axes_[(size_t)it->second].values[current[(size_t)it->second]];
	};
	auto applies = [&](const Ax& a) {
		for (auto& c : a.requires_all) if (!c.holds_value(value_of(c.key))) return false;
		for (auto& c : a.unless_any) if (c.holds_value(value_of(c.key))) return false;
		return true;
	};
	auto emit = [&] {
		rows_.insert(rows_.end(), current.begin(), current.end());
		count_++;
	};
	std::function<void(size_t)> generate = [&](size_t level) {
		if (level == n) { emit(); return; }
		const Ax& a = axes_[level];
		if (!applies(a)) { current[level] = UNSET; generate(level + 1); return; } // the option stays unset, in one branch
		for (size_t v = 0; v < a.values.size(); v++) { current[level] = (uint8_t)v; generate(level + 1); }
		current[level] = UNSET;
	};
	generate(0);

	// Layers and the way back from a row to its combination.
	index_.reserve(count_);
	std::vector<uint32_t> mid_of;
	for (uint32_t c = 0; c < count_; c++) {
		const uint8_t* r = row(c);
		size_t mids = 0;
		for (int p : prunable_) {
			const uint8_t v = r[p];
			mids += v != UNSET && (int)v != axes_[(size_t)p].lo && (int)v != axes_[(size_t)p].hi;
		}
		if (layers_.size() <= mids) layers_.resize(mids + 1);
		layers_[mids].push_back(c);
		index_[hash(r, n)] = c;
	}
	if (layers_.empty()) layers_.emplace_back();
}

Options ComboTable::options(uint32_t c) const {
	Options o;
	const uint8_t* r = row(c);
	for (size_t k = 0; k < axes_.size(); k++) if (r[k] != UNSET) o[axes_[k].key] = axes_[k].values[r[k]];
	return o;
}

int64_t ComboTable::find(const std::vector<uint8_t>& r) const {
	auto it = index_.find(hash(r.data(), r.size()));
	if (it == index_.end() || memcmp(row(it->second), r.data(), r.size()) != 0) return -1;
	return it->second;
}

void ComboTable::mid_lines(uint32_t c, std::vector<Line>& out) const {
	out.clear();
	std::vector<uint8_t> r(row(c), row(c) + axes_.size());
	for (int p : prunable_) {
		const Ax& a = axes_[(size_t)p];
		const uint8_t v = r[(size_t)p];
		if (v == UNSET || (int)v == a.lo || (int)v == a.hi) continue;
		r[(size_t)p] = (uint8_t)a.lo;
		const int64_t lo = find(r);
		r[(size_t)p] = (uint8_t)a.hi;
		const int64_t hi = find(r);
		r[(size_t)p] = v;
		out.push_back({p, lo, hi});
	}
}
