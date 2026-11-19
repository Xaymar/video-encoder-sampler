#include "combos.h"

#include <algorithm>
#include <cstdio>
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
	if (!have_value || have_value->empty()) return false; // "" is the option left out
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

bool is_default_key(const std::string& k) { return k == "preset" || k == "tune" || k == "profile" || k == "chroma" || k == "bitdepth" || k == "gpu"; }

ComboTable::ComboTable(const std::string& encoder, const std::vector<Axis>& axes, const std::vector<Axis>& free_axes, const std::set<std::string>& fixed, bool prune,
                       UnsetValue unset_value, Rules rules) : unset_value_(std::move(unset_value)), rules_(std::move(rules)) {
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
	if (unset_value_) // what defaults depend on first, so that a condition can ask for an unset option's default
		for (size_t k = 0; k < config.size(); k++)
			if (is_default_key(config[k].key) && config[k].requires_all.empty() && config[k].unless_any.empty()) { placed[k] = true; axes_.push_back(config[k]); }
	while (axes_.size() < config.size()) {
		bool progress = false;
		for (size_t k = 0; k < config.size(); k++) if (!placed[k] && !waits(k)) { placed[k] = true; axes_.push_back(config[k]); progress = true; }
		if (!progress) for (size_t k = 0; k < config.size(); k++) if (!placed[k]) { placed[k] = true; axes_.push_back(config[k]); } // a circle
	}
	std::map<std::string, int>& position = position_;
	for (size_t k = 0; k < axes_.size(); k++) position[axes_[k].key] = (int)k;
	for (size_t k = 0; k < axes_.size(); k++) if (is_default_key(axes_[k].key)) default_axes_.push_back((int)k);
	for (auto& a : axes_) {
		auto at = [&](const Cond& c) { auto it = position.find(c.key); return it == position.end() ? -1 : it->second; };
		for (auto& c : a.requires_all) a.conds.push_back({at(c), &c});
		a.requires_n = a.conds.size();
		for (auto& c : a.unless_any) a.conds.push_back({at(c), &c});
	}

	// A combination's number: one digit per option, the first option the highest, so that numbers go in the order the combinations are made in.
	Id weight = 1;
	for (size_t k = axes_.size(); k-- > 0;) {
		Ax& a = axes_[k];
		if (!a.conds.empty()) { // only an option with conditions can be left unset
			const auto blank = std::find(a.values.begin(), a.values.end(), std::string());
			a.unset_digit = (int)(blank - a.values.begin());
		}
		a.radix = a.values.size() + (a.unset_digit == (int)a.values.size());
		a.weight = weight;
		if (weight > (Id)INT64_MAX / a.radix) {
			long double n = 1;
			for (auto& b : axes_) n *= (long double)b.values.size();
			char count[32];
			snprintf(count, sizeof count, "%.2Le", n);
			throw std::runtime_error("the options of encoder '" + encoder + "' make about " + count + " combinations, more than can be counted (9.2e18): list fewer values");
		}
		weight *= a.radix;
	}

	// Which options can be pruned: those with something between their lowest and highest value.
	for (size_t k = 0; k < axes_.size(); k++) {
		Ax& a = axes_[k];
		// "" (the option left out) is no point of the range: it is always run, like the ends, and never pruned
		std::vector<int> set;
		for (int i = 0; i < (int)a.values.size(); i++) if (!a.values[(size_t)i].empty()) set.push_back(i);
		if (!prune || fixed.count(a.key) || a.key == "gpu" || a.key == "chroma" || a.key == "bitdepth" || a.key == "range" || a.key == "trc" || set.size() < 3) continue;
		bool numeric = true;
		for (int i : set) { double d; numeric &= Cond::number(a.values[(size_t)i], d); }
		if (numeric) {
			auto less = [&](int x, int y) { return atof(a.values[(size_t)x].c_str()) < atof(a.values[(size_t)y].c_str()); };
			int lo = set.front(), hi = set.front();
			for (int i : set) { if (less(i, lo)) lo = i; if (less(hi, i)) hi = i; }
			a.lo = lo; a.hi = hi;
		} else { a.lo = set.front(); a.hi = set.back(); } // not numbers (presets): the order of the list
		a.prunable = true;
		prunable_.push_back((int)k);
	}
	// A rule is judged once every option it reads is chosen; an unset option reads as its default, which depends on the options of is_default_key.
	rules_at_.assign(axes_.size(), {});
	for (size_t i = 0; i < rules_.size(); i++) {
		int at = -1;
		for (auto& name : rules_.names(i)) { auto it = position_.find(name); if (it != position_.end()) at = std::max(at, it->second); }
		for (int k : default_axes_) at = std::max(at, k);
		all_rules_.push_back((int)i);
		if (at >= 0) rules_at_[(size_t)at].push_back((int)i);
		else if (!rules_hold({(int)i}, std::vector<uint8_t>(axes_.size(), UNSET))) none_ = true;
	}
	layers_.resize(layers());
	listed_.assign(layers(), false);
}

std::vector<uint8_t> ComboTable::row(Id c) const {
	std::vector<uint8_t> r(axes_.size(), UNSET); // as layer() sees a row while making it: the options after k not known yet
	for (size_t k = 0; k < axes_.size(); k++) { // in order: whether "" is listed or not set depends on the options before it
		const Ax& a = axes_[k];
		const Id d = c / a.weight % a.radix;
		r[k] = (int)d != a.unset_digit ? (uint8_t)d : d < a.values.size() && applies(k, r) ? (uint8_t)d : UNSET;
	}
	return r;
}

ComboTable::Id ComboTable::id(const std::vector<uint8_t>& r) const {
	Id c = 0;
	for (size_t k = 0; k < axes_.size(); k++) c += (r[k] == UNSET ? (Id)axes_[k].unset_digit : (Id)r[k]) * axes_[k].weight; // valid rows only: UNSET has a digit
	return c;
}

// An option is set when its conditions hold for the options before it; one that comes later (a circle) or is not set counts as not set.
bool ComboTable::applies(size_t k, const std::vector<uint8_t>& r) const {
	const Ax& a = axes_[k];
	for (size_t i = 0; i < a.conds.size(); i++) {
		const auto [at, c] = a.conds[i];
		const std::string* have = at >= 0 && (size_t)at >= k ? nullptr // comes later (a circle): not known
		                        : at < 0 || r[(size_t)at] == UNSET || axes_[(size_t)at].values[r[(size_t)at]].empty() ? unset(c->key, r) : &axes_[(size_t)at].values[r[(size_t)at]];
		if (c->holds_value(have) != (i < a.requires_n)) return false;
	}
	return true;
}

const std::string* ComboTable::unset(const std::string& key, const std::vector<uint8_t>& r) const {
	if (!unset_value_) return nullptr;
	Options base;
	std::string id;
	for (int k : default_axes_) {
		const uint8_t v = r[(size_t)k];
		if (v == UNSET || axes_[(size_t)k].values[v].empty()) continue;
		base[axes_[(size_t)k].key] = axes_[(size_t)k].values[v];
		id += axes_[(size_t)k].key + "=" + axes_[(size_t)k].values[v] + ";";
	}
	id += "|" + key;
	auto it = unset_known_.find(id);
	if (it == unset_known_.end()) it = unset_known_.emplace(id, unset_value_(base, key)).first;
	return it->second.empty() ? nullptr : &it->second;
}

bool ComboTable::valid(const std::vector<uint8_t>& r) const {
	if (none_) return false;
	for (size_t k = 0; k < axes_.size(); k++) if ((r[k] != UNSET) != applies(k, r)) return false;
	return rules_hold(all_rules_, r);
}

bool ComboTable::rules_hold(const std::vector<int>& which, const std::vector<uint8_t>& r) const {
	if (which.empty()) return true;
	const Rules::Lookup look = [&](const std::string& key, bool& set) -> const std::string* {
		auto it = position_.find(key);
		if (it != position_.end()) {
			const uint8_t v = r[(size_t)it->second];
			if (v != UNSET && !axes_[(size_t)it->second].values[v].empty()) { set = true; return &axes_[(size_t)it->second].values[v]; }
		}
		set = false;
		return unset(key, r);
	};
	for (int i : which) if (!rules_.holds((size_t)i, look)) return false;
	return true;
}

const std::vector<ComboTable::Id>& ComboTable::layer(size_t want) const {
	if (listed_[want]) return layers_[want];
	listed_[want] = true;
	std::vector<Id>& out = layers_[want];
	// Depth first, as the combinations are made, keeping only branches that can still end with `want` options in the middle of their range.
	const size_t n = axes_.size();
	std::vector<size_t> prunable_after(n + 1, 0); // prunable options from k on
	for (size_t k = n; k-- > 0;) prunable_after[k] = prunable_after[k + 1] + axes_[k].prunable;
	std::vector<uint8_t> current(n, UNSET);
	if (none_) return out;
	std::function<void(size_t, size_t, Id)> generate = [&](size_t level, size_t mids, Id c) {
		if (mids > want || mids + prunable_after[level] < want) return;
		if (level == n) { out.push_back(c); return; }
		const Ax& a = axes_[level];
		if (!applies(level, current)) { // the option stays unset, in one branch
			current[level] = UNSET;
			if (rules_hold(rules_at_[level], current)) generate(level + 1, mids, c + (Id)a.unset_digit * a.weight);
			return;
		}
		for (size_t v = 0; v < a.values.size(); v++) {
			current[level] = (uint8_t)v;
			if (!rules_hold(rules_at_[level], current)) continue; // a rule between options says this cannot work
			const bool mid = a.prunable && (int)v != a.lo && (int)v != a.hi && !a.values[v].empty();
			generate(level + 1, mids + mid, c + v * a.weight);
		}
		current[level] = UNSET;
	};
	generate(0, 0, 0);
	return out;
}

Options ComboTable::options(Id c) const {
	Options o;
	const std::vector<uint8_t> r = row(c);
	for (size_t k = 0; k < axes_.size(); k++) if (r[k] != UNSET) o[axes_[k].key] = axes_[k].values[r[k]];
	return o;
}

void ComboTable::mid_lines(Id c, std::vector<Line>& out) const {
	out.clear();
	std::vector<uint8_t> r = row(c);
	for (int p : prunable_) {
		const Ax& a = axes_[(size_t)p];
		const uint8_t v = r[(size_t)p];
		if (v == UNSET || (int)v == a.lo || (int)v == a.hi || a.values[v].empty()) continue;
		r[(size_t)p] = (uint8_t)a.lo;
		const int64_t lo = find(r);
		r[(size_t)p] = (uint8_t)a.hi;
		const int64_t hi = find(r);
		r[(size_t)p] = v;
		out.push_back({p, lo, hi});
	}
}
