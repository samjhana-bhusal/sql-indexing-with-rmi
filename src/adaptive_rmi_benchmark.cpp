// Adaptive RMI benchmark with LSM-style merge/compaction.
//
// Design (rewritten to make retraining actually affect lookups):
//   Inserts land in a delta buffer. Every window we MERGE the buffer into the
//   searchable sorted array, which shifts the true rank (position) of keys.
//   The Stage-1 router (key -> segment) is unchanged by a merge, but each key's
//   true position moves, so the Stage-2 linear models and their error bounds go
//   stale. A lookup predicts a position, searches the bounded window
//   [pred-eps, pred+eps], and — if the key isn't there because the bound was
//   violated — falls back to a full search (guaranteeing correctness but paying
//   a large latency + a "bound miss").
//
//   Three maintenance policies decide what to refit after each merge:
//     NEVER    — refit nothing. Bounds drift out of date; misses accumulate.
//     PERIODIC — refit every segment every merge. Tight bounds, maximal cost.
//     ADAPTIVE — refit only drift-flagged segments (per-segment error + KS test).
//
//   Per window we record read MQPS, p99 latency, bound-miss rate, and mean
//   search-window width, so the cost of NOT maintaining the index is measured,
//   not asserted.

#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <map>
#include <set>
#include <chrono>
#include <random>
#include <cmath>
#include <cassert>
#include <algorithm>
#include <numeric>
#include <string>
#include "drift_monitor.hpp"

enum class RetrainPolicy { NEVER, PERIODIC, ADAPTIVE };

static const char* policy_name(RetrainPolicy p) {
    switch (p) {
        case RetrainPolicy::NEVER:    return "never";
        case RetrainPolicy::PERIODIC: return "periodic";
        case RetrainPolicy::ADAPTIVE: return "adaptive";
    }
    return "unknown";
}

struct RMIParams {
    uint64_t min_key;
    uint64_t max_key;
    uint64_t N;
    uint64_t M;

    double fc1_weight[32];
    double fc1_bias[32];
    double fc2_weight[32];
    double fc2_bias[1];

    std::vector<double> slopes;
    std::vector<double> intercepts;
    std::vector<uint64_t> errors;
};

bool load_binary_file(const std::string& path, std::vector<uint64_t>& vec) {
    std::ifstream in(path, std::ios::binary);
    if (!in) return false;
    uint64_t N = 0;
    in.read(reinterpret_cast<char*>(&N), sizeof(uint64_t));
    vec.resize(N);
    in.read(reinterpret_cast<char*>(vec.data()), N * sizeof(uint64_t));
    return in.good();
}

bool load_rmi_params(const std::string& path, RMIParams& params) {
    std::ifstream in(path, std::ios::binary);
    if (!in) return false;
    in.read(reinterpret_cast<char*>(&params.min_key), sizeof(uint64_t));
    in.read(reinterpret_cast<char*>(&params.max_key), sizeof(uint64_t));
    in.read(reinterpret_cast<char*>(&params.N), sizeof(uint64_t));
    in.read(reinterpret_cast<char*>(&params.M), sizeof(uint64_t));

    in.read(reinterpret_cast<char*>(params.fc1_weight), 32 * sizeof(double));
    in.read(reinterpret_cast<char*>(params.fc1_bias), 32 * sizeof(double));
    in.read(reinterpret_cast<char*>(params.fc2_weight), 32 * sizeof(double));
    in.read(reinterpret_cast<char*>(params.fc2_bias), 1 * sizeof(double));

    uint64_t M = params.M;
    params.slopes.resize(M);
    params.intercepts.resize(M);
    params.errors.resize(M);

    in.read(reinterpret_cast<char*>(params.slopes.data()), M * sizeof(double));
    in.read(reinterpret_cast<char*>(params.intercepts.data()), M * sizeof(double));
    in.read(reinterpret_cast<char*>(params.errors.data()), M * sizeof(uint64_t));
    return in.good();
}

inline uint64_t predict_segment(uint64_t key, const RMIParams& params) {
    double x = static_cast<double>(key - params.min_key) /
               static_cast<double>(params.max_key - params.min_key);
    double hidden[32];
    for (int i = 0; i < 32; ++i) {
        double val = x * params.fc1_weight[i] + params.fc1_bias[i];
        hidden[i] = val > 0.0 ? val : 0.0;
    }
    double y = 0.0;
    for (int i = 0; i < 32; ++i) y += hidden[i] * params.fc2_weight[i];
    y += params.fc2_bias[0];

    int idx = static_cast<int>(y * params.M);
    if (idx < 0) idx = 0;
    if (idx >= static_cast<int>(params.M)) idx = params.M - 1;
    return static_cast<uint64_t>(idx);
}

inline double predict_pos(uint64_t key, const RMIParams& params, uint64_t& leaf_idx) {
    leaf_idx = predict_segment(key, params);
    return params.slopes[leaf_idx] * static_cast<double>(key) + params.intercepts[leaf_idx];
}

// Refit one Stage-2 OLS segment from sorted keys + their ranks (positions).
static void refit_segment(size_t seg_id,
                          const std::vector<uint64_t>& seg_keys,
                          const std::vector<uint64_t>& seg_positions,
                          RMIParams& params) {
    size_t n = seg_keys.size();
    if (n == 0) return;

    if (n == 1) {
        params.slopes[seg_id] = 0.0;
        params.intercepts[seg_id] = static_cast<double>(seg_positions[0]);
        params.errors[seg_id] = 0;
        return;
    }

    double x_sum = 0.0, y_sum = 0.0;
    for (size_t i = 0; i < n; ++i) {
        x_sum += static_cast<double>(seg_keys[i]);
        y_sum += static_cast<double>(seg_positions[i]);
    }
    double x_mean = x_sum / n;
    double y_mean = y_sum / n;

    double num = 0.0, den = 0.0;
    for (size_t i = 0; i < n; ++i) {
        double dx = static_cast<double>(seg_keys[i]) - x_mean;
        double dy = static_cast<double>(seg_positions[i]) - y_mean;
        num += dx * dy;
        den += dx * dx;
    }

    double m = (den == 0.0) ? 0.0 : num / den;
    double c = y_mean - m * x_mean;

    uint64_t max_err = 0;
    for (size_t i = 0; i < n; ++i) {
        double pred = m * static_cast<double>(seg_keys[i]) + c;
        double ae = std::fabs(pred - static_cast<double>(seg_positions[i]));
        uint64_t e = static_cast<uint64_t>(std::ceil(ae));
        if (e > max_err) max_err = e;
    }

    params.slopes[seg_id] = m;
    params.intercepts[seg_id] = c;
    params.errors[seg_id] = max_err;
}

// Per-lookup diagnostics used to build correctness/latency metrics.
struct LookupStat {
    bool found;
    bool bound_miss;    // bounded window failed -> full-search fallback
    uint64_t window;    // elements covered by the search window this lookup used
};

class AdaptiveRMI {
public:
    AdaptiveRMI(std::vector<uint64_t>& keys,
                std::vector<uint64_t>& positions,
                RMIParams& params,
                RetrainPolicy policy,
                double error_multiplier = 1.5)
        : keys_(keys), positions_(positions), params_(params),
          policy_(policy),
          total_inserts_(0), total_merges_(0),
          total_retrains_(0), total_segments_retrained_(0),
          retrain_wall_clock_(0.0) {

        monitor_ = std::make_unique<DriftMonitor>(
            params.M, params.errors, error_multiplier);

        // Per-segment reference key lists (for the KS drift test).
        training_keys_per_seg_.resize(params.M);
        for (size_t i = 0; i < keys_.size(); ++i) {
            uint64_t seg = predict_segment(keys_[i], params_);
            training_keys_per_seg_[seg].push_back(keys_[i]);
        }
    }

    void insert(uint64_t key, uint64_t value) {
        delta_buffer_[key] = value;
        total_inserts_++;
    }

    // Merge the delta buffer into the sorted array and reassign ranks.
    // The router (Stage 1) is untouched, so any staleness this introduces is
    // exactly what a maintenance policy is responsible for repairing.
    void merge_buffer() {
        if (delta_buffer_.empty()) return;

        std::vector<uint64_t> merged;
        merged.reserve(keys_.size() + delta_buffer_.size());

        size_t ki = 0;
        auto it = delta_buffer_.begin();
        while (ki < keys_.size() && it != delta_buffer_.end()) {
            if (keys_[ki] < it->first) {
                merged.push_back(keys_[ki++]);
            } else if (it->first < keys_[ki]) {
                merged.push_back((it++)->first);
            } else {              // key already present: keep one
                merged.push_back(keys_[ki++]);
                ++it;
            }
        }
        while (ki < keys_.size()) merged.push_back(keys_[ki++]);
        while (it != delta_buffer_.end()) merged.push_back((it++)->first);

        keys_ = std::move(merged);
        positions_.resize(keys_.size());
        std::iota(positions_.begin(), positions_.end(), uint64_t{0});
        delta_buffer_.clear();
        total_merges_++;
    }

    // Decide-and-refit for one window. Drift detection (ADAPTIVE) must run on
    // the buffer BEFORE it is merged away, so call order in the driver is:
    //   insert batch -> detect (here) -> merge -> refit (here).
    // We fold both phases into maintain() to keep that contract in one place.
    void maintain() {
        std::vector<size_t> flagged;

        if (policy_ == RetrainPolicy::NEVER) {
            merge_buffer();
            return;
        } else if (policy_ == RetrainPolicy::PERIODIC) {
            merge_buffer();
            flagged.resize(params_.M);
            std::iota(flagged.begin(), flagged.end(), size_t{0});
        } else { // ADAPTIVE
            flagged = detect_drift();     // uses buffer
            merge_buffer();               // clears buffer
        }

        if (flagged.empty()) return;

        auto t0 = std::chrono::high_resolution_clock::now();

        // One pass to bucket the (now merged) array into the flagged segments.
        std::set<size_t> want(flagged.begin(), flagged.end());
        std::vector<std::vector<uint64_t>> seg_keys(params_.M);
        std::vector<std::vector<uint64_t>> seg_pos(params_.M);
        for (size_t i = 0; i < keys_.size(); ++i) {
            uint64_t seg = predict_segment(keys_[i], params_);
            if (want.count(seg)) {
                seg_keys[seg].push_back(keys_[i]);
                seg_pos[seg].push_back(positions_[i]);
            }
        }

        for (size_t seg : flagged) {
            refit_segment(seg, seg_keys[seg], seg_pos[seg], params_);
            monitor_->update_training_error(seg, static_cast<double>(params_.errors[seg]));
            monitor_->reset_segment_stats(seg);
            training_keys_per_seg_[seg] = seg_keys[seg];   // refresh KS reference
        }

        auto t1 = std::chrono::high_resolution_clock::now();
        retrain_wall_clock_ += std::chrono::duration<double>(t1 - t0).count();
        total_retrains_++;
        total_segments_retrained_ += flagged.size();
        last_segments_retrained_ = flagged.size();
    }

    LookupStat lookup(uint64_t key, uint64_t& value) {
        // Recent inserts not yet merged live in the buffer.
        auto it = delta_buffer_.find(key);
        if (it != delta_buffer_.end()) {
            value = it->second;
            return {true, false, 0};
        }

        uint64_t leaf_idx = 0;
        double pred_pos = predict_pos(key, params_, leaf_idx);
        uint64_t eps = params_.errors[leaf_idx];

        // Clamp the search window to [0, N-1] at BOTH ends, in double, before
        // casting. Under drift a stale model can predict a position far outside
        // the array (or overflow to inf/NaN); if low is not also clamped from
        // above, keys_.begin()+low becomes a wild pointer past end() and the
        // bounded search reads unmapped memory. The `!(x >= 0)` tests also
        // reject NaN. A fully out-of-range prediction collapses to a 1-element
        // window that misses, correctly routing the lookup to the full-search
        // fallback below.
        double n_minus_1 = static_cast<double>(keys_.size() - 1);
        double lo_d = pred_pos - static_cast<double>(eps);
        double hi_d = pred_pos + static_cast<double>(eps);
        if (!(lo_d >= 0.0)) lo_d = 0.0;
        if (lo_d > n_minus_1) lo_d = n_minus_1;
        if (!(hi_d >= 0.0)) hi_d = 0.0;
        if (hi_d > n_minus_1) hi_d = n_minus_1;
        int64_t low = static_cast<int64_t>(lo_d);
        int64_t high = static_cast<int64_t>(hi_d);
        uint64_t window = static_cast<uint64_t>(high - low + 1);

        auto begin = keys_.begin() + low;
        auto end = keys_.begin() + high + 1;
        auto bit = std::lower_bound(begin, end, key);
        if (bit != end && *bit == key) {
            size_t idx = std::distance(keys_.begin(), bit);
            value = positions_[idx];
            monitor_->record_lookup_error(leaf_idx, std::fabs(pred_pos - static_cast<double>(idx)));
            return {true, false, window};
        }

        // Bound violated -> full search. Correct, but O(N)-wide: a bound miss.
        auto fit = std::lower_bound(keys_.begin(), keys_.end(), key);
        if (fit != keys_.end() && *fit == key) {
            size_t idx = std::distance(keys_.begin(), fit);
            value = positions_[idx];
            monitor_->record_lookup_error(leaf_idx, std::fabs(pred_pos - static_cast<double>(idx)));
            return {true, true, static_cast<uint64_t>(keys_.size())};
        }
        return {false, true, static_cast<uint64_t>(keys_.size())};
    }

    size_t array_size() const { return keys_.size(); }
    size_t buffer_size() const { return delta_buffer_.size(); }
    size_t total_inserts() const { return total_inserts_; }
    size_t total_merges() const { return total_merges_; }
    size_t total_retrains() const { return total_retrains_; }
    size_t total_segments_retrained() const { return total_segments_retrained_; }
    size_t last_segments_retrained() const { return last_segments_retrained_; }
    double retrain_wall_clock() const { return retrain_wall_clock_; }

private:
    std::vector<uint64_t>& keys_;
    std::vector<uint64_t>& positions_;
    RMIParams& params_;
    RetrainPolicy policy_;
    std::map<uint64_t, uint64_t> delta_buffer_;
    std::unique_ptr<DriftMonitor> monitor_;
    std::vector<std::vector<uint64_t>> training_keys_per_seg_;

    size_t total_inserts_;
    size_t total_merges_;
    size_t total_retrains_;
    size_t total_segments_retrained_;
    size_t last_segments_retrained_ = 0;
    double retrain_wall_clock_;

    std::vector<size_t> detect_drift() {
        std::vector<std::pair<size_t, uint64_t>> buffer_seg_assignments;
        buffer_seg_assignments.reserve(delta_buffer_.size());
        for (auto& [key, val] : delta_buffer_) {
            (void)val;
            buffer_seg_assignments.emplace_back(predict_segment(key, params_), key);
        }

        auto hot = monitor_->segments_needing_retrain(
            training_keys_per_seg_, delta_buffer_,
            params_.slopes, params_.intercepts, params_.errors,
            buffer_seg_assignments);

        std::vector<size_t> flagged;
        flagged.reserve(hot.size());
        for (auto& e : hot) flagged.push_back(e.segment_id);
        return flagged;
    }
};

// Drifting insert workload: first half within the original key range, second
// half shifted beyond max_key — a genuine covariate shift the router did not
// see at training time.
static std::vector<uint64_t> generate_drifting_inserts(
    uint64_t min_key, uint64_t max_key, size_t count, uint64_t seed) {
    std::mt19937_64 gen(seed);
    uint64_t range = max_key - min_key;
    std::vector<uint64_t> inserts;
    inserts.reserve(count);

    size_t half = count / 2;
    std::uniform_int_distribution<uint64_t> in_range(min_key, max_key);
    for (size_t i = 0; i < half; ++i) inserts.push_back(in_range(gen));

    uint64_t shift = range / 2;
    std::uniform_int_distribution<uint64_t> shifted(max_key, max_key + shift);
    for (size_t i = half; i < count; ++i) inserts.push_back(shifted(gen));

    return inserts;
}

static double percentile(std::vector<double>& v, double p) {
    if (v.empty()) return 0.0;
    std::sort(v.begin(), v.end());
    double rank = p / 100.0 * (v.size() - 1);
    size_t lo = static_cast<size_t>(std::floor(rank));
    size_t hi = static_cast<size_t>(std::ceil(rank));
    if (lo == hi) return v[lo];
    double frac = rank - lo;
    return v[lo] * (1.0 - frac) + v[hi] * frac;
}

int main(int argc, char* argv[]) {
    std::string data_dir = "data";
    if (argc > 1 && argv[1][0] != '-') data_dir = argv[1];

    size_t num_inserts = 200000;
    size_t check_interval = 10000;
    double error_multiplier = 1.5;
    size_t num_reads_per_window = 10000;
    RetrainPolicy policy = RetrainPolicy::ADAPTIVE;

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--inserts" && i + 1 < argc) num_inserts = std::stoull(argv[++i]);
        else if (arg == "--interval" && i + 1 < argc) check_interval = std::stoull(argv[++i]);
        else if (arg == "--alpha" && i + 1 < argc) error_multiplier = std::stod(argv[++i]);
        else if (arg == "--reads" && i + 1 < argc) num_reads_per_window = std::stoull(argv[++i]);
        else if (arg == "--policy" && i + 1 < argc) {
            std::string p = argv[++i];
            if (p == "never") policy = RetrainPolicy::NEVER;
            else if (p == "periodic") policy = RetrainPolicy::PERIODIC;
            else if (p == "adaptive") policy = RetrainPolicy::ADAPTIVE;
            else { std::cerr << "Unknown policy: " << p << std::endl; return 1; }
        }
    }

    std::vector<uint64_t> keys, positions;
    RMIParams params;

    std::cout << "Loading dataset from: " << data_dir << std::endl;
    if (!load_binary_file(data_dir + "/keys.bin", keys) ||
        !load_binary_file(data_dir + "/positions.bin", positions) ||
        !load_rmi_params(data_dir + "/rmi_params.bin", params)) {
        std::cerr << "Error: Could not load data files from: " << data_dir << std::endl;
        return 1;
    }

    size_t N = keys.size();
    std::cout << "Loaded " << N << " keys, M=" << params.M << " segments." << std::endl;

    AdaptiveRMI index(keys, positions, params, policy, error_multiplier);
    auto insert_keys = generate_drifting_inserts(
        params.min_key, params.max_key, num_inserts, 42);

    std::cout << "\n==========================================================" << std::endl;
    std::cout << "  Adaptive RMI Benchmark (merge/compaction + maintenance)" << std::endl;
    std::cout << "==========================================================" << std::endl;
    std::cout << "  Policy: " << policy_name(policy) << std::endl;
    std::cout << "  Inserts: " << num_inserts << ", interval: " << check_interval
              << ", alpha: " << error_multiplier << std::endl;

    std::mt19937 gen(123);
    size_t windows = num_inserts / check_interval;

    std::cout << "\nWin | ArraySize | MQPS | p99(us) | BoundMiss% | MeanWin | SegsRetr" << std::endl;
    std::cout << "----|-----------|------|---------|------------|---------|---------" << std::endl;

    // Per-window CSV for the evaluation driver / plots.
    std::string win_csv = data_dir + "/adaptive_windows_" + policy_name(policy) + ".csv";
    std::ofstream wc(win_csv);
    wc << "window,array_size,mqps,p99_us,bound_miss_pct,mean_window,segs_retrained\n";

    for (size_t w = 0; w < windows; ++w) {
        size_t ins_start = w * check_interval;
        size_t ins_end = std::min(ins_start + check_interval, num_inserts);

        for (size_t i = ins_start; i < ins_end; ++i)
            index.insert(insert_keys[i], N + i);   // value = synthetic rank tag

        // Merge + policy-driven maintenance for this window.
        index.maintain();

        // Read workload: half over the whole array (background), half over the
        // already-merged insert keys (the drifted hot set) so a stale model's
        // degradation is actually exercised rather than diluted.
        std::vector<double> latencies_us;
        latencies_us.reserve(num_reads_per_window);
        size_t bound_misses = 0;
        double window_sum = 0.0;
        size_t merged_inserts = ins_end;   // all inserted so far are merged

        for (size_t i = 0; i < num_reads_per_window; ++i) {
            uint64_t target;
            if (i % 2 == 0 && merged_inserts > 0) {
                target = insert_keys[gen() % merged_inserts];
            } else {
                target = keys[gen() % keys.size()];
            }

            uint64_t val = 0;
            auto q0 = std::chrono::high_resolution_clock::now();
            LookupStat st = index.lookup(target, val);
            auto q1 = std::chrono::high_resolution_clock::now();

            assert(st.found);
            latencies_us.push_back(std::chrono::duration<double, std::micro>(q1 - q0).count());
            if (st.bound_miss) ++bound_misses;
            window_sum += static_cast<double>(st.window);
        }

        double total_us = std::accumulate(latencies_us.begin(), latencies_us.end(), 0.0);
        double mqps = (static_cast<double>(num_reads_per_window) / (total_us / 1e6)) / 1e6;
        double p99 = percentile(latencies_us, 99.0);
        double miss_pct = 100.0 * bound_misses / num_reads_per_window;
        double mean_win = window_sum / num_reads_per_window;

        std::cout << "  " << (w + 1) << " | " << index.array_size()
                  << " | " << mqps << " | " << p99
                  << " | " << miss_pct << " | " << mean_win
                  << " | " << index.last_segments_retrained() << std::endl;

        wc << (w + 1) << "," << index.array_size() << "," << mqps << ","
           << p99 << "," << miss_pct << "," << mean_win << ","
           << index.last_segments_retrained() << "\n";
    }
    wc.close();

    std::cout << "\n==========================================================" << std::endl;
    std::cout << "  Summary (" << policy_name(policy) << ")" << std::endl;
    std::cout << "==========================================================" << std::endl;
    std::cout << "  Total inserts:            " << index.total_inserts() << std::endl;
    std::cout << "  Total merges:             " << index.total_merges() << std::endl;
    std::cout << "  Total retrain events:     " << index.total_retrains() << std::endl;
    std::cout << "  Total segments retrained: " << index.total_segments_retrained() << std::endl;
    std::cout << "  Retrain wall-clock (s):   " << index.retrain_wall_clock() << std::endl;
    std::cout << "  Final array size:         " << index.array_size() << std::endl;
    std::cout << "  Per-window CSV:           " << win_csv << std::endl;

    std::string results_path = data_dir + "/adaptive_rmi_results_" + policy_name(policy) + ".txt";
    std::ofstream out(results_path);
    if (out) {
        out << "policy: " << policy_name(policy) << "\n";
        out << "total_inserts: " << index.total_inserts() << "\n";
        out << "total_merges: " << index.total_merges() << "\n";
        out << "total_retrain_events: " << index.total_retrains() << "\n";
        out << "total_segments_retrained: " << index.total_segments_retrained() << "\n";
        out << "retrain_wall_clock_s: " << index.retrain_wall_clock() << "\n";
        out << "array_size_final: " << index.array_size() << "\n";
    }

    std::cout << "==========================================================" << std::endl;
    return 0;
}
