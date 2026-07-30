#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <map>
#include <chrono>
#include <random>
#include <cmath>
#include <cassert>
#include <algorithm>
#include <numeric>
#include "drift_monitor.hpp"

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

// Refit a single Stage 2 OLS segment from a merged set of sorted keys + positions.
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

class AdaptiveRMI {
public:
    AdaptiveRMI(std::vector<uint64_t>& keys,
                std::vector<uint64_t>& positions,
                RMIParams& params,
                size_t check_interval = 10000,
                double error_multiplier = 1.5)
        : keys_(keys), positions_(positions), params_(params),
          check_interval_(check_interval),
          inserts_since_check_(0),
          total_inserts_(0),
          total_retrains_(0),
          total_segments_retrained_(0),
          retrain_wall_clock_(0.0) {

        monitor_ = std::make_unique<DriftMonitor>(
            params.M, params.errors, error_multiplier);

        // Build per-segment key + position lists for KS reference and refit
        training_keys_per_seg_.resize(params.M);
        training_pos_per_seg_.resize(params.M);
        for (size_t i = 0; i < keys.size(); ++i) {
            uint64_t seg = predict_segment(keys[i], params);
            training_keys_per_seg_[seg].push_back(keys[i]);
            training_pos_per_seg_[seg].push_back(positions[i]);
        }
    }

    void insert(uint64_t key, uint64_t value) {
        delta_buffer_[key] = value;
        inserts_since_check_++;
        total_inserts_++;

        if (inserts_since_check_ >= check_interval_) {
            run_drift_check();
            inserts_since_check_ = 0;
        }
    }

    bool lookup(uint64_t key, uint64_t& value) {
        auto it = delta_buffer_.find(key);
        if (it != delta_buffer_.end()) {
            value = it->second;
            // Still probe the RMI to record prediction error for monitoring.
            // The key is in the buffer (not in the sorted array), so the RMI
            // prediction is "wrong" by definition — this is exactly the drift
            // signal we want to capture.
            uint64_t leaf_idx = 0;
            double pred_pos = predict_pos(key, params_, leaf_idx);
            double abs_err = std::fabs(pred_pos - static_cast<double>(keys_.size()));
            monitor_->record_lookup_error(leaf_idx, abs_err);
            return true;
        }

        uint64_t leaf_idx = 0;
        double pred_pos = predict_pos(key, params_, leaf_idx);
        uint64_t eps = params_.errors[leaf_idx];

        int64_t low = static_cast<int64_t>(pred_pos) - static_cast<int64_t>(eps);
        int64_t high = static_cast<int64_t>(pred_pos) + static_cast<int64_t>(eps);
        if (low < 0) low = 0;
        if (high >= static_cast<int64_t>(keys_.size())) high = keys_.size() - 1;

        auto bit = std::lower_bound(keys_.begin() + low, keys_.begin() + high + 1, key);
        if (bit != keys_.end() && *bit == key) {
            size_t found_idx = std::distance(keys_.begin(), bit);
            value = positions_[found_idx];

            double abs_err = std::fabs(pred_pos - static_cast<double>(found_idx));
            monitor_->record_lookup_error(leaf_idx, abs_err);
            return true;
        }
        return false;
    }

    size_t buffer_size() const { return delta_buffer_.size(); }

    // Summary stats
    size_t total_inserts() const { return total_inserts_; }
    size_t total_retrains() const { return total_retrains_; }
    size_t total_segments_retrained() const { return total_segments_retrained_; }
    double retrain_wall_clock() const { return retrain_wall_clock_; }

    const std::vector<std::string>& retrain_log() const { return retrain_csv_lines_; }

private:
    std::vector<uint64_t>& keys_;
    std::vector<uint64_t>& positions_;
    RMIParams& params_;
    std::map<uint64_t, uint64_t> delta_buffer_;
    std::unique_ptr<DriftMonitor> monitor_;
    std::vector<std::vector<uint64_t>> training_keys_per_seg_;
    std::vector<std::vector<uint64_t>> training_pos_per_seg_;

    size_t check_interval_;
    size_t inserts_since_check_;
    size_t total_inserts_;
    size_t total_retrains_;
    size_t total_segments_retrained_;
    double retrain_wall_clock_;
    std::vector<std::string> retrain_csv_lines_;

    void run_drift_check() {
        // Build segment assignments for buffer keys
        std::vector<std::pair<size_t, uint64_t>> buffer_seg_assignments;
        buffer_seg_assignments.reserve(delta_buffer_.size());
        for (auto& [key, val] : delta_buffer_) {
            uint64_t seg = predict_segment(key, params_);
            buffer_seg_assignments.emplace_back(seg, key);
        }

        auto hot = monitor_->segments_needing_retrain(
            training_keys_per_seg_, delta_buffer_,
            params_.slopes, params_.intercepts, params_.errors,
            buffer_seg_assignments);

        if (hot.empty()) return;

        // Check if >50% segments are hot → flag full retrain recommended
        bool full_retrain_flag = hot.size() > params_.M / 2;

        auto retrain_start = std::chrono::high_resolution_clock::now();

        for (auto& event : hot) {
            size_t seg = event.segment_id;

            // Merge training keys/positions + buffer keys for this segment
            std::vector<uint64_t> merged_keys = training_keys_per_seg_[seg];
            std::vector<uint64_t> merged_positions = training_pos_per_seg_[seg];

            for (auto& [bk, bv] : delta_buffer_) {
                uint64_t bs = predict_segment(bk, params_);
                if (bs == seg) {
                    merged_keys.push_back(bk);
                    merged_positions.push_back(bv);
                }
            }

            // Sort merged keys and positions together
            std::vector<size_t> order(merged_keys.size());
            std::iota(order.begin(), order.end(), 0);
            std::sort(order.begin(), order.end(), [&](size_t a, size_t b) {
                return merged_keys[a] < merged_keys[b];
            });
            std::vector<uint64_t> sorted_keys(merged_keys.size());
            std::vector<uint64_t> sorted_pos(merged_keys.size());
            for (size_t i = 0; i < order.size(); ++i) {
                sorted_keys[i] = merged_keys[order[i]];
                sorted_pos[i] = merged_positions[order[i]];
            }

            double err_before = static_cast<double>(params_.errors[seg]);
            refit_segment(seg, sorted_keys, sorted_pos, params_);
            double err_after = static_cast<double>(params_.errors[seg]);

            monitor_->update_training_error(seg, err_after);
            monitor_->reset_segment_stats(seg);

            // Update training keys and positions for this segment
            training_keys_per_seg_[seg] = sorted_keys;
            training_pos_per_seg_[seg] = sorted_pos;

            // CSV log line
            std::ostringstream line;
            line << total_inserts_ << "," << seg << ","
                 << err_before << "," << err_after << ","
                 << (event.ks_triggered ? "true" : "false") << ","
                 << (event.error_triggered ? "true" : "false") << ","
                 << (full_retrain_flag ? "true" : "false");
            retrain_csv_lines_.push_back(line.str());
        }

        auto retrain_end = std::chrono::high_resolution_clock::now();
        double elapsed = std::chrono::duration<double>(retrain_end - retrain_start).count();
        retrain_wall_clock_ += elapsed;
        total_retrains_++;
        total_segments_retrained_ += hot.size();
    }
};

// Generate insert keys that simulate distribution drift:
// first half from the original key range, second half shifted right.
static std::vector<uint64_t> generate_drifting_inserts(
    uint64_t min_key, uint64_t max_key, size_t count, uint64_t seed) {
    std::mt19937_64 gen(seed);
    uint64_t range = max_key - min_key;
    std::vector<uint64_t> inserts;
    inserts.reserve(count);

    size_t half = count / 2;

    // First half: keys within the original range (minor drift)
    std::uniform_int_distribution<uint64_t> in_range(min_key, max_key);
    for (size_t i = 0; i < half; ++i) {
        inserts.push_back(in_range(gen));
    }

    // Second half: keys shifted right — genuine covariate shift
    uint64_t shift = range / 2;
    std::uniform_int_distribution<uint64_t> shifted(max_key, max_key + shift);
    for (size_t i = half; i < count; ++i) {
        inserts.push_back(shifted(gen));
    }

    return inserts;
}

int main(int argc, char* argv[]) {
    std::string data_dir = "data";
    if (argc > 1) data_dir = argv[1];

    std::string keys_path   = data_dir + "/keys.bin";
    std::string pos_path    = data_dir + "/positions.bin";
    std::string params_path = data_dir + "/rmi_params.bin";

    std::vector<uint64_t> keys;
    std::vector<uint64_t> positions;
    RMIParams params;

    std::cout << "Loading dataset from: " << data_dir << std::endl;
    if (!load_binary_file(keys_path, keys) ||
        !load_binary_file(pos_path, positions) ||
        !load_rmi_params(params_path, params)) {
        std::cerr << "Error: Could not load data files from: " << data_dir << std::endl;
        return 1;
    }

    size_t N = keys.size();
    std::cout << "Loaded " << N << " keys, M=" << params.M << " segments." << std::endl;

    // Configuration — CLI overrides or defaults
    size_t num_inserts = 200000;
    size_t check_interval = 10000;
    double error_multiplier = 1.5;
    size_t num_reads_per_window = 10000;

    // Parse optional flags: --inserts N --interval N --alpha F
    for (int i = 2; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--inserts" && i + 1 < argc) num_inserts = std::stoull(argv[++i]);
        else if (arg == "--interval" && i + 1 < argc) check_interval = std::stoull(argv[++i]);
        else if (arg == "--alpha" && i + 1 < argc) error_multiplier = std::stod(argv[++i]);
    }

    AdaptiveRMI index(keys, positions, params, check_interval, error_multiplier);

    // Generate drifting insert workload
    auto insert_keys = generate_drifting_inserts(
        params.min_key, params.max_key, num_inserts, 42);

    std::cout << "\n==========================================================" << std::endl;
    std::cout << "  Adaptive RMI Benchmark (Drift-Triggered Localized Retrain)" << std::endl;
    std::cout << "==========================================================" << std::endl;
    std::cout << "  Inserts: " << num_inserts << std::endl;
    std::cout << "  Check interval: " << check_interval << std::endl;
    std::cout << "  Error multiplier (alpha): " << error_multiplier << std::endl;

    // Interleaved insert + read workload
    std::mt19937 gen(123);
    std::uniform_int_distribution<size_t> key_dis(0, N - 1);

    size_t windows = num_inserts / check_interval;

    std::cout << "\nWindow | Inserts | Buffer | Reads MQPS | Retrains | Segs Retrained" << std::endl;
    std::cout << "-------|---------|--------|------------|----------|---------------" << std::endl;

    for (size_t w = 0; w < windows; ++w) {
        size_t ins_start = w * check_interval;
        size_t ins_end = ins_start + check_interval;

        size_t retrains_before = index.total_retrains();
        size_t segs_before = index.total_segments_retrained();

        // Insert batch
        for (size_t i = ins_start; i < ins_end && i < num_inserts; ++i) {
            index.insert(insert_keys[i], i);
        }

        // Read throughput measurement
        // Mix: 80% existing keys (correctness-checked), 20% insert keys
        // (exercises drifted segments so DriftMonitor records errors).
        auto read_start = std::chrono::high_resolution_clock::now();
        for (size_t i = 0; i < num_reads_per_window; ++i) {
            uint64_t val = 0;
            if (i % 5 == 0 && ins_end > 0) {
                size_t buf_idx = gen() % ins_end;
                uint64_t target = insert_keys[buf_idx];
                index.lookup(target, val);
            } else {
                size_t idx = key_dis(gen);
                uint64_t target = keys[idx];
                bool found = index.lookup(target, val);
                assert(found);
            }
        }
        auto read_end = std::chrono::high_resolution_clock::now();

        double read_time = std::chrono::duration<double>(read_end - read_start).count();
        double mqps = (static_cast<double>(num_reads_per_window) / read_time) / 1e6;

        std::cout << "  " << (w + 1) << "    | " << (ins_end) << "   | "
                  << index.buffer_size() << "  | "
                  << mqps << "   | "
                  << (index.total_retrains() - retrains_before) << "        | "
                  << (index.total_segments_retrained() - segs_before) << std::endl;
    }

    std::cout << "\n==========================================================" << std::endl;
    std::cout << "  Summary" << std::endl;
    std::cout << "==========================================================" << std::endl;
    std::cout << "  Total inserts:            " << index.total_inserts() << std::endl;
    std::cout << "  Total retrain events:     " << index.total_retrains() << std::endl;
    std::cout << "  Total segments retrained: " << index.total_segments_retrained() << std::endl;
    std::cout << "  Retrain wall-clock (s):   " << index.retrain_wall_clock() << std::endl;
    std::cout << "  Final buffer size:        " << index.buffer_size() << std::endl;

    // Write retrain log CSV
    std::string csv_path = data_dir + "/adaptive_rmi_retrains.csv";
    std::ofstream csv(csv_path);
    if (csv) {
        csv << "insert_count,segment_id,error_before,error_after,ks_triggered,error_triggered,full_retrain_recommended\n";
        for (auto& line : index.retrain_log()) {
            csv << line << "\n";
        }
        std::cout << "  Retrain log written to: " << csv_path << std::endl;
    }

    // Write summary results
    std::string results_path = data_dir + "/adaptive_rmi_results.txt";
    std::ofstream out(results_path);
    if (out) {
        out << "total_inserts: " << index.total_inserts() << "\n";
        out << "total_retrain_events: " << index.total_retrains() << "\n";
        out << "total_segments_retrained: " << index.total_segments_retrained() << "\n";
        out << "retrain_wall_clock_s: " << index.retrain_wall_clock() << "\n";
        out << "buffer_size_final: " << index.buffer_size() << "\n";
    }

    std::cout << "==========================================================" << std::endl;
    return 0;
}
