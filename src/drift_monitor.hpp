#pragma once
#include <vector>
#include <cmath>
#include <algorithm>
#include <cstdint>
#include <numeric>
#include <map>

struct KSResult {
    double statistic;
    double p_value;
    bool is_drifted;
};

// Two-sample Kolmogorov-Smirnov test.
// Port of the core logic from proj2/src/drift_lab/drift/detectors.py
// (kolmogorov_smirnov_test), adapted from multi-feature to single-feature (key).
// Both inputs must be sorted ascending.
inline KSResult ks_2samp(const std::vector<uint64_t>& ref_sorted,
                         const std::vector<uint64_t>& test_sorted,
                         double alpha = 0.01) {
    size_t n = ref_sorted.size();
    size_t m = test_sorted.size();

    if (n == 0 || m == 0) return {0.0, 1.0, false};

    double d_max = 0.0;
    size_t i = 0, j = 0;

    while (i < n || j < m) {
        uint64_t val;
        if (i < n && (j >= m || ref_sorted[i] <= test_sorted[j])) {
            val = ref_sorted[i];
        } else {
            val = test_sorted[j];
        }

        while (i < n && ref_sorted[i] <= val) ++i;
        while (j < m && test_sorted[j] <= val) ++j;

        double f_ref  = static_cast<double>(i) / n;
        double f_test = static_cast<double>(j) / m;
        double diff = std::fabs(f_ref - f_test);
        if (diff > d_max) d_max = diff;
    }

    // Asymptotic p-value: Kolmogorov distribution
    // lambda = D * sqrt(n*m / (n+m))
    double en = std::sqrt(static_cast<double>(n) * m / (n + m));
    double lambda = (en + 0.12 + 0.11 / en) * d_max;

    double p_value = 0.0;
    if (lambda > 0.0) {
        // P(D > d) ≈ 2 * sum_{k=1}^{inf} (-1)^{k+1} * exp(-2*k^2*lambda^2)
        double two_lam_sq = 2.0 * lambda * lambda;
        for (int k = 1; k <= 100; ++k) {
            double sign = (k % 2 == 1) ? 1.0 : -1.0;
            double term = sign * std::exp(-two_lam_sq * k * k);
            p_value += term;
        }
        p_value *= 2.0;
        p_value = std::max(0.0, std::min(1.0, p_value));
    } else {
        p_value = 1.0;
    }

    return {d_max, p_value, p_value < alpha};
}

struct SegmentStats {
    double training_max_error;
    double running_error_sum;
    uint64_t running_error_count;
};

struct RetrainEvent {
    size_t segment_id;
    double error_before;
    double error_after;
    bool ks_triggered;
    bool error_triggered;
};

class DriftMonitor {
public:
    DriftMonitor(size_t num_segments,
                 const std::vector<uint64_t>& training_errors,
                 double error_multiplier = 1.5,
                 double ks_alpha = 0.01,
                 size_t min_segment_samples = 30)
        : num_segments_(num_segments),
          error_multiplier_(error_multiplier),
          ks_alpha_(ks_alpha),
          min_segment_samples_(min_segment_samples) {

        seg_stats_.resize(num_segments);
        for (size_t i = 0; i < num_segments; ++i) {
            seg_stats_[i].training_max_error = static_cast<double>(training_errors[i]);
            seg_stats_[i].running_error_sum = 0.0;
            seg_stats_[i].running_error_count = 0;
        }
    }

    void record_lookup_error(size_t segment_id, double abs_error) {
        if (segment_id >= num_segments_) return;
        seg_stats_[segment_id].running_error_sum += abs_error;
        seg_stats_[segment_id].running_error_count++;
    }

    void reset_segment_stats(size_t segment_id) {
        if (segment_id >= num_segments_) return;
        seg_stats_[segment_id].running_error_sum = 0.0;
        seg_stats_[segment_id].running_error_count = 0;
    }

    void update_training_error(size_t segment_id, double new_max_error) {
        if (segment_id >= num_segments_) return;
        seg_stats_[segment_id].training_max_error = new_max_error;
    }

    // Identify segments needing retrain.
    // training_keys_per_segment: the original sorted keys per segment (for KS).
    // buffer: the delta buffer contents.
    // predict_segment: callable that maps a key to a segment index.
    std::vector<RetrainEvent> segments_needing_retrain(
        const std::vector<std::vector<uint64_t>>& training_keys_per_segment,
        const std::map<uint64_t, uint64_t>& buffer,
        const std::vector<double>& slopes,
        const std::vector<double>& intercepts,
        const std::vector<uint64_t>& errors,
        // Stage 1 predict: key -> segment_id (provided as function pointer or
        // inlined in the caller). We accept segment assignments for buffer keys
        // directly to avoid coupling with the NN.
        const std::vector<std::pair<size_t, uint64_t>>& buffer_segment_assignments
    ) {
        // Build per-segment buffer key lists
        std::vector<std::vector<uint64_t>> buffer_keys_per_seg(num_segments_);
        for (auto& [seg, key] : buffer_segment_assignments) {
            if (seg < num_segments_) {
                buffer_keys_per_seg[seg].push_back(key);
            }
        }
        // Sort buffer keys per segment (needed for KS test)
        for (auto& v : buffer_keys_per_seg) {
            std::sort(v.begin(), v.end());
        }

        std::vector<RetrainEvent> hot;

        for (size_t i = 0; i < num_segments_; ++i) {
            bool error_trigger = false;
            bool ks_trigger = false;

            // Check 1: prediction error ratio
            auto& st = seg_stats_[i];
            if (st.running_error_count > 0 && st.training_max_error > 0) {
                double mean_err = st.running_error_sum / st.running_error_count;
                if (mean_err > error_multiplier_ * st.training_max_error) {
                    error_trigger = true;
                }
            }

            // Check 2: KS test on key distribution
            if (buffer_keys_per_seg[i].size() >= min_segment_samples_ &&
                training_keys_per_segment[i].size() >= min_segment_samples_) {
                auto ks = ks_2samp(training_keys_per_segment[i],
                                   buffer_keys_per_seg[i],
                                   ks_alpha_);
                if (ks.is_drifted) {
                    ks_trigger = true;
                }
            }

            if (error_trigger || ks_trigger) {
                double err_before = (st.running_error_count > 0)
                    ? st.running_error_sum / st.running_error_count
                    : static_cast<double>(errors[i]);
                hot.push_back({i, err_before, 0.0, ks_trigger, error_trigger});
            }
        }

        return hot;
    }

    size_t num_segments() const { return num_segments_; }

private:
    size_t num_segments_;
    double error_multiplier_;
    double ks_alpha_;
    size_t min_segment_samples_;
    std::vector<SegmentStats> seg_stats_;
};
