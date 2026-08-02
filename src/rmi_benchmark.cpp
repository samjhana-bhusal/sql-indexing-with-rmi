#include <iostream>
#include <fstream>
#include <vector>
#include <chrono>
#include <random>
#include <cmath>
#include <cassert>
#include <algorithm>
#include <numeric>
#include <string>

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
    if (!in) {
        std::cerr << "Failed to open file: " << path << std::endl;
        return false;
    }
    uint64_t N = 0;
    in.read(reinterpret_cast<char*>(&N), sizeof(uint64_t));
    if (!in) return false;
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

inline double predict_pos(uint64_t key, const RMIParams& params, uint64_t& leaf_idx) {
    double x = (double)(key - params.min_key) / (double)(params.max_key - params.min_key);
    double hidden[32];
    for (int i = 0; i < 32; ++i) {
        double val = x * params.fc1_weight[i] + params.fc1_bias[i];
        hidden[i] = val > 0.0 ? val : 0.0;
    }
    double y = 0.0;
    for (int i = 0; i < 32; ++i) y += hidden[i] * params.fc2_weight[i];
    y += params.fc2_bias[0];

    int idx = (int)(y * params.M);
    if (idx < 0) idx = 0;
    if (idx >= (int)params.M) idx = params.M - 1;
    leaf_idx = idx;

    return params.slopes[leaf_idx] * (double)key + params.intercepts[leaf_idx];
}

// Evict L2/L3 caches by cycling a dummy buffer through memory.
static void flush_caches() {
    const size_t sz = 128 * 1024 * 1024; // 128 MB
    auto* buf = new char[sz];
    for (size_t i = 0; i < sz; ++i) {
        reinterpret_cast<volatile char*>(buf)[i] = static_cast<char>(i & 0xFF);
    }
    delete[] buf;
}

int main(int argc, char* argv[]) {
    std::string data_dir   = "data";
    std::string params_dir = "data";
    int num_trials = 5;

    // Parse arguments
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--trials" && i + 1 < argc) {
            num_trials = std::stoi(argv[++i]);
        } else if (data_dir == "data" && arg[0] != '-') {
            data_dir = arg;
            params_dir = arg;
        }
    }
    if (argc > 2 && std::string(argv[2])[0] != '-') params_dir = argv[2];

    std::string keys_path   = data_dir + "/keys.bin";
    std::string pos_path    = data_dir + "/positions.bin";
    std::string params_path = params_dir + "/rmi_params.bin";

    std::vector<uint64_t> keys;
    std::vector<uint64_t> positions;
    RMIParams params;

    std::cout << "Loading keys/positions from: " << data_dir << std::endl;
    if (!load_binary_file(keys_path, keys) || !load_binary_file(pos_path, positions)) {
        std::cerr << "Error: Could not load data files from: " << data_dir << std::endl;
        return 1;
    }
    if (!load_rmi_params(params_path, params)) {
        std::cerr << "Error: Could not load RMI params from: " << params_dir << std::endl;
        return 1;
    }

    size_t N = keys.size();
    std::cout << "Loaded " << N << " elements." << std::endl;
    assert(N == positions.size());
    assert(N == params.N);

    size_t rmi_mem_bytes = 4 * sizeof(uint64_t)
                         + 32 * sizeof(double) + 32 * sizeof(double)
                         + 32 * sizeof(double) + 1 * sizeof(double)
                         + params.M * sizeof(double) * 2
                         + params.M * sizeof(uint64_t);
    std::cout << "RMI Memory Footprint: " << rmi_mem_bytes << " bytes ("
              << (double)rmi_mem_bytes / 1024.0 << " KB)" << std::endl;

    size_t num_queries = 100000;

    std::cout << "Running " << num_trials << " trials of " << num_queries << " RMI lookups..." << std::endl;

    std::vector<double> throughputs;
    std::vector<double> latencies;

    size_t fallback_count = 0;

    for (int trial = 0; trial < num_trials; ++trial) {
        flush_caches();

        std::mt19937 gen(trial);
        std::uniform_int_distribution<size_t> dis(0, N - 1);
        std::vector<size_t> query_indices(num_queries);
        for (size_t i = 0; i < num_queries; ++i) query_indices[i] = dis(gen);

        // Warm-up
        for (size_t i = 0; i < 10000; ++i) {
            uint64_t leaf = 0;
            predict_pos(keys[dis(gen)], params, leaf);
        }

        auto start = std::chrono::high_resolution_clock::now();
        for (size_t i = 0; i < num_queries; ++i) {
            size_t idx = query_indices[i];
            uint64_t target_key = keys[idx];
            uint64_t leaf_idx = 0;
            double pred_pos = predict_pos(target_key, params, leaf_idx);
            uint64_t eps = params.errors[leaf_idx];

            int64_t low = (int64_t)pred_pos - (int64_t)eps;
            int64_t high = (int64_t)pred_pos + (int64_t)eps;
            if (low < 0) low = 0;
            if (high >= (int64_t)N) high = N - 1;

            auto hi_it = keys.begin() + high + 1;
            auto it = std::lower_bound(keys.begin() + low, hi_it, target_key);
            // Last-mile fallback: if Stage 1 assigned a leaf whose error bound
            // does not cover this key, the bounded window misses. Widen to a
            // full search rather than returning a wrong answer. Rare, but must
            // be counted and reported, not swallowed.
            if (it == hi_it || *it != target_key) {
                it = std::lower_bound(keys.begin(), keys.end(), target_key);
                ++fallback_count;
            }
            assert(it != keys.end() && *it == target_key);
            uint64_t found_pos = std::distance(keys.begin(), it);
            assert(found_pos == positions[idx]);
        }
        auto end = std::chrono::high_resolution_clock::now();

        double elapsed = std::chrono::duration<double>(end - start).count();
        double avg_lat = (elapsed * 1e6) / num_queries;
        double mqps = (static_cast<double>(num_queries) / elapsed) / 1e6;

        throughputs.push_back(mqps);
        latencies.push_back(avg_lat);
        std::cout << "  Trial " << (trial + 1) << ": " << mqps << " MQPS, "
                  << avg_lat << " us/query" << std::endl;
    }

    auto mean = [](const std::vector<double>& v) {
        return std::accumulate(v.begin(), v.end(), 0.0) / v.size();
    };
    auto stddev = [&mean](const std::vector<double>& v) {
        double m = mean(v);
        double sq_sum = 0.0;
        for (auto x : v) sq_sum += (x - m) * (x - m);
        return std::sqrt(sq_sum / v.size());
    };

    double mean_mqps = mean(throughputs);
    double std_mqps  = stddev(throughputs);
    double mean_lat  = mean(latencies);
    double std_lat   = stddev(latencies);
    double min_mqps  = *std::min_element(throughputs.begin(), throughputs.end());
    double max_mqps  = *std::max_element(throughputs.begin(), throughputs.end());

    std::cout << "\nResults (" << num_trials << " trials):" << std::endl;
    std::cout << "  Throughput: " << mean_mqps << " +/- " << std_mqps << " MQPS"
              << " [min=" << min_mqps << ", max=" << max_mqps << "]" << std::endl;
    std::cout << "  Latency:    " << mean_lat << " +/- " << std_lat << " us/query" << std::endl;

    size_t total_queries = num_queries * (size_t)num_trials;
    double fallback_rate = 100.0 * (double)fallback_count / (double)total_queries;
    std::cout << "  Bound misses: " << fallback_count << " / " << total_queries
              << " (" << fallback_rate << "%)" << std::endl;

    std::ofstream out(params_dir + "/rmi_results.txt");
    if (out) {
        out << "memory_bytes: " << rmi_mem_bytes << "\n";
        out << "avg_latency_us: " << mean_lat << "\n";
        out << "throughput_mqps: " << mean_mqps << "\n";
        out << "stddev_mqps: " << std_mqps << "\n";
        out << "bound_misses: " << fallback_count << "\n";
        out << "bound_miss_rate_pct: " << fallback_rate << "\n";
        out << "trials: " << num_trials << "\n";
    }

    return 0;
}
