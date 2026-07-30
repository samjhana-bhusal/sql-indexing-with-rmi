#include <iostream>
#include <fstream>
#include <vector>
#include <map>
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

class ReadWriteLearnedIndex {
private:
    const std::vector<uint64_t>& keys;
    const std::vector<uint64_t>& positions;
    const RMIParams& params;
    std::map<uint64_t, uint64_t> delta_buffer;

public:
    ReadWriteLearnedIndex(const std::vector<uint64_t>& k, const std::vector<uint64_t>& p, const RMIParams& rmi)
        : keys(k), positions(p), params(rmi) {}

    void insert(uint64_t key, uint64_t value) {
        delta_buffer[key] = value;
    }

    bool lookup(uint64_t key, uint64_t& value) {
        auto it = delta_buffer.find(key);
        if (it != delta_buffer.end()) {
            value = it->second;
            return true;
        }

        uint64_t leaf_idx = 0;
        double pred_pos = predict_pos(key, params, leaf_idx);
        uint64_t eps = params.errors[leaf_idx];

        int64_t low = (int64_t)pred_pos - (int64_t)eps;
        int64_t high = (int64_t)pred_pos + (int64_t)eps;
        if (low < 0) low = 0;
        if (high >= (int64_t)keys.size()) high = keys.size() - 1;

        auto bit = std::lower_bound(keys.begin() + low, keys.begin() + high + 1, key);
        if (bit != keys.end() && *bit == key) {
            value = positions[std::distance(keys.begin(), bit)];
            return true;
        }
        return false;
    }

    size_t buffer_size() const { return delta_buffer.size(); }
    void clear_buffer() { delta_buffer.clear(); }
};

static void flush_caches() {
    const size_t sz = 128 * 1024 * 1024;
    auto* buf = new char[sz];
    for (size_t i = 0; i < sz; ++i) {
        reinterpret_cast<volatile char*>(buf)[i] = static_cast<char>(i & 0xFF);
    }
    delete[] buf;
}

int main(int argc, char* argv[]) {
    std::string data_dir = "data";
    int num_trials = 5;

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--trials" && i + 1 < argc) {
            num_trials = std::stoi(argv[++i]);
        } else if (data_dir == "data" && arg[0] != '-') {
            data_dir = arg;
        }
    }

    std::string keys_path = data_dir + "/keys.bin";
    std::string pos_path = data_dir + "/positions.bin";
    std::string params_path = data_dir + "/rmi_params.bin";

    std::vector<uint64_t> keys;
    std::vector<uint64_t> positions;
    RMIParams params;

    std::cout << "Loading dataset binaries..." << std::endl;
    if (!load_binary_file(keys_path, keys) || !load_binary_file(pos_path, positions) || !load_rmi_params(params_path, params)) {
        std::cerr << "Error: Could not load data files." << std::endl;
        return 1;
    }

    ReadWriteLearnedIndex rw_index(keys, positions, params);

    std::cout << "\n==========================================================" << std::endl;
    std::cout << "  Read-Write Learned Index (LSM Delta Buffer) Benchmark   " << std::endl;
    std::cout << "  Trials: " << num_trials << std::endl;
    std::cout << "==========================================================" << std::endl;

    // Insertion latency
    size_t num_writes = 10000;
    std::cout << "Executing " << num_writes << " dynamic writes..." << std::endl;

    auto write_start = std::chrono::high_resolution_clock::now();
    for (size_t i = 0; i < num_writes; ++i) {
        rw_index.insert(2000000000000 + i * 3, i * 100);
    }
    auto write_end = std::chrono::high_resolution_clock::now();

    double write_time = std::chrono::duration<double>(write_end - write_start).count();
    std::cout << "  Average Insertion Latency: " << (write_time * 1e6) / num_writes << " us/write" << std::endl;

    // Read latency with varying buffer sizes — repeated trials
    std::vector<size_t> buffer_sizes = {0, 100, 1000, 5000, 10000};
    size_t num_reads = 50000;

    std::cout << "\nEvaluating lookup latency vs. delta buffer size (" << num_trials << " trials)..." << std::endl;

    for (size_t size : buffer_sizes) {
        std::vector<double> trial_latencies;

        for (int trial = 0; trial < num_trials; ++trial) {
            rw_index.clear_buffer();
            for (size_t i = 0; i < size; ++i) {
                rw_index.insert(2000000000000 + i * 3, i * 100);
            }

            flush_caches();

            std::mt19937 gen(trial);
            std::uniform_int_distribution<size_t> dis(0, keys.size() - 1);

            auto read_start = std::chrono::high_resolution_clock::now();
            for (size_t i = 0; i < num_reads; ++i) {
                uint64_t target_key;
                uint64_t target_val = 0;
                if (size > 0 && i % 5 == 0) {
                    target_key = 2000000000000 + (i % size) * 3;
                    bool found = rw_index.lookup(target_key, target_val);
                    assert(found);
                } else {
                    size_t idx = dis(gen);
                    target_key = keys[idx];
                    bool found = rw_index.lookup(target_key, target_val);
                    assert(found && target_val == positions[idx]);
                }
            }
            auto read_end = std::chrono::high_resolution_clock::now();

            double read_time = std::chrono::duration<double>(read_end - read_start).count();
            double latency = (read_time * 1e6) / num_reads;
            trial_latencies.push_back(latency);
        }

        double mean = std::accumulate(trial_latencies.begin(), trial_latencies.end(), 0.0) / trial_latencies.size();
        double sq_sum = 0.0;
        for (auto x : trial_latencies) sq_sum += (x - mean) * (x - mean);
        double sd = std::sqrt(sq_sum / trial_latencies.size());

        std::cout << "  Buffer Size: " << size << " | Avg Lookup Latency: "
                  << mean << " +/- " << sd << " us/read" << std::endl;
    }

    std::cout << "==========================================================" << std::endl;
    return 0;
}
