#include <iostream>
#include <fstream>
#include <vector>
#include <chrono>
#include <random>
#include <cassert>
#include <algorithm>
#include <numeric>
#include <cmath>
#include <string>

#include "lipp.h"

bool load_binary_file(const std::string& path, std::vector<uint64_t>& vec) {
    std::ifstream in(path, std::ios::binary);
    if (!in) return false;
    uint64_t N = 0;
    in.read(reinterpret_cast<char*>(&N), sizeof(uint64_t));
    vec.resize(N);
    in.read(reinterpret_cast<char*>(vec.data()), N * sizeof(uint64_t));
    return in.good();
}

static void flush_caches() {
    const size_t sz = 128 * 1024 * 1024;
    auto* buf = new char[sz];
    for (size_t i = 0; i < sz; ++i) {
        reinterpret_cast<volatile char*>(buf)[i] = static_cast<char>(i & 0xFF);
    }
    delete[] buf;
}

int main(int argc, char* argv[]) {
    std::string run_dir = "data";
    int num_trials = 5;

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--trials" && i + 1 < argc) {
            num_trials = std::stoi(argv[++i]);
        } else if (run_dir == "data" && arg[0] != '-') {
            run_dir = arg;
        }
    }

    std::string keys_path = run_dir + "/keys.bin";
    std::string pos_path  = run_dir + "/positions.bin";

    std::vector<uint64_t> keys;
    std::vector<uint64_t> positions;

    std::cout << "Loading dataset from " << run_dir << "..." << std::endl;
    if (!load_binary_file(keys_path, keys) || !load_binary_file(pos_path, positions)) {
        std::cerr << "Error: Could not load data files." << std::endl;
        return 1;
    }

    size_t N = keys.size();
    std::cout << "Loaded " << N << " elements." << std::endl;
    assert(N == positions.size());

    // Bulk-load LIPP
    std::cout << "Building LIPP index (bulk load)..." << std::endl;
    LIPP<uint64_t, uint64_t> index;

    std::vector<std::pair<uint64_t, uint64_t>> kv_pairs(N);
    for (size_t i = 0; i < N; ++i) {
        kv_pairs[i] = {keys[i], positions[i]};
    }

    auto build_start = std::chrono::high_resolution_clock::now();
    index.bulk_load(kv_pairs.data(), static_cast<int>(N));
    auto build_end = std::chrono::high_resolution_clock::now();
    double build_time = std::chrono::duration<double>(build_end - build_start).count();
    std::cout << "LIPP built in " << build_time << " seconds." << std::endl;

    size_t num_queries = 100000;
    std::cout << "Running " << num_trials << " trials of " << num_queries
              << " random lookups..." << std::endl;

    std::vector<double> throughputs;
    std::vector<double> latencies_vec;

    for (int trial = 0; trial < num_trials; ++trial) {
        flush_caches();

        std::mt19937 gen(trial);
        std::uniform_int_distribution<size_t> dis(0, N - 1);
        std::vector<size_t> query_indices(num_queries);
        for (size_t i = 0; i < num_queries; ++i) query_indices[i] = dis(gen);

        auto start = std::chrono::high_resolution_clock::now();
        for (size_t i = 0; i < num_queries; ++i) {
            size_t idx = query_indices[i];
            uint64_t target_key = keys[idx];
            assert(index.exists(target_key));
            uint64_t val = index.at(target_key);
            assert(val == positions[idx]);
        }
        auto end = std::chrono::high_resolution_clock::now();

        double elapsed = std::chrono::duration<double>(end - start).count();
        double avg_lat = (elapsed * 1e6) / num_queries;
        double mqps = (static_cast<double>(num_queries) / elapsed) / 1e6;

        throughputs.push_back(mqps);
        latencies_vec.push_back(avg_lat);
        std::cout << "  Trial " << (trial + 1) << ": " << mqps << " MQPS, "
                  << avg_lat << " us/query" << std::endl;
    }

    auto mean = [](const std::vector<double>& v) {
        return std::accumulate(v.begin(), v.end(), 0.0) / v.size();
    };
    auto stddev = [&mean](const std::vector<double>& v) {
        double m = mean(v);
        double sq = 0.0;
        for (auto x : v) sq += (x - m) * (x - m);
        return std::sqrt(sq / v.size());
    };

    double mean_mqps = mean(throughputs);
    double std_mqps  = stddev(throughputs);
    double mean_lat  = mean(latencies_vec);

    std::cout << "\nResults (" << num_trials << " trials):" << std::endl;
    std::cout << "  Throughput: " << mean_mqps << " +/- " << std_mqps << " MQPS" << std::endl;
    std::cout << "  Latency:    " << mean_lat << " us/query" << std::endl;

    std::ofstream out(run_dir + "/lipp_results.txt");
    if (out) {
        out << "avg_latency_us: " << mean_lat << "\n";
        out << "throughput_mqps: " << mean_mqps << "\n";
        out << "stddev_mqps: " << std_mqps << "\n";
        out << "trials: " << num_trials << "\n";
    }

    return 0;
}
