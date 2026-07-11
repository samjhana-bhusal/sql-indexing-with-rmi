#include <iostream>
#include <fstream>
#include <vector>
#include <chrono>
#include <random>
#include <cmath>
#include <cassert>
#include <algorithm>

// Structure to hold RMI parameters
struct RMIParams {
    uint64_t min_key;
    uint64_t max_key;
    uint64_t N;
    uint64_t M;
    
    // Stage 1 Root Model (1 -> 32 -> 1)
    double fc1_weight[32];
    double fc1_bias[32];
    double fc2_weight[32];
    double fc2_bias[1];
    
    // Stage 2 Leaf Models
    std::vector<double> slopes;
    std::vector<double> intercepts;
    std::vector<uint64_t> errors;
};

// Utility function to load SOSD-format binary file
bool load_binary_file(const std::string& path, std::vector<uint64_t>& vec) {
    std::ifstream in(path, std::ios::binary);
    if (!in) {
        std::cerr << "Failed to open file: " << path << std::endl;
        return false;
    }
    uint64_t N = 0;
    in.read(reinterpret_cast<char*>(&N), sizeof(uint64_t));
    if (!in) {
        std::cerr << "Failed to read preamble N from: " << path << std::endl;
        return false;
    }
    
    vec.resize(N);
    in.read(reinterpret_cast<char*>(vec.data()), N * sizeof(uint64_t));
    if (!in) {
        std::cerr << "Failed to read data payload from: " << path << std::endl;
        return false;
    }
    return true;
}

// Load RMI parameters from binary file
bool load_rmi_params(const std::string& path, RMIParams& params) {
    std::ifstream in(path, std::ios::binary);
    if (!in) {
        std::cerr << "Failed to open RMI parameters: " << path << std::endl;
        return false;
    }
    
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
    
    if (!in) {
        std::cerr << "Failed to read RMI parameters body." << std::endl;
        return false;
    }
    return true;
}

// RMI predict function
inline double predict_pos(uint64_t key, const RMIParams& params, uint64_t& leaf_idx) {
    // 1. Stage 1 Neural Network Inference
    double x = (double)(key - params.min_key) / (double)(params.max_key - params.min_key);
    
    double hidden[32];
    for (int i = 0; i < 32; ++i) {
        double val = x * params.fc1_weight[i] + params.fc1_bias[i];
        hidden[i] = val > 0.0 ? val : 0.0; // ReLU
    }
    
    double y = 0.0;
    for (int i = 0; i < 32; ++i) {
        y += hidden[i] * params.fc2_weight[i];
    }
    y += params.fc2_bias[0];
    
    // 2. Select Leaf Model index and clamp
    int idx = (int)(y * params.M);
    if (idx < 0) idx = 0;
    if (idx >= (int)params.M) idx = params.M - 1;
    leaf_idx = idx;
    
    // 3. Stage 2 Linear Spline Inference
    double pred_pos = params.slopes[leaf_idx] * (double)key + params.intercepts[leaf_idx];
    return pred_pos;
}

int main(int argc, char* argv[]) {
    std::string run_dir = "data";
    if (argc > 1) {
        run_dir = argv[1];
    }
    
    std::string keys_path   = run_dir + "/keys.bin";
    std::string pos_path    = run_dir + "/positions.bin";
    std::string params_path = run_dir + "/rmi_params.bin";
    
    std::vector<uint64_t> keys;
    std::vector<uint64_t> positions;
    RMIParams params;
    
    std::cout << "Loading dataset binaries from " << run_dir << "..." << std::endl;
    if (!load_binary_file(keys_path, keys) || !load_binary_file(pos_path, positions)) {
        std::cerr << "Error: Could not load data files from: " << run_dir << std::endl;
        return 1;
    }
    
    std::cout << "Loading RMI parameters from " << run_dir << "..." << std::endl;
    if (!load_rmi_params(params_path, params)) {
        std::cerr << "Error: Could not load RMI params from: " << run_dir << std::endl;
        return 1;
    }
    
    size_t N = keys.size();
    std::cout << "Successfully loaded " << N << " elements." << std::endl;
    assert(N == positions.size());
    assert(N == params.N);
    
    // Calculate RMI memory footprint
    size_t rmi_mem_bytes = 4 * sizeof(uint64_t) // min_key, max_key, N, M
                         + 32 * sizeof(double)  // fc1_weight
                         + 32 * sizeof(double)  // fc1_bias
                         + 32 * sizeof(double)  // fc2_weight
                         + 1 * sizeof(double)   // fc2_bias
                         + params.M * sizeof(double)     // slopes
                         + params.M * sizeof(double)     // intercepts
                         + params.M * sizeof(uint64_t);  // errors
                         
    double rmi_mem_kb = (double)rmi_mem_bytes / 1024.0;
    std::cout << "RMI Memory Footprint: " << rmi_mem_bytes << " bytes (" << rmi_mem_kb << " KB)" << std::endl;
    
    // Select 100,000 random lookup queries
    size_t num_queries = 100000;
    std::vector<size_t> query_indices(num_queries);
    std::mt19937 gen(42); // Same seed for fair comparison
    std::uniform_int_distribution<size_t> dis(0, N - 1);
    for (size_t i = 0; i < num_queries; ++i) {
        query_indices[i] = dis(gen);
    }
    
    std::cout << "Running " << num_queries << " RMI lookups..." << std::endl;
    
    // Warm-up to ensure caching
    for (size_t i = 0; i < 10000; ++i) {
        size_t idx = dis(gen);
        uint64_t target_key = keys[idx];
        uint64_t leaf_idx = 0;
        predict_pos(target_key, params, leaf_idx);
    }
    
    auto lookup_start = std::chrono::high_resolution_clock::now();
    for (size_t i = 0; i < num_queries; ++i) {
        size_t idx = query_indices[i];
        uint64_t target_key = keys[idx];
        
        // Step 1 & 2: Predict position and get leaf bucket
        uint64_t leaf_idx = 0;
        double pred_pos = predict_pos(target_key, params, leaf_idx);
        uint64_t eps = params.errors[leaf_idx];
        
        // Step 3: Define search bounds
        int64_t low = (int64_t)pred_pos - (int64_t)eps;
        int64_t high = (int64_t)pred_pos + (int64_t)eps;
        
        // Clamp bounds to actual data array range
        if (low < 0) low = 0;
        if (high >= (int64_t)N) high = N - 1;
        
        // Step 4: Run local binary search
        auto it = std::lower_bound(keys.begin() + low, keys.begin() + high + 1, target_key);
        
        // Verify correctness
        assert(it != keys.end() && *it == target_key && "RMI lookup key must exist in keys array!");
        uint64_t found_pos = std::distance(keys.begin(), it);
        assert(found_pos == positions[idx] && "Returned index does not match true position!");
    }
    auto lookup_end = std::chrono::high_resolution_clock::now();
    
    double total_lookup_time = std::chrono::duration<double>(lookup_end - lookup_start).count();
    double avg_latency_us = (total_lookup_time * 1e6) / num_queries;
    double throughput_mqps = (static_cast<double>(num_queries) / total_lookup_time) / 1e6;
    
    std::cout << "Benchmarking results:" << std::endl;
    std::cout << "  Total Lookup Time: " << total_lookup_time << " seconds" << std::endl;
    std::cout << "  Average Latency:   " << avg_latency_us << " us/query" << std::endl;
    std::cout << "  Throughput:        " << throughput_mqps << " Million queries/sec" << std::endl;
    
    // Write results to file for Phase 5 comparison
    std::ofstream out(run_dir + "/rmi_results.txt");
    if (out) {
        out << "memory_bytes: " << rmi_mem_bytes << "\n";
        out << "avg_latency_us: " << avg_latency_us << "\n";
        out << "throughput_mqps: " << throughput_mqps << "\n";
    }
    
    return 0;
}
