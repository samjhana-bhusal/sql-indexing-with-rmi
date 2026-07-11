#include <iostream>
#include <fstream>
#include <vector>
#include <chrono>
#include <random>
#include <cassert>
#include "btree.hpp"

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

int main(int argc, char* argv[]) {
    std::string run_dir = "data";
    if (argc > 1) {
        run_dir = argv[1];
    }
    
    std::string keys_path = run_dir + "/keys.bin";
    std::string pos_path = run_dir + "/positions.bin";
    
    std::vector<uint64_t> keys;
    std::vector<uint64_t> positions;
    
    std::cout << "Loading dataset binaries from " << run_dir << "..." << std::endl;
    if (!load_binary_file(keys_path, keys) || !load_binary_file(pos_path, positions)) {
        std::cerr << "Error: Could not load data files from: " << run_dir << ". Please run data_prep.py first." << std::endl;
        return 1;
    }
    
    size_t N = keys.size();
    std::cout << "Successfully loaded " << N << " elements." << std::endl;
    assert(N == positions.size());
    
    std::cout << "Building B+Tree..." << std::endl;
    auto build_start = std::chrono::high_resolution_clock::now();
    BPlusTree tree;
    for (size_t i = 0; i < N; ++i) {
        tree.insert(keys[i], positions[i]);
    }
    auto build_end = std::chrono::high_resolution_clock::now();
    double build_time = std::chrono::duration<double>(build_end - build_start).count();
    
    std::cout << "B+Tree built in " << build_time << " seconds." << std::endl;
    
    size_t mem_bytes = tree.get_memory_size();
    double mem_mb = static_cast<double>(mem_bytes) / (1024.0 * 1024.0);
    std::cout << "Memory footprint: " << mem_bytes << " bytes (" << mem_mb << " MB)" << std::endl;
    
    // Select 100,000 random lookup queries
    size_t num_queries = 100000;
    std::vector<size_t> query_indices(num_queries);
    std::mt19937 gen(42); // Seed for deterministic evaluation
    std::uniform_int_distribution<size_t> dis(0, N - 1);
    for (size_t i = 0; i < num_queries; ++i) {
        query_indices[i] = dis(gen);
    }
    
    std::cout << "Running " << num_queries << " random lookups..." << std::endl;
    auto lookup_start = std::chrono::high_resolution_clock::now();
    for (size_t i = 0; i < num_queries; ++i) {
        size_t idx = query_indices[i];
        uint64_t target_key = keys[idx];
        uint64_t target_val = 0;
        
        bool found = tree.lookup(target_key, target_val);
        
        // Assert correctness
        assert(found && "Key must be found in the tree");
        assert(target_val == positions[idx] && "Returned value must match index position");
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
    std::ofstream out(run_dir + "/btree_results.txt");
    if (out) {
        out << "memory_bytes: " << mem_bytes << "\n";
        out << "avg_latency_us: " << avg_latency_us << "\n";
        out << "throughput_mqps: " << throughput_mqps << "\n";
    }
    
    return 0;
}
