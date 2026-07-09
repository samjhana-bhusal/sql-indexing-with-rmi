#include <iostream>
#include <vector>
#include <random>
#include <algorithm>
#include <cassert>
#include "btree.hpp"

// Test sequential insertions (bulk loading pattern)
void test_sequential_insert() {
    std::cout << "Running test_sequential_insert..." << std::endl;
    BPlusTree tree;
    const size_t num_elements = 10000;
    
    // Insert sequential keys
    for (size_t i = 0; i < num_elements; ++i) {
        tree.insert(i * 10, i);
    }
    
    assert(tree.size() == num_elements);
    
    // Verify lookups
    for (size_t i = 0; i < num_elements; ++i) {
        uint64_t val = 0;
        bool found = tree.lookup(i * 10, val);
        assert(found);
        assert(val == i);
    }
    std::cout << "  [PASS] Sequential insertion and lookup verified." << std::endl;
}

// Test random insertions (verifies split logic for leaves and internal nodes)
void test_random_insert() {
    std::cout << "Running test_random_insert..." << std::endl;
    BPlusTree tree;
    const size_t num_elements = 5000;
    
    std::vector<uint64_t> keys;
    for (size_t i = 0; i < num_elements; ++i) {
        keys.push_back(i);
    }
    
    // Shuffle keys randomly
    std::mt19937 g(123);
    std::shuffle(keys.begin(), keys.end(), g);
    
    // Insert in random order
    for (size_t i = 0; i < num_elements; ++i) {
        tree.insert(keys[i], keys[i] * 2);
    }
    
    assert(tree.size() == num_elements);
    
    // Verify all keys are found and hold correct values
    for (size_t i = 0; i < num_elements; ++i) {
        uint64_t val = 0;
        bool found = tree.lookup(keys[i], val);
        assert(found);
        assert(val == keys[i] * 2);
    }
    std::cout << "  [PASS] Random insertion and splitting verified." << std::endl;
}

// Test looking up keys that do not exist
void test_missing_keys() {
    std::cout << "Running test_missing_keys..." << std::endl;
    BPlusTree tree;
    
    // Insert some keys
    tree.insert(10, 100);
    tree.insert(20, 200);
    tree.insert(30, 300);
    
    uint64_t val = 0;
    
    // Look up missing keys
    assert(!tree.lookup(5, val));
    assert(!tree.lookup(15, val));
    assert(!tree.lookup(25, val));
    assert(!tree.lookup(35, val));
    
    // Look up present keys
    assert(tree.lookup(10, val) && val == 100);
    assert(tree.lookup(20, val) && val == 200);
    assert(tree.lookup(30, val) && val == 300);
    
    std::cout << "  [PASS] Missing key failure and present key success verified." << std::endl;
}

// Test large scale insertions to trigger deeper tree splits
void test_large_scale() {
    std::cout << "Running test_large_scale..." << std::endl;
    BPlusTree tree;
    const size_t num_elements = 100000;
    
    for (size_t i = 0; i < num_elements; ++i) {
        tree.insert(i, i);
    }
    
    assert(tree.size() == num_elements);
    
    // Random sample lookup for verification
    std::mt19937 g(42);
    std::uniform_int_distribution<size_t> dis(0, num_elements - 1);
    
    for (int i = 0; i < 10000; ++i) {
        size_t target = dis(g);
        uint64_t val = 0;
        bool found = tree.lookup(target, val);
        assert(found);
        assert(val == target);
    }
    std::cout << "  [PASS] Large-scale (100k elements) splits and queries verified." << std::endl;
}

int main() {
    std::cout << "========================================" << std::endl;
    std::cout << "      B+TREE UNIT TEST SUITE            " << std::endl;
    std::cout << "========================================" << std::endl;
    
    test_sequential_insert();
    test_random_insert();
    test_missing_keys();
    test_large_scale();
    
    std::cout << "========================================" << std::endl;
    std::cout << "    ALL UNIT TESTS PASSED SUCCESSFULLY! " << std::endl;
    std::cout << "========================================" << std::endl;
    return 0;
}
