CXX = clang++
CXXFLAGS = -O3 -march=native -std=c++17 -Wall

all: btree_benchmark rmi_benchmark buffered_rmi_benchmark adaptive_rmi_benchmark alex_benchmark lipp_benchmark btree_test

btree_benchmark: src/btree_benchmark.cpp src/btree.hpp
	$(CXX) $(CXXFLAGS) src/btree_benchmark.cpp -o btree_benchmark

rmi_benchmark: src/rmi_benchmark.cpp
	$(CXX) $(CXXFLAGS) src/rmi_benchmark.cpp -o rmi_benchmark

buffered_rmi_benchmark: src/buffered_rmi_benchmark.cpp
	$(CXX) $(CXXFLAGS) src/buffered_rmi_benchmark.cpp -o buffered_rmi_benchmark

adaptive_rmi_benchmark: src/adaptive_rmi_benchmark.cpp src/drift_monitor.hpp
	$(CXX) $(CXXFLAGS) src/adaptive_rmi_benchmark.cpp -o adaptive_rmi_benchmark

alex_benchmark: src/alex_benchmark.cpp
	$(CXX) -O3 -std=c++17 -Wall -I./external/alex/src/core src/alex_benchmark.cpp -o alex_benchmark

lipp_benchmark: src/lipp_benchmark.cpp
	$(CXX) -O3 -std=c++17 -Wall -I./external/lipp/src/core src/lipp_benchmark.cpp -o lipp_benchmark

btree_test: src/btree_test.cpp src/btree.hpp
	$(CXX) $(CXXFLAGS) src/btree_test.cpp -o btree_test

test: btree_test
	./btree_test

clean:
	rm -f btree_benchmark rmi_benchmark buffered_rmi_benchmark adaptive_rmi_benchmark alex_benchmark lipp_benchmark btree_test
