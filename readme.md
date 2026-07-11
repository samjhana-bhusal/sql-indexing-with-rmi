# Hybrid Learned Index (RMI) vs. B+Tree

A performance comparison of a degree-optimized ($B=64$) C++ B+Tree against a two-stage hybrid **Recursive Model Index (RMI)** equipped with an LSM-style Delta Buffer. 

This project demonstrates how database workloads can be shifted from memory-bound pointer-chasing to cache-friendly arithmetic pipeline calculations.

---

## Performance Summary

Evaluated on a 10-million row lognormal distribution dataset using Apple Silicon unified hardware (`run_20260711_105955`).

### 1. Index Size & Read Throughput
* **B+Tree Baseline**: **334.71 MB** footprint | 4.74 MQPS (0.210 $\mu$s/query)
* **Hybrid RMI ($M=5000$)**: **117.97 KB** footprint | **14.91 MQPS** (0.067 $\mu$s/query)
* **Result**: RMI achieves a **99.96% memory reduction** and scales **3.14$\times$ faster** by fitting entirely within L2 cache lines.

### 2. Dynamic LSM-Buffer Scaling
Real-time inserts run at a consistent **0.0230 $\mu$s/write**. As un-indexed keys accumulate in the delta store, read latency adjusts across the tracking blocks:

| Buffer Size (Keys) | Avg Read Latency ($\mu$s) |
|--------------------|--------------------------|
| 0 (Base RMI)       | 0.0589                   |
| 100                | 0.0646                   |
| 1,000              | 0.0784                   |
| 5,000              | 0.0991                   |
| 10,000             | 0.1050                   |

### 3. GPU Parallel Batch Sweep (MPS)
Throughput profiles evaluating 100,000 randomized lookups show the parallel hardware sweet spot at a batch size of 512:

| Batch Size | Execution Time (s) | Throughput (MQPS) |
|------------|--------------------|-------------------|
| 1          | 12.0505            | 0.0083            |
| 128        | 0.1276             | 0.7837            |
| 512        | 0.0367             | **2.7215**        |
| 1024       | 0.0483             | 2.0698            |
| 4096       | 0.0823             | 1.2145            |
| 8192       | 0.0539             | 1.8562            |

---
---

## Quick Start

### Run the Pipeline
To generate data, compile code, train models, and output all benchmarking metrics in one step:
```bash
./run_pipeline.sh
```