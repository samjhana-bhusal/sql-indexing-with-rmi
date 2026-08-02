# Hybrid Learned Index (RMI) vs. B+Tree

A performance comparison of a degree-optimized ($B=64$) C++ B+Tree against a two-stage hybrid **Recursive Model Index (RMI)** equipped with an LSM-style Delta Buffer, drift-triggered adaptive retraining, and benchmarked against **ALEX** and **LIPP** baselines.

This project demonstrates how database workloads can be shifted from memory-bound pointer-chasing to cache-friendly arithmetic pipeline calculations.

---

## Performance Summary

Evaluated on a 10-million row lognormal distribution dataset using Apple Silicon unified hardware. All throughput numbers are **5-trial mean ± stddev** with inter-trial cache flushing.

### 1. Index Size & Read Throughput

Synthetic lognormal, N=10M:

| Index | Memory | Throughput (MQPS) | Latency (μs) |
|-------|--------|-------------------|--------------|
| B+Tree (B=64) | 334.72 MB | 4.00 ± 0.72 | 0.259 |
| **RMI (M=5000)** | **117.98 KB** | 11.07 ± 1.21 | 0.091 |
| ALEX (Ding et al. 2020) | 218.31 MB | 14.98 ± 4.51 | 0.080 |
| LIPP (Wu et al. 2021) | 710.07 MB | **22.74 ± 5.80** | **0.049** |

**Result**: LIPP and ALEX win on raw throughput, but the RMI is **three orders of magnitude smaller** (118 KB vs 218–710 MB) — small enough to sit entirely in L2 cache. That is the RMI's actual selling point, not peak MQPS.

### 1b. Real-World Validation (SOSD)

Same pipeline on real key distributions — [SOSD](https://zenodo.org/records/15240501) `books` (11.42M unique keys) and `wiki_ts` (3.37M unique keys):

| Dataset | Index | Memory | MQPS | Latency (μs) |
|---------|-------|--------|------|--------------|
| books | B+Tree | 384.66 MB | 4.00 ± 0.65 | 0.259 |
| books | **RMI** | **117.98 KB** | 10.42 ± 1.23 | 0.097 |
| books | ALEX | 250.88 MB | **11.08 ± 1.78** | **0.093** |
| books | LIPP | — | n/a¹ | — |
| wiki_ts | B+Tree | 113.35 MB | 5.49 ± 0.39 | 0.183 |
| wiki_ts | **RMI** | **117.98 KB** | 14.08 ± 0.59 | 0.071 |
| wiki_ts | ALEX | 78.07 MB | 15.32 ± 0.84 | 0.065 |
| wiki_ts | LIPP | 331.79 MB | **18.46 ± 1.89** | **0.055** |

The synthetic findings hold on real data: the RMI stays within 6–9% of ALEX's throughput at a tiny fraction of the memory.

> ¹ **LIPP cannot bulk-load `books` on Apple Silicon.** LIPP computes node models in `long double`, which is 64-bit on AArch64 (53-bit mantissa) but 80-bit on x86-64. 83.1% of `books` keys exceed 2^53, and one adjacent key pair — distinct as `uint64` — collides when cast, producing a divide-by-zero and a non-finite slope. This is an upstream portability limitation, not an algorithmic result, so it is reported as unavailable rather than worked around. LIPP is unaffected on `wiki_ts` (keys are O(10⁹)).

> **Precision fix found via real data.** Stage 1 bucket assignment was originally done in `float32` at training time while the C++ runtime evaluates the exported weights in `float64`. On `books` this put 0.015% of keys (1,742 of 11.42M) in a different leaf than the one whose error bound was fitted for them, so bounded search missed. Fixed by doing bucket assignment in `float64`, plus a last-mile fallback that widens to a full search on a bound miss — so a boundary disagreement costs throughput, never correctness. Post-fix bound-miss rate: **0 / 500,000** on both datasets.

### 2. Dynamic LSM-Buffer Scaling
Real-time inserts run at a consistent **0.0230 μs/write**. As un-indexed keys accumulate in the delta store, read latency adjusts:

| Buffer Size (Keys) | Avg Read Latency (μs) |
|--------------------|-----------------------|
| 0 (Base RMI)       | 0.0589                |
| 100                | 0.0646                |
| 1,000              | 0.0784                |
| 5,000              | 0.0991                |
| 10,000             | 0.1050                |

### 3. The Cost of GPU Dispatch (Batch Size Sensitivity)
Sweeping batch sizes from 1 to 65,536 over 100K lookups (5-trial mean ± stddev, MPS):

| Batch | MQPS | Batch | MQPS |
|-------|------|-------|------|
| 1 | 0.008 ± 0.000 | 2048 | 8.28 ± 2.87 |
| 64 | 0.478 ± 0.036 | 4096 | 10.25 ± 2.79 |
| 128 | 0.963 ± 0.027 | **8192** | **11.77 ± 2.87** |
| 256 | 1.686 ± 0.203 | 16384 | 10.57 ± 3.07 |
| 512 | 3.308 ± 0.490 | 32768 | 8.48 ± 2.19 |
| 1024 | 5.380 ± 1.234 | 65536 | 6.01 ± 1.87 |

At batch=1 the GPU is ~1400× slower than CPU — every lookup pays a full kernel launch for ~100 FLOPs of work. Throughput climbs monotonically to **11.77 ± 2.87 MQPS at batch=8192**, which is *statistically tied* with CPU RMI (11.07 ± 1.21), then falls off.

The real finding is the **break-even batch size**: you need 8,192 concurrent lookups per dispatch just to *match* one CPU core — a concurrency level point-lookup workloads rarely reach, and which costs latency while the batch fills.

> ⚠️ **This result was previously reported incorrectly.** The earlier version synchronized only on CUDA (`if device.type == 'cuda': torch.cuda.synchronize()`), so on MPS the timer measured *queue-submission* time, not execution time — understating GPU throughput ~4× at large batches and producing a bogus "GPU always loses by 4.1×" conclusion. Fixed with an explicit `torch.mps.synchronize()` inside the timed region.

### 4. Adaptive Retraining (Drift-Triggered)

The delta buffer defers insert cost but the static RMI's accuracy degrades as
key distributions shift. Our **drift-triggered localized retraining** monitors
per-segment prediction error and a two-sample KS test (ported from
[proj2/Drift-Lab](../proj2)), selectively refitting only degraded Stage 2 leaf
models.

| Config | MQPS | Segments Retrained | Retrain Cost (s) |
|---|---|---|---|
| A — Never retrain | 8.04 ± 0.26 | 26 | 0.025 ± 0.003 |
| B — Periodic retrain | 4.00 ± 0.15 | 4501 | 3.557 ± 0.047 |
| C — **Adaptive retrain** | 4.44 ± 0.34 | 583 | **0.211 ± 0.010** |

Adaptive retraining retrained 13% of the segments periodic did, at **5.9% of
the wall-clock cost**, while delivering 11% higher read throughput.

![Adaptive vs Periodic](report/adaptive_vs_periodic.png)

## Visualizations

### Pareto Frontier (Memory vs. Latency)
![Pareto Frontier](report/pareto_frontier.png)

### GPU Hardware Throughput Scaling Curve
![GPU Throughput Scaling](report/gpu_throughput.png)

## Quick Start

### Run the Full Pipeline
Generates data (synthetic + SOSD), compiles all benchmarks (B+Tree, RMI, ALEX, LIPP), trains models, and runs all evaluations:
```bash
./run_pipeline.sh
```

### Run Individual Benchmarks
```bash
make                              # compile all
./btree_benchmark data/run_dir    # B+Tree with 5-trial stats
./rmi_benchmark data/run_dir      # RMI with 5-trial stats
./alex_benchmark data/run_dir     # ALEX baseline
./lipp_benchmark data/run_dir     # LIPP baseline
./adaptive_rmi_benchmark data/run_dir  # Drift-triggered adaptive RMI
```

All benchmarks accept `--trials N` to control trial count (default 5).

## Repository Layout

```
proj1/
├── readme.md
├── UPGRADE_TODO.md       # Sprint plan + definition of done
├── DESIGN.md             # Adaptive retraining design document
├── CITATION.cff          # GitHub citation metadata
├── run_pipeline.sh       # One-step end-to-end pipeline
├── Makefile              # Builds all C++ benchmarks
├── report/
│   ├── main.tex          # Full LaTeX manuscript
│   ├── references.bib
│   └── *.png             # Figures (Pareto, GPU, adaptive)
├── src/
│   ├── btree.hpp                    # Degree-64 B+Tree (header-only)
│   ├── btree_benchmark.cpp          # B+Tree benchmark (5-trial)
│   ├── rmi_benchmark.cpp            # Static RMI benchmark (5-trial)
│   ├── buffered_rmi_benchmark.cpp   # LSM delta-buffer benchmark
│   ├── adaptive_rmi_benchmark.cpp   # Drift-triggered adaptive RMI
│   ├── drift_monitor.hpp            # KS-test + error monitor (C++ port from proj2)
│   ├── alex_benchmark.cpp           # ALEX baseline
│   ├── lipp_benchmark.cpp           # LIPP baseline
│   ├── train_rmi.py                 # PyTorch Stage 1 NN + OLS Stage 2
│   ├── data_prep.py                 # Synthetic + SOSD data generation
│   ├── gpu_rmi_benchmark.py         # GPU batch-size sensitivity sweep
│   ├── adaptive_evaluation.py       # 3-config A/B/C eval driver
│   └── parameter_sweep.py           # M-sweep analysis
└── external/
    ├── alex/              # Microsoft ALEX (header-only, ARM-patched)
    └── lipp/              # LIPP (header-only)
```

## Cross-Project Connection

The drift-triggered retraining mechanism in `src/drift_monitor.hpp` is a **C++ port** of the KS-test routine from our companion project [proj2/Drift-Lab](../proj2) (`proj2/src/drift_lab/drift/detectors.py`). This shared machinery is what makes the two projects tell one coherent research story: *confidence- and cost-aware ML for data systems — retrain the learned model only where it degrades, not on a fixed schedule.*
