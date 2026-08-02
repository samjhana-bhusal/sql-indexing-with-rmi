#!/bin/bash
set -e

# Auto-generate a unique run ID using current timestamp
RUN_ID="run_$(date +%Y%m%d_%H%M%S)"
RUN_DIR="data/$RUN_ID"

echo "=========================================================="
echo "    Learned Index Pipeline: B+Tree vs. RMI               "
echo "    Run ID : $RUN_ID                                      "
echo "    Run Dir: $RUN_DIR                                     "
echo "=========================================================="

# 1. Environment Checks
echo -e "\n--> Checking dependencies..."
if ! command -v python3 &> /dev/null; then
    echo "Error: python3 is not installed."
    exit 1
fi

# 2. Phase 1: Data Acquisition & Preprocessing
# Choose your active evaluation distribution target here: [uniform | lognormal | clustered]
DISTRIBUTION="lognormal" 
OPTIMIZED_M=5000 # Lock in your optimized sweep parameters here

echo -e "\n--> Phase 1: Data Acquisition & Preprocessing ({$DISTRIBUTION})..."
python3 src/data_prep.py \
    --distribution "$DISTRIBUTION" \
    --num-keys 10000000 \
    --out-dir "$RUN_DIR"

# 3. Phase 2: Compile C++ Benchmarks
echo -e "\n--> Phase 2: Compiling C++ Benchmarks..."
make

# 4. Phase 3: Train Hybrid RMI (Uses your new adaptive training loop)
echo -e "\n--> Phase 3: Training Hybrid Learned Index (RMI)..."
python3 src/train_rmi.py \
    --keys "$RUN_DIR/keys.bin" \
    --positions "$RUN_DIR/positions.bin" \
    --num-leaves $OPTIMIZED_M \
    --out-dir "$RUN_DIR"

# 5. Run CPU Benchmarks 
echo -e "\n--> Running CPU B+Tree Benchmark..."
./btree_benchmark "$RUN_DIR"

echo -e "\n--> Running CPU RMI Benchmark..."
./rmi_benchmark "$RUN_DIR"

echo -e "\n--> Running Dynamic Read-Write (Delta-Buffered) RMI Benchmark..."
./buffered_rmi_benchmark "$RUN_DIR"

echo -e "\n--> Running Adaptive RMI Benchmark (Drift-Triggered Retrain)..."
./adaptive_rmi_benchmark "$RUN_DIR"

echo -e "\n--> Running ALEX Baseline Benchmark..."
./alex_benchmark "$RUN_DIR"

echo -e "\n--> Running LIPP Baseline Benchmark..."
./lipp_benchmark "$RUN_DIR"

# 6. Phase 4: GPU Benchmarks
echo -e "\n--> Phase 4: Running GPU Batch Benchmark..."
python3 src/gpu_rmi_benchmark.py \
    --keys    "$RUN_DIR/keys.bin" \
    --positions "$RUN_DIR/positions.bin" \
    --params  "$RUN_DIR/rmi_params.bin" \
    --out-dir "$RUN_DIR"

# 7. Summary Table Processing
echo -e "\n=========================================================="
echo "    Benchmark Complete!  Run: $RUN_ID                    "
echo "=========================================================="

# 8. SOSD Real-Data Benchmarks (books + wiki_ts)
SOSD_DATASETS="books wiki_ts"
for DATASET in $SOSD_DATASETS; do
    SOSD_DIR="data/sosd_${DATASET}"
    echo -e "\n--> SOSD: Preparing ${DATASET} dataset..."
    python3 src/data_prep.py \
        --sosd-dataset "$DATASET" \
        --num-keys 10000000 \
        --out-dir "$SOSD_DIR"

    echo "--> SOSD: Training RMI on ${DATASET}..."
    python3 src/train_rmi.py \
        --keys "$SOSD_DIR/keys.bin" \
        --positions "$SOSD_DIR/positions.bin" \
        --num-leaves $OPTIMIZED_M \
        --out-dir "$SOSD_DIR"

    echo "--> SOSD: Running benchmarks on ${DATASET}..."
    # Do not abort the whole pipeline if one index cannot handle a dataset.
    # Known: LIPP fails to bulk-load `books` on AArch64 because its long double
    # is 64-bit and 83% of books keys exceed 2^53 (see report Section 6).
    ./btree_benchmark "$SOSD_DIR" || echo "  [WARN] btree_benchmark failed on ${DATASET}"
    ./rmi_benchmark   "$SOSD_DIR" || echo "  [WARN] rmi_benchmark failed on ${DATASET}"
    ./alex_benchmark  "$SOSD_DIR" || echo "  [WARN] alex_benchmark failed on ${DATASET}"
    ./lipp_benchmark  "$SOSD_DIR" || echo "  [WARN] lipp_benchmark failed on ${DATASET}"

    echo "  --> SOSD ${DATASET} done. Results in ${SOSD_DIR}/"
done

# 9. Summary Table Processing
BT_FILE="$RUN_DIR/btree_results.txt"
RMI_FILE="$RUN_DIR/rmi_results.txt"

if [ -f "$BT_FILE" ] && [ -f "$RMI_FILE" ]; then
    BT_MEM=$(grep  "memory_bytes"   "$BT_FILE"  | awk '{print $2}')
    BT_LAT=$(grep  "avg_latency_us" "$BT_FILE"  | awk '{print $2}')
    BT_THR=$(grep  "throughput_mqps" "$BT_FILE" | awk '{print $2}')
    BT_STD=$(grep  "stddev_mqps"    "$BT_FILE"  | awk '{print $2}')

    RMI_MEM=$(grep  "memory_bytes"   "$RMI_FILE" | awk '{print $2}')
    RMI_LAT=$(grep  "avg_latency_us" "$RMI_FILE" | awk '{print $2}')
    RMI_THR=$(grep  "throughput_mqps" "$RMI_FILE" | awk '{print $2}')
    RMI_STD=$(grep  "stddev_mqps"    "$RMI_FILE"  | awk '{print $2}')

    BT_MEM_MB=$(awk "BEGIN{printf \"%.2f\", $BT_MEM/1048576}")
    RMI_MEM_KB=$(awk "BEGIN{printf \"%.2f\", $RMI_MEM/1024}")

    printf "\n%-28s %-24s %-24s\n" "Metric" "B+Tree (CPU)" "RMI (CPU)"
    printf "%-28s %-24s %-24s\n" "--------" "----------" "---------"
    printf "%-28s %-24s %-24s\n" "Index Memory" "${BT_MEM_MB} MB" "${RMI_MEM_KB} KB"
    printf "%-28s %-24s %-24s\n" "Avg Latency" "${BT_LAT} us" "${RMI_LAT} us"
    printf "%-28s %-24s %-24s\n" "Throughput" "${BT_THR} ± ${BT_STD} MQPS" "${RMI_THR} ± ${RMI_STD} MQPS"
fi

# 10. Full Comparison Table (B+Tree vs RMI vs ALEX vs LIPP)
echo -e "\n==> Full Comparison Table (synthetic + SOSD):"
echo -e "\n--- Synthetic ($DISTRIBUTION) ---"
python3 src/generate_comparison_table.py "$RUN_DIR"

for DATASET in $SOSD_DATASETS; do
    SOSD_DIR="data/sosd_${DATASET}"
    if [ -d "$SOSD_DIR" ]; then
        echo -e "\n--- SOSD: ${DATASET} ---"
        python3 src/generate_comparison_table.py "$SOSD_DIR"
    fi
done