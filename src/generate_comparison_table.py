"""
Reads benchmark result files from a pipeline run directory and prints
a unified comparison table (LaTeX-ready) for the paper.

Usage:
    python3 src/generate_comparison_table.py data/run_YYYYMMDD_HHMMSS
"""
import os
import sys

def parse_results(path):
    results = {}
    if not os.path.exists(path):
        return None
    with open(path) as f:
        for line in f:
            if ':' in line:
                key, val = line.strip().split(':', 1)
                results[key.strip()] = val.strip()
    return results

def fmt_mem(mem_bytes_str):
    b = float(mem_bytes_str)
    if b > 1024 * 1024:
        return f"{b / (1024*1024):.2f} MB"
    return f"{b / 1024:.2f} KB"

def main():
    if len(sys.argv) < 2:
        print("Usage: python3 src/generate_comparison_table.py <run_dir>")
        sys.exit(1)

    run_dir = sys.argv[1]

    benchmarks = {
        "B+Tree ($B{=}64$)": "btree_results.txt",
        "RMI ($M{=}5000$)": "rmi_results.txt",
        "ALEX": "alex_results.txt",
        "LIPP": "lipp_results.txt",
    }

    print(f"\n{'Index':<22} {'Memory':<14} {'MQPS':>12}  {'Latency (us)':>14}")
    print("-" * 66)

    latex_rows = []
    for name, fname in benchmarks.items():
        r = parse_results(os.path.join(run_dir, fname))
        if r is None:
            print(f"{name:<22} {'(not run)':<14} {'--':>12}  {'--':>14}")
            continue

        mem = fmt_mem(r.get("memory_bytes", "0")) if "memory_bytes" in r else "--"
        mqps = r.get("throughput_mqps", "--")
        std = r.get("stddev_mqps", "")
        lat = r.get("avg_latency_us", "--")
        trials = r.get("trials", "1")

        mqps_str = f"{float(mqps):.2f}" if mqps != "--" else "--"
        if std:
            mqps_str += f" ± {float(std):.2f}"
        lat_str = f"{float(lat):.3f}" if lat != "--" else "--"

        print(f"{name:<22} {mem:<14} {mqps_str:>12}  {lat_str:>14}")

        latex_rows.append(
            f"{name} & {mem} & ${mqps_str}$ & ${lat_str}$ \\\\"
        )

    print("\n--- LaTeX table rows ---")
    for row in latex_rows:
        print(row)

if __name__ == "__main__":
    main()
