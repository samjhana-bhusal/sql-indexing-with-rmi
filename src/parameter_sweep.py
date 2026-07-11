import os
import sys
import subprocess
import time
import matplotlib.pyplot as plt
import numpy as np

def run_cmd(cmd):
    print(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if result.returncode != 0:
        print(f"Error executing command: {' '.join(cmd)}")
        print(result.stderr)
        return False
    return True

def parse_results_file(path):
    metrics = {}
    if os.path.exists(path):
        with open(path, 'r') as f:
            for line in f:
                parts = line.strip().split(': ')
                if len(parts) == 2:
                    metrics[parts[0]] = float(parts[1])
    return metrics

def main():
    print("==========================================================")
    # 1. Compile binaries
    print("Compiling C++ benchmarks...")
    if not run_cmd(["make"]):
        print("Compilation failed.")
        sys.exit(1)
        
    distributions = ["uniform", "lognormal", "clustered"]
    leaf_counts = [100, 500, 1000, 5000, 10000]
    
    # Store all data
    # { dist: { "btree": {mem, lat}, "rmi": [ {M, mem, lat, max_err, mean_err} ] } }
    sweep_data = {}
    
    for dist in distributions:
        print(f"\n==========================================================")
        print(f"   Starting Sweep for Distribution: {dist.upper()}        ")
        print("==========================================================")
        
        sweep_data[dist] = {"btree": None, "rmi": []}
        
        # A. Preprocess data for this distribution (10M keys)
        print(f"Generating data ({dist})...")
        if not run_cmd(["python3", "src/data_prep.py", "--distribution", dist, "--num-keys", "10000000"]):
            continue
            
        # B. Run B+Tree baseline
        print("Running B+Tree baseline...")
        if run_cmd(["./btree_benchmark"]):
            btree_metrics = parse_results_file("data/btree_results.txt")
            sweep_data[dist]["btree"] = btree_metrics
            print(f"  B+Tree Memory: {btree_metrics.get('memory_bytes', 0)/1024/1024:.2f} MB | Latency: {btree_metrics.get('avg_latency_us', 0):.4f} us")
            
        # C. Loop over Leaf count M for RMI
        for M in leaf_counts:
            print(f"\n--- Training RMI with M={M} ({dist}) ---")
            # Train RMI with 8 epochs for faster sweep speed
            if not run_cmd(["python3", "src/train_rmi.py", "--num-leaves", str(M), "--epochs", "8"]):
                continue
                
            print(f"Running CPU RMI benchmark for M={M}...")
            if run_cmd(["./rmi_benchmark"]):
                rmi_metrics = parse_results_file("data/rmi_results.txt")
                err_metrics = parse_results_file("data/rmi_error_stats.txt")
                
                combined = {
                    "M": M,
                    "memory_bytes": rmi_metrics.get("memory_bytes", 0),
                    "avg_latency_us": rmi_metrics.get("avg_latency_us", 0),
                    "throughput_mqps": rmi_metrics.get("throughput_mqps", 0),
                    "max_error": err_metrics.get("max_error", 0),
                    "mean_error": err_metrics.get("mean_error", 0)
                }
                sweep_data[dist]["rmi"].append(combined)
                print(f"  RMI M={M} Memory: {combined['memory_bytes']/1024:.2f} KB | Latency: {combined['avg_latency_us']:.4f} us | Max Error: {combined['max_error']}")

    # 2. Generate Pareto Frontier Plot
    print("\nGenerating Pareto Frontier Plot...")
    plt.figure(figsize=(10, 6))
    
    colors = {"uniform": "#2ca02c", "lognormal": "#1f77b4", "clustered": "#d62728"}
    markers = {"uniform": "^", "lognormal": "o", "clustered": "s"}
    
    # Plot RMI Curves
    for dist in distributions:
        rmi_results = sweep_data[dist]["rmi"]
        if not rmi_results:
            continue
            
        m_vals = [r["M"] for r in rmi_results]
        # Memory in KB
        mem_kb = [r["memory_bytes"] / 1024.0 for r in rmi_results]
        latencies = [r["avg_latency_us"] for r in rmi_results]
        
        plt.plot(mem_kb, latencies, label=f"RMI - {dist.capitalize()}", 
                 color=colors[dist], marker=markers[dist], linewidth=2)
        
        # Annotate M values
        for i, M in enumerate(m_vals):
            plt.annotate(f"M={M}", (mem_kb[i], latencies[i]), textcoords="offset points", 
                         xytext=(0,10), ha='center', fontsize=8, alpha=0.7)

    # Plot B+Tree reference points (usually flat across distributions)
    # B+Tree memory is around 150-200 MB, so we plot it at the right side of the graph
    for dist in distributions:
        btree = sweep_data[dist]["btree"]
        if btree:
            bt_mem_kb = btree["memory_bytes"] / 1024.0
            bt_lat = btree["avg_latency_us"]
            plt.scatter([bt_mem_kb], [bt_lat], color=colors[dist], marker="X", s=100, 
                        label=f"B+Tree - {dist.capitalize()}", edgecolors='black', zorder=5)

    plt.xscale('log')
    plt.xlabel('Index Parameter Footprint (KB, Log Scale)', fontsize=12)
    plt.ylabel('Average Lookup Latency (microseconds)', fontsize=12)
    plt.title('Pareto Frontier: Memory Footprint vs. Lookup Latency', fontsize=14)
    plt.grid(True, which="both", ls="--", alpha=0.5)
    plt.legend(loc="best", frameon=True, shadow=True)
    
    # Save the Pareto Frontier plot
    plot_path = "data/pareto_frontier.png"
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"Pareto Frontier plot saved to {plot_path}.")
    
    # 3. Generate Error Bounding Reduction Plot
    plt.figure(figsize=(10, 6))
    for dist in distributions:
        rmi_results = sweep_data[dist]["rmi"]
        if not rmi_results:
            continue
        m_vals = [r["M"] for r in rmi_results]
        max_errors = [r["max_error"] for r in rmi_results]
        plt.plot(m_vals, max_errors, label=f"{dist.capitalize()} (Max Error)", 
                 color=colors[dist], marker=markers[dist], linewidth=2)
                 
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel('Number of Leaf Models (M, Log Scale)', fontsize=12)
    plt.ylabel('Max Prediction Error (epsilon, Log Scale)', fontsize=12)
    plt.title('RMI Error Window Reduction vs. Model Count (M)', fontsize=14)
    plt.grid(True, which="both", ls="--", alpha=0.5)
    plt.legend()
    
    err_plot_path = "data/error_reduction.png"
    plt.savefig(err_plot_path, dpi=300, bbox_inches='tight')
    print(f"Error reduction plot saved to {err_plot_path}.")
    
    # 4. Generate Summary Report Table
    report_path = "data/parameter_sweep_report.md"
    print(f"Writing summary report to {report_path}...")
    with open(report_path, 'w') as f:
        f.write("# RMI Parameter Sweep & Sensitivity Report\n\n")
        f.write("This report displays performance and size metrics gathered during the parameter sweep over different distributions.\n\n")
        
        for dist in distributions:
            f.write(f"## Distribution: {dist.upper()}\n\n")
            
            bt = sweep_data[dist]["btree"]
            if bt:
                f.write(f"* **B+Tree Baseline**: Size = {bt['memory_bytes']/1024/1024:.2f} MB, Latency = {bt['avg_latency_us']:.4f} us, Throughput = {bt['throughput_mqps']:.4f} MQPS\n\n")
            
            f.write("| Leaf Count (M) | Index Memory (KB) | Avg Latency (us) | Throughput (MQPS) | Max Error (epsilon) |\n")
            f.write("| --- | --- | --- | --- | --- |\n")
            for r in sweep_data[dist]["rmi"]:
                f.write(f"| {r['M']} | {r['memory_bytes']/1024:.2f} KB | {r['avg_latency_us']:.4f} us | {r['throughput_mqps']:.4f} MQPS | {r['max_error']} |\n")
            f.write("\n")

    print("\nSweep Complete! Check data/pareto_frontier.png and data/parameter_sweep_report.md.")

if __name__ == "__main__":
    main()
