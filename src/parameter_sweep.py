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
    run_id = f"sweep_{time.strftime('%Y%m%d_%H%M%S')}"
    base_run_dir = os.path.join("data", run_id)
    os.makedirs(base_run_dir, exist_ok=True)
    print(f"Global sweep session root directory: {base_run_dir}")
    print("==========================================================")

    print("Compiling C++ benchmarks...")
    if not run_cmd(["make"]):
        print("Compilation failed.")
        sys.exit(1)
        
    distributions = ["uniform", "lognormal", "clustered"]
    leaf_counts = [100, 500, 1000, 5000, 10000]
    
    sweep_data = {}
    
    for dist in distributions:
        print(f"\n==========================================================")
        print(f"   Starting Sweep for Distribution: {dist.upper()}        ")
        print("==========================================================")
        
        sweep_data[dist] = {"btree": None, "rmi": []}
        
        # Isolate by distribution inside our parent run session
        dist_dir = os.path.join(base_run_dir, dist)
        os.makedirs(dist_dir, exist_ok=True)
        
        # A. Preprocess data uniquely for this distribution branch
        # data_prep writes keys.bin/positions.bin directly into dist_dir
        if not run_cmd(["python3", "src/data_prep.py",
                        "--distribution", dist,
                        "--num-keys", "10000000",
                        "--out-dir", base_run_dir,
                        "--run-id", dist]):
            continue
            
        # B. Run B+Tree baseline (Pass isolated dist directory to C++)
        print("Running B+Tree baseline...")
        if run_cmd(["./btree_benchmark", dist_dir]):
            btree_metrics = parse_results_file(os.path.join(dist_dir, "btree_results.txt"))
            sweep_data[dist]["btree"] = btree_metrics
            if btree_metrics:
                print(f"  B+Tree Memory: {btree_metrics.get('memory_bytes', 0)/1024/1024:.2f} MB | Latency: {btree_metrics.get('avg_latency_us', 0):.4f} us")
            
        # C. Loop over Leaf count M for RMI
        for M in leaf_counts:
            print(f"\n--- Training RMI with M={M} ({dist}) ---")
            
            # Isolate model checkpoint configs to prevent overwrites across configuration sweeps
            config_dir = os.path.join(dist_dir, f"M_{M}")
            os.makedirs(config_dir, exist_ok=True)
            
            # Point training script to the structural keys generated under the active dist_dir
            if not run_cmd(["python3", "src/train_rmi.py", 
                            "--keys", os.path.join(dist_dir, "keys.bin"), 
                            "--positions", os.path.join(dist_dir, "positions.bin"), 
                            "--num-leaves", str(M), 
                            "--out-dir", config_dir]):
                continue
                
            print(f"Running CPU RMI benchmark for M={M}...")
            # rmi_benchmark argv[1]=data dir (keys/positions), argv[2]=config dir (params + results)
            if run_cmd(["./rmi_benchmark", dist_dir, config_dir]):
                rmi_metrics = parse_results_file(os.path.join(config_dir, "rmi_results.txt"))
                err_metrics = parse_results_file(os.path.join(config_dir, "rmi_error_stats.txt"))
                
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
    
    for dist in distributions:
        rmi_results = sweep_data[dist]["rmi"]
        if not rmi_results:
            continue
        m_vals = [r["M"] for r in rmi_results]
        mem_kb = [r["memory_bytes"] / 1024.0 for r in rmi_results]
        latencies = [r["avg_latency_us"] for r in rmi_results]
        
        plt.plot(mem_kb, latencies, label=f"RMI - {dist.capitalize()}", color=colors[dist], marker=markers[dist], linewidth=2)
        for i, M in enumerate(m_vals):
            plt.annotate(f"M={M}", (mem_kb[i], latencies[i]), textcoords="offset points", xytext=(0,10), ha='center', fontsize=8, alpha=0.7)

    for dist in distributions:
        btree = sweep_data[dist]["btree"]
        if btree:
            bt_mem_kb = btree["memory_bytes"] / 1024.0
            bt_lat = btree["avg_latency_us"]
            plt.scatter([bt_mem_kb], [bt_lat], color=colors[dist], marker="X", s=100, label=f"B+Tree - {dist.capitalize()}", edgecolors='black', zorder=5)

    plt.xscale('log')
    plt.xlabel('Index Parameter Footprint (KB, Log Scale)', fontsize=12)
    plt.ylabel('Average Lookup Latency (microseconds)', fontsize=12)
    plt.title('Pareto Frontier: Memory Footprint vs. Lookup Latency', fontsize=14)
    plt.grid(True, which="both", ls="--", alpha=0.5)
    plt.legend(loc="best", frameon=True, shadow=True)
    
    plot_path = os.path.join(base_run_dir, "pareto_frontier.png")
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    # 3. Generate Summary Report Table
    report_path = os.path.join(base_run_dir, "parameter_sweep_report.md")
    with open(report_path, 'w') as f:
        f.write(f"# RMI Sweep Report ({run_id})\n\n")
        for dist in distributions:
            f.write(f"## Distribution: {dist.upper()}\n\n")
            bt = sweep_data[dist]["btree"]
            if bt:
                f.write(f"* **B+Tree**: Size = {bt['memory_bytes']/1024/1024:.2f} MB, Latency = {bt['avg_latency_us']:.4f} us\n\n")
            f.write("| Leaf Count (M) | Index Memory (KB) | Avg Latency (us) | Throughput (MQPS) | Max Error |\n")
            f.write("| --- | --- | --- | --- | --- |\n")
            for r in sweep_data[dist]["rmi"]:
                f.write(f"| {r['M']} | {r['memory_bytes']/1024:.2f} KB | {r['avg_latency_us']:.4f} us | {r['throughput_mqps']:.4f} MQPS | {r['max_error']} |\n")
            f.write("\n")

    print(f"\nSweep Complete! Visualizations and tables generated inside {base_run_dir}/")

if __name__ == "__main__":
    main()