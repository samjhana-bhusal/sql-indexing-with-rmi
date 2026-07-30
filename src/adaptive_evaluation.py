"""
S1.4 — Adaptive vs. Periodic vs. Never-Retrain Evaluation Driver.

Runs three RMI configurations on the same insert-heavy drifting workload:
  A) Never retrain   (existing buffered_rmi_benchmark)
  B) Periodic retrain (adaptive_rmi_benchmark --alpha 0 → always retrain)
  C) Adaptive retrain (adaptive_rmi_benchmark --alpha 1.5)

Reports 5-run mean ± stddev for read throughput (MQPS) and retrain cost.
Produces report/adaptive_vs_periodic.png.
"""

import argparse
import os
import re
import subprocess
import sys
from pathlib import Path

import numpy as np

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    HAS_MPL = True
except ImportError:
    HAS_MPL = False


ROOT = Path(__file__).resolve().parent.parent

# Prefer the project venv (has torch) over the system python.
VENV_PYTHON = ROOT / "venv" / "bin" / "python"
PYTHON = str(VENV_PYTHON) if VENV_PYTHON.exists() else sys.executable


def run_pipeline_phase(run_dir: str, distribution: str = "drifting",
                       num_keys: int = 1_000_000, num_leaves: int = 1000):
    """Generate data + train RMI for a single run."""
    os.makedirs(run_dir, exist_ok=True)

    subprocess.run([
        PYTHON, str(ROOT / "src" / "data_prep.py"),
        "--distribution", distribution,
        "--num-keys", str(num_keys),
        "--out-dir", run_dir,
    ], check=True, capture_output=True)

    subprocess.run([
        PYTHON, str(ROOT / "src" / "train_rmi.py"),
        "--keys", f"{run_dir}/keys.bin",
        "--positions", f"{run_dir}/positions.bin",
        "--num-leaves", str(num_leaves),
        "--out-dir", run_dir,
    ], check=True, capture_output=True)


def parse_adaptive_output(stdout: str):
    """Extract per-window MQPS and summary stats from adaptive_rmi_benchmark."""
    mqps_values = []
    for line in stdout.splitlines():
        # Match lines like "  1    | 10000   | 10000  | 9.43879   | ..."
        m = re.match(r'\s+\d+\s+\|\s+\d+\s+\|\s+\d+\s+\|\s+([\d.]+)', line)
        if m:
            mqps_values.append(float(m.group(1)))

    total_retrains = 0
    total_segs = 0
    retrain_time = 0.0
    for line in stdout.splitlines():
        if "Total retrain events:" in line:
            total_retrains = int(line.split(":")[-1].strip())
        elif "Total segments retrained:" in line:
            total_segs = int(line.split(":")[-1].strip())
        elif "Retrain wall-clock (s):" in line:
            retrain_time = float(line.split(":")[-1].strip())

    return {
        "mqps_per_window": mqps_values,
        "mean_mqps": np.mean(mqps_values) if mqps_values else 0.0,
        "total_retrains": total_retrains,
        "total_segments_retrained": total_segs,
        "retrain_wall_clock_s": retrain_time,
    }


def run_config(config_name: str, run_dir: str, alpha: float,
               num_inserts: int = 100000, interval: int = 10000):
    """Run adaptive_rmi_benchmark with given alpha."""
    binary = str(ROOT / "adaptive_rmi_benchmark")
    cmd = [binary, run_dir,
           "--inserts", str(num_inserts),
           "--interval", str(interval),
           "--alpha", str(alpha)]

    result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
    if result.returncode != 0:
        print(f"  [{config_name}] FAILED: {result.stderr[:200]}")
        return None

    return parse_adaptive_output(result.stdout)


def main():
    parser = argparse.ArgumentParser(description="Adaptive RMI Evaluation (A/B/C)")
    parser.add_argument("--trials", type=int, default=5)
    parser.add_argument("--num-keys", type=int, default=1_000_000)
    parser.add_argument("--num-inserts", type=int, default=100000)
    parser.add_argument("--num-leaves", type=int, default=500)
    parser.add_argument("--distribution", type=str, default="drifting")
    parser.add_argument("--out-dir", type=str, default="data/eval_adaptive")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    configs = {
        "A_never_retrain":   {"alpha": 1e12},  # effectively never triggers
        "B_periodic_retrain": {"alpha": 0.0},   # always triggers → periodic
        "C_adaptive_retrain": {"alpha": 1.5},   # drift-triggered
    }

    all_results = {name: [] for name in configs}

    for trial in range(args.trials):
        print(f"\n=== Trial {trial + 1}/{args.trials} ===")
        run_dir = os.path.join(args.out_dir, f"trial_{trial}")

        # Generate fresh data + train RMI
        run_pipeline_phase(run_dir, args.distribution, args.num_keys, args.num_leaves)

        for name, cfg in configs.items():
            print(f"  Running {name} (alpha={cfg['alpha']}) ...")
            result = run_config(name, run_dir, cfg["alpha"],
                                num_inserts=args.num_inserts)
            if result:
                all_results[name].append(result)

    # Aggregate
    print("\n" + "=" * 70)
    print("  EVALUATION SUMMARY (mean ± stddev over {} trials)".format(args.trials))
    print("=" * 70)
    print(f"{'Config':<25} {'MQPS':>12} {'Retrains':>12} {'Segs Retrained':>16} {'Retrain Time (s)':>18}")
    print("-" * 85)

    summary = {}
    for name in configs:
        results = all_results[name]
        if not results:
            print(f"{name:<25} {'FAILED':>12}")
            continue

        mqps_arr = np.array([r["mean_mqps"] for r in results])
        retrains_arr = np.array([r["total_retrains"] for r in results])
        segs_arr = np.array([r["total_segments_retrained"] for r in results])
        time_arr = np.array([r["retrain_wall_clock_s"] for r in results])

        summary[name] = {
            "mqps_mean": np.mean(mqps_arr), "mqps_std": np.std(mqps_arr),
            "retrains_mean": np.mean(retrains_arr),
            "segs_mean": np.mean(segs_arr),
            "time_mean": np.mean(time_arr), "time_std": np.std(time_arr),
            "per_window": [r["mqps_per_window"] for r in results],
        }

        print(f"{name:<25} {np.mean(mqps_arr):>8.3f}±{np.std(mqps_arr):.3f}"
              f" {np.mean(retrains_arr):>12.1f}"
              f" {np.mean(segs_arr):>16.1f}"
              f" {np.mean(time_arr):>14.3f}±{np.std(time_arr):.3f}")

    print("=" * 85)

    # Generate plot
    if HAS_MPL and summary:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

        colors = {"A_never_retrain": "#e74c3c",
                  "B_periodic_retrain": "#3498db",
                  "C_adaptive_retrain": "#2ecc71"}
        labels = {"A_never_retrain": "Never Retrain",
                  "B_periodic_retrain": "Periodic Retrain",
                  "C_adaptive_retrain": "Adaptive Retrain"}

        for name in configs:
            if name not in summary:
                continue
            windows_list = summary[name]["per_window"]
            max_len = max(len(w) for w in windows_list)
            padded = np.array([w + [np.nan] * (max_len - len(w)) for w in windows_list])
            mean_curve = np.nanmean(padded, axis=0)
            std_curve = np.nanstd(padded, axis=0)
            x = np.arange(1, max_len + 1)

            ax1.plot(x, mean_curve, label=labels[name], color=colors[name], linewidth=2)
            ax1.fill_between(x, mean_curve - std_curve, mean_curve + std_curve,
                             alpha=0.2, color=colors[name])

        ax1.set_xlabel("Window (each = 10k inserts)")
        ax1.set_ylabel("Read Throughput (MQPS)")
        ax1.set_title("Read Throughput Over Time")
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # Bar chart: retrain cost
        bar_names = [labels[n] for n in configs if n in summary]
        bar_times = [summary[n]["time_mean"] for n in configs if n in summary]
        bar_stds = [summary[n]["time_std"] for n in configs if n in summary]
        bar_colors = [colors[n] for n in configs if n in summary]

        ax2.bar(bar_names, bar_times, yerr=bar_stds, color=bar_colors, alpha=0.8, capsize=5)
        ax2.set_ylabel("Retrain Wall-Clock (seconds)")
        ax2.set_title("Retrain Cost Comparison")
        ax2.grid(True, alpha=0.3, axis="y")

        plt.tight_layout()
        plot_path = str(ROOT / "report" / "adaptive_vs_periodic.png")
        plt.savefig(plot_path, dpi=150)
        print(f"\nPlot saved to {plot_path}")

    # Write results CSV
    csv_path = os.path.join(args.out_dir, "evaluation_summary.csv")
    with open(csv_path, "w") as f:
        f.write("config,mqps_mean,mqps_std,retrains_mean,segs_retrained_mean,retrain_time_mean,retrain_time_std\n")
        for name in configs:
            if name not in summary:
                continue
            s = summary[name]
            f.write(f"{name},{s['mqps_mean']:.4f},{s['mqps_std']:.4f},"
                    f"{s['retrains_mean']:.1f},{s['segs_mean']:.1f},"
                    f"{s['time_mean']:.4f},{s['time_std']:.4f}\n")
    print(f"Summary CSV written to {csv_path}")


if __name__ == "__main__":
    main()
