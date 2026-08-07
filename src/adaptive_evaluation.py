"""
Adaptive vs. Periodic vs. Never-Retrain evaluation (merge/compaction design).

Runs three maintenance policies on the same drifting insert workload, where the
delta buffer is periodically merged into the searchable array so a stale model
genuinely degrades:

  never    — refit nothing after a merge
  periodic — refit every segment after every merge
  adaptive — refit only drift-flagged segments

Reports, per policy, the trajectory of read throughput, p99 latency, bound-miss
rate, and mean search-window width over time (mean ± stddev across trials), plus
the total maintenance cost. Produces report/adaptive_vs_periodic.png.
"""

import argparse
import csv
import os
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
VENV_PYTHON = ROOT / "venv" / "bin" / "python"
PYTHON = str(VENV_PYTHON) if VENV_PYTHON.exists() else sys.executable

POLICIES = ["never", "periodic", "adaptive"]
COLORS = {"never": "#e74c3c", "periodic": "#3498db", "adaptive": "#2ecc71"}
LABELS = {"never": "Never retrain", "periodic": "Periodic retrain", "adaptive": "Adaptive retrain"}


def prepare_run(run_dir, distribution, num_keys, num_leaves):
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


def run_policy(policy, run_dir, num_inserts, interval, alpha):
    binary = str(ROOT / "adaptive_rmi_benchmark")
    cmd = [binary, run_dir, "--policy", policy,
           "--inserts", str(num_inserts), "--interval", str(interval),
           "--alpha", str(alpha)]
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
    if r.returncode != 0:
        print(f"  [{policy}] FAILED (rc={r.returncode}): "
              f"stderr={r.stderr[:300]!r} stdout_tail={r.stdout[-300:]!r}")
        return None

    # Per-window metrics from the CSV the benchmark writes.
    win_csv = os.path.join(run_dir, f"adaptive_windows_{policy}.csv")
    windows = {"mqps": [], "p99_us": [], "bound_miss_pct": [], "mean_window": [], "segs": []}
    with open(win_csv) as f:
        for row in csv.DictReader(f):
            windows["mqps"].append(float(row["mqps"]))
            windows["p99_us"].append(float(row["p99_us"]))
            windows["bound_miss_pct"].append(float(row["bound_miss_pct"]))
            windows["mean_window"].append(float(row["mean_window"]))
            windows["segs"].append(int(row["segs_retrained"]))

    # Scalar summary from the results file.
    summary = {}
    res = os.path.join(run_dir, f"adaptive_rmi_results_{policy}.txt")
    with open(res) as f:
        for line in f:
            if ":" in line:
                k, v = line.split(":", 1)
                summary[k.strip()] = v.strip()

    return {
        "windows": windows,
        "total_segments_retrained": int(summary.get("total_segments_retrained", 0)),
        "retrain_wall_clock_s": float(summary.get("retrain_wall_clock_s", 0.0)),
    }


def stack_windows(per_trial, key):
    """Stack a per-window metric across trials into a (trials, windows) array."""
    seqs = [np.array(t["windows"][key]) for t in per_trial]
    min_len = min(len(s) for s in seqs)
    return np.vstack([s[:min_len] for s in seqs])


def main():
    ap = argparse.ArgumentParser(description="Adaptive RMI maintenance-policy evaluation")
    ap.add_argument("--trials", type=int, default=5)
    ap.add_argument("--num-keys", type=int, default=1_000_000)
    ap.add_argument("--num-inserts", type=int, default=200000)
    ap.add_argument("--interval", type=int, default=10000)
    ap.add_argument("--num-leaves", type=int, default=500)
    ap.add_argument("--alpha", type=float, default=1.5)
    ap.add_argument("--distribution", type=str, default="drifting")
    ap.add_argument("--out-dir", type=str, default="data/eval_adaptive")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    results = {p: [] for p in POLICIES}

    for trial in range(args.trials):
        print(f"\n=== Trial {trial + 1}/{args.trials} ===")
        run_dir = os.path.join(args.out_dir, f"trial_{trial}")
        prepare_run(run_dir, args.distribution, args.num_keys, args.num_leaves)
        for policy in POLICIES:
            print(f"  Running {policy} ...")
            r = run_policy(policy, run_dir, args.num_inserts, args.interval, args.alpha)
            if r:
                results[policy].append(r)

    # ---- Aggregate + print ----
    print("\n" + "=" * 92)
    print(f"  MAINTENANCE-POLICY EVALUATION (mean ± stddev over {args.trials} trials)")
    print("=" * 92)
    print(f"{'Policy':<12}{'MQPS':>16}{'p99 (us)':>16}{'BoundMiss %':>16}"
          f"{'Segs Retrained':>18}{'Retrain (s)':>14}")
    print("-" * 92)

    agg = {}
    for policy in POLICIES:
        trials = results[policy]
        if not trials:
            print(f"{policy:<12}{'FAILED':>16}")
            continue

        # Time-averaged per-trial scalars, then across trials.
        mqps = np.array([np.mean(t["windows"]["mqps"]) for t in trials])
        p99 = np.array([np.mean(t["windows"]["p99_us"]) for t in trials])
        miss = np.array([np.mean(t["windows"]["bound_miss_pct"]) for t in trials])
        segs = np.array([t["total_segments_retrained"] for t in trials])
        rtime = np.array([t["retrain_wall_clock_s"] for t in trials])

        agg[policy] = {
            "mqps": mqps, "p99": p99, "miss": miss, "segs": segs, "rtime": rtime,
            "miss_curve": stack_windows(trials, "bound_miss_pct"),
            "p99_curve": stack_windows(trials, "p99_us"),
            "meanwin_curve": stack_windows(trials, "mean_window"),
        }

        print(f"{policy:<12}{mqps.mean():>9.2f}±{mqps.std():<5.2f}"
              f"{p99.mean():>9.3f}±{p99.std():<5.3f}"
              f"{miss.mean():>9.2f}±{miss.std():<5.2f}"
              f"{segs.mean():>13.0f}±{segs.std():<4.0f}"
              f"{rtime.mean():>9.3f}±{rtime.std():<4.3f}")
    print("=" * 92)

    # Cost-efficiency callout.
    if "periodic" in agg and "adaptive" in agg:
        per = agg["periodic"]["segs"].mean()
        ada = agg["adaptive"]["segs"].mean()
        if per > 0:
            print(f"\n  Adaptive retrains {100*ada/per:.1f}% of the segments periodic does "
                  f"({ada:.0f} vs {per:.0f}).")
        print(f"  Never-retrain bound-miss rate: {agg['never']['miss'].mean():.1f}%  "
              f"vs adaptive {agg['adaptive']['miss'].mean():.1f}%  "
              f"vs periodic {agg['periodic']['miss'].mean():.1f}%")

    # ---- Figure: the degradation story ----
    if HAS_MPL and agg:
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(16, 4.5))

        for policy in POLICIES:
            if policy not in agg:
                continue
            mc = agg[policy]["miss_curve"]
            x = np.arange(1, mc.shape[1] + 1)
            m, s = mc.mean(0), mc.std(0)
            ax1.plot(x, m, label=LABELS[policy], color=COLORS[policy], linewidth=2)
            ax1.fill_between(x, m - s, m + s, alpha=0.15, color=COLORS[policy])

            pc = agg[policy]["p99_curve"]
            m, s = pc.mean(0), pc.std(0)
            ax2.plot(x, m, label=LABELS[policy], color=COLORS[policy], linewidth=2)
            ax2.fill_between(x, m - s, m + s, alpha=0.15, color=COLORS[policy])

        ax1.set_xlabel("Window (each = 10k inserts merged)")
        ax1.set_ylabel("Bound-miss rate (%)")
        ax1.set_title("Correctness cost of stale model")
        ax1.legend(); ax1.grid(True, alpha=0.3)

        ax2.set_xlabel("Window (each = 10k inserts merged)")
        ax2.set_ylabel("p99 lookup latency (μs)")
        ax2.set_title("Tail latency over time")
        ax2.legend(); ax2.grid(True, alpha=0.3)

        names = [LABELS[p] for p in POLICIES if p in agg]
        segs = [agg[p]["segs"].mean() for p in POLICIES if p in agg]
        errs = [agg[p]["segs"].std() for p in POLICIES if p in agg]
        cols = [COLORS[p] for p in POLICIES if p in agg]
        ax3.bar(names, segs, yerr=errs, color=cols, alpha=0.85, capsize=5)
        ax3.set_ylabel("Total segments retrained")
        ax3.set_title("Maintenance cost")
        ax3.grid(True, alpha=0.3, axis="y")
        ax3.tick_params(axis="x", rotation=15)

        plt.tight_layout()
        out = str(ROOT / "report" / "adaptive_vs_periodic.png")
        plt.savefig(out, dpi=150)
        print(f"\nFigure saved to {out}")

    # ---- Summary CSV ----
    csv_path = os.path.join(args.out_dir, "evaluation_summary.csv")
    with open(csv_path, "w") as f:
        f.write("policy,mqps_mean,mqps_std,p99_mean,p99_std,"
                "bound_miss_mean,bound_miss_std,segs_mean,segs_std,rtime_mean,rtime_std\n")
        for policy in POLICIES:
            if policy not in agg:
                continue
            a = agg[policy]
            f.write(f"{policy},{a['mqps'].mean():.4f},{a['mqps'].std():.4f},"
                    f"{a['p99'].mean():.4f},{a['p99'].std():.4f},"
                    f"{a['miss'].mean():.4f},{a['miss'].std():.4f},"
                    f"{a['segs'].mean():.1f},{a['segs'].std():.1f},"
                    f"{a['rtime'].mean():.5f},{a['rtime'].std():.5f}\n")
    print(f"Summary CSV written to {csv_path}")


if __name__ == "__main__":
    main()
