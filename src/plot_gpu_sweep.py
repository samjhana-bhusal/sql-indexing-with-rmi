"""
Regenerate the GPU batch-size sweep figure (report/gpu_throughput.png) from the
reported 5-run sweep in report/gpu_sweep.csv.

Single panel, log2 x-axis (matching the "log-scale sweep" description), with
error bars and the CPU RMI reference line. GPU memory is essentially constant
across the sweep (76.6-78.3 MB), so it is stated in the caption rather than
given its own (flat, uninformative) panel.
"""
import csv
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
CSV = os.path.join(HERE, "..", "report", "gpu_sweep.csv")
OUT = os.path.join(HERE, "..", "report", "gpu_throughput.png")
CPU_RMI = 11.07  # CPU RMI throughput baseline (MQPS), tab:baselines

batch, mqps, std, mem = [], [], [], []
with open(CSV) as f:
    for row in csv.DictReader(f):
        batch.append(int(row["batch"]))
        mqps.append(float(row["mqps"]))
        std.append(float(row["std"]))
        mem.append(float(row["mem_mb"]))

fig, ax = plt.subplots(figsize=(7.5, 4.2))
ax.errorbar(batch, mqps, yerr=std, marker="o", linewidth=2, capsize=3,
            color="#1f77b4", label="GPU (MPS), 5-run mean $\\pm$ stddev")
ax.axhline(CPU_RMI, color="#d62728", linestyle="--", linewidth=1.5,
           label=f"CPU RMI ({CPU_RMI:.2f} MQPS)")

ax.set_xscale("log", base=2)
ax.set_xticks(batch)
ax.set_xticklabels([str(b) for b in batch], rotation=45, fontsize=9)
ax.set_xlabel("Batch size (log$_2$ scale)", fontsize=12)
ax.set_ylabel("Throughput (MQPS)", fontsize=12)
ax.set_title("GPU Dispatch Cost: Break-Even Batch Size vs. CPU RMI", fontsize=12)
ax.grid(True, which="both", ls="--", alpha=0.4)
ax.set_ylim(bottom=0)
ax.legend(fontsize=10, loc="upper left")

plt.tight_layout()
plt.savefig(OUT, dpi=200, bbox_inches="tight")
print(f"Wrote {OUT}  (GPU memory across sweep: {min(mem):.1f}-{max(mem):.1f} MB)")
