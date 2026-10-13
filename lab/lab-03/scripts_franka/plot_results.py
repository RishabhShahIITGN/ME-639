#!/usr/bin/env python3
"""
plot_results.py — IK Solver Comparison Plots (Seed 19, 250 Episodes)
=====================================================================
Generates a 2×2 figure comparing Mink, DLS, and QP IK solvers:
  1. Success Rate (Bar Chart)
  2. IK Planning Time Distribution (Box Plot)
  3. Total IK Iterations Distribution (Box Plot)
  4. Minimum Clearance Distribution (Box Plot)

Usage:
    cd lab/lab-03
    python3 scripts/plot_results.py
"""

import json
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

# ---------------------------------------------------------------------------
# Locate the seed-19 batch folders
# ---------------------------------------------------------------------------
LAB = Path(__file__).resolve().parent.parent
LOGS = LAB / "logs"

BATCH_FOLDERS = {
    "Mink": "ik_batch_20261012_010325_seed19",
    "DLS":  "ik_dls_batch_20261011_230632_seed19",
    "QP":   "ik_qp_batch_20261011_235407_seed19",
}

COLORS = {
    "Mink": "#2196F3",   # blue
    "DLS":  "#4CAF50",   # green
    "QP":   "#FF9800",   # orange
}


def load_episodes(folder: str) -> list[dict]:
    """Load all episode_*.json files from a batch folder."""
    batch_dir = LOGS / folder
    episodes = []
    for fname in sorted(batch_dir.glob("episode_*.json")):
        with open(fname) as f:
            episodes.append(json.load(f))
    return episodes


# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------
print("Loading episode data …")
data: dict[str, list[dict]] = {}
for solver, folder in BATCH_FOLDERS.items():
    data[solver] = load_episodes(folder)
    n = len(data[solver])
    succ = sum(1 for e in data[solver] if e.get("success"))
    print(f"  {solver:6s}: {n} episodes, {succ} successes  (folder: {folder})")

# ---------------------------------------------------------------------------
# Extract metrics (successful episodes only for distributions)
# ---------------------------------------------------------------------------
def success_rate(episodes):
    total = len(episodes)
    succ  = sum(1 for e in episodes if e.get("success"))
    return 100.0 * succ / total if total else 0.0

def get_planning_times(episodes):
    return [e["planning_time_s"] for e in episodes if e.get("success") and e.get("planning_time_s") is not None]

def get_ik_iterations(episodes):
    return [e["ik_iterations"] for e in episodes if e.get("success") and e.get("ik_iterations") is not None]

def get_min_clearance(episodes):
    vals = []
    for e in episodes:
        mc = e.get("minimum_clearance")
        if mc and isinstance(mc, dict) and mc.get("distance_m") is not None:
            vals.append(mc["distance_m"])
    return vals

solvers     = list(data.keys())
rates       = [success_rate(data[s])      for s in solvers]
plan_times  = [get_planning_times(data[s]) for s in solvers]
ik_iters    = [get_ik_iterations(data[s])  for s in solvers]
clearances  = [get_min_clearance(data[s])  for s in solvers]

# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle(
    "Performance Comparison of IK Solvers (250 Episodes, Seed 19)",
    fontsize=15, fontweight="bold", y=1.01
)

colors_list = [COLORS[s] for s in solvers]

# ── 1. Success Rate Bar Chart ────────────────────────────────────────────────
ax1 = axes[0, 0]
bars = ax1.bar(solvers, rates, color=colors_list, width=0.5, edgecolor="white", linewidth=1.2)
ax1.set_ylim(0, 110)
ax1.set_ylabel("Success Rate (%)", fontsize=11)
ax1.set_title("1. Success Rate", fontsize=12, fontweight="bold")
ax1.yaxis.grid(True, linestyle="--", alpha=0.7)
ax1.set_axisbelow(True)
for bar, val in zip(bars, rates):
    ax1.text(bar.get_x() + bar.get_width() / 2, val + 1.5,
             f"{val:.1f}%", ha="center", va="bottom", fontsize=11, fontweight="bold")

# ── 2. IK Planning Time Box Plot ────────────────────────────────────────────
ax2 = axes[0, 1]
bp2 = ax2.boxplot(
    plan_times, tick_labels=solvers, patch_artist=True,
    medianprops=dict(color="black", linewidth=2),
    flierprops=dict(marker="o", markersize=3, alpha=0.4),
    widths=0.45,
)
for patch, color in zip(bp2["boxes"], colors_list):
    patch.set_facecolor(color)
    patch.set_alpha(0.8)
ax2.set_ylabel("Planning Time (Seconds)", fontsize=11)
ax2.set_title("2. IK Planning Time Distribution", fontsize=12, fontweight="bold")
ax2.yaxis.grid(True, linestyle="--", alpha=0.7)
ax2.set_axisbelow(True)

# ── 3. Total IK Iterations Box Plot ─────────────────────────────────────────
ax3 = axes[1, 0]
bp3 = ax3.boxplot(
    ik_iters, tick_labels=solvers, patch_artist=True,
    medianprops=dict(color="black", linewidth=2),
    flierprops=dict(marker="o", markersize=3, alpha=0.4),
    widths=0.45,
)
for patch, color in zip(bp3["boxes"], colors_list):
    patch.set_facecolor(color)
    patch.set_alpha(0.8)
ax3.set_ylabel("IK Iterations (Count)", fontsize=11)
ax3.set_title("3. Total IK Iterations Distribution", fontsize=12, fontweight="bold")
ax3.yaxis.grid(True, linestyle="--", alpha=0.7)
ax3.set_axisbelow(True)

# ── 4. Minimum Clearance Box Plot ───────────────────────────────────────────
ax4 = axes[1, 1]
# Convert to millimeters for readability
clearances_mm = [[v * 1000 for v in vals] for vals in clearances]
bp4 = ax4.boxplot(
    clearances_mm, tick_labels=solvers, patch_artist=True,
    medianprops=dict(color="black", linewidth=2),
    flierprops=dict(marker="o", markersize=3, alpha=0.4),
    widths=0.45,
)
for patch, color in zip(bp4["boxes"], colors_list):
    patch.set_facecolor(color)
    patch.set_alpha(0.8)
ax4.set_ylabel("Minimum Clearance (mm)", fontsize=11)
ax4.set_title("4. Minimum Signed Clearance Distribution", fontsize=12, fontweight="bold")
ax4.yaxis.grid(True, linestyle="--", alpha=0.7)
ax4.set_axisbelow(True)
ax4.axhline(y=0, color="red", linestyle="--", linewidth=1.2, alpha=0.8, label="Contact threshold")
ax4.legend(fontsize=9)

# ── Legend patches ───────────────────────────────────────────────────────────
legend_patches = [mpatches.Patch(color=COLORS[s], label=s, alpha=0.85) for s in solvers]
fig.legend(
    handles=legend_patches, loc="lower center",
    ncol=3, fontsize=11, frameon=True,
    bbox_to_anchor=(0.5, -0.04), title="IK Solver", title_fontsize=10
)

# ── Style tweaks ─────────────────────────────────────────────────────────────
for ax in axes.flat:
    ax.tick_params(labelsize=10)
    for spine in ax.spines.values():
        spine.set_linewidth(0.8)

plt.tight_layout(rect=[0, 0.04, 1, 1])

# ---------------------------------------------------------------------------
# Save
# ---------------------------------------------------------------------------
out_path = LOGS / "ik_comparison_plots_seed19.png"
fig.savefig(out_path, dpi=300, bbox_inches="tight")
print(f"\nPlot saved → {out_path}")

# ── Print a quick stats table ─────────────────────────────────────────────────
print("\n── Summary statistics ──────────────────────────────────────────────────")
print(f"{'Metric':<35} {'Mink':>12} {'DLS':>12} {'QP':>12}")
print("-" * 75)
print(f"{'Success Rate (%)':35} {rates[0]:12.1f} {rates[1]:12.1f} {rates[2]:12.1f}")
for label, vals_list in [
    ("Planning Time – median (s)",    [[np.median(v) for v in plan_times]]),
    ("Planning Time – mean (s)",      [[np.mean(v) for v in plan_times]]),
    ("IK Iterations – median",        [[np.median(v) for v in ik_iters]]),
    ("Min Clearance – median (mm)",   [[np.median(v)*1000 for v in clearances]]),
]:
    row = vals_list[0]
    print(f"{label:35} {row[0]:12.3f} {row[1]:12.3f} {row[2]:12.3f}")
print("-" * 75)
