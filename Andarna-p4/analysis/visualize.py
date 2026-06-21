#!/usr/bin/env python
"""
visualize.py — Phase 13: generate the summary figures from all collected results.

Reads the committed JSON under results/ and writes figures to figures/:
  fig1_component_breakdown.png   stacked component time (see note: BF16 only — see below)
  fig2_latency_vs_precision.png  mean latency vs precision, error bars = std
  fig3_memory_vs_precision.png   peak GPU memory vs precision
  fig4_accuracy_vs_precision.png cosine similarity vs BF16, lower whisker = worst-case (min)
  fig5_control_freq.png          achievable control Hz per precision, with a 10 Hz target line

CPU-only: pure plotting, no GPU/torch.

NOTE on fig1: per-stage component timing (vision/projector/prefill/decode) was measured only at
BF16 (Phase 2). FP16/INT8/INT4 have end-to-end totals (Phase 5) but were never decomposed, so a
truthful per-precision stacked chart isn't possible without GPU re-runs. fig1 therefore shows the
BF16 decomposition (the data we have); regenerate with --component-files once per-precision
breakdowns exist.

Usage:
    python analysis/visualize.py
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

R = Path("results")
FIG = Path("figures")
PRECS = ["bf16", "fp16", "int8", "int4"]
PCOLOR = {"bf16": "#2c3e50", "fp16": "#2980b9", "int8": "#e67e22", "int4": "#c0392b"}


def load(p: str) -> dict:
    return json.loads((R / p).read_text())


def main() -> None:
    FIG.mkdir(parents=True, exist_ok=True)
    sweep = load("quantization/precision_sweep.json")["results"]
    comp = load("baseline/component_breakdown_bf16.json")
    acc = load("quantization/accuracy_validation.json")["comparison_vs_bf16"]

    # --- fig1: BF16 component breakdown (stacked) ------------------------------------
    stages = comp["stages"]
    order = [("vision", "Vision Encoder"), ("projector", "MLP Projector"),
             ("prefill", "LLM Prefill"), ("decode", "LLM Decode")]
    seg_colors = ["#2980b9", "#16a085", "#f39c12", "#c0392b", "#95a5a6"]
    fig, ax = plt.subplots(figsize=(4.5, 6))
    bottom = 0.0
    for (key, label), col in zip(order, seg_colors):
        v = stages[key]["mean_ms"]
        ax.bar("BF16", v, bottom=bottom, color=col, label=f"{label} ({v:.1f} ms)")
        if v > 8:
            ax.text(0, bottom + v / 2, f"{v:.0f}", ha="center", va="center", color="white", fontsize=9)
        bottom += v
    ov = comp["unaccounted_overhead_ms"]
    ax.bar("BF16", ov, bottom=bottom, color=seg_colors[4], label=f"overhead ({ov:.1f} ms)")
    ax.set_ylabel("latency (ms)")
    ax.set_title("Component time breakdown (BF16)\n[per-precision split not measured — see note]")
    ax.legend(fontsize=8, loc="upper right")
    fig.tight_layout(); fig.savefig(FIG / "fig1_component_breakdown.png", dpi=120); plt.close(fig)

    # --- fig2: latency vs precision (error bars = std) -------------------------------
    means = [sweep[p]["latency"]["mean_ms"] for p in PRECS]
    stds = [sweep[p]["latency"]["std_ms"] for p in PRECS]
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.errorbar(PRECS, means, yerr=stds, marker="o", capsize=5, lw=2, color="#2c3e50")
    for x, m in zip(PRECS, means):
        ax.annotate(f"{m:.0f} ms", (x, m), textcoords="offset points", xytext=(0, 10), ha="center")
    ax.set_ylabel("mean inference latency (ms)")
    ax.set_title("Inference latency vs precision (error bars = std over 100 iters)\n"
                 "quantization is SLOWER — INT8 worst, INT4 < INT8")
    ax.grid(True, alpha=0.3); ax.set_ylim(0, max(means) * 1.18)
    fig.tight_layout(); fig.savefig(FIG / "fig2_latency_vs_precision.png", dpi=120); plt.close(fig)

    # --- fig3: memory vs precision ---------------------------------------------------
    mem = [sweep[p]["peak_mem_allocated_mb"] / 1000.0 for p in PRECS]
    fig, ax = plt.subplots(figsize=(7, 4.5))
    bars = ax.bar(PRECS, mem, color=[PCOLOR[p] for p in PRECS])
    for b, m in zip(bars, mem):
        ax.text(b.get_x() + b.get_width() / 2, m + 0.2, f"{m:.1f} GB", ha="center", fontsize=10)
    ax.set_ylabel("peak GPU memory (GB)")
    ax.set_title("Peak GPU memory vs precision\nINT4 = 0.31x BF16 (the one clear quantization win)")
    ax.grid(True, axis="y", alpha=0.3); ax.set_ylim(0, max(mem) * 1.18)
    fig.tight_layout(); fig.savefig(FIG / "fig3_memory_vs_precision.png", dpi=120); plt.close(fig)

    # --- fig4: accuracy (cosine) vs precision ----------------------------------------
    cos = [acc[p]["mean_cosine"] for p in PRECS]
    cmin = [acc[p]["min_cosine"] for p in PRECS]
    lower = [c - lo for c, lo in zip(cos, cmin)]  # whisker down to worst-case
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.errorbar(PRECS, cos, yerr=[lower, [0, 0, 0, 0]], marker="s", capsize=5, lw=2,
                color="#8e44ad", label="mean cosine (whisker → worst-case)")
    for x, c in zip(PRECS, cos):
        ax.annotate(f"{c:.3f}", (x, c), textcoords="offset points", xytext=(0, 10), ha="center")
    ax.set_ylabel("cosine similarity vs BF16 action")
    ax.set_title("Action fidelity vs precision\nFP16 ≈ identical; INT4 drifts most (worst-case 0.74)")
    ax.grid(True, alpha=0.3); ax.set_ylim(0.7, 1.005); ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(FIG / "fig4_accuracy_vs_precision.png", dpi=120); plt.close(fig)

    # --- fig5: control frequency vs precision, with 10 Hz target ---------------------
    hz = [1000.0 / sweep[p]["latency"]["mean_ms"] for p in PRECS]
    fig, ax = plt.subplots(figsize=(7, 4.5))
    bars = ax.bar(PRECS, hz, color=[PCOLOR[p] for p in PRECS])
    for b, h in zip(bars, hz):
        ax.text(b.get_x() + b.get_width() / 2, h + 0.15, f"{h:.2f} Hz", ha="center", fontsize=10)
    ax.axhline(10, color="#c0392b", ls="--", lw=2, label="10 Hz control target")
    ax.set_ylabel("achievable control frequency (Hz)")
    ax.set_title("Control frequency vs precision\nEVERY precision is far below a 10 Hz loop "
                 "(best ~2.8 Hz)")
    ax.grid(True, axis="y", alpha=0.3); ax.set_ylim(0, 11); ax.legend(loc="upper right")
    fig.tight_layout(); fig.savefig(FIG / "fig5_control_freq.png", dpi=120); plt.close(fig)

    figs = ["fig1_component_breakdown", "fig2_latency_vs_precision", "fig3_memory_vs_precision",
            "fig4_accuracy_vs_precision", "fig5_control_freq"]
    print("Saved figures:")
    for f in figs:
        print(f"  figures/{f}.png")
    print("\nData snapshot used:")
    print(f"  latency (ms): " + ", ".join(f"{p} {sweep[p]['latency']['mean_ms']:.0f}" for p in PRECS))
    print(f"  memory (GB) : " + ", ".join(f"{p} {sweep[p]['peak_mem_allocated_mb']/1000:.1f}" for p in PRECS))
    print(f"  cosine      : " + ", ".join(f"{p} {acc[p]['mean_cosine']:.3f}" for p in PRECS))
    print(f"  Hz          : " + ", ".join(f"{p} {1000/sweep[p]['latency']['mean_ms']:.2f}" for p in PRECS))


if __name__ == "__main__":
    main()
