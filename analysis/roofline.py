#!/usr/bin/env python
"""
roofline.py — Phase 12: A10G roofline + arithmetic-intensity classification for OpenVLA kernels.

Computes:
  1. Ridge points (arithmetic intensity where the GPU flips memory-bound -> compute-bound) at each
     precision, from the A10G's peak compute and 600 GB/s bandwidth.
  2. Arithmetic intensity (FLOP per byte of HBM traffic) for the dominant GEMMs in each regime
     (LLM prefill vs LLM decode, plus vision), and classifies each memory- vs compute-bound.
  3. A cross-check of analytic GEMM FLOPs/step against the PyTorch Profiler's measured aten::mm
     FLOPs (Phase 3), and a roofline plot.

CPU-only: pure arithmetic on the published specs + Phase-3 FLOP estimates. No GPU, no torch.

Method note: the PyTorch Profiler gives FLOPs but not HBM bytes moved, so byte traffic is derived
from the GEMM tensor shapes and dtype (the standard way to place kernels on a roofline):
    AI = FLOPs / bytes = (2*M*K*N) / (bytes_per_elem * (M*K + K*N + M*N))

Usage:
    python analysis/roofline.py
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# --- A10G specs (as given) -----------------------------------------------------------
PEAK_FLOPS = {  # FLOP/s (or OP/s for INT8)
    "FP32": 31.2e12,
    "FP16/BF16": 62.5e12,
    "INT8": 125e12,
}
BW_BYTES_S = 600e9  # 600 GB/s

# --- Llama-2-7B (OpenVLA backbone) dims + OpenVLA prefill sequence length -------------
H = 4096        # hidden size
FFN = 11008     # MLP intermediate size
LAYERS = 32
VOCAB = 32064   # OpenVLA-extended vocab (adds action tokens)
SEQ_PREFILL = 281   # OpenVLA prompt: ~256 visual patch tokens + text (matches Phase-4 nsys mask)
DECODE_TOKENS = 6   # action tokens generated after the first (Phase 2: ~6 decode steps)
BYTES_BF16 = 2

PROFILER_TXT = Path("results/baseline/torch_profiler_top20.txt")
PROFILER_STEPS = 20  # the Phase-3 profiler ran 20 steps


def ridge_points() -> dict:
    return {prec: round(peak / BW_BYTES_S, 2) for prec, peak in PEAK_FLOPS.items()}


def gemm_flops(M, K, N):
    return 2.0 * M * K * N


def gemm_ai(M, K, N, bytes_per=BYTES_BF16):
    """Arithmetic intensity (FLOP/byte) for C[M,N] = A[M,K] @ B[K,N]."""
    bytes_moved = bytes_per * (M * K + K * N + M * N)
    return gemm_flops(M, K, N) / bytes_moved


def parse_profiler_mm_flops() -> float | None:
    """Pull aten::mm 'Total MFLOPs' from the committed Phase-3 table -> FLOPs per step."""
    if not PROFILER_TXT.exists():
        return None
    for line in PROFILER_TXT.read_text().splitlines():
        if "aten::mm" in line:
            nums = re.findall(r"[\d.]+", line)
            mflops = float(nums[-1])  # last column is Total MFLOPs
            return mflops * 1e6 / PROFILER_STEPS
    return None


def main() -> None:
    bf16_ridge = PEAK_FLOPS["FP16/BF16"] / BW_BYTES_S
    int8_ridge = PEAK_FLOPS["INT8"] / BW_BYTES_S
    ridges = ridge_points()

    # --- kernel table: (label, regime, M, K, N) --------------------------------------
    kernels = [
        # LLM DECODE (one new token, M=1) — weight read per token, reused once
        ("LLM decode · attn QKV/O", "decode", 1, H, H),
        ("LLM decode · MLP gate/up", "decode", 1, H, FFN),
        ("LLM decode · MLP down",    "decode", 1, FFN, H),
        ("LLM decode · LM head",     "decode", 1, H, VOCAB),
        # LLM PREFILL (full prompt, M=seq) — weights amortized over many tokens
        ("LLM prefill · attn QKV/O", "prefill", SEQ_PREFILL, H, H),
        ("LLM prefill · MLP gate/up", "prefill", SEQ_PREFILL, H, FFN),
        ("LLM prefill · MLP down",   "prefill", SEQ_PREFILL, FFN, H),
        # Vision encoder (ViT block linear; 256 patches batched) — representative
        ("Vision · ViT MLP (256 patch)", "vision", 256, 1024, 4096),
    ]

    rows = []
    for label, regime, M, K, N in kernels:
        ai = gemm_ai(M, K, N)
        bound = "memory-bound" if ai < bf16_ridge else "compute-bound"
        rows.append({"kernel": label, "regime": regime, "M": M, "K": K, "N": N,
                     "arithmetic_intensity_flop_per_byte": round(ai, 2),
                     "bound_at_bf16": bound,
                     "x_ridge_distance": round(ai / bf16_ridge, 3)})

    # --- analytic FLOPs/step, split prefill vs decode (the FLOP-vs-time paradox) ------
    def layer_gemm_flops(M):
        attn = 4 * gemm_flops(M, H, H)                      # Q,K,V,O
        mlp = 2 * gemm_flops(M, H, FFN) + gemm_flops(M, FFN, H)  # gate, up, down
        return attn + mlp

    prefill_flops = LAYERS * layer_gemm_flops(SEQ_PREFILL) + gemm_flops(SEQ_PREFILL, H, VOCAB)
    decode_flops = DECODE_TOKENS * (LAYERS * layer_gemm_flops(1) + gemm_flops(1, H, VOCAB))
    total_flops_step = prefill_flops + decode_flops
    profiler_flops_step = parse_profiler_mm_flops()

    result = {
        "a10g_specs": {"peak_flops": PEAK_FLOPS, "memory_bandwidth_bytes_s": BW_BYTES_S},
        "ridge_points_flop_per_byte": ridges,
        "model": {"backbone": "Llama-2-7B", "hidden": H, "ffn": FFN, "layers": LAYERS,
                  "vocab": VOCAB, "prefill_seq": SEQ_PREFILL, "decode_tokens": DECODE_TOKENS,
                  "compute_dtype": "bf16"},
        "kernels": rows,
        "flops_per_inference": {
            "prefill_flops": prefill_flops, "decode_flops": decode_flops,
            "total_analytic": total_flops_step,
            "prefill_share_pct": round(prefill_flops / total_flops_step * 100, 1),
            "decode_share_pct": round(decode_flops / total_flops_step * 100, 1),
            "profiler_aten_mm_flops_per_step": profiler_flops_step,
            "analytic_vs_profiler_ratio": (round(total_flops_step / profiler_flops_step, 3)
                                           if profiler_flops_step else None),
        },
    }
    out = Path("results/analysis/roofline.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2))

    # --- print tables -----------------------------------------------------------------
    print("=== A10G ridge points (peak compute / 600 GB/s) ===")
    for prec, rp in ridges.items():
        print(f"  {prec:10}: {rp:6.1f} {'OP' if prec=='INT8' else 'FLOP'}/byte")
    print(f"\n  Operating precision is BF16 -> ridge = {bf16_ridge:.1f} FLOP/byte.")
    print("  AI below the ridge = memory-bound (bandwidth-limited); above = compute-bound.\n")

    print("=== Kernel arithmetic intensity & classification (BF16) ===")
    print(f"  {'kernel':30}{'AI (F/B)':>10}{'vs ridge':>10}  bound")
    print("  " + "-" * 66)
    for r in rows:
        print(f"  {r['kernel']:30}{r['arithmetic_intensity_flop_per_byte']:>10.1f}"
              f"{r['x_ridge_distance']:>9.2f}x  {r['bound_at_bf16']}")

    print("\n=== FLOPs vs time paradox ===")
    print(f"  prefill: {prefill_flops/1e12:.2f} TFLOP/step ({result['flops_per_inference']['prefill_share_pct']}% of FLOPs) — but ~35% of TIME (compute-bound, efficient)")
    print(f"  decode : {decode_flops/1e12:.3f} TFLOP/step ({result['flops_per_inference']['decode_share_pct']}% of FLOPs) — but ~59% of TIME (memory-bound, starves the tensor cores)")
    if profiler_flops_step:
        print(f"  cross-check: analytic {total_flops_step/1e12:.2f} TFLOP/step vs profiler aten::mm "
              f"{profiler_flops_step/1e12:.2f} TFLOP/step "
              f"(ratio {result['flops_per_inference']['analytic_vs_profiler_ratio']})")

    # --- roofline plot ----------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(8.5, 6))
    ai_axis = np.logspace(-1, 3.2, 400)
    colors = {"FP32": "#7f8c8d", "FP16/BF16": "#2c3e50", "INT8": "#8e44ad"}
    for prec, peak in PEAK_FLOPS.items():
        attainable = np.minimum(BW_BYTES_S * ai_axis, peak)
        ax.plot(ai_axis, attainable / 1e12, color=colors[prec], lw=1.8,
                label=f"{prec} roof (ridge {ridges[prec]:.0f})")
    regime_color = {"decode": "#c0392b", "prefill": "#27ae60", "vision": "#2980b9"}
    seen = set()
    for r in rows:
        ai = r["arithmetic_intensity_flop_per_byte"]
        perf = min(BW_BYTES_S * ai, PEAK_FLOPS["FP16/BF16"]) / 1e12
        lbl = r["regime"] if r["regime"] not in seen else None
        seen.add(r["regime"])
        ax.scatter([ai], [perf], color=regime_color[r["regime"]], s=55, zorder=5, label=lbl)
    ax.axvline(bf16_ridge, color="#2c3e50", ls="--", alpha=0.4)
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("arithmetic intensity (FLOP / byte)")
    ax.set_ylabel("attainable performance (TFLOP/s)")
    ax.set_title("OpenVLA kernels on the A10G roofline\n"
                 "decode GEMMs are deeply memory-bound; prefill/vision are compute-bound")
    ax.grid(True, which="both", alpha=0.25)
    ax.legend(fontsize=8, loc="lower right")
    fig.tight_layout()
    figp = Path("figures/roofline.png")
    figp.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(figp, dpi=120)

    print(f"\nSaved data -> {out}\nSaved plot -> {figp}")


if __name__ == "__main__":
    main()
