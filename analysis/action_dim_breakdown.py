#!/usr/bin/env python
"""
action_dim_breakdown.py — attribute quantization action error to specific action dimensions.

Phase 6 reported aggregate MAE / max-abs-error / cosine per precision. The INT4 max abs error
(0.83) exceeded the whole action scale, which suggested a single dimension — likely the gripper
bit (grasp <-> release) — was flipping. This script decomposes the per-precision error across
OpenVLA's 7 action dimensions to confirm where it lives.

OpenVLA action layout (LIBERO): [dx, dy, dz, droll, dpitch, dyaw, gripper].
The first 6 are continuous end-effector deltas; the 7th is the gripper command (open/close).

Reads results/quantization/accuracy_validation.json (the saved action vectors). No GPU needed.

Usage:
    python analysis/action_dim_breakdown.py
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

DIM_NAMES = ["dx", "dy", "dz", "d_roll", "d_pitch", "d_yaw", "gripper"]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Per-dimension action error breakdown vs BF16.")
    p.add_argument("--input", default="results/quantization/accuracy_validation.json")
    p.add_argument("--output", default="results/quantization/action_dim_breakdown.json")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    data = json.loads(Path(args.input).read_text())
    vecs = data["action_vectors"]
    if "bf16" not in vecs:
        raise SystemExit("No bf16 baseline in action_vectors.")
    base = np.asarray(vecs["bf16"], dtype=np.float64)          # (n_obs, 7)
    n_obs, n_dim = base.shape

    report = {"baseline": "bf16", "n_obs": n_obs, "dim_names": DIM_NAMES, "precisions": {}}

    print(f"=== Per-dimension |error| vs BF16 (mean over {n_obs} obs) ===")
    head = "  " + "prec".ljust(6) + "".join(d.rjust(9) for d in DIM_NAMES) + "   worst-dim"
    print(head)
    print("  " + "-" * (len(head) - 2))

    for prec, arr in vecs.items():
        if prec == "bf16":
            continue
        a = np.asarray(arr, dtype=np.float64)
        abs_err = np.abs(a - base)                              # (n_obs, 7)
        per_dim_mae = abs_err.mean(axis=0)                      # (7,)
        worst = int(np.argmax(per_dim_mae))

        # Gripper-specific: sign flips between baseline and this precision on dim -1.
        grip_base = np.sign(base[:, -1])
        grip_prec = np.sign(a[:, -1])
        grip_flips = int(np.sum(grip_base != grip_prec))

        # How much of the total abs error (summed over dims & obs) is the gripper dim?
        grip_share = float(abs_err[:, -1].sum() / abs_err.sum()) if abs_err.sum() > 0 else 0.0

        report["precisions"][prec] = {
            "per_dim_mae": {DIM_NAMES[i]: round(float(per_dim_mae[i]), 5) for i in range(n_dim)},
            "worst_dim": DIM_NAMES[worst],
            "gripper_sign_flips": grip_flips,
            "gripper_error_share": round(grip_share, 4),
            "gripper_base_values": [round(float(x), 4) for x in base[:, -1]],
            "gripper_prec_values": [round(float(x), 4) for x in a[:, -1]],
        }
        row = "  " + prec.ljust(6) + "".join(f"{per_dim_mae[i]:9.4f}" for i in range(n_dim))
        print(row + f"   {DIM_NAMES[worst]}")

    print("\n=== Gripper dimension (grasp/release) detail ===")
    for prec, r in report["precisions"].items():
        print(f"  {prec}: {r['gripper_sign_flips']}/{n_obs} sign flips | "
              f"gripper carries {r['gripper_error_share']*100:.0f}% of total abs error")
        print(f"     bf16 gripper : {report['precisions'][prec]['gripper_base_values']}")
        print(f"     {prec:4} gripper : {report['precisions'][prec]['gripper_prec_values']}")

    out = Path(args.output)
    out.write_text(json.dumps(report, indent=2))
    print(f"\nSaved -> {out}")


if __name__ == "__main__":
    main()
