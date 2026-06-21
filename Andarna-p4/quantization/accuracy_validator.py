#!/usr/bin/env python
"""
accuracy_validator.py — Phase 6: does quantization change what the robot would do?

Runs the SAME 5 fixed observations through OpenVLA at each precision (BF16/FP16/INT8/INT4)
with greedy decoding (do_sample=False), then compares each precision's 7-DoF action vectors
against the BF16 baseline. Reports, per precision:
  - mean absolute error (MAE) across the action dimensions,
  - max absolute error (the worst single dimension), and
  - cosine similarity vs the BF16 action.

Latency (Phase 5) told us quantization is slower; this tells us whether it's also *different* —
i.e. whether the memory savings of INT4 come at the cost of changed policy behavior. This is
the bridge to Project 2: quantization is just another perturbation on the policy.

Each precision runs in its own subprocess (clean CUDA context, and INT8/INT4 can't coexist
with BF16 weights in one process). Loader is reused from precision_runner.

Usage:
    python quantization/accuracy_validator.py
    python quantization/accuracy_validator.py --n-obs 5 --only bf16 int4
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from quantization.precision_runner import (  # noqa: E402
    DEFAULT_MODEL, DEFAULT_UNNORM_KEY, INPUT_DTYPE, PRECISIONS, PROMPT_TEMPLATE, load_model,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="OpenVLA quantization accuracy validation.")
    p.add_argument("--model-id", default=DEFAULT_MODEL)
    p.add_argument("--unnorm-key", default=DEFAULT_UNNORM_KEY)
    p.add_argument("--n-obs", type=int, default=5, help="Number of fixed observations.")
    p.add_argument("--instruction", default="pick up the object and place it in the basket")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--only", nargs="+", choices=PRECISIONS, default=PRECISIONS)
    p.add_argument("--output", default="results/quantization/accuracy_validation.json")
    # Worker-mode flags (internal):
    p.add_argument("--precision", choices=PRECISIONS)
    p.add_argument("--fragment-out")
    return p.parse_args()


def build_observations(n: int) -> list[Image.Image]:
    """n deterministic synthetic 224x224 RGB frames (seed = observation index)."""
    obs = []
    for s in range(n):
        rng = np.random.default_rng(s)
        obs.append(Image.fromarray(rng.integers(0, 256, (224, 224, 3), dtype=np.uint8), "RGB"))
    return obs


# --------------------------------------------------------------------------- worker
def run_worker(args) -> dict:
    device = args.device
    precision = args.precision
    torch.cuda.set_device(device)
    processor, model = load_model(args.model_id, precision, device)

    observations = build_observations(args.n_obs)
    prompt = PROMPT_TEMPLATE.format(instruction=args.instruction)
    unnorm_key = args.unnorm_key or None
    in_dtype = INPUT_DTYPE[precision]

    actions = []
    with torch.inference_mode():
        for img in observations:
            inputs = processor(prompt, img).to(device, dtype=in_dtype)
            a = model.predict_action(**inputs, unnorm_key=unnorm_key, do_sample=False)
            actions.append([float(x) for x in np.asarray(a).flatten().tolist()])
    return {"status": "ok", "precision": precision, "actions": actions}


# --------------------------------------------------------------------------- metrics
def compare_to_baseline(actions: list[list[float]], baseline: list[list[float]]) -> dict:
    per_obs = []
    for i, (a_l, b_l) in enumerate(zip(actions, baseline)):
        a, b = np.asarray(a_l, dtype=np.float64), np.asarray(b_l, dtype=np.float64)
        diff = a - b
        denom = np.linalg.norm(a) * np.linalg.norm(b)
        cosine = float(np.dot(a, b) / denom) if denom > 0 else float("nan")
        per_obs.append({
            "obs": i,
            "mae": float(np.mean(np.abs(diff))),
            "max_ae": float(np.max(np.abs(diff))),
            "cosine": cosine,
        })
    maes = [o["mae"] for o in per_obs]
    cosines = [o["cosine"] for o in per_obs]
    return {
        "mean_mae": float(np.mean(maes)),
        "max_ae": float(np.max([o["max_ae"] for o in per_obs])),
        "mean_cosine": float(np.mean(cosines)),
        "min_cosine": float(np.min(cosines)),
        "per_obs": per_obs,
    }


# --------------------------------------------------------------------------- orchestrator
def run_worker_subprocess(precision: str, args, frag: Path) -> dict:
    cmd = [
        sys.executable, os.path.abspath(__file__),
        "--precision", precision, "--fragment-out", str(frag),
        "--model-id", args.model_id, "--unnorm-key", args.unnorm_key,
        "--n-obs", str(args.n_obs), "--instruction", args.instruction, "--device", args.device,
    ]
    print(f"[accuracy] === {precision.upper()} === (collecting {args.n_obs} action vectors)")
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0 or not frag.exists():
        tail = "\n".join(proc.stderr.strip().splitlines()[-8:])
        print(f"[accuracy] {precision} FAILED (rc={proc.returncode}):\n{tail}")
        return {"status": "failed", "precision": precision, "stderr_tail": tail}
    return json.loads(frag.read_text())


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is not available — this requires a GPU.")

    if args.precision:  # worker mode
        result = run_worker(args)
        Path(args.fragment_out).write_text(json.dumps(result))
        return

    # Always need bf16 as the baseline; ensure it's first and present.
    order = [p for p in PRECISIONS if p in (["bf16"] + args.only)]
    if "bf16" not in order:
        order = ["bf16"] + order

    frag_dir = Path(args.output).parent / "_acc_fragments"
    frag_dir.mkdir(parents=True, exist_ok=True)
    raw: dict[str, dict] = {}
    for precision in order:
        frag = frag_dir / f"{precision}.json"
        if frag.exists():
            frag.unlink()
        raw[precision] = run_worker_subprocess(precision, args, frag)

    base = raw.get("bf16")
    if not base or base.get("status") != "ok":
        raise SystemExit("BF16 baseline failed — cannot compute relative accuracy.")
    baseline_actions = base["actions"]
    baseline_mean_abs = float(np.mean(np.abs(np.asarray(baseline_actions))))

    comparisons = {}
    for precision, r in raw.items():
        if r.get("status") == "ok":
            comparisons[precision] = compare_to_baseline(r["actions"], baseline_actions)

    summary = {
        "metadata": {
            "model_id": args.model_id, "device_name": torch.cuda.get_device_name(args.device),
            "torch_version": torch.__version__, "python_version": platform.python_version(),
            "n_observations": args.n_obs, "instruction": args.instruction,
            "baseline": "bf16", "do_sample": False,
            "baseline_mean_abs_action": round(baseline_mean_abs, 5),
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        },
        "action_vectors": {p: r["actions"] for p, r in raw.items() if r.get("status") == "ok"},
        "comparison_vs_bf16": comparisons,
    }
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, indent=2))

    # -- table --------------------------------------------------------------------------
    print(f"\n=== Accuracy vs BF16 ({args.n_obs} obs, greedy; "
          f"baseline mean |action| = {baseline_mean_abs:.4f}) ===")
    hdr = f"  {'prec':6}{'mean MAE':>12}{'max abs err':>14}{'mean cosine':>14}{'min cosine':>13}"
    print(hdr)
    print("  " + "-" * (len(hdr) - 2))
    for p in PRECISIONS:
        c = comparisons.get(p)
        if not c:
            continue
        print(f"  {p:6}{c['mean_mae']:>12.5f}{c['max_ae']:>14.5f}"
              f"{c['mean_cosine']:>14.5f}{c['min_cosine']:>13.5f}")
    print(f"\nSaved -> {out}")


if __name__ == "__main__":
    main()
