#!/usr/bin/env python
"""
memory_profiler.py — Phase 10: GPU memory-stability / leak check over a long run.

Runs OpenVLA inference for N iterations (default 500), logging GPU memory every `--log-every`
steps, then verifies that **peak memory plateaus** (doesn't grow) — i.e. no leak across a long
control session. Saves a memory-over-time plot to figures/ and the raw samples to results/.

Why this matters for deployment: a robot runs its policy in a loop for hours. If GPU memory
creeps up (un-freed activations, KV-cache not released, fragmentation), it eventually OOMs mid-
task. A flat `max_memory_allocated` after warmup is the evidence that the inference loop is
allocation-stable.

Usage:
    python profiling/memory_profiler.py --iters 500 --log-every 50
"""
from __future__ import annotations

import argparse
import json
import platform
from datetime import datetime, timezone
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # headless — no display on the EC2 box
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image

DEFAULT_MODEL = "openvla/openvla-7b-finetuned-libero-object"
DEFAULT_UNNORM_KEY = "libero_object"
PROMPT_TEMPLATE = "In: What action should the robot take to {instruction}?\nOut:"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="OpenVLA GPU memory-stability check.")
    p.add_argument("--model-id", default=DEFAULT_MODEL)
    p.add_argument("--unnorm-key", default=DEFAULT_UNNORM_KEY)
    p.add_argument("--iters", type=int, default=500)
    p.add_argument("--warmup", type=int, default=20)
    p.add_argument("--log-every", type=int, default=50)
    p.add_argument("--instruction", default="pick up the object and place it in the basket")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--output", default="results/memory/memory_stability.json")
    p.add_argument("--figure", default="figures/memory_stability.png")
    return p.parse_args()


def load_model(model_id: str, device: str):
    from transformers import AutoModelForVision2Seq, AutoProcessor
    processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
    model = AutoModelForVision2Seq.from_pretrained(
        model_id, attn_implementation="sdpa", torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True, trust_remote_code=True).to(device)
    model.eval()
    return processor, model


def mb(x: int) -> float:
    return round(x / 1e6, 1)


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("CUDA not available.")
    device = args.device
    torch.cuda.set_device(device)

    print(f"[load] {args.model_id} (BF16)")
    processor, model = load_model(args.model_id, device)
    rng = np.random.default_rng(0)
    image = Image.fromarray(rng.integers(0, 256, (224, 224, 3), dtype=np.uint8), "RGB")
    prompt = PROMPT_TEMPLATE.format(instruction=args.instruction)
    unnorm_key = args.unnorm_key or None

    def run_step():
        inputs = processor(prompt, image).to(device, dtype=torch.bfloat16)
        return model.predict_action(**inputs, unnorm_key=unnorm_key, do_sample=False)

    print(f"[warmup] {args.warmup} steps ...")
    with torch.inference_mode():
        for _ in range(args.warmup):
            run_step()
    torch.cuda.synchronize(device)

    # Baseline after warmup: model weights + steady-state working set.
    samples = []  # (iter, allocated_mb, reserved_mb, peak_allocated_mb)

    def log(step: int):
        torch.cuda.synchronize(device)
        samples.append((step,
                        mb(torch.cuda.memory_allocated(device)),
                        mb(torch.cuda.memory_reserved(device)),
                        mb(torch.cuda.max_memory_allocated(device))))

    print(f"[run] {args.iters} iters, logging every {args.log_every} ...")
    log(0)
    with torch.inference_mode():
        for i in range(1, args.iters + 1):
            run_step()
            if i % args.log_every == 0:
                log(i)
                s = samples[-1]
                print(f"  step {s[0]:4d}: allocated {s[1]} MB | reserved {s[2]} MB | peak {s[3]} MB")

    steps = [s[0] for s in samples]
    alloc = [s[1] for s in samples]
    reserved = [s[2] for s in samples]
    peak = [s[3] for s in samples]

    # --- leak verdict: peak plateau + allocated drift after the first logged point ----
    post = [(st, a) for st, a in zip(steps, alloc) if st >= args.log_every]
    drift_mb = round(post[-1][1] - post[0][1], 2) if len(post) >= 2 else 0.0
    span = (post[-1][0] - post[0][0]) if len(post) >= 2 else 1
    drift_per_100 = round(drift_mb / span * 100, 4) if span else 0.0
    peak_plateau = (max(peak) - min(p for st, _, _, p in
                                    [(s[0], s[1], s[2], s[3]) for s in samples] if st >= args.log_every)) \
        if len(post) >= 2 else 0.0
    peak_growth_mb = round(max(peak) - peak[1] if len(peak) > 1 else 0.0, 2)
    leak_free = abs(drift_mb) < 50.0 and peak_growth_mb < 50.0  # <50 MB over the run = stable

    result = {
        "metadata": {
            "model_id": args.model_id, "precision": "bfloat16", "iters": args.iters,
            "warmup": args.warmup, "log_every": args.log_every,
            "device_name": torch.cuda.get_device_name(device), "torch_version": torch.__version__,
            "python_version": platform.python_version(),
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        },
        "samples": [{"iter": s[0], "allocated_mb": s[1], "reserved_mb": s[2], "peak_mb": s[3]}
                    for s in samples],
        "verdict": {
            "allocated_drift_mb_over_run": drift_mb,
            "allocated_drift_mb_per_100_iters": drift_per_100,
            "peak_growth_mb_after_first_log": peak_growth_mb,
            "leak_free": bool(leak_free),
        },
    }
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2))

    # --- plot --------------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(steps, peak, "-o", label="peak allocated", color="#c0392b", linewidth=2, markersize=3)
    ax.plot(steps, reserved, "-s", label="reserved", color="#2980b9", linewidth=1.5, markersize=3)
    ax.plot(steps, alloc, "-^", label="allocated", color="#27ae60", linewidth=1.5, markersize=3)
    ax.set_xlabel("inference iteration")
    ax.set_ylabel("GPU memory (MB)")
    ax.set_title(f"OpenVLA BF16 GPU memory over {args.iters} iters (A10G)\n"
                 f"drift {drift_mb:+.1f} MB · peak growth {peak_growth_mb:+.1f} MB · "
                 f"{'LEAK-FREE' if leak_free else 'GROWTH DETECTED'}")
    ax.set_ylim(0, max(reserved) * 1.15)
    ax.legend(loc="lower right")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    figp = Path(args.figure)
    figp.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(figp, dpi=120)

    print("\n=== Memory stability verdict ===")
    print(f"  allocated drift over run : {drift_mb:+.1f} MB ({drift_per_100:+.3f} MB / 100 iters)")
    print(f"  peak growth after warmup : {peak_growth_mb:+.1f} MB")
    print(f"  => {'LEAK-FREE (peak plateaus)' if leak_free else 'POSSIBLE GROWTH — investigate'}")
    print(f"\nSaved data -> {out}\nSaved plot -> {figp}")


if __name__ == "__main__":
    main()
