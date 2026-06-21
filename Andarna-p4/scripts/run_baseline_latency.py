#!/usr/bin/env python
"""
run_baseline_latency.py — End-to-end BF16 inference latency baseline for OpenVLA.

Loads the pretrained `openvla/openvla-7b-finetuned-libero-object` policy in BF16, runs a
warm-up burst (to absorb CUDA context creation, autotuning, and allocator warm-up), then
times N steady-state inference steps with `torch.cuda.Event`. Reports mean / p50 / p95 / max
latency in milliseconds and writes a JSON summary.

Why CUDA events (not time.perf_counter): CUDA kernel launches are asynchronous. Wall-clock
around a `predict_action` call measures launch + Python overhead, not GPU execution. Event
timing records markers *on the stream* and reads elapsed GPU time after a synchronize — the
real device-side latency, which is what a robot's control loop actually waits on.

Usage:
    python scripts/run_baseline_latency.py
    python scripts/run_baseline_latency.py --warmup 20 --iters 100 \
        --output results/baseline/baseline_latency_bf16.json

Run on the EC2 A10G with the project venv active (see scripts/install_ec2.sh).
"""
from __future__ import annotations

import argparse
import json
import platform
import statistics
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from PIL import Image


DEFAULT_MODEL = "openvla/openvla-7b-finetuned-libero-object"
DEFAULT_UNNORM_KEY = "libero_object"
# OpenVLA's prompt template; the finetuned LIBERO checkpoints expect this exact format.
PROMPT_TEMPLATE = "In: What action should the robot take to {instruction}?\nOut:"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="OpenVLA BF16 baseline latency benchmark.")
    p.add_argument("--model-id", default=DEFAULT_MODEL, help="HF model id to load.")
    p.add_argument("--unnorm-key", default=DEFAULT_UNNORM_KEY,
                   help="Action de-normalization key (dataset name). Pass '' to let the model auto-select.")
    p.add_argument("--warmup", type=int, default=20, help="Warm-up steps (not timed).")
    p.add_argument("--iters", type=int, default=100, help="Timed inference steps.")
    p.add_argument("--instruction", default="pick up the object and place it in the basket",
                   help="Task instruction fed to the policy.")
    p.add_argument("--image", default=None,
                   help="Optional path to a 224x224-ish RGB image. If omitted, a synthetic image is used.")
    p.add_argument("--attn-impl", default="flash_attention_2",
                   choices=["flash_attention_2", "sdpa", "eager"],
                   help="Attention backend. Falls back to sdpa if flash-attn is unavailable.")
    p.add_argument("--device", default="cuda:0", help="CUDA device.")
    p.add_argument("--output", default="results/baseline/baseline_latency_bf16.json",
                   help="Where to write the JSON summary.")
    p.add_argument("--save-raw", action="store_true",
                   help="Include the full per-step latency array in the JSON.")
    return p.parse_args()


def build_observation(image_path: str | None, device: str):
    """Return a single fixed (image, instruction-ready) observation reused for every step.

    A fixed input isolates inference cost from input variation — we want the *latency
    distribution of the model*, not of the data pipeline.
    """
    if image_path:
        img = Image.open(image_path).convert("RGB")
    else:
        # Deterministic synthetic 224x224 image (seeded) — stands in for a camera frame.
        rng = np.random.default_rng(0)
        arr = rng.integers(0, 256, size=(224, 224, 3), dtype=np.uint8)
        img = Image.fromarray(arr, mode="RGB")
    return img


def load_model(model_id: str, attn_impl: str, device: str):
    from transformers import AutoModelForVision2Seq, AutoProcessor

    processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)

    def _load(impl: str):
        return AutoModelForVision2Seq.from_pretrained(
            model_id,
            attn_implementation=impl,
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
            trust_remote_code=True,
        ).to(device)

    try:
        model = _load(attn_impl)
        used_impl = attn_impl
    except (ImportError, RuntimeError, ValueError) as e:
        if attn_impl == "flash_attention_2":
            print(f"[warn] flash_attention_2 unavailable ({e}); falling back to sdpa.")
            model = _load("sdpa")
            used_impl = "sdpa"
        else:
            raise
    model.eval()
    return processor, model, used_impl


def percentiles(samples_ms: list[float]) -> dict:
    arr = np.asarray(samples_ms, dtype=np.float64)
    return {
        "mean_ms": float(arr.mean()),
        "p50_ms": float(np.percentile(arr, 50)),
        "p95_ms": float(np.percentile(arr, 95)),
        "max_ms": float(arr.max()),
        "min_ms": float(arr.min()),
        "std_ms": float(arr.std(ddof=1)) if len(arr) > 1 else 0.0,
    }


def main() -> None:
    args = parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("CUDA is not available — this benchmark requires a GPU.")
    device = args.device
    torch.cuda.set_device(device)

    print(f"[load] {args.model_id} (BF16) on {torch.cuda.get_device_name(device)} ...")
    t0 = time.perf_counter()
    processor, model, used_impl = load_model(args.model_id, args.attn_impl, device)
    load_seconds = time.perf_counter() - t0
    print(f"[load] done in {load_seconds:.1f}s (attn={used_impl})")

    image = build_observation(args.image, device)
    prompt = PROMPT_TEMPLATE.format(instruction=args.instruction)
    unnorm_key = args.unnorm_key or None

    def run_step():
        """One full inference: image+prompt -> 7-DoF action. Mirrors a single control tick."""
        inputs = processor(prompt, image).to(device, dtype=torch.bfloat16)
        return model.predict_action(**inputs, unnorm_key=unnorm_key, do_sample=False)

    # --- Warm-up (not timed): CUDA context, kernel autotuning, allocator growth ---------
    print(f"[warmup] {args.warmup} steps ...")
    with torch.inference_mode():
        for _ in range(args.warmup):
            run_step()
    torch.cuda.synchronize(device)

    # --- Timed loop: one CUDA event pair per step ---------------------------------------
    print(f"[bench] timing {args.iters} steps ...")
    latencies_ms: list[float] = []
    with torch.inference_mode():
        for _ in range(args.iters):
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            run_step()
            end.record()
            torch.cuda.synchronize(device)
            latencies_ms.append(start.elapsed_time(end))  # ms, GPU-side

    stats = percentiles(latencies_ms)
    peak_mem_mb = torch.cuda.max_memory_allocated(device) / 1e6

    summary = {
        "metadata": {
            "model_id": args.model_id,
            "precision": "bfloat16",
            "attn_implementation": used_impl,
            "unnorm_key": unnorm_key,
            "instruction": args.instruction,
            "synthetic_image": args.image is None,
            "warmup_steps": args.warmup,
            "timed_steps": args.iters,
            "device_name": torch.cuda.get_device_name(device),
            "torch_version": torch.__version__,
            "cuda_version": torch.version.cuda,
            "python_version": platform.python_version(),
            "model_load_seconds": round(load_seconds, 2),
            "peak_gpu_mem_mb": round(peak_mem_mb, 1),
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        },
        "latency": {k: round(v, 3) for k, v in stats.items()},
        # Embedded framing: a robot control loop at f Hz has a 1000/f ms budget per tick.
        "achieved_control_freq_hz_mean": round(1000.0 / stats["mean_ms"], 2),
    }
    if args.save_raw:
        summary["latency"]["raw_ms"] = [round(x, 3) for x in latencies_ms]

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, indent=2))

    print("\n=== BF16 baseline latency ===")
    print(f"  mean  : {stats['mean_ms']:.2f} ms   ({summary['achieved_control_freq_hz_mean']} Hz)")
    print(f"  p50   : {stats['p50_ms']:.2f} ms")
    print(f"  p95   : {stats['p95_ms']:.2f} ms")
    print(f"  max   : {stats['max_ms']:.2f} ms")
    print(f"  peak GPU mem: {peak_mem_mb:.0f} MB")
    print(f"\nSaved -> {out}")


if __name__ == "__main__":
    main()
