#!/usr/bin/env python
"""
batch_scaling_test.py — Phase 9: batch-size scaling behavior of OpenVLA on the A10G.

Robotics inference is batch=1, but batch scaling reveals how well the model's GEMMs use the GPU.
Two findings here:

  A. End-to-end action *generation* is hard-locked to batch=1. OpenVLA's modeling_prismatic.py
     asserts `input_ids.shape[0] == 1` in the cached-decode path ("Generation is only currently
     supported for batch size of 1!"). We probe this so the limit is documented, not assumed.

  B. The *prefill forward* (vision encoder + projector + LLM prefill — one forward, no decode loop)
     DOES accept batch B. We sweep B in {1,2,4,8} and report per-call latency, per-item latency,
     and throughput (items/s) + peak memory. This is the meaningful scaling signal: if per-item
     latency drops as B grows, batch=1 is under-utilizing the GPU (a deployment lever for
     multi-robot / parallel-sim serving, even though a single robot can't use it).

Usage:
    python optimization/batch_scaling_test.py
    python optimization/batch_scaling_test.py --batch-sizes 1 2 4 8 16
"""
from __future__ import annotations

import argparse
import json
import platform
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from PIL import Image

DEFAULT_MODEL = "openvla/openvla-7b-finetuned-libero-object"
DEFAULT_UNNORM_KEY = "libero_object"
PROMPT_TEMPLATE = "In: What action should the robot take to {instruction}?\nOut:"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="OpenVLA batch-size scaling test.")
    p.add_argument("--model-id", default=DEFAULT_MODEL)
    p.add_argument("--unnorm-key", default=DEFAULT_UNNORM_KEY)
    p.add_argument("--batch-sizes", type=int, nargs="+", default=[1, 2, 4, 8])
    p.add_argument("--warmup", type=int, default=10)
    p.add_argument("--iters", type=int, default=30)
    p.add_argument("--instruction", default="pick up the object and place it in the basket")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--output", default="results/optimization/batch_scaling.json")
    return p.parse_args()


def load_model(model_id: str, device: str):
    from transformers import AutoModelForVision2Seq, AutoProcessor
    processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
    model = AutoModelForVision2Seq.from_pretrained(
        model_id, attn_implementation="sdpa", torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True, trust_remote_code=True).to(device)
    model.eval()
    return processor, model


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

    base = processor(prompt, image).to(device, dtype=torch.bfloat16)

    def make_batch(b: int) -> dict:
        """Replicate the single observation B times (identical prompt+image, no padding needed)."""
        out = {}
        for k, v in base.items():
            if torch.is_tensor(v):
                out[k] = v.repeat(b, *([1] * (v.dim() - 1)))
            else:
                out[k] = v
        return out

    result = {
        "metadata": {
            "model_id": args.model_id, "precision": "bfloat16",
            "device_name": torch.cuda.get_device_name(device), "torch_version": torch.__version__,
            "python_version": platform.python_version(), "warmup": args.warmup, "iters": args.iters,
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        },
    }

    # --- A. Probe the batch-1 generation limit ----------------------------------------
    print("[probe] attempting batched action generation at B=2 ...")
    gen_probe = {}
    try:
        b2 = make_batch(2)
        with torch.inference_mode():
            model.predict_action(**b2, unnorm_key=unnorm_key, do_sample=False)
        gen_probe = {"batched_generation_supported": True}
        print("[probe] batched generation worked (unexpected).")
    except Exception as e:  # noqa: BLE001
        msg = str(e).splitlines()[0] if str(e) else type(e).__name__
        gen_probe = {"batched_generation_supported": False, "error": f"{type(e).__name__}: {msg}"}
        print(f"[probe] batched generation BLOCKED: {gen_probe['error']}")
    result["generation_batch_probe"] = gen_probe

    # --- B. Prefill-forward batch sweep -----------------------------------------------
    print(f"[sweep] prefill forward, batch sizes {args.batch_sizes} ...")
    rows = {}
    for b in args.batch_sizes:
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)
        try:
            batch = make_batch(b)
            fwd_kwargs = {k: v for k, v in batch.items() if torch.is_tensor(v)}
            fwd_kwargs["use_cache"] = False

            def run_fwd():
                return model(**fwd_kwargs)

            with torch.inference_mode():
                for _ in range(args.warmup):
                    run_fwd()
            torch.cuda.synchronize(device)

            lat = []
            with torch.inference_mode():
                for _ in range(args.iters):
                    s = torch.cuda.Event(enable_timing=True)
                    e = torch.cuda.Event(enable_timing=True)
                    s.record(); run_fwd(); e.record(); torch.cuda.synchronize(device)
                    lat.append(s.elapsed_time(e))
            arr = np.asarray(lat, dtype=np.float64)
            mean_ms = float(arr.mean())
            rows[b] = {
                "batch_size": b,
                "latency_per_call_ms": round(mean_ms, 3),
                "latency_per_item_ms": round(mean_ms / b, 3),
                "throughput_items_per_s": round(b * 1000.0 / mean_ms, 2),
                "peak_mem_mb": round(torch.cuda.max_memory_allocated(device) / 1e6, 1),
            }
            r = rows[b]
            print(f"  B={b}: {r['latency_per_call_ms']} ms/call | {r['latency_per_item_ms']} ms/item"
                  f" | {r['throughput_items_per_s']} items/s | {r['peak_mem_mb']} MB")
        except torch.cuda.OutOfMemoryError:
            rows[b] = {"batch_size": b, "status": "OOM"}
            print(f"  B={b}: OOM — stopping sweep.")
            torch.cuda.empty_cache()
            break
    result["prefill_forward_sweep"] = rows

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2))

    print("\n=== Batch scaling (prefill forward) ===")
    print(f"  {'B':>3}{'ms/call':>11}{'ms/item':>11}{'items/s':>11}{'peak MB':>11}")
    print("  " + "-" * 44)
    base_item = rows.get(1, {}).get("latency_per_item_ms")
    for b in args.batch_sizes:
        r = rows.get(b)
        if not r or r.get("status") == "OOM":
            if r:
                print(f"  {b:>3}{'OOM':>11}")
            continue
        print(f"  {b:>3}{r['latency_per_call_ms']:>11.2f}{r['latency_per_item_ms']:>11.2f}"
              f"{r['throughput_items_per_s']:>11.2f}{r['peak_mem_mb']:>11.0f}")
    if base_item:
        print(f"\n  per-item speedup at largest batch vs B=1: "
              f"shown by ms/item dropping from {base_item} ms")
    print(f"\n  NOTE: end-to-end action generation is batch-1-locked "
          f"(supported={gen_probe.get('batched_generation_supported')}).")
    print(f"\nSaved -> {out}")


if __name__ == "__main__":
    main()
