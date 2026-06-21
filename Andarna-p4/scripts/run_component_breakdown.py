#!/usr/bin/env python
"""
run_component_breakdown.py — Phase 2: per-stage latency breakdown for OpenVLA (BF16).

Times the four inference stages (vision encoder, MLP projector, LLM prefill, LLM decode)
with CUDA-event hooks over 100 iterations and reports each stage's mean latency and its
percentage of total inference time. Also brackets the whole `predict_action` call to surface
"unaccounted" overhead (tokenization, embedding merge, sampling, action de-normalization).

Usage:
    python scripts/run_component_breakdown.py
    python scripts/run_component_breakdown.py --warmup 20 --iters 100 \
        --output results/baseline/component_breakdown_bf16.json
"""
from __future__ import annotations

import argparse
import json
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from PIL import Image

# Make the repo root importable so `profiling/` resolves regardless of CWD.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from profiling.component_timer import STAGES, ComponentTimer  # noqa: E402

DEFAULT_MODEL = "openvla/openvla-7b-finetuned-libero-object"
DEFAULT_UNNORM_KEY = "libero_object"
PROMPT_TEMPLATE = "In: What action should the robot take to {instruction}?\nOut:"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="OpenVLA per-stage latency breakdown (BF16).")
    p.add_argument("--model-id", default=DEFAULT_MODEL)
    p.add_argument("--unnorm-key", default=DEFAULT_UNNORM_KEY)
    p.add_argument("--warmup", type=int, default=20)
    p.add_argument("--iters", type=int, default=100)
    p.add_argument("--instruction", default="pick up the object and place it in the basket")
    p.add_argument("--attn-impl", default="sdpa", choices=["flash_attention_2", "sdpa", "eager"])
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--output", default="results/baseline/component_breakdown_bf16.json")
    return p.parse_args()


def load_model(model_id: str, attn_impl: str, device: str):
    from transformers import AutoModelForVision2Seq, AutoProcessor

    processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)

    def _load(impl):
        return AutoModelForVision2Seq.from_pretrained(
            model_id, attn_implementation=impl, torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True, trust_remote_code=True,
        ).to(device)

    try:
        model, used = _load(attn_impl), attn_impl
    except (ImportError, RuntimeError, ValueError) as e:
        if attn_impl == "flash_attention_2":
            print(f"[warn] flash_attention_2 unavailable ({e}); falling back to sdpa.")
            model, used = _load("sdpa"), "sdpa"
        else:
            raise
    model.eval()
    return processor, model, used


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is not available — this benchmark requires a GPU.")
    device = args.device
    torch.cuda.set_device(device)

    print(f"[load] {args.model_id} (BF16) on {torch.cuda.get_device_name(device)} ...")
    processor, model, used_impl = load_model(args.model_id, args.attn_impl, device)
    print(f"[load] done (attn={used_impl})")

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

    print(f"[bench] timing {args.iters} steps with per-stage hooks ...")
    timer = ComponentTimer(model)
    e2e_ms: list[float] = []
    with timer.attached(), torch.inference_mode():
        for _ in range(args.iters):
            timer.start_iter()
            total_start = torch.cuda.Event(enable_timing=True)
            total_end = torch.cuda.Event(enable_timing=True)
            total_start.record()
            run_step()
            total_end.record()
            timer.end_iter()                       # synchronizes
            e2e_ms.append(total_start.elapsed_time(total_end))

    summary = timer.summarize()
    mean_e2e = float(np.mean(e2e_ms))
    summed_stages = summary["summed_stage_mean_ms"]
    overhead_ms = mean_e2e - summed_stages

    result = {
        "metadata": {
            "model_id": args.model_id,
            "precision": "bfloat16",
            "attn_implementation": used_impl,
            "warmup_steps": args.warmup,
            "timed_steps": args.iters,
            "mean_decode_steps": round(summary["mean_decode_steps"], 2),
            "device_name": torch.cuda.get_device_name(device),
            "torch_version": torch.__version__,
            "python_version": platform.python_version(),
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        },
        "end_to_end_mean_ms": round(mean_e2e, 3),
        "summed_stage_mean_ms": round(summed_stages, 3),
        "unaccounted_overhead_ms": round(overhead_ms, 3),
        "unaccounted_overhead_pct": round(overhead_ms / mean_e2e * 100.0, 2) if mean_e2e else 0.0,
        "stages": {
            s: {k: round(v, 3) for k, v in summary["stages"][s].items()} for s in STAGES
        },
    }

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2))

    # -- pretty table -------------------------------------------------------------------
    print("\n=== OpenVLA BF16 per-stage breakdown (mean over "
          f"{args.iters} iters, {summary['mean_decode_steps']:.1f} decode steps) ===")
    label = {"vision": "Vision Encoder", "projector": "MLP Projector",
             "prefill": "LLM Prefill", "decode": "LLM Decode"}
    print(f"  {'stage':16}{'mean ms':>10}{'p95 ms':>10}{'% of total':>12}")
    for s in STAGES:
        st = result["stages"][s]
        print(f"  {label[s]:16}{st['mean_ms']:>10.2f}{st['p95_ms']:>10.2f}{st['pct_of_total']:>11.1f}%")
    print(f"  {'-'*48}")
    print(f"  {'summed stages':16}{summed_stages:>10.2f}{'':>10}{'':>12}")
    print(f"  {'overhead (other)':16}{overhead_ms:>10.2f}{'':>10}"
          f"{result['unaccounted_overhead_pct']:>11.1f}%")
    print(f"  {'end-to-end':16}{mean_e2e:>10.2f}")
    print(f"\nSaved -> {out}")


if __name__ == "__main__":
    main()
