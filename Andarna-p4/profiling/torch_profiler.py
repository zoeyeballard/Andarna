#!/usr/bin/env python
"""
torch_profiler.py — Phase 3: PyTorch Profiler pass over OpenVLA inference (BF16).

Wraps the inference loop in `torch.profiler.profile` with CPU+CUDA activity tracing, shape
recording, and memory tracking. Produces three artifacts:
  1. A Chrome trace JSON (open in chrome://tracing or https://ui.perfetto.dev) — gitignored.
  2. Top-20 operators by total GPU (CUDA) time.
  3. Top-20 operators by GPU memory usage.
The two tables are printed and saved to a text file under results/.

Where component_timer.py answered "which of the four stages is slow", this answers "which
CUDA kernels / aten ops inside those stages dominate" — the next zoom level for optimization.

Usage:
    python profiling/torch_profiler.py
    python profiling/torch_profiler.py --warmup 20 --profile-steps 20 \
        --trace traces/baseline_trace.json --output results/baseline/torch_profiler_top20.txt
"""
from __future__ import annotations

import argparse
import platform
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.profiler import ProfilerActivity, profile, record_function

DEFAULT_MODEL = "openvla/openvla-7b-finetuned-libero-object"
DEFAULT_UNNORM_KEY = "libero_object"
PROMPT_TEMPLATE = "In: What action should the robot take to {instruction}?\nOut:"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="PyTorch Profiler pass over OpenVLA inference.")
    p.add_argument("--model-id", default=DEFAULT_MODEL)
    p.add_argument("--unnorm-key", default=DEFAULT_UNNORM_KEY)
    p.add_argument("--warmup", type=int, default=20, help="Untimed warmup steps before profiling.")
    p.add_argument("--profile-steps", type=int, default=20, help="Steps recorded by the profiler.")
    p.add_argument("--instruction", default="pick up the object and place it in the basket")
    p.add_argument("--attn-impl", default="sdpa", choices=["flash_attention_2", "sdpa", "eager"])
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--row-limit", type=int, default=20)
    p.add_argument("--trace", default="traces/baseline_trace.json")
    p.add_argument("--output", default="results/baseline/torch_profiler_top20.txt")
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

    # Warm up outside the profiler so JIT/autotune/allocator effects don't pollute the trace.
    print(f"[warmup] {args.warmup} steps ...")
    with torch.inference_mode():
        for _ in range(args.warmup):
            run_step()
    torch.cuda.synchronize(device)

    print(f"[profile] recording {args.profile_steps} steps "
          "(activities=CPU+CUDA, record_shapes, profile_memory) ...")
    with torch.inference_mode(), profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        record_shapes=True,
        profile_memory=True,
        with_flops=True,
    ) as prof:
        for _ in range(args.profile_steps):
            with record_function("predict_action"):
                run_step()
            torch.cuda.synchronize(device)

    # 1) Chrome trace --------------------------------------------------------------------
    trace_path = Path(args.trace)
    trace_path.parent.mkdir(parents=True, exist_ok=True)
    prof.export_chrome_trace(str(trace_path))

    # 2) + 3) Top-20 tables --------------------------------------------------------------
    ka = prof.key_averages()
    table_gpu = ka.table(sort_by="cuda_time_total", row_limit=args.row_limit)
    table_mem = ka.table(sort_by="cuda_memory_usage", row_limit=args.row_limit)

    header = (
        f"OpenVLA PyTorch Profiler summary\n"
        f"  model      : {args.model_id}  (BF16, attn={used_impl})\n"
        f"  device     : {torch.cuda.get_device_name(device)}\n"
        f"  torch/py   : {torch.__version__} / {platform.python_version()}\n"
        f"  warmup     : {args.warmup} steps   profiled: {args.profile_steps} steps\n"
        f"  trace      : {trace_path}\n"
        f"  generated  : {datetime.now(timezone.utc).isoformat()}\n"
    )
    body = (
        f"{header}\n"
        f"{'='*100}\nTOP {args.row_limit} OPERATORS BY GPU TIME (sort_by=cuda_time_total)\n{'='*100}\n"
        f"{table_gpu}\n\n"
        f"{'='*100}\nTOP {args.row_limit} OPERATORS BY GPU MEMORY (sort_by=cuda_memory_usage)\n{'='*100}\n"
        f"{table_mem}\n"
    )

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(body)

    print("\n" + body)
    print(f"Saved tables  -> {out}")
    print(f"Saved trace   -> {trace_path}  (open in chrome://tracing or ui.perfetto.dev)")


if __name__ == "__main__":
    main()
