#!/usr/bin/env python
"""
torch_compile_test.py — Phase 8: does torch.compile(mode="reduce-overhead") help OpenVLA?

Measures, in one process:
  1. Uncompiled BF16 baseline (warmup + timed).
  2. torch.compile on the model's forward (the function generate() calls every decode step),
     with mode="reduce-overhead" (CUDA-graph capture + kernel fusion).
  3. First-call latency (which includes the one-time Dynamo trace + Inductor compile) reported
     SEPARATELY from steady-state latency.
  4. Whether compilation errored or fell back, and how many times Dynamo recompiled.

Why this is the interesting case: OpenVLA inference is `predict_action -> generate()`, an
autoregressive loop. Prefill (long sequence) and each decode step (1 new token, growing KV cache)
present *changing shapes* to the graph. reduce-overhead uses CUDA graphs, which dislike dynamic
shapes — so VLM generation commonly triggers recompiles, graph breaks, or cudagraph skips. We
record exactly what happens rather than assuming a speedup.

Usage:
    python optimization/torch_compile_test.py
    python optimization/torch_compile_test.py --baseline-iters 50 --steady-iters 30
"""
from __future__ import annotations

import argparse
import json
import platform
import traceback
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from PIL import Image

DEFAULT_MODEL = "openvla/openvla-7b-finetuned-libero-object"
DEFAULT_UNNORM_KEY = "libero_object"
PROMPT_TEMPLATE = "In: What action should the robot take to {instruction}?\nOut:"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="torch.compile reduce-overhead test for OpenVLA.")
    p.add_argument("--model-id", default=DEFAULT_MODEL)
    p.add_argument("--unnorm-key", default=DEFAULT_UNNORM_KEY)
    p.add_argument("--warmup", type=int, default=20)
    p.add_argument("--baseline-iters", type=int, default=50)
    p.add_argument("--steady-iters", type=int, default=30)
    p.add_argument("--instruction", default="pick up the object and place it in the basket")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--mode", default="reduce-overhead",
                   choices=["reduce-overhead", "default", "max-autotune"])
    p.add_argument("--output", default="results/optimization/torch_compile.json")
    return p.parse_args()


def load_model(model_id: str, device: str):
    from transformers import AutoModelForVision2Seq, AutoProcessor
    processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
    model = AutoModelForVision2Seq.from_pretrained(
        model_id, attn_implementation="sdpa", torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True, trust_remote_code=True).to(device)
    model.eval()
    return processor, model


def time_iters(run_step, n: int, device: str) -> list[float]:
    out = []
    with torch.inference_mode():
        for _ in range(n):
            s = torch.cuda.Event(enable_timing=True)
            e = torch.cuda.Event(enable_timing=True)
            s.record(); run_step(); e.record(); torch.cuda.synchronize(device)
            out.append(s.elapsed_time(e))
    return out


def stats(ms: list[float]) -> dict:
    a = np.asarray(ms, dtype=np.float64)
    return {"mean_ms": round(float(a.mean()), 3), "p50_ms": round(float(np.percentile(a, 50)), 3),
            "p95_ms": round(float(np.percentile(a, 95)), 3), "min_ms": round(float(a.min()), 3),
            "n": len(ms)}


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

    # --- 1) Uncompiled baseline -------------------------------------------------------
    print(f"[baseline] warmup {args.warmup}, time {args.baseline_iters} ...")
    with torch.inference_mode():
        for _ in range(args.warmup):
            run_step()
    torch.cuda.synchronize(device)
    baseline = stats(time_iters(run_step, args.baseline_iters, device))
    print(f"[baseline] mean {baseline['mean_ms']} ms")

    # --- 2) Apply torch.compile to the forward generate() calls each step -------------
    import torch._dynamo as dynamo
    dynamo.reset()
    dynamo.config.cache_size_limit = 64  # allow recompiles for the prefill/decode shape set
    counters = dynamo.utils.counters

    result = {
        "metadata": {
            "model_id": args.model_id, "precision": "bfloat16", "compile_mode": args.mode,
            "device_name": torch.cuda.get_device_name(device), "torch_version": torch.__version__,
            "python_version": platform.python_version(),
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        },
        "baseline_uncompiled": baseline,
        "compile": {},
    }

    print(f"[compile] wrapping model.forward with torch.compile(mode='{args.mode}') ...")
    model.forward = torch.compile(model.forward, mode=args.mode, dynamic=None)

    compile_error = None
    first_call_ms = None
    try:
        # First call triggers tracing + Inductor compilation (and CUDA-graph capture).
        torch.cuda.synchronize(device)
        t0 = torch.cuda.Event(enable_timing=True)
        t1 = torch.cuda.Event(enable_timing=True)
        # Use perf_counter for the (long, mostly-CPU) compile wall time.
        import time as _time
        wall0 = _time.perf_counter()
        with torch.inference_mode():
            run_step()
        torch.cuda.synchronize(device)
        first_call_ms = (_time.perf_counter() - wall0) * 1000.0
        print(f"[compile] first call (compile + run): {first_call_ms/1000:.1f} s")
    except Exception as e:  # noqa: BLE001 - we WANT to capture compile failures
        compile_error = f"{type(e).__name__}: {e}"
        print(f"[compile] FIRST-CALL ERROR: {compile_error}")
        result["compile"]["error"] = compile_error
        result["compile"]["traceback_tail"] = "".join(
            traceback.format_exc().splitlines(keepends=True)[-12:])

    if compile_error is None:
        # A few more calls to settle recompiles, then time steady state.
        with torch.inference_mode():
            for _ in range(5):
                run_step()
        torch.cuda.synchronize(device)
        steady = stats(time_iters(run_step, args.steady_iters, device))
        result["compile"].update({
            "first_call_s": round(first_call_ms / 1000.0, 2),
            "steady_state": steady,
            "speedup_vs_baseline": round(baseline["mean_ms"] / steady["mean_ms"], 3),
        })
        print(f"[compile] steady-state mean {steady['mean_ms']} ms "
              f"(baseline {baseline['mean_ms']} ms -> {result['compile']['speedup_vs_baseline']}x)")

    # Dynamo health: recompiles and graph breaks (key for the VLM-generation caveat).
    result["compile"]["dynamo"] = {
        "recompiles": int(counters["stats"].get("unique_graphs", 0)),
        "graph_breaks": int(sum(counters["graph_break"].values())) if "graph_break" in counters else 0,
        "frames_compiled": int(counters["stats"].get("calls_captured", 0)),
        "top_graph_break_reasons": dict(sorted(
            counters.get("graph_break", {}).items(), key=lambda kv: -kv[1])[:5]),
    }

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2))

    print("\n=== torch.compile summary ===")
    print(f"  baseline (eager)   : {baseline['mean_ms']} ms")
    if compile_error:
        print(f"  compiled           : ERRORED -> {compile_error}")
    else:
        print(f"  first call (compile): {result['compile']['first_call_s']} s")
        print(f"  compiled steady    : {result['compile']['steady_state']['mean_ms']} ms "
              f"({result['compile']['speedup_vs_baseline']}x vs baseline)")
    d = result["compile"]["dynamo"]
    print(f"  dynamo: {d['recompiles']} graphs, {d['graph_breaks']} graph breaks")
    print(f"\nSaved -> {out}")


if __name__ == "__main__":
    main()
