#!/usr/bin/env python
"""
precision_runner.py — Phase 5: precision sweep for OpenVLA (BF16 / FP16 / INT8 / INT4).

For each precision level we measure, on the A10G:
  - model load time (from_pretrained -> on GPU),
  - end-to-end inference latency over 100 iterations after warmup (torch.cuda.Event),
  - peak GPU memory (allocated + reserved).

Why a subprocess per precision: bitsandbytes-quantized models cannot be `.to()`-moved and
the CUDA caching allocator retains freed blocks, so loading four models in one process gives
dirty peak-memory and load-time numbers. A fresh process per precision = a fresh CUDA context
= clean, comparable measurements. The orchestrator (default mode) spawns the workers; each
worker writes a JSON fragment that the orchestrator aggregates into the comparison table.

Usage:
    python quantization/precision_runner.py            # run the full sweep
    python quantization/precision_runner.py --only int4 int8
    # (internal) single-precision worker:
    python quantization/precision_runner.py --precision int8 --fragment-out /tmp/frag.json
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from PIL import Image

DEFAULT_MODEL = "openvla/openvla-7b-finetuned-libero-object"
DEFAULT_UNNORM_KEY = "libero_object"
PROMPT_TEMPLATE = "In: What action should the robot take to {instruction}?\nOut:"
PRECISIONS = ["bf16", "fp16", "int8", "int4"]

# Activation/compute dtype fed to the processor for each precision.
INPUT_DTYPE = {
    "bf16": torch.bfloat16,
    "fp16": torch.float16,
    "int8": torch.float16,   # bnb int8 computes in fp16
    "int4": torch.bfloat16,  # nf4 compute dtype = bf16
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="OpenVLA precision sweep (latency / memory / load).")
    p.add_argument("--model-id", default=DEFAULT_MODEL)
    p.add_argument("--unnorm-key", default=DEFAULT_UNNORM_KEY)
    p.add_argument("--warmup", type=int, default=20)
    p.add_argument("--iters", type=int, default=100)
    p.add_argument("--instruction", default="pick up the object and place it in the basket")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--only", nargs="+", choices=PRECISIONS, default=PRECISIONS,
                   help="Subset of precisions to run (default: all four).")
    p.add_argument("--output", default="results/quantization/precision_sweep.json")
    # Worker-mode flags (internal):
    p.add_argument("--precision", choices=PRECISIONS, help="Run a single precision (worker mode).")
    p.add_argument("--fragment-out", help="Worker: path to write this precision's JSON fragment.")
    return p.parse_args()


# --------------------------------------------------------------------------- model loading
def load_model(model_id: str, precision: str, device: str):
    from transformers import AutoModelForVision2Seq, AutoProcessor, BitsAndBytesConfig

    processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
    common = dict(trust_remote_code=True, low_cpu_mem_usage=True, attn_implementation="sdpa")

    if precision == "bf16":
        # Non-quantized: load then move to GPU explicitly.
        model = AutoModelForVision2Seq.from_pretrained(
            model_id, torch_dtype=torch.bfloat16, **common).to(device)
    elif precision == "fp16":
        model = AutoModelForVision2Seq.from_pretrained(
            model_id, torch_dtype=torch.float16, **common).to(device)
    elif precision == "int8":
        # bitsandbytes models must NOT be .to()-moved and must NOT use device_map
        # (device_map triggers accelerate dispatch_model -> .to(), which bnb forbids).
        # transformers places the quantized weights on the active CUDA device automatically.
        qc = BitsAndBytesConfig(load_in_8bit=True)
        model = AutoModelForVision2Seq.from_pretrained(
            model_id, quantization_config=qc, torch_dtype=torch.float16, **common)
    elif precision == "int4":
        qc = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True)
        model = AutoModelForVision2Seq.from_pretrained(
            model_id, quantization_config=qc, torch_dtype=torch.bfloat16, **common)
    else:
        raise ValueError(f"unknown precision {precision}")

    model.eval()
    return processor, model


def percentiles(samples_ms: list[float]) -> dict:
    arr = np.asarray(samples_ms, dtype=np.float64)
    return {
        "mean_ms": float(arr.mean()), "p50_ms": float(np.percentile(arr, 50)),
        "p95_ms": float(np.percentile(arr, 95)), "max_ms": float(arr.max()),
        "min_ms": float(arr.min()), "std_ms": float(arr.std(ddof=1)) if len(arr) > 1 else 0.0,
    }


# --------------------------------------------------------------------------- worker
def run_one_precision(args) -> dict:
    """Load one precision, benchmark it, return a result dict. Runs in its own process."""
    device = args.device
    precision = args.precision
    torch.cuda.set_device(device)
    torch.cuda.reset_peak_memory_stats(device)
    torch.cuda.empty_cache()

    t0 = time.perf_counter()
    processor, model = load_model(args.model_id, precision, device)
    torch.cuda.synchronize(device)
    load_seconds = time.perf_counter() - t0

    rng = np.random.default_rng(0)
    image = Image.fromarray(rng.integers(0, 256, (224, 224, 3), dtype=np.uint8), "RGB")
    prompt = PROMPT_TEMPLATE.format(instruction=args.instruction)
    unnorm_key = args.unnorm_key or None
    in_dtype = INPUT_DTYPE[precision]

    def run_step():
        inputs = processor(prompt, image).to(device, dtype=in_dtype)
        return model.predict_action(**inputs, unnorm_key=unnorm_key, do_sample=False)

    with torch.inference_mode():
        for _ in range(args.warmup):
            run_step()
    torch.cuda.synchronize(device)

    latencies_ms: list[float] = []
    with torch.inference_mode():
        for _ in range(args.iters):
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            run_step()
            end.record()
            torch.cuda.synchronize(device)
            latencies_ms.append(start.elapsed_time(end))

    return {
        "status": "ok",
        "precision": precision,
        "input_dtype": str(in_dtype).replace("torch.", ""),
        "load_seconds": round(load_seconds, 2),
        "latency": {k: round(v, 3) for k, v in percentiles(latencies_ms).items()},
        "peak_mem_allocated_mb": round(torch.cuda.max_memory_allocated(device) / 1e6, 1),
        "peak_mem_reserved_mb": round(torch.cuda.max_memory_reserved(device) / 1e6, 1),
    }


# --------------------------------------------------------------------------- orchestrator
def run_worker_subprocess(precision: str, args, frag_path: Path) -> dict:
    cmd = [
        sys.executable, os.path.abspath(__file__),
        "--precision", precision, "--fragment-out", str(frag_path),
        "--model-id", args.model_id, "--unnorm-key", args.unnorm_key,
        "--warmup", str(args.warmup), "--iters", str(args.iters),
        "--instruction", args.instruction, "--device", args.device,
    ]
    print(f"\n[sweep] === {precision.upper()} === (subprocess for clean CUDA context)")
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0 or not frag_path.exists():
        tail = "\n".join(proc.stderr.strip().splitlines()[-8:])
        print(f"[sweep] {precision} FAILED (rc={proc.returncode}). stderr tail:\n{tail}")
        return {"status": "failed", "precision": precision,
                "returncode": proc.returncode, "stderr_tail": tail}
    result = json.loads(frag_path.read_text())
    lat = result["latency"]["mean_ms"]
    print(f"[sweep] {precision}: load {result['load_seconds']}s | mean {lat} ms | "
          f"peak {result['peak_mem_allocated_mb']} MB")
    return result


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is not available — this benchmark requires a GPU.")

    # Worker mode: run a single precision and write the fragment.
    if args.precision:
        result = run_one_precision(args)
        Path(args.fragment_out).write_text(json.dumps(result))
        return

    # Orchestrator mode: spawn one worker per precision.
    device_name = torch.cuda.get_device_name(args.device)
    frag_dir = Path(args.output).parent / "_fragments"
    frag_dir.mkdir(parents=True, exist_ok=True)

    results: dict[str, dict] = {}
    for precision in args.only:
        frag = frag_dir / f"{precision}.json"
        if frag.exists():
            frag.unlink()
        results[precision] = run_worker_subprocess(precision, args, frag)

    summary = {
        "metadata": {
            "model_id": args.model_id, "device_name": device_name,
            "torch_version": torch.__version__, "python_version": platform.python_version(),
            "warmup_steps": args.warmup, "timed_steps": args.iters,
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        },
        "results": results,
    }
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, indent=2))

    # -- comparison table ---------------------------------------------------------------
    ok = {p: r for p, r in results.items() if r.get("status") == "ok"}
    base = ok.get("bf16")
    print("\n=== OpenVLA precision sweep (A10G, "
          f"{args.iters} timed iters) ===")
    hdr = f"  {'prec':5}{'load s':>9}{'mean ms':>10}{'p95 ms':>9}{'peak MB':>10}{'vs bf16 lat':>13}{'vs bf16 mem':>13}"
    print(hdr)
    print("  " + "-" * (len(hdr) - 2))
    for p in PRECISIONS:
        r = results.get(p)
        if not r or r.get("status") != "ok":
            if r:
                print(f"  {p:5}{'FAILED':>9}")
            continue
        lat = r["latency"]["mean_ms"]
        mem = r["peak_mem_allocated_mb"]
        lat_rel = f"{lat / base['latency']['mean_ms']:.2f}x" if base else "-"
        mem_rel = f"{mem / base['peak_mem_allocated_mb']:.2f}x" if base else "-"
        print(f"  {p:5}{r['load_seconds']:>9.1f}{lat:>10.2f}{r['latency']['p95_ms']:>9.2f}"
              f"{mem:>10.0f}{lat_rel:>13}{mem_rel:>13}")
    print(f"\nSaved -> {out}")


if __name__ == "__main__":
    main()
