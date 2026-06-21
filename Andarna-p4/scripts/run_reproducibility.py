#!/usr/bin/env python
"""
run_reproducibility.py — Phase 11: cross-run reproducibility of the latency benchmark.

Runs the full BF16 latency benchmark (20 warmup + 100 timed steps) FIVE separate times, each in
its own fresh process (fresh CUDA context + model load — so the spread captures real run-to-run
variation: clock/thermal state, context init, allocator layout), then reports the coefficient of
variation (CV = std/mean) of the five per-run mean latencies.

A CV under ~5% means the benchmark is reproducible and our single-run numbers are trustworthy. If
it's above 5%, the script dumps GPU clock/thermal/process state to point at the likely cause
(thermal throttling, clock drift, a competing process, or missing persistence mode).

Reuses scripts/run_baseline_latency.py as the per-run worker (no duplicate timing logic).

Usage:
    python scripts/run_reproducibility.py --runs 5 --iters 100
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
BASELINE = HERE / "run_baseline_latency.py"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Cross-run reproducibility / CV of the latency benchmark.")
    p.add_argument("--runs", type=int, default=5)
    p.add_argument("--warmup", type=int, default=20)
    p.add_argument("--iters", type=int, default=100)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--cv-threshold-pct", type=float, default=5.0)
    p.add_argument("--output", default="results/baseline/reproducibility.json")
    return p.parse_args()


def gpu_state_dump() -> str:
    """Snapshot clocks / temp / power / other processes — only used if CV is high."""
    try:
        q = subprocess.run(
            ["nvidia-smi",
             "--query-gpu=clocks.sm,clocks.max.sm,temperature.gpu,power.draw,power.limit,"
             "utilization.gpu,clocks_throttle_reasons.active",
             "--format=csv,noheader"],
            capture_output=True, text=True, timeout=15)
        procs = subprocess.run(
            ["nvidia-smi", "--query-compute-apps=pid,used_memory", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=15)
        return f"gpu: {q.stdout.strip()}\ncompute-procs: {procs.stdout.strip() or '(none)'}"
    except Exception as e:  # noqa: BLE001
        return f"(nvidia-smi unavailable: {e})"


def main() -> None:
    args = parse_args()
    frag_dir = Path(args.output).parent / "_repro_runs"
    frag_dir.mkdir(parents=True, exist_ok=True)

    runs = []
    for i in range(1, args.runs + 1):
        frag = frag_dir / f"run{i}.json"
        if frag.exists():
            frag.unlink()
        print(f"[repro] run {i}/{args.runs} (fresh process) ...", flush=True)
        cmd = [sys.executable, str(BASELINE),
               "--warmup", str(args.warmup), "--iters", str(args.iters),
               "--device", args.device, "--output", str(frag)]
        proc = subprocess.run(cmd, capture_output=True, text=True)
        if proc.returncode != 0 or not frag.exists():
            tail = "\n".join(proc.stderr.strip().splitlines()[-6:])
            raise SystemExit(f"run {i} failed (rc={proc.returncode}):\n{tail}")
        data = json.loads(frag.read_text())
        lat = data["latency"]
        runs.append({"run": i, "mean_ms": lat["mean_ms"], "p50_ms": lat["p50_ms"],
                     "p95_ms": lat["p95_ms"], "max_ms": lat["max_ms"]})
        print(f"          mean {lat['mean_ms']} ms | p95 {lat['p95_ms']} ms", flush=True)

    means = np.asarray([r["mean_ms"] for r in runs], dtype=np.float64)
    grand_mean = float(means.mean())
    across_std = float(means.std(ddof=1))
    cv_pct = float(across_std / grand_mean * 100.0)
    p95s = np.asarray([r["p95_ms"] for r in runs], dtype=np.float64)
    cv_p95_pct = float(p95s.std(ddof=1) / p95s.mean() * 100.0)

    high = cv_pct > args.cv_threshold_pct
    result = {
        "metadata": {
            "runs": args.runs, "warmup": args.warmup, "iters": args.iters,
            "device": args.device, "cv_threshold_pct": args.cv_threshold_pct,
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        },
        "per_run": runs,
        "across_run_mean_ms": round(grand_mean, 3),
        "across_run_std_ms": round(across_std, 4),
        "cv_pct_of_mean_latency": round(cv_pct, 4),
        "cv_pct_of_p95_latency": round(cv_p95_pct, 4),
        "reproducible": (not high),
    }
    if high:
        result["gpu_state_when_high_cv"] = gpu_state_dump()

    out = Path(args.output)
    out.write_text(json.dumps(result, indent=2))

    print("\n=== Reproducibility (5 fresh runs, 100 iters each) ===")
    for r in runs:
        print(f"  run {r['run']}: mean {r['mean_ms']:.2f} ms")
    print(f"  across-run mean : {grand_mean:.2f} ms")
    print(f"  across-run std  : {across_std:.3f} ms")
    print(f"  CV (mean lat)   : {cv_pct:.3f}%   [threshold {args.cv_threshold_pct}%]")
    print(f"  CV (p95 lat)    : {cv_p95_pct:.3f}%")
    print(f"  => {'REPRODUCIBLE (CV under threshold)' if not high else 'HIGH VARIANCE — see gpu_state'}")
    if high:
        print("\n[investigate] GPU state:\n" + result["gpu_state_when_high_cv"])
    print(f"\nSaved -> {out}")


if __name__ == "__main__":
    main()
