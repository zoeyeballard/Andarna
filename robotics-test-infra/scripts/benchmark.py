#!/usr/bin/env python
"""Performance benchmarking suite (5b).

Measures the timing/throughput profile of the test harness itself — the numbers that
decide whether SIL testing stays fast and cheap, and the ones an embedded engineer
reads as a WCET/latency budget:

  - per-step policy inference latency distribution (mean/p50/p95/p99/max)
  - policy-inference vs MuJoCo-step time breakdown (which dominates the loop?)
  - peak resident memory (RSS) during evaluation
  - episode throughput (episodes/minute)

Writes benchmark.json so a PR that, say, doubles inference time is caught as a number,
not a vibe.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import EvalConfig  # noqa: E402


def _stats(ms: list[float]) -> dict:
    a = np.asarray(ms, dtype=float)
    if a.size == 0:
        return {k: float("nan") for k in ("mean", "p50", "p95", "p99", "max")}
    return {
        "mean": float(a.mean()),
        "p50": float(np.percentile(a, 50)),
        "p95": float(np.percentile(a, 95)),
        "p99": float(np.percentile(a, 99)),
        "max": float(a.max()),
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Benchmark the evaluation harness.")
    ap.add_argument("--num_episodes", type=int, default=3)
    ap.add_argument("--max_episode_steps", type=int, default=200)
    ap.add_argument("--seed", type=int, default=100_000)
    ap.add_argument("--checkpoint_path", default=None)
    ap.add_argument("--output", default="artifacts/benchmark.json")
    args = ap.parse_args(argv)

    overrides = {"max_episode_steps": args.max_episode_steps, "video_capture_on_nonsuccess": False}
    if args.checkpoint_path:
        overrides["checkpoint_path"] = args.checkpoint_path
    config = EvalConfig.from_env(**overrides)

    import time

    try:
        import psutil

        proc = psutil.Process()
    except ImportError:
        proc = None

    from src.evaluator import PolicyEvaluator

    ev = PolicyEvaluator(config)
    ev.load()

    infer_ms: list[float] = []
    step_ms: list[float] = []
    peak_rss = 0
    start = time.perf_counter()
    for i in range(args.num_episodes):
        result, _ = ev._run_episode(
            seed=args.seed + i, episode_index=i, step_times_ms=step_ms
        )
        infer_ms.extend(result.inference_times_ms)
        if proc is not None:
            peak_rss = max(peak_rss, proc.memory_info().rss)
    wall = time.perf_counter() - start
    ev.close()

    infer = _stats(infer_ms)
    step = _stats(step_ms)
    report = {
        "num_episodes": args.num_episodes,
        "max_episode_steps": args.max_episode_steps,
        "checkpoint": ev._resolved_checkpoint,
        "inference_ms": infer,
        "env_step_ms": step,
        "loop_breakdown": {
            # total time spent in each, and which fraction of the (timed) loop it is
            "inference_total_s": sum(infer_ms) / 1000.0,
            "env_step_total_s": sum(step_ms) / 1000.0,
            "inference_fraction": (
                sum(infer_ms) / (sum(infer_ms) + sum(step_ms))
                if (infer_ms or step_ms)
                else float("nan")
            ),
        },
        "peak_rss_mb": round(peak_rss / 1e6, 1) if proc else None,
        "throughput_eps_per_min": args.num_episodes / wall * 60.0 if wall > 0 else 0.0,
        "wall_s": wall,
    }

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2) + "\n")

    print(json.dumps(report, indent=2))
    print(
        f"\n[benchmark] inference p95={infer['p95']:.2f}ms max={infer['max']:.0f}ms | "
        f"env-step mean={step['mean']:.2f}ms | "
        f"inference is {report['loop_breakdown']['inference_fraction'] * 100:.0f}% of loop | "
        f"peak RSS={report['peak_rss_mb']}MB | {report['throughput_eps_per_min']:.1f} ep/min"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
