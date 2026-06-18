#!/usr/bin/env python
"""Determinism validation (5a): the same seed must produce the same trajectory.

MuJoCo + a deterministic policy are reproducible given a seed. If two runs of the same
seeded episode diverge, something is injecting nondeterminism (an unseeded RNG, a race,
nondeterministic kernels) — which would turn every regression signal into noise. This
fails loudly when that happens.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import EvalConfig  # noqa: E402


def run_twice(config: EvalConfig, seed: int, max_steps: int):
    from src.evaluator import PolicyEvaluator

    ev = PolicyEvaluator(config)
    ev.load()
    trace_a: list = []
    trace_b: list = []
    ev._run_episode(seed=seed, episode_index=0, action_trace=trace_a)
    ev._run_episode(seed=seed, episode_index=0, action_trace=trace_b)
    ev.close()
    return trace_a, trace_b


def compare_traces(a: list, b: list) -> tuple[bool, str]:
    if len(a) != len(b):
        return False, f"length mismatch: {len(a)} vs {len(b)} steps"
    for i, (x, y) in enumerate(zip(a, b, strict=True)):
        if not np.array_equal(x, y):
            return False, f"action mismatch at step {i}: max|Δ|={np.max(np.abs(x - y)):.3e}"
    return True, f"identical across {len(a)} steps"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Verify seeded determinism of the rollout.")
    ap.add_argument("--seed", type=int, default=100_000)
    ap.add_argument("--max_steps", type=int, default=60)
    ap.add_argument("--checkpoint_path", default=None)
    args = ap.parse_args(argv)

    overrides = {"max_episode_steps": args.max_steps, "video_capture_on_nonsuccess": False}
    if args.checkpoint_path:
        overrides["checkpoint_path"] = args.checkpoint_path
    config = EvalConfig.from_env(**overrides)

    print(f"[determinism] running seed {args.seed} twice for {args.max_steps} steps...", flush=True)
    a, b = run_twice(config, args.seed, args.max_steps)
    ok, detail = compare_traces(a, b)
    print(f"[determinism] {'PASS' if ok else 'FAIL'} — {detail}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
