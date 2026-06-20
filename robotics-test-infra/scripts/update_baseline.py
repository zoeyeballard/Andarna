#!/usr/bin/env python
"""Run a full evaluation and write baselines/baseline_metrics.json.

The baseline is the reference the Tier-3 regression check compares against. Run this
deliberately (a clean checkpoint, enough episodes for statistical significance) and
commit the result — it is the contract that says "this is how the policy behaves; fail
the build if a change makes it meaningfully worse."
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import EvalConfig  # noqa: E402
from src.metrics import compute_metrics  # noqa: E402

BASELINE_PATH = Path(__file__).resolve().parents[1] / "baselines" / "baseline_metrics.json"


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Record an evaluation baseline.")
    p.add_argument("--num_episodes", type=int, default=50)
    p.add_argument("--max_episode_steps", type=int, default=400)
    p.add_argument("--seed", type=int, default=100_000)
    p.add_argument("--checkpoint_path", default=None)
    p.add_argument("--render_backend", default="egl", choices=["egl", "osmesa", "glfw"])
    p.add_argument("--output", default=str(BASELINE_PATH))
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    overrides = {"num_episodes": args.num_episodes, "max_episode_steps": args.max_episode_steps,
                 "seed": args.seed, "render_backend": args.render_backend,
                 "video_capture_on_nonsuccess": False}
    if args.checkpoint_path:
        overrides["checkpoint_path"] = args.checkpoint_path
    config = EvalConfig.from_env(**overrides)

    from src.evaluator import PolicyEvaluator  # lazy heavy import

    ev = PolicyEvaluator(config)
    ev.load()
    print(f"[update_baseline] {config.num_episodes} episodes @ {config.max_episode_steps} steps...", flush=True)
    result = ev.evaluate(video_dir=None)
    ev.close()

    metrics = compute_metrics(result)
    baseline = {
        "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "checkpoint": result.checkpoint,
        "env_type": config.env_type,
        "task": config.task,
        "seed": config.seed,
        "max_episode_steps": config.max_episode_steps,
        **metrics.to_dict(),
        # per-task breakdown — single task today; structured for future multi-task suites
        "per_task_results": {
            config.task: {
                "num_episodes": metrics.num_episodes,
                "success_rate": metrics.success_rate,
                "avg_episode_length": metrics.avg_episode_length,
            }
        },
    }
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(baseline, indent=2) + "\n")
    print(f"[update_baseline] wrote {out}")
    print(json.dumps({k: baseline[k] for k in ("success_rate", "avg_episode_length",
          "inference_latency_p95_ms", "consistency_score")}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
