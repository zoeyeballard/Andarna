#!/usr/bin/env python
"""CLI entry point: run a policy evaluation in simulation and emit results + a report.

Writes to ``--output_dir``:
  results.json            full per-episode results + aggregate metrics
  validation_comment.md   the Tier-2 PR comment (markdown)
  videos/*.mp4            non-success episode recordings (if enabled)

Exit code is 0 unless ``--fail_on_threshold`` is set and a threshold check failed
(used by the CI sim-validation tier to set the build status).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Make `import src.*` work no matter where this is invoked from.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import EvalConfig  # noqa: E402
from src.metrics import compute_metrics, passes_threshold  # noqa: E402
from src.reporter import render_markdown_summary, render_validation_comment  # noqa: E402


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run a SIL policy evaluation in MuJoCo.")
    p.add_argument("--checkpoint_path", default=None)
    p.add_argument("--task", default=None)
    p.add_argument("--num_episodes", type=int, default=None)
    p.add_argument("--max_episode_steps", type=int, default=None)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--render_backend", default=None, choices=["egl", "osmesa", "glfw"])
    p.add_argument("--output_dir", default="artifacts")
    p.add_argument("--no_video", action="store_true", help="disable non-success video capture")
    p.add_argument(
        "--fail_on_threshold",
        action="store_true",
        help="exit non-zero if any threshold check fails (for CI)",
    )
    return p.parse_args(argv)


def build_config(args: argparse.Namespace) -> EvalConfig:
    overrides = {
        k: v
        for k, v in {
            "checkpoint_path": args.checkpoint_path,
            "task": args.task,
            "num_episodes": args.num_episodes,
            "max_episode_steps": args.max_episode_steps,
            "seed": args.seed,
            "render_backend": args.render_backend,
        }.items()
        if v is not None
    }
    if args.no_video:
        overrides["video_capture_on_nonsuccess"] = False
    # from_env applies RTI_* env vars first, then CLI overrides win.
    return EvalConfig.from_env(**overrides)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    config = build_config(args)
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    video_dir = out / "videos"

    from src.evaluator import PolicyEvaluator  # lazy: heavy sim import

    evaluator = PolicyEvaluator(config)
    print(f"[run_evaluation] loading {config.checkpoint_path} ({config.render_backend})...", flush=True)
    evaluator.load()
    print(
        f"[run_evaluation] running {config.num_episodes} episodes "
        f"(<= {config.max_episode_steps} steps, seed {config.seed})...",
        flush=True,
    )
    result = evaluator.evaluate(video_dir=video_dir if config.video_capture_on_nonsuccess else None)
    evaluator.close()

    metrics = compute_metrics(result)
    thresholds = passes_threshold(metrics, config)

    n_videos = len(list(video_dir.glob("*.mp4"))) if video_dir.exists() else 0
    comment = render_validation_comment(
        metrics, thresholds, config,
        num_video_artifacts=n_videos,
        checkpoint_label=result.checkpoint,
    )

    (out / "results.json").write_text(
        json.dumps({"config": config.__dict__, "metrics": metrics.to_dict(), "result": result.to_dict()}, indent=2)
    )
    (out / "validation_comment.md").write_text(comment)

    print("\n" + render_markdown_summary(metrics, config, checkpoint_label=result.checkpoint))
    print("PASS" if thresholds.passed else "FAIL", "thresholds:", thresholds.reasons() or "all ok")
    print(f"[run_evaluation] wrote results.json + validation_comment.md to {out}/ ({n_videos} videos)")

    if args.fail_on_threshold and not thresholds.passed:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
