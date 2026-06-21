#!/usr/bin/env python
"""Run a sensor-perturbation sweep (Project 2 ↔ Project 3 bridge) and report the
success-rate-vs-severity curve plus the degradation cliff.

Reuses Project 2's degradation module (pure NumPy) against Project 3's ACT policy, so
it runs on CPU — no GPU. Writes a CSV, a results JSON, and a markdown comment.

    python scripts/run_perturbation_sweep.py --axis resolution --num_episodes 10
    python scripts/run_perturbation_sweep.py --axis latency --levels 0 1 2 3 5
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import EvalConfig  # noqa: E402
from src.perturbation_tests import find_cliff, perturbation_sweep  # noqa: E402

AXES = ["noise", "latency", "gap", "resolution"]


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Sensor-perturbation sweep for the ACT policy.")
    p.add_argument("--axis", required=True, choices=AXES)
    p.add_argument("--levels", type=float, nargs="*", default=None,
                   help="severity levels to sweep (default: Project 2's standard sweep)")
    p.add_argument("--num_episodes", type=int, default=10)
    p.add_argument("--max_episode_steps", type=int, default=400)
    p.add_argument("--seed", type=int, default=100_000)
    p.add_argument("--checkpoint_path", default=None)
    p.add_argument("--render_backend", default="osmesa", choices=["egl", "osmesa", "glfw"])
    p.add_argument("--output_dir", default="artifacts")
    return p.parse_args(argv)


def render_markdown(axis: str, points: list[dict], cliff) -> str:
    lines = [
        "<!-- proj3-perturbation -->",
        f"## 🌪️ Perturbation Sweep — `{axis}`",
        "",
        "Project 2's sensor-degradation module applied to the ACT policy "
        "(CPU; degradation injected upstream of the policy, model untouched).",
        "",
        "| Severity | Success rate | Avg ep. length | p95 latency |",
        "|---|---|---|---|",
    ]
    for pt in points:
        lines.append(
            f"| {pt['level']:g} | {pt['success_rate']:.2f} ({int(round(pt['success_rate']*pt['num_episodes']))}/"
            f"{pt['num_episodes']}) | {pt['avg_episode_length']:.0f} | "
            f"{pt['inference_latency_p95_ms']:.2f} ms |"
        )
    clean = points[0]["success_rate"] if points else 0.0
    cliff_txt = (f"**severity {cliff:g}**" if cliff is not None
                 else "not reached in this sweep")
    lines += [
        "",
        f"**Degradation cliff** (success < 50% of the clean {clean:.2f}): {cliff_txt}.",
        f"_ACT policy × {points[0]['num_episodes'] if points else 0} episodes/level, "
        f"seed-locked, OSMesa._",
    ]
    return "\n".join(lines) + "\n"


def main(argv=None) -> int:
    args = parse_args(argv)
    overrides = {
        "num_episodes": args.num_episodes,
        "max_episode_steps": args.max_episode_steps,
        "seed": args.seed,
        "render_backend": args.render_backend,
        "video_capture_on_nonsuccess": False,
    }
    if args.checkpoint_path:
        overrides["checkpoint_path"] = args.checkpoint_path
    config = EvalConfig.from_env(**overrides)

    print(f"[perturbation] axis={args.axis} levels={args.levels or 'default'} "
          f"episodes={config.num_episodes} ckpt={config.checkpoint_path}", flush=True)
    points = perturbation_sweep(config, args.axis, levels=args.levels)
    cliff = find_cliff(points)

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    csv_path = out / f"perturbation_{args.axis}.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(points[0].keys()))
        w.writeheader()
        w.writerows(points)
    (out / f"perturbation_{args.axis}.json").write_text(
        json.dumps({"axis": args.axis, "cliff": cliff, "points": points}, indent=2) + "\n")
    md = render_markdown(args.axis, points, cliff)
    (out / "perturbation_comment.md").write_text(md)

    print("\n" + md)
    print(f"[perturbation] wrote {csv_path} (+ .json, perturbation_comment.md)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
