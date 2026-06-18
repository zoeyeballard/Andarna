#!/usr/bin/env python
"""Standalone report generation from a results.json (+ optional baseline).

Decouples reporting from evaluation: CI runs the eval once, uploads results.json as an
artifact, and this turns it into the validation and/or regression PR-comment markdown
without re-running the simulation. Also used to regenerate a report locally.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import EvalConfig  # noqa: E402
from src.metrics import (  # noqa: E402
    AggregateMetrics,
    LatencyStats,
    compare_to_baseline,
    passes_threshold,
)
from src.reporter import render_regression_comment, render_validation_comment  # noqa: E402


def _metrics_from_dict(d: dict) -> AggregateMetrics:
    return AggregateMetrics(
        num_episodes=d["num_episodes"],
        num_success=d["num_success"],
        success_rate=d["success_rate"],
        avg_episode_length=d["avg_episode_length"],
        consistency_score=d["consistency_score"],
        latency=LatencyStats(
            mean_ms=d["inference_latency_mean_ms"],
            p50_ms=d["inference_latency_p50_ms"],
            p95_ms=d["inference_latency_p95_ms"],
            p99_ms=d["inference_latency_p99_ms"],
            max_ms=d["inference_latency_max_ms"],
            n_samples=0,
        ),
        throughput_eps_per_min=d["throughput_eps_per_min"],
        timeouts=d["timeouts"],
    )


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Generate PR-comment reports from results.json.")
    p.add_argument("--results", required=True, help="path to results.json from run_evaluation")
    p.add_argument("--baseline", default=None, help="path to baseline_metrics.json (enables regression report)")
    p.add_argument("--out_validation", default=None)
    p.add_argument("--out_regression", default=None)
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    data = json.loads(Path(args.results).read_text())
    metrics = _metrics_from_dict(data["metrics"])
    config = EvalConfig(**data["config"]) if "config" in data else EvalConfig()

    thresholds = passes_threshold(metrics, config)
    validation = render_validation_comment(metrics, thresholds, config, checkpoint_label=data.get("result", {}).get("checkpoint"))
    print(validation)
    if args.out_validation:
        Path(args.out_validation).write_text(validation)

    if args.baseline:
        baseline = json.loads(Path(args.baseline).read_text())
        report = compare_to_baseline(metrics, baseline)
        regression = render_regression_comment(report, baseline_meta=baseline)
        print("\n" + regression)
        if args.out_regression:
            Path(args.out_regression).write_text(regression)
        if report.has_regression:
            return 1
    return 0 if thresholds.passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
