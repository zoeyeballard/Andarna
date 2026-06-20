"""Human-readable test reports: markdown summaries and GitHub PR comments.

The PR comment is the headline deliverable — when someone opens a PR against this repo
the first thing they see is an automated simulation-validation comment with a metrics
table and a clear pass/fail. Pure string formatting; no heavy deps.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.config import EvalConfig
    from src.metrics import AggregateMetrics, RegressionReport, ThresholdResult

PASS, FAIL, INFO, WARN = "✅", "❌", "ℹ️", "⚠️"


def _fmt(value: float, spec: str = ".3g", dash_if_nan: bool = True) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return "—" if dash_if_nan else "nan"
    return format(value, spec)


def render_validation_comment(
    metrics: AggregateMetrics,
    threshold_result: ThresholdResult,
    config: EvalConfig,
    *,
    num_video_artifacts: int = 0,
    checkpoint_label: str | None = None,
) -> str:
    """Render the Tier-2 simulation-validation PR comment."""
    overall = PASS if threshold_result.passed else FAIL
    sr = _status_for(threshold_result, "success_rate")
    al = _status_for(threshold_result, "avg_episode_length")
    lat = _status_for(threshold_result, "inference_latency_p95_ms")

    lines = [
        f"## 🤖 Simulation Validation Results {overall}",
        "",
        "| Metric | Value | Threshold | Status |",
        "|--------|-------|-----------|--------|",
        f"| Success Rate | {_fmt(metrics.success_rate, '.2f')} "
        f"({metrics.num_success}/{metrics.num_episodes}) "
        f"| ≥ {config.success_rate_threshold:.2f} | {sr} |",
        f"| Avg Episode Length | {_fmt(metrics.avg_episode_length, '.0f')} steps "
        f"| ≤ {config.max_avg_episode_length:.0f} | {al} |",
        f"| Inference Latency (p95) | {_fmt(metrics.latency.p95_ms, '.2f')}ms "
        f"| ≤ {config.inference_latency_ceiling_ms:.0f}ms | {lat} |",
        f"| Inference Latency (max) | {_fmt(metrics.latency.max_ms, '.1f')}ms | — | {INFO} |",
        f"| Consistency Score | {_fmt(metrics.consistency_score, '.1f')} | — | {INFO} |",
        f"| Throughput | {_fmt(metrics.throughput_eps_per_min, '.1f')} ep/min | — | {INFO} |",
        f"| Timeouts | {metrics.timeouts}/{metrics.num_episodes} | — | {INFO} |",
        "",
    ]
    if not threshold_result.passed:
        lines.append(f"{FAIL} **Threshold failures:** " + "; ".join(threshold_result.reasons()))
        lines.append("")
    if num_video_artifacts > 0:
        lines.append(f"📹 {num_video_artifacts} non-success episode video(s) attached as artifacts.")
        lines.append("")

    ckpt = checkpoint_label or config.checkpoint_path
    lines.append(
        f"_Evaluated: ACT policy (`{ckpt}`) on {config.task} × {metrics.num_episodes} episodes "
        f"(seed {config.seed}, {config.render_backend})._"
    )
    return "\n".join(lines)


def render_regression_comment(
    report: RegressionReport,
    *,
    baseline_meta: dict | None = None,
) -> str:
    """Render the Tier-3 regression-analysis PR comment."""
    overall = WARN if report.has_regression else PASS
    title = "Regression detected" if report.has_regression else "No regressions"
    lines = [
        f"## 📊 Regression Analysis {overall} — {title}",
        "",
        "| Metric | Baseline | Current | Delta | Status |",
        "|--------|----------|---------|-------|--------|",
    ]
    for d in report.deltas:
        status = f"{WARN} Regression" if d.is_regression else f"{PASS} Within tolerance"
        sign = "+" if d.delta >= 0 else ""
        lines.append(
            f"| {d.name} | {_fmt(d.baseline, '.3g')} | {_fmt(d.current, '.3g')} "
            f"| {sign}{_fmt(d.delta, '.3g', dash_if_nan=False)} | {status} |"
        )
    lines.append("")
    if baseline_meta:
        ts = baseline_meta.get("timestamp", "?")
        n = baseline_meta.get("num_episodes", "?")
        lines.append(f"_Baseline: {n} episodes, recorded {ts}._")
    if report.has_regression:
        names = ", ".join(d.name for d in report.regressions())
        lines.append("")
        lines.append(f"{FAIL} **Build failed** — regression in: {names}.")
    return "\n".join(lines)


def render_markdown_summary(
    metrics: AggregateMetrics, config: EvalConfig, checkpoint_label: str | None = None
) -> str:
    """A compact summary for logs / standalone report generation."""
    m = metrics
    ckpt = checkpoint_label or config.checkpoint_path
    return (
        f"### Evaluation summary — {config.task} ({ckpt})\n"
        f"- episodes: {m.num_episodes} (success {m.num_success}, timeouts {m.timeouts})\n"
        f"- success_rate: {_fmt(m.success_rate, '.3f')}\n"
        f"- avg_episode_length: {_fmt(m.avg_episode_length, '.1f')} steps "
        f"(consistency {_fmt(m.consistency_score, '.1f')})\n"
        f"- inference latency ms: mean {_fmt(m.latency.mean_ms, '.2f')}, "
        f"p50 {_fmt(m.latency.p50_ms, '.2f')}, p95 {_fmt(m.latency.p95_ms, '.2f')}, "
        f"p99 {_fmt(m.latency.p99_ms, '.2f')}, max {_fmt(m.latency.max_ms, '.1f')}\n"
        f"- throughput: {_fmt(m.throughput_eps_per_min, '.1f')} ep/min\n"
    )


def _status_for(threshold_result: ThresholdResult, name: str) -> str:
    for c in threshold_result.checks:
        if c.name == name:
            return PASS if c.passed else FAIL
    return INFO
