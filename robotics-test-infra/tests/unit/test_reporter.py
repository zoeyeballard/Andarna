"""Unit tests for src.reporter — PR-comment formatting and pass/fail emoji."""

from src.config import EvalConfig
from src.metrics import AggregateMetrics, LatencyStats, compare_to_baseline, passes_threshold
from src.reporter import (
    FAIL,
    PASS,
    WARN,
    render_regression_comment,
    render_validation_comment,
)


def _metrics(success_rate=0.3, p95=12.0):
    return AggregateMetrics(
        num_episodes=10,
        num_success=int(success_rate * 10),
        success_rate=success_rate,
        avg_episode_length=187.0,
        consistency_score=45.2,
        latency=LatencyStats(mean_ms=5.0, p50_ms=0.5, p95_ms=p95, p99_ms=900.0, max_ms=1800.0, n_samples=100),
        throughput_eps_per_min=4.0,
        timeouts=2,
    )


def test_validation_comment_pass():
    cfg = EvalConfig(success_rate_threshold=0.2, inference_latency_ceiling_ms=50)
    m = _metrics(success_rate=0.3, p95=12.0)
    tr = passes_threshold(m, cfg)
    out = render_validation_comment(m, tr, cfg, num_video_artifacts=2)
    assert "Simulation Validation Results" in out
    assert PASS in out
    assert "Success Rate" in out and "0.30" in out
    assert "2 non-success episode video(s)" in out
    assert FAIL not in out.split("\n")[0]  # header shows pass


def test_validation_comment_fail_lists_reasons():
    cfg = EvalConfig(success_rate_threshold=0.5, inference_latency_ceiling_ms=50)
    m = _metrics(success_rate=0.1)
    tr = passes_threshold(m, cfg)
    out = render_validation_comment(m, tr, cfg)
    assert FAIL in out
    assert "Threshold failures" in out
    assert "success_rate" in out


def test_regression_comment_flags_and_fails():
    baseline = {"success_rate": 0.5, "avg_episode_length": 180.0,
                "inference_latency_p95_ms": 10.0, "consistency_score": 30.0}
    rep = compare_to_baseline(_metrics(success_rate=0.2), baseline)
    out = render_regression_comment(rep, baseline_meta={"timestamp": "2026-06-17", "num_episodes": 50})
    assert "Regression Analysis" in out
    assert WARN in out
    assert "Build failed" in out


def test_regression_comment_clean_when_no_regression():
    baseline = {"success_rate": 0.2, "avg_episode_length": 200.0,
                "inference_latency_p95_ms": 20.0, "consistency_score": 60.0}
    rep = compare_to_baseline(_metrics(success_rate=0.3, p95=12.0), baseline)
    out = render_regression_comment(rep)
    assert "No regressions" in out
    assert "Build failed" not in out
