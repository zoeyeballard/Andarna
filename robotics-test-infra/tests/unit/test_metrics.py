"""Unit tests for src.metrics — aggregation, percentiles, regression, edge cases."""

import math

import numpy as np
import pytest

from src.config import EvalConfig
from src.metrics import (
    EpisodeResult,
    EvalResult,
    compare_to_baseline,
    compute_metrics,
    passes_threshold,
)


def _episode(idx, success, length, latencies, timed_out=False):
    return EpisodeResult(
        episode_index=idx,
        seed=1000 + idx,
        success=success,
        length=length,
        timed_out=timed_out,
        inference_times_ms=list(latencies),
    )


def _result(episodes, wall_time_s=60.0):
    return EvalResult(
        episodes=episodes,
        wall_time_s=wall_time_s,
        checkpoint="ckpt",
        env_type="aloha",
        task="AlohaTransferCube-v0",
        num_episodes_requested=len(episodes),
    )


# --- success rate -------------------------------------------------------- #
def test_success_rate_all_succeed():
    m = compute_metrics(_result([_episode(i, True, 100, [1.0]) for i in range(3)]))
    assert m.success_rate == 1.0
    assert m.num_success == 3


def test_success_rate_none_succeed():
    m = compute_metrics(_result([_episode(i, False, 100, [1.0]) for i in range(3)]))
    assert m.success_rate == 0.0
    assert m.num_success == 0


def test_success_rate_mixed():
    eps = [_episode(0, True, 100, [1.0]), _episode(1, False, 100, [1.0]),
           _episode(2, True, 100, [1.0]), _episode(3, False, 100, [1.0])]
    m = compute_metrics(_result(eps))
    assert m.success_rate == 0.5
    assert m.num_success == 2


# --- latency percentiles vs hand-computed -------------------------------- #
def test_latency_percentiles_hand_computed():
    # one episode whose per-step latencies are 10,20,...,100 (linear interp percentiles)
    eps = [_episode(0, False, 10, [10, 20, 30, 40, 50, 60, 70, 80, 90, 100])]
    m = compute_metrics(_result(eps))
    assert m.latency.mean_ms == pytest.approx(55.0)
    assert m.latency.p50_ms == pytest.approx(55.0)   # 0.5*(9) -> between 50 and 60
    assert m.latency.p95_ms == pytest.approx(95.5)   # 0.95*9=8.55 -> 90 + .55*10
    assert m.latency.p99_ms == pytest.approx(99.1)   # 0.99*9=8.91 -> 90 + .91*10
    assert m.latency.max_ms == pytest.approx(100.0)
    assert m.latency.n_samples == 10


def test_latency_aggregates_across_episodes():
    eps = [_episode(0, False, 2, [1.0, 3.0]), _episode(1, False, 2, [5.0, 7.0])]
    m = compute_metrics(_result(eps))
    assert m.latency.n_samples == 4
    assert m.latency.max_ms == pytest.approx(7.0)
    assert m.latency.mean_ms == pytest.approx(4.0)


# --- consistency / throughput -------------------------------------------- #
def test_consistency_is_population_std_of_lengths():
    eps = [_episode(0, False, 100, [1]), _episode(1, False, 200, [1]), _episode(2, False, 300, [1])]
    m = compute_metrics(_result(eps))
    assert m.avg_episode_length == pytest.approx(200.0)
    assert m.consistency_score == pytest.approx(np.std([100, 200, 300]))  # ddof=0


def test_throughput_eps_per_min():
    eps = [_episode(i, False, 100, [1]) for i in range(6)]
    m = compute_metrics(_result(eps, wall_time_s=120.0))  # 6 eps / 2 min
    assert m.throughput_eps_per_min == pytest.approx(3.0)


# --- edge cases ---------------------------------------------------------- #
def test_empty_results():
    m = compute_metrics(_result([], wall_time_s=0.0))
    assert m.num_episodes == 0
    assert math.isnan(m.success_rate)
    assert m.latency.n_samples == 0
    assert m.throughput_eps_per_min == 0.0


def test_single_episode_consistency_zero():
    m = compute_metrics(_result([_episode(0, True, 150, [2.0])]))
    assert m.consistency_score == 0.0
    assert m.success_rate == 1.0


def test_all_timeouts():
    eps = [_episode(i, False, 300, [1.0], timed_out=True) for i in range(4)]
    m = compute_metrics(_result(eps))
    assert m.timeouts == 4
    assert m.success_rate == 0.0


# --- regression detection ------------------------------------------------ #
def _metrics_with(success_rate, avg_len, p95, consistency):
    # build an AggregateMetrics via a synthetic run that yields these exact stats
    from src.metrics import AggregateMetrics, LatencyStats

    return AggregateMetrics(
        num_episodes=10,
        num_success=int(round(success_rate * 10)),
        success_rate=success_rate,
        avg_episode_length=avg_len,
        consistency_score=consistency,
        latency=LatencyStats(mean_ms=p95, p50_ms=p95, p95_ms=p95, p99_ms=p95, max_ms=p95, n_samples=10),
        throughput_eps_per_min=5.0,
        timeouts=0,
    )


BASELINE = {
    "success_rate": 0.50,
    "avg_episode_length": 200.0,
    "inference_latency_p95_ms": 10.0,
    "consistency_score": 30.0,
}


def test_regression_flags_success_rate_drop():
    cur = _metrics_with(0.30, 200.0, 10.0, 30.0)  # 0.30 < 0.50 - 0.10
    rep = compare_to_baseline(cur, BASELINE)
    assert rep.has_regression
    assert any(d.name == "success_rate" and d.is_regression for d in rep.deltas)


def test_no_regression_when_within_tolerance():
    cur = _metrics_with(0.45, 210.0, 11.0, 32.0)  # all within tolerance
    rep = compare_to_baseline(cur, BASELINE)
    assert not rep.has_regression


def test_regression_flags_latency_growth():
    # ~2x slowdown: 20 vs 10 baseline clears max(10*0.75, 5)=7.5
    cur = _metrics_with(0.50, 200.0, 20.0, 30.0)
    rep = compare_to_baseline(cur, BASELINE)
    assert any(d.name == "inference_latency_p95_ms" and d.is_regression for d in rep.deltas)


def test_zero_baseline_consistency_not_flagged_by_noise():
    # all-timeout baseline has consistency 0; a small spread must NOT trip a regression
    cur = _metrics_with(0.0, 200.0, 3.0, 5.0)
    baseline = {"success_rate": 0.0, "avg_episode_length": 200.0,
                "inference_latency_p95_ms": 2.7, "consistency_score": 0.0}
    rep = compare_to_baseline(cur, baseline)
    assert not rep.has_regression


def test_improvement_is_not_regression():
    cur = _metrics_with(0.70, 150.0, 5.0, 10.0)  # better on every axis
    rep = compare_to_baseline(cur, BASELINE)
    assert not rep.has_regression


# --- threshold pass/fail ------------------------------------------------- #
def test_passes_threshold_true():
    cur = _metrics_with(0.30, 180.0, 12.0, 40.0)
    cfg = EvalConfig(success_rate_threshold=0.2, max_avg_episode_length=300, inference_latency_ceiling_ms=50)
    assert passes_threshold(cur, cfg).passed


def test_passes_threshold_fails_on_success_rate():
    cur = _metrics_with(0.10, 180.0, 12.0, 40.0)
    cfg = EvalConfig(success_rate_threshold=0.2)
    res = passes_threshold(cur, cfg)
    assert not res.passed
    assert any("success_rate" in r for r in res.reasons())


def test_nan_metric_fails_threshold():
    # a degenerate (empty) run must fail loudly, not pass via NaN comparison
    m = compute_metrics(_result([], wall_time_s=0.0))
    cfg = EvalConfig(success_rate_threshold=0.0)
    assert not passes_threshold(m, cfg).passed
