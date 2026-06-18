"""Aggregate metrics, baseline comparison, and threshold pass/fail.

This module is intentionally dependency-light (numpy only) so Tier-1 CI can run its
unit tests on a bare runner. It also defines the shared result dataclasses
(``EpisodeResult`` / ``EvalResult``) that the evaluator produces and everything else
consumes — keeping them here (not in the heavy ``evaluator`` module) means tests and
the reporter can import them without pulling in MuJoCo/torch/lerobot.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:  # avoid importing config at runtime; only needed for type hints
    from src.config import EvalConfig


# --------------------------------------------------------------------------- #
# Raw results (produced by the evaluator)
# --------------------------------------------------------------------------- #
@dataclass
class EpisodeResult:
    """Per-episode outcome from a single simulation rollout."""

    episode_index: int
    seed: int | None
    success: bool
    length: int  # number of environment steps actually executed
    timed_out: bool  # hit the step cap without the env terminating on its own
    inference_times_ms: list[float] = field(default_factory=list)
    sum_reward: float = 0.0
    max_reward: float = 0.0
    # cube free-joint pose at episode end: [x, y, z, qw, qx, qy, qz]
    final_object_position: list[float] | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class EvalResult:
    """Full result of an evaluation run (N episodes)."""

    episodes: list[EpisodeResult]
    wall_time_s: float
    checkpoint: str
    env_type: str
    task: str
    num_episodes_requested: int
    render_backend: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "checkpoint": self.checkpoint,
            "env_type": self.env_type,
            "task": self.task,
            "num_episodes_requested": self.num_episodes_requested,
            "render_backend": self.render_backend,
            "wall_time_s": self.wall_time_s,
            "episodes": [e.to_dict() for e in self.episodes],
        }


# --------------------------------------------------------------------------- #
# Aggregate metrics
# --------------------------------------------------------------------------- #
def _percentile(values: list[float] | np.ndarray, q: float) -> float:
    """Linear-interpolated percentile; returns NaN for an empty input."""
    arr = np.asarray(values, dtype=float)
    if arr.size == 0:
        return float("nan")
    return float(np.percentile(arr, q, method="linear"))


@dataclass
class LatencyStats:
    """Per-step policy-inference latency, in milliseconds.

    Note: with an action-chunking policy (ACT, ``n_action_steps=100``) real inference
    only runs on chunk-boundary steps; the rest are cheap queue pops. The distribution
    is therefore strongly bimodal, which is exactly why p95/p99 (steady-state) and max
    (the periodic spike) are reported separately — see TESTING_PHILOSOPHY.md (WCET).
    """

    mean_ms: float
    p50_ms: float
    p95_ms: float
    p99_ms: float
    max_ms: float
    n_samples: int

    @classmethod
    def from_samples(cls, samples: list[float]) -> LatencyStats:
        arr = np.asarray(samples, dtype=float)
        if arr.size == 0:
            return cls(float("nan"), float("nan"), float("nan"), float("nan"), float("nan"), 0)
        return cls(
            mean_ms=float(arr.mean()),
            p50_ms=_percentile(arr, 50),
            p95_ms=_percentile(arr, 95),
            p99_ms=_percentile(arr, 99),
            max_ms=float(arr.max()),
            n_samples=int(arr.size),
        )


@dataclass
class AggregateMetrics:
    """Aggregate metrics computed across all episodes of one evaluation run."""

    num_episodes: int
    num_success: int
    success_rate: float
    avg_episode_length: float
    consistency_score: float  # std-dev of episode lengths (lower = more consistent)
    latency: LatencyStats
    throughput_eps_per_min: float
    timeouts: int

    def to_dict(self) -> dict[str, Any]:
        """Flat, JSON-friendly dict. Keys match the committed baseline schema so a
        baseline file is just this dict plus a little provenance metadata."""
        return {
            "num_episodes": self.num_episodes,
            "num_success": self.num_success,
            "success_rate": self.success_rate,
            "avg_episode_length": self.avg_episode_length,
            "consistency_score": self.consistency_score,
            "inference_latency_mean_ms": self.latency.mean_ms,
            "inference_latency_p50_ms": self.latency.p50_ms,
            "inference_latency_p95_ms": self.latency.p95_ms,
            "inference_latency_p99_ms": self.latency.p99_ms,
            "inference_latency_max_ms": self.latency.max_ms,
            "throughput_eps_per_min": self.throughput_eps_per_min,
            "timeouts": self.timeouts,
        }


def compute_metrics(result: EvalResult) -> AggregateMetrics:
    """Compute aggregate metrics from a completed evaluation run.

    Handles the edge cases the unit tests exercise: zero episodes (everything NaN/0),
    a single episode (consistency = 0), and all-timeout runs.
    """
    episodes = result.episodes
    n = len(episodes)
    if n == 0:
        return AggregateMetrics(
            num_episodes=0,
            num_success=0,
            success_rate=float("nan"),
            avg_episode_length=float("nan"),
            consistency_score=float("nan"),
            latency=LatencyStats.from_samples([]),
            throughput_eps_per_min=0.0,
            timeouts=0,
        )

    successes = [bool(e.success) for e in episodes]
    lengths = np.asarray([e.length for e in episodes], dtype=float)
    all_latencies: list[float] = []
    for e in episodes:
        all_latencies.extend(e.inference_times_ms)

    num_success = int(sum(successes))
    # population std (ddof=0): well-defined for a single episode (=> 0.0)
    consistency = float(lengths.std(ddof=0))
    throughput = (n / result.wall_time_s * 60.0) if result.wall_time_s > 0 else 0.0

    return AggregateMetrics(
        num_episodes=n,
        num_success=num_success,
        success_rate=num_success / n,
        avg_episode_length=float(lengths.mean()),
        consistency_score=consistency,
        latency=LatencyStats.from_samples(all_latencies),
        throughput_eps_per_min=throughput,
        timeouts=int(sum(e.timed_out for e in episodes)),
    )


# --------------------------------------------------------------------------- #
# Threshold pass/fail (current run vs. configured thresholds)
# --------------------------------------------------------------------------- #
@dataclass
class ThresholdCheck:
    name: str
    value: float
    threshold: float
    comparison: str  # ">=" or "<="
    passed: bool


@dataclass
class ThresholdResult:
    checks: list[ThresholdCheck]

    @property
    def passed(self) -> bool:
        return all(c.passed for c in self.checks)

    def failures(self) -> list[ThresholdCheck]:
        return [c for c in self.checks if not c.passed]

    def reasons(self) -> list[str]:
        return [
            f"{c.name}={c.value:.4g} fails {c.comparison} {c.threshold:.4g}"
            for c in self.failures()
        ]


def passes_threshold(current: AggregateMetrics, config: EvalConfig) -> ThresholdResult:
    """Simple pass/fail of a run against the configured thresholds.

    Gates on p95 inference latency (steady-state), not max — the periodic chunk-boundary
    spike is expected and is tracked separately as a benchmark/WCET signal.
    """
    checks = [
        ThresholdCheck(
            "success_rate", current.success_rate, config.success_rate_threshold, ">=",
            _ge(current.success_rate, config.success_rate_threshold),
        ),
        ThresholdCheck(
            "avg_episode_length", current.avg_episode_length, config.max_avg_episode_length, "<=",
            _le(current.avg_episode_length, config.max_avg_episode_length),
        ),
        ThresholdCheck(
            "inference_latency_p95_ms", current.latency.p95_ms, config.inference_latency_ceiling_ms,
            "<=", _le(current.latency.p95_ms, config.inference_latency_ceiling_ms),
        ),
    ]
    return ThresholdResult(checks=checks)


def _ge(value: float, threshold: float) -> bool:
    # NaN never passes (a degenerate run should fail loudly, not slip through)
    return (not math.isnan(value)) and value >= threshold


def _le(value: float, threshold: float) -> bool:
    return (not math.isnan(value)) and value <= threshold


# --------------------------------------------------------------------------- #
# Regression comparison (current run vs. committed baseline)
# --------------------------------------------------------------------------- #
@dataclass
class MetricDelta:
    name: str
    baseline: float
    current: float
    delta: float  # current - baseline
    higher_is_better: bool
    is_regression: bool

    @property
    def status(self) -> str:
        return "regression" if self.is_regression else "ok"


@dataclass
class RegressionReport:
    deltas: list[MetricDelta]

    @property
    def has_regression(self) -> bool:
        return any(d.is_regression for d in self.deltas)

    def regressions(self) -> list[MetricDelta]:
        return [d for d in self.deltas if d.is_regression]


# Default regression tolerances. A lower-is-better metric is only a regression when it
# worsens by more than BOTH its fractional slack AND its absolute floor — i.e. the move
# must clear ``max(baseline * frac, abs_floor)``. The absolute floor is what keeps a
# near-zero baseline (p95 ≈ a few ms, consistency = 0 when all episodes time out) from
# flagging ordinary shared-runner noise as a regression.
DEFAULT_SUCCESS_RATE_TOL = 0.10  # absolute drop allowed before success-rate regression
# key -> (fractional_tol, absolute_floor)
DEFAULT_LOWER_IS_BETTER_TOL: dict[str, tuple[float, float]] = {
    "avg_episode_length": (0.25, 20.0),  # +25% or +20 steps
    "inference_latency_p95_ms": (0.75, 5.0),  # +75% or +5 ms (catches a ~2x slowdown)
    "consistency_score": (0.50, 30.0),  # +50% or +30 steps of length spread
}


def _baseline_value(baseline: dict[str, Any] | AggregateMetrics, key: str) -> float:
    if isinstance(baseline, AggregateMetrics):
        baseline = baseline.to_dict()
    return float(baseline[key])


def compare_to_baseline(
    current: AggregateMetrics,
    baseline: dict[str, Any] | AggregateMetrics,
    *,
    success_rate_tol: float = DEFAULT_SUCCESS_RATE_TOL,
    lower_is_better_tol: dict[str, tuple[float, float]] | None = None,
) -> RegressionReport:
    """Compare a current run to a stored baseline and flag regressions.

    - ``success_rate`` (higher is better): regression if it drops by more than
      ``success_rate_tol`` (absolute).
    - ``avg_episode_length`` / ``inference_latency_p95_ms`` / ``consistency_score``
      (lower is better): regression if the increase over baseline exceeds
      ``max(baseline * frac, abs_floor)`` for that metric.
    """
    tol = lower_is_better_tol or DEFAULT_LOWER_IS_BETTER_TOL
    cur = current.to_dict()
    deltas: list[MetricDelta] = []

    # success rate — higher is better, absolute tolerance
    b_sr = _baseline_value(baseline, "success_rate")
    c_sr = cur["success_rate"]
    deltas.append(
        MetricDelta(
            name="success_rate",
            baseline=b_sr,
            current=c_sr,
            delta=c_sr - b_sr,
            higher_is_better=True,
            is_regression=(not math.isnan(c_sr)) and c_sr < (b_sr - success_rate_tol),
        )
    )

    # lower-is-better metrics: must clear both fractional and absolute slack
    for key, (frac, abs_floor) in tol.items():
        b = _baseline_value(baseline, key)
        c = cur[key]
        allowed_increase = max(b * frac, abs_floor)
        deltas.append(
            MetricDelta(
                name=key,
                baseline=b,
                current=c,
                delta=c - b,
                higher_is_better=False,
                is_regression=(not math.isnan(c)) and (c - b) > allowed_increase,
            )
        )

    return RegressionReport(deltas=deltas)
