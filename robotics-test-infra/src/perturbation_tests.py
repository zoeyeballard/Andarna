"""Sensor-perturbation tests — the Project 2 ↔ Project 3 bridge.

This is the seam where Project 2's observation-degradation study
(``openvla-robustness/robustness/degradations.py``) plugs into this CI framework, so
the test pyramid answers not just *"does the policy work?"* but *"how gracefully does
it degrade as the sensor pipeline gets worse?"*

**It runs on CPU.** The earlier scaffold assumed a GPU was required — that was true of
Project 2's *7B OpenVLA* study, but the thing we reuse here is Project 2's
**degradation module**, which is pure NumPy (Gaussian noise, resolution loss, a
frame-drop coin flip, a FIFO latency buffer). Applied to Project 3's small **ACT**
policy, the whole perturbation sweep runs on the same CPU runners the rest of the
framework targets — no GPU, no self-hosted runner. The degradation is injected into
the camera frame *before* the policy sees it (see ``PolicyEvaluator._perturb_observation``),
i.e. it models the sensor + transport layer and never touches the model.

The metrics / regression / reporter machinery is reused unchanged: a perturbation run
still produces an ``EvalResult`` → ``AggregateMetrics``, still compares to a baseline,
still posts a PR comment. Only the observation pipeline differs.

See ARCHITECTURE.md § "Extending the framework".
"""

from __future__ import annotations

import sys
from pathlib import Path

from .config import EvalConfig
from .metrics import AggregateMetrics, compute_metrics

PERTURBATION_TESTS_IMPLEMENTED = True


# --- locate Project 2's degradation module (sibling project in the same repo) --------
def _import_degradations():
    """Import ``robustness.degradations`` from the sibling ``openvla-robustness`` project.

    Both projects live in the Andarna monorepo, so the module is found relative to this
    file. Kept as a function (not a top-level import) with a clear error so a checkout
    that is missing Project 2 fails loudly instead of mysteriously."""
    repo_root = Path(__file__).resolve().parents[2]
    p2 = repo_root / "openvla-robustness"
    if str(p2) not in sys.path:
        sys.path.insert(0, str(p2))
    try:
        from robustness import degradations  # type: ignore
    except ImportError as e:  # pragma: no cover
        raise RuntimeError(
            f"Project 2's degradation module not importable from {p2}. The perturbation "
            "bridge needs the openvla-robustness/ project present in the repo."
        ) from e
    return degradations


# --- single perturbed evaluation ----------------------------------------------------
def evaluate_under_perturbation(
    config: EvalConfig,
    axis: str,
    level: float | None = None,
    *,
    profile: str | None = None,
    evaluator=None,
) -> AggregateMetrics:
    """Run the standard eval with one degradation applied to every observation frame.

    Pass either ``(axis, level)`` for a single-axis point (axis ∈
    noise/latency/gap/resolution) or ``profile=<name>`` for a stacked deployment
    profile. ``evaluator`` lets a caller reuse one already-``load()``-ed
    :class:`PolicyEvaluator` across a sweep (the policy is loaded once)."""
    deg = _import_degradations()
    if profile is not None:
        degrader = deg.profile_degrader(profile)
    else:
        if level is None:
            raise ValueError("pass level=<severity> for a single-axis perturbation")
        degrader = deg.make_degrader(axis, level)

    own = evaluator is None
    if own:
        from .evaluator import PolicyEvaluator  # lazy: pulls the sim stack
        evaluator = PolicyEvaluator(config)
        evaluator.load()

    prev = evaluator._image_degrader
    evaluator._image_degrader = degrader
    try:
        result = evaluator.evaluate(video_dir=None)
    finally:
        evaluator._image_degrader = prev
        if own:
            evaluator.close()
    return compute_metrics(result)


# --- full-axis sweep ----------------------------------------------------------------
def perturbation_sweep(
    config: EvalConfig,
    axis: str,
    levels: list[float] | None = None,
) -> list[dict]:
    """Sweep one degradation axis and return the success-rate-vs-severity curve.

    Loads the policy once and reuses it across every level. Returns one dict per level
    with ``level``, ``success_rate``, ``avg_episode_length`` and the p95 latency, so the
    reporter/regression layer can gate on the *cliff* (where success collapses) rather
    than a single nominal number."""
    deg = _import_degradations()
    axis_key = "delay" if axis == "latency" else axis
    if levels is None:
        if axis_key not in deg.SWEEPS:
            raise ValueError(f"axis must be one of {list(deg.SWEEPS)} (or 'latency')")
        levels = list(deg.SWEEPS[axis_key])

    from .evaluator import PolicyEvaluator  # lazy: pulls the sim stack
    evaluator = PolicyEvaluator(config)
    evaluator.load()
    try:
        points = []
        for lvl in levels:
            m = evaluate_under_perturbation(config, axis_key, lvl, evaluator=evaluator)
            points.append({
                "axis": axis_key,
                "level": lvl,
                "success_rate": m.success_rate,
                "avg_episode_length": m.avg_episode_length,
                "inference_latency_p95_ms": m.latency.p95_ms,
                "num_episodes": m.num_episodes,
            })
    finally:
        evaluator.close()
    return points


def find_cliff(points: list[dict], frac: float = 0.5) -> float | None:
    """The first severity level whose success rate falls below ``frac`` of the clean
    (lowest-severity) rate — the degradation 'cliff' the gate should watch."""
    if not points:
        return None
    clean = points[0]["success_rate"]
    if clean <= 0:
        return None
    for pt in points:
        if pt["success_rate"] < frac * clean:
            return pt["level"]
    return None
