"""Sensor-perturbation tests — SCAFFOLDING ONLY (Project 2 integration point).

NOT IMPLEMENTED YET. This is the seam where Project 2's observation-degradation study
(``../openvla-robustness/robustness/degradations.py``) plugs into this CI framework, so
the test pyramid can answer not just "does the policy work?" but "how gracefully does it
degrade as the sensor pipeline gets worse?"

WHY IT'S A STUB: the robustness sweep evaluates a 7B VLA (OpenVLA) and needs a GPU —
it cannot run on the free CPU runners this framework targets. It would run on a
self-hosted GPU runner or a Colab-backed job (see
``.github/workflows/proj3_perturbation_sweep.yml``), on a schedule rather than per-PR.

PLANNED INTERFACE (when Project 2's GPU results are in hand):

    from robustness.degradations import apply_degradation, PROFILES  # Project 2

    def evaluate_under_perturbation(config, axis, level) -> AggregateMetrics:
        '''Run the standard eval, but transform each observation frame with the given
        degradation (gaussian noise / latency / frame-drop / resolution) before the
        policy sees it — i.e. perturb the sensor+transport layer, not the model.'''

    def perturbation_sweep(config, axis) -> dict[level, AggregateMetrics]:
        '''Sweep one axis and return the success-rate-vs-severity curve. The regression
        check then gates on the *cliff* (the level at which success collapses) moving
        adversely — a far stronger contract than a single nominal-condition number.'''

The metrics / regression / reporter machinery here is reused unchanged: a perturbation
run still produces an ``EvalResult`` -> ``AggregateMetrics``, still compares to a
baseline, still posts a PR comment. Only the observation pipeline differs.

See ARCHITECTURE.md § "Extending the framework" for how this slots in.
"""

from __future__ import annotations

# Intentionally no runtime code yet. Importing this module is a no-op so the rest of
# the framework (and Tier-1 CI) is unaffected by the not-yet-present GPU dependencies.
PERTURBATION_TESTS_IMPLEMENTED = False
