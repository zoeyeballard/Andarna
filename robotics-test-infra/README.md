# robotics-test-infra 🧪🤖

**An automated test & validation framework for robotics software, built around a
three-tier GitHub Actions CI/CD pipeline.**

This project wraps the MuJoCo simulation and the LeRobot **ACT** policy from
[Project 1 (Andarna)](../README.md) in a continuous-integration pipeline that, on
every push or PR, runs evaluation episodes in simulation, reports performance
metrics, captures video of episodes that don't succeed, and **fails the build when
metrics regress below an established baseline**.

This is **software-in-the-loop (SIL) testing** — the first layer of the robotics
testing pyramid. No physical robot, no GPU: everything runs on CPU in simulation,
which is exactly how production robotics CI works before hardware-in-the-loop (HIL)
stages.

> Part of a robotics-AI portfolio. Project 1 = train/eval an ACT policy (LeRobot +
> MuJoCo). Project 2 = OpenVLA robustness study. **Project 3 (this) = the test
> infrastructure that would gate both in CI.**

## Why this exists

A model that scores well in a notebook tells you nothing about whether it still
works after the next commit. Shipping robots needs the *infrastructure* around the
model: deterministic evaluation, latency budgets, regression gates, reproducible
environments, and a clear pass/fail signal on every change. That infrastructure —
not the policy — is the deliverable here.

The Project 1 ACT policy is deliberately **undertrained** (a few hundred CPU steps);
it approaches the cube but usually overshoots. That's a feature: a policy with a
non-trivial, noisy success rate gives the regression detector something real to
track, where a perfect policy would not.

## The three CI tiers

| Tier | Workflow | Triggers | Budget | What it catches |
|------|----------|----------|--------|-----------------|
| **1 — Fast checks** | `fast_checks` | every push & PR | < 3 min | lint / type / unit-test breakage. No sim. |
| **2 — Sim validation** | `sim_validation` | PRs to `main`, manual, nightly | < 15 min | policy/eval breakage; success rate & latency vs. thresholds. Runs in Docker. |
| **3 — Regression check** | `regression_check` | PRs to `main` | < 15 min | silent metric drift vs. a committed baseline. Fails the build on regression. |

## Repo layout

```
robotics-test-infra/
├── src/
│   ├── config.py          # test configuration + thresholds (env-overridable for CI)
│   ├── evaluator.py        # loads the policy, runs N MuJoCo episodes, instruments latency
│   ├── metrics.py          # aggregate metrics + baseline comparison / pass-fail
│   ├── reporter.py          # markdown + PR-comment report generation
│   └── video_capture.py     # records non-success episodes to MP4 with overlays
├── tests/
│   ├── unit/                # fast, deps-light (Tier 1)
│   └── integration/         # full-pipeline (Tier 2/3, in Docker)
├── baselines/baseline_metrics.json   # committed baseline for regression comparison
├── scripts/                 # run_evaluation / update_baseline / generate_report CLIs
├── Dockerfile               # reproducible CPU simulation environment
└── artifacts/               # local run outputs (gitignored; summaries kept)
```

> **Note on workflow location:** GitHub Actions only reads `.github/workflows/` at
> the *repository* root. Because this project lives in a subdirectory of the Andarna
> repo, the Project-3 workflow YAMLs live at the repo root prefixed `proj3_*.yml`
> and `cd robotics-test-infra` before running. See [ARCHITECTURE.md](ARCHITECTURE.md).

## Quick start (local, CPU)

```bash
cd robotics-test-infra
# light deps (lint + unit tests) — no MuJoCo needed
uv pip install --python ../.venv/bin/python -e ".[dev]"
pytest tests/unit -v

# full sim stack (already present in the shared Project-1 venv)
MUJOCO_GL=egl python scripts/run_evaluation.py --num_episodes 3
```

See [ARCHITECTURE.md](ARCHITECTURE.md) for the testing-pyramid rationale and
[TESTING_PHILOSOPHY.md](TESTING_PHILOSOPHY.md) for how this maps to real-time /
embedded validation (WCET, jitter, deadline monitoring).
