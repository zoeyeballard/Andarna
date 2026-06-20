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

The Project 1 ACT policy was retrained to convergence (100k steps on an A10G) and now
earns a **real, non-saturated success rate (~66% over 50 episodes)**. That middle band
is exactly what a regression detector wants: high enough to be a meaningful signal, far
enough from 100% that a genuine degradation has room to show up. (An earlier
few-hundred-step CPU checkpoint scored 0% — useful only as a "does the pipeline run"
placeholder; the gate now tracks a policy that actually works.)

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

## Status / results

The pipeline is live: **Tier 1 runs green on GitHub** on every push. Tiers 2/3 build
the Docker sim image and run real MuJoCo episodes; they are now **active** —
`RTI_CHECKPOINT_PATH` points at the retrained ACT policy on the HF Hub
([`zoeyeballard/act-aloha-cube`](https://huggingface.co/zoeyeballard/act-aloha-cube)),
public so no `HF_TOKEN` secret is needed.

Committed baseline (`baselines/baseline_metrics.json`) — 50 episodes × 400 steps,
seed 100000, OSMesa render, CPU (the canonical CI environment):

| Metric | Value | Note |
|--------|-------|------|
| Success rate | **0.66** (33/50) | retrained ACT (100k steps); non-saturated by design |
| Avg episode length | **330 steps** | task completes in ~280–330; 400-step env horizon |
| Inference latency p95 | **~0.85 ms** | steady-state (action-chunk queue replay) — the gated budget |
| Inference latency max | **~222 ms** | periodic chunk-boundary forward pass (WCET) |
| Loop breakdown | inference ≪ sim step | MuJoCo step+render dominates wall-clock |
| Determinism | ✅ identical | same seed → byte-identical trajectory |

Because eval is **seed-locked**, the per-PR check (5 episodes, same seeds) is
deterministic — the Tier-2 comment reports **0.80 (4/5)** for these first five seeds,
and Tier-3 compares it to the 50-episode baseline (0.66) within a 0.10 tolerance. We
now **gate on task success** (`success_rate_threshold = 0.4`, a conservative floor well
below the ~0.66 operating point); the **p95 latency budget** and the **regression
check** remain the primary gates. See [ARCHITECTURE.md](ARCHITECTURE.md) for the
rationale.

## Quick start (local, CPU)

```bash
cd robotics-test-infra
# Tier-1 deps only (lint + unit tests) — no MuJoCo needed
uv pip install --python ../.venv/bin/python -e ".[dev]"
pytest tests/unit -v            # 56 unit tests, < 1s
ruff check src tests scripts && mypy src

# Full sim stack is already present in the shared Project-1 venv. Render with EGL
# locally (osmesa in the CI container).
export MUJOCO_GL=egl
python scripts/preflight.py                              # checkpoint + env health (5d/5e)
python scripts/run_evaluation.py --num_episodes 3        # eval → results.json + PR comment
python scripts/check_determinism.py --max_steps 50       # same seed → same trajectory (5a)
python scripts/benchmark.py --num_episodes 3             # latency/throughput profile (5b)
```

## Updating the baseline

The regression gate compares each run to `baselines/baseline_metrics.json`. Re-record
it whenever the checkpoint or the eval profile changes (use the **same** profile the
PR check runs at):

```bash
python scripts/update_baseline.py --num_episodes 5 --max_episode_steps 200 --seed 100000
```

In CI, the Tier-3 workflow has an `update_baseline` `workflow_dispatch` input that
records and commits a fresh baseline in the canonical (Docker/OSMesa) environment.

## Adding a test scenario

`EvalConfig` carries `env_type`/`task`, and the baseline schema has a `per_task_results`
slot, so a new task is mostly configuration:

```bash
python scripts/run_evaluation.py --task AlohaInsertion-v0 --num_episodes 5
```

New metrics go in `metrics.py` (compute + a tolerance entry in
`DEFAULT_LOWER_IS_BETTER_TOL`); new gates go in `passes_threshold`. See
[ARCHITECTURE.md](ARCHITECTURE.md) § *Extending the framework* for multi-task,
perturbation (Project 2), and hardware-in-the-loop paths.

## CI setup (one-time, to enable Tiers 2/3)

1. Push the ACT checkpoint to the Hugging Face Hub.
2. Repo *Settings → Secrets and variables → Actions*: set variable
   `RTI_CHECKPOINT_PATH` (the Hub repo id) and, if private, secret `HF_TOKEN`.
3. Open a PR to `main` — the sim-validation and regression comments appear on it.

## Further reading

- [ARCHITECTURE.md](ARCHITECTURE.md) — the testing pyramid, Docker/OSMesa rationale,
  regression-tolerance tuning, and how this extends to HIL.
- [TESTING_PHILOSOPHY.md](TESTING_PHILOSOPHY.md) — how this maps to real-time /
  embedded validation: WCET vs. p95, jitter, deadline monitoring, why determinism is
  non-negotiable, and why the harness — not the model — is the product.
