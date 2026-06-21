# Architecture

How `robotics-test-infra` is put together, why it's shaped this way, and how it
extends toward hardware. This is a **software-in-the-loop (SIL)** test framework: it
exercises a real policy in a real physics simulator on CPU, and turns the result into
a pass/fail signal on every change.

## The testing pyramid

Robotics testing is layered, cheapest-and-fastest at the bottom. This project builds
the bottom three layers in CI; the top two are where it would grow toward hardware.

```
                ▲  slower, costlier, higher-fidelity
   ┌─────────────────────────────────────────────┐
   │  Field / shadow-mode      (real robot, real world)   ← future
   ├─────────────────────────────────────────────┤
   │  HIL  hardware-in-the-loop (real compute + actuators) ← future
   ├─────────────────────────────────────────────┤
   │  Tier 3  Regression check  (vs. committed baseline)   │ this
   │  Tier 2  Sim validation    (MuJoCo episodes, Docker)  │ repo
   │  Tier 1  Fast checks       (lint, types, unit tests)  │
   └─────────────────────────────────────────────┘
                ▼  faster, cheaper, lower-fidelity
```

| Tier | Trigger | Budget | Runs | Catches |
|------|---------|--------|------|---------|
| **1 — Fast checks** | every push & PR | < 3 min | bare runner, no sim | a broken import, a lint/type error, a unit-test failure — before you pay for a sim run |
| **2 — Sim validation** | PR→main, nightly, manual | < 15 min | Docker + MuJoCo (CPU) | the policy/eval crashing, success rate or **p95 latency** breaching thresholds |
| **3 — Regression check** | PR→main | < 15 min | Docker + MuJoCo (CPU) | *silent drift*: metrics that quietly got worse vs. the committed baseline |

Each tier is a gate. A change that fails Tier 1 never reaches Tier 2. The principle is
**fail fast and cheap**: the most common breakage (code) is caught in seconds; the
expensive sim runs only happen once the cheap checks are green.

### What each tier actually does

- **Tier 1** (`proj3_fast_checks.yml`): `ruff` + `mypy src` + `pytest tests/unit`. No
  MuJoCo, no torch, no checkpoint — the `config`/`metrics`/`reporter` modules are
  pure-Python + numpy, and `evaluator` imports the sim stack lazily, so 56 unit tests
  run in well under a second.
- **Tier 2** (`proj3_sim_validation.yml`): builds the Docker image, runs N episodes in
  MuJoCo, uploads non-success episode videos as artifacts, posts a metrics PR comment,
  and fails the build if any threshold check fails (`--fail_on_threshold`).
- **Tier 3** (`proj3_regression_check.yml`): runs the same eval, compares it to
  `baselines/baseline_metrics.json`, posts a regression-analysis comment, and fails on
  a flagged regression. A `workflow_dispatch` path re-records and commits the baseline.
- **Matrix** (`proj3_matrix.yml`): the same eval across seeds {42, 123, 456} to prove
  the harness isn't passing on one lucky setup. Informational; weekly + manual.

## Why Docker

GitHub's runners ship no MuJoCo, no GPU, and no display. The image
(`Dockerfile`) pins the exact sim stack (LeRobot 0.5.1 / MuJoCo 3.8 / gym-aloha) and —
critically — installs **OSMesa**, a software OpenGL implementation, so MuJoCo can
render offscreen with no GPU. Locally we render with **EGL** (fast, GPU-backed); in CI
we set `MUJOCO_GL=osmesa`. The render backend is a single config knob
(`EvalConfig.render_backend`), and the evaluator sets `MUJOCO_GL`/`PYOPENGL_PLATFORM`
*before* MuJoCo is imported, because the GL backend can't be changed after import.

The image also installs **CPU-only torch first**, so the sim deps resolve against it
instead of pulling multi-GB CUDA wheels (a lesson carried over from Project 1). The
198 MB checkpoint is **not** baked in — it's pulled from the Hugging Face Hub at
runtime (or bind-mounted locally), keeping the image lean and the build cacheable.

## How regression detection works (and how to tune it)

A run produces an `AggregateMetrics`; `metrics.compare_to_baseline` diffs it against
the committed baseline and flags each metric:

- **`success_rate`** (higher is better): regression if it drops by more than an
  *absolute* tolerance (default 0.10).
- **`avg_episode_length`, `inference_latency_p95_ms`, `consistency_score`** (lower is
  better): regression only if the increase clears **both** a fractional and an
  absolute floor — `delta > max(baseline × frac, abs_floor)`.

The absolute floor is the important part. The baseline here is `p95 ≈ 2.6 ms` and
`consistency = 0` (all episodes time out at the same step). With *pure* fractional
tolerances, ordinary shared-runner jitter — or any nonzero spread — would be flagged
as a "regression." The floors (`+5 ms` latency, `+30` steps consistency,
`+20` steps length) absorb noise so the gate fires on real drift, not weather. Tune
the table in `metrics.DEFAULT_LOWER_IS_BETTER_TOL`; loosen on noisier runners, tighten
once you trust the environment.

The baseline must be recorded at the **same profile** (episodes, step cap, seed) the
PR check runs at, or the comparison is apples-to-oranges. Both use the canonical CI
profile (5 episodes × 200 steps, seed 100000); regenerate the baseline with the
Tier-3 `update_baseline` dispatch whenever the checkpoint or profile changes.

> **On the success threshold being 0.0.** The Project-1 checkpoint is undertrained and
> empirically scores **0/10** on the cube transfer. So we don't gate on task success
> yet — the binding, *passing* gates are the **p95 latency budget** and the
> **regression check**. This is honest, and it's how early-stage robotics CI actually
> looks: the policy isn't good yet, but the harness and the timing must be green, and
> the day a change makes either worse, the build goes red. Raise
> `success_rate_threshold` once a better checkpoint earns a real success rate.

## Determinism

MuJoCo + a deterministic policy are reproducible given a seed; `check_determinism.py`
asserts that two runs of the same seeded episode produce *byte-identical* action
trajectories (verified: identical across 50 steps). This is a prerequisite for the
whole pyramid — if the harness weren't deterministic, every regression signal would be
indistinguishable from noise, and the floors above would have to be so wide as to be
useless.

## Repo / module layout

```
src/
  config.py        EvalConfig dataclass; validation; RTI_* env overrides for CI
  evaluator.py     PolicyEvaluator: load policy/env/processors, instrumented rollout
                   (per-step inference latency, timeout, final cube pose, frames).
                   Heavy sim imports are lazy → mocks without MuJoCo.
  metrics.py       EpisodeResult/EvalResult dataclasses; aggregate metrics; percentiles;
                   compare_to_baseline; passes_threshold
  reporter.py      validation + regression PR-comment markdown
  video_capture.py downscaled MP4 with step/latency overlays (non-success episodes)
  validation.py    checkpoint integrity (5d) + env health (5e) reports
  perturbation_tests.py  Project 2 bridge — observation-degradation sweep (see below)
scripts/           run_evaluation · update_baseline · generate_report ·
                   preflight · check_determinism · benchmark
tests/unit/        fast, deps-light (Tier 1)        tests/integration/  full pipeline (Tier 2/3)
baselines/         baseline_metrics.json (the regression contract)
Dockerfile         reproducible CPU sim image (OSMesa)
```

Workflows live at the **repository root** `.github/workflows/proj3_*.yml` (GitHub only
reads workflows there, not in subdirectories) and `cd robotics-test-infra` before
running. They're prefixed `proj3_` and path-filtered so they coexist with the other
projects in this repo.

## CI setup (one-time)

Tiers 2/3 need the checkpoint. They **no-op with a warning until configured**, so the
repo stays green before setup. To enable them:

1. Push the ACT checkpoint to the Hugging Face Hub (the 198 MB `pretrained_model/`).
2. Repo **Settings → Secrets and variables → Actions**:
   - **Variable** `RTI_CHECKPOINT_PATH` = the Hub repo id (e.g. `you/act-aloha-cube`).
   - **Secret** `HF_TOKEN` = an HF read token (only if the model repo is private).
3. Open a PR to `main` — Tier 2/3 build the image, run the eval, and post their
   comments. Re-record the baseline in this environment via the Tier-3
   `update_baseline` workflow dispatch.

## Extending the framework

The metrics / regression / reporter / Docker machinery is policy-agnostic. Growth
paths, easiest first:

- **More tasks / models.** `EvalConfig` already carries `env_type`/`task`; the baseline
  schema has a `per_task_results` slot. Add `AlohaInsertion-v0` or a second policy and
  the same gates apply per task.
- **Project 2's perturbation sweep — implemented** (`src/perturbation_tests.py` +
  `proj3_perturbation_sweep.yml`, Tier 4). Project 2's degradation module degrades the
  *observation pipeline* (sensor noise, latency, frame-drops, resolution) upstream of
  the policy; `perturbation_sweep()` maps the success-rate-vs-severity curve and
  `find_cliff()` locates where it collapses. It reuses this framework unchanged — a
  perturbed run still yields an `EvalResult` → `AggregateMetrics` → CSV/markdown comment;
  only `PolicyEvaluator._perturb_observation` (a per-camera, per-episode hook) differs.
  **It runs on CPU.** Project 2's *7B OpenVLA* study needed a GPU, but the reused piece
  is its degradation code (pure NumPy); applied to the small ACT policy it runs on the
  same `ubuntu-latest` runners as Tiers 1-3. Tier 4 is `workflow_dispatch` + weekly
  (a multi-level sweep is heavier than a single eval), gating on the *cliff location*
  rather than a single nominal number. Run it locally with
  `python scripts/run_perturbation_sweep.py --axis resolution`.
- **Hardware-in-the-loop (HIL).** Swap the MuJoCo env for a driver that talks to real
  compute/actuators. The pyramid is identical: bring-up health checks (`validation.py`)
  become real power-on self-tests; the latency budget becomes a real-time deadline
  measured on the target; the regression gate guards the real system.
- **RTOS scheduling / timing validation.** The benchmark suite already separates policy
  inference from the rest of the loop and surfaces the periodic chunk-boundary WCET
  spike — the inputs a real-time scheduler needs to budget a control task. See
  [TESTING_PHILOSOPHY.md](TESTING_PHILOSOPHY.md).
