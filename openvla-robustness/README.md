# OpenVLA Robustness Analysis (Andarna — Project 2)

Fine-tune / evaluate **OpenVLA-7B** on the **LIBERO** simulation benchmark, then
systematically **degrade the observation pipeline** to map *where and why* the
policy breaks. The fine-tuning is table stakes; the **robustness analysis is the
portfolio piece**.

> Sibling to Project 1 (LeRobot ACT, in the repo root). Where Project 1 trained a
> single-task specialist (ACT, ~50 M params) from scratch, this project probes a
> language-conditioned **VLA generalist** (~7 B params) and asks an
> embedded-systems question: *how gracefully does perception failure propagate to
> control failure?*

## The question

A real robot's camera feed is never clean: sensors add noise, buses add latency,
frames drop, cheap optics lose resolution. We hold the **model fixed** and corrupt
only **what it sees**, then measure the success-rate cliff for each failure mode:

| Degradation | Sweep | Simulates |
|---|---|---|
| Gaussian image noise | σ ∈ {0, .01, .02, .05, .1, .15, .2, .3} | sensor noise, low light, EMI |
| Observation latency | delay ∈ {0,1,2,3,5,8,10} steps | processing/bus delay, compute backpressure |
| Frame drops | rate ∈ {0,.05,.1,.2,.3,.5} | sensor faults, USB packet loss |
| Resolution | downscale ∈ {1,2,4,8} | cheaper cameras, bandwidth limits |
| Combined profiles | mild → edge-case | realistic stacked degradation |

Deliverable: degradation curves, per-task vulnerability heatmaps, a failure-mode
taxonomy, and `ROBUSTNESS_REPORT.md` — including an **embedded-systems
perspective** (latency budgets, WCET, RTOS scheduling, sensor-fusion implications).

## Environment

- **Local (this WSL2 box):** all code authoring, the degradation module, analysis,
  plotting, report writing. CPU-only (Intel, no CUDA).
- **Remote GPU (Colab via `google-colab-cli`):** only the GPU-bound work — model
  eval, sweeps, optional LoRA fine-tune. Scripts are authored locally and shipped
  with `colab exec`.

## Layout

```
openvla-robustness/
├── setup/                  install_remote.py, verify.py   (run on the Colab VM)
├── scripts/                run_baseline.py, run_degradation.py, run_full_sweep.py, finetune.py
├── robustness/             degradations.py (obs-level wrappers), analysis.py (plots)
├── results/                baseline/  robustness/        (raw per-trial JSON gitignored, summary CSV/JSON committed)
├── figures/                generated plots (committed)
└── ROBUSTNESS_REPORT.md    final report
```

## Workflow

```bash
# one-time: GPU session
colab new --gpu A100 -n openvla-session
colab exec -s openvla-session -f setup/install_remote.py
colab exec -s openvla-session -f setup/verify.py

# baseline + sweeps
colab exec -s openvla-session -f scripts/run_baseline.py
colab exec -s openvla-session -f scripts/run_full_sweep.py -- --degradation latency
colab download -s openvla-session results/ -o ./results/

# ALWAYS stop when idle — sessions burn compute units
colab stop -s openvla-session
```

## Status

**Code complete; results pending GPU execution.** The full harness is built and the
non-GPU pieces are unit-tested locally:

- `robustness/degradations.py` — the four-axis degradation module (+ stacked profiles).
  Tests: `python -m robustness.test_degradations` (7 pass).
- `scripts/libero_eval.py` — shared OpenVLA+LIBERO rollout engine with the obs hook.
- `scripts/run_baseline.py`, `run_degradation.py`, `run_full_sweep.py`, `finetune.py`.
- `robustness/analysis.py` — curves, heatmaps, latency budget, profiles, behavioral.
  Tests: `python -m robustness.test_analysis` (figure pipeline verified on synthetic data).
- `setup/install_remote.py`, `setup/verify.py` — VM provisioning + sanity check.
- `ROBUSTNESS_REPORT.md` — methodology + embedded-systems analysis written; results
  tables marked ⟨PENDING⟩ until the sweeps run.

Remaining: the GPU-bound steps (baseline, sweeps, optional LoRA). `colab auth` is
interactive, so those run with a human in the loop — see [Workflow](#workflow).

Local dev deps: `pip install -r requirements-local.txt`. See the repo-root `NOTES.md`
§6 for how this follows from Project 1.
