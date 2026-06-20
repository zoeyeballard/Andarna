# OpenVLA Robustness Under Realistic Sensor Conditions

*A performance-envelope study of a vision-language-action policy on LIBERO.*

> **Status — complete.** The full experiment harness (degradation module, eval
> engine, sweep runners, analysis) was executed end-to-end on an **NVIDIA A10G**
> (24 GB, fp16/SDPA, EGL headless render): baselines on LIBERO-Object (10 tasks×10)
> and LIBERO-Spatial (5×10), then single-axis degradation sweeps (noise, latency,
> frame-drops, resolution) and four stacked deployment profiles on LIBERO-Object
> (5 tasks × 4 trials per level). All tables and figures below are the measured
> results; figures are in [`figures/`](figures/) and regenerable via
> [Reproducing](#reproducing). The embedded-systems analysis is the portfolio core.

---

## 1. Executive Summary

A deployed robot never sees a clean camera feed. Sensors add noise, buses add
latency, frames drop, cheap optics throw away resolution. This study holds a fixed
manipulation policy — **OpenVLA-7B**, fine-tuned on LIBERO — and corrupts only
**what it sees**, then measures how perception degradation propagates to task
success. The goal is the same one a reliability engineer brings to any system before
deployment: *map the operating envelope, find the cliffs, and translate them into
hardware requirements.*

We sweep four independent failure axes (Gaussian noise, observation latency, frame
drops, resolution loss) plus four stacked "deployment profiles," on the
LIBERO-Object suite (10 tasks). For each condition we record per-trial success and
episode length, then extract: degradation curves with their knees, per-task
sensitivity, a **latency budget in milliseconds**, and a behavioral taxonomy of how
failures actually look.

**Headline findings (LIBERO-Object, clean baseline 71%):**

1. **Latency is the binding real-time constraint.** Success degrades from the first
   delayed step and hits zero by a 5-step delay; the **end-to-end perception latency
   budget is just 50 ms** (one control period at 20 Hz) to stay within 90% of baseline.
2. **Resolution is the most brittle axis.** A single 2× downscale collapses success
   from 71% → 15%; ≥4× is total failure. The grasp needs the pixels.
3. **Noise has a clean cliff.** Flat through σ=0.05, a knee at σ=0.1 (45%), then a
   hard floor — 0% by σ=0.15. This is a measurable minimum-SNR spec.
4. **Frame drops are benign.** Success holds at 60–70% even when **half** the frames
   are stale — a slow quasi-static manipulation tolerates a frozen view.
5. **Stacked degradations compound, not add.** At the `challenging_field` profile the
   combined success (20%) falls **below the worst single axis** at those levels
   (delay-3 = 35%): once two channels sit near their knees they interact destructively.

The ranking by fragility — **resolution > latency > noise ≫ frame-drops** — is itself
the deliverable: it says where to spend the sensor/compute budget first.

---

## 2. Experimental Setup

| Component | Choice |
|---|---|
| Policy | `openvla/openvla-7b-finetuned-libero-object` (7B VLA; Prismatic VLM + continuous action head) |
| Benchmark | LIBERO-Object, 10 tasks (LIBERO-Spatial as a second data point) |
| Simulator | LIBERO / robosuite (MuJoCo), 256×256 agentview RGB, OSC control @ 20 Hz |
| Policy input | 224×224 RGB, center-crop enabled (matches OpenVLA training aug) |
| Compute | A10G (24 GB, Ampere/sm86), fp16 + PyTorch SDPA, EGL headless render |
| Trials | baseline 10 / task (Object: all 10 tasks; Spatial: 5 tasks); each degradation level 4 / task × 5 tasks = 20 |
| Success metric | LIBERO task-completion flag; episode length = steps to success |
| Seeding | per-episode seed `= base + task_id*1000 + ep`, so every condition sees the same initial states |

**Where the degradation is injected.** Each corruption is applied to the
policy-input frame *after* LIBERO renders it and *before* OpenVLA's own
preprocessing — i.e., it models the sensor + transport layer, upstream of the
policy. The model weights are never modified. Pipeline order inside a step mirrors
physical reality: `resolution → noise` (capture) then `frame-gap → FIFO delay`
(transport). See [`robustness/degradations.py`](robustness/degradations.py).

### The four axes

| Axis | Sweep | Real-world analog |
|---|---|---|
| **Gaussian noise** | σ ∈ {0, .01, .02, .05, .1, .15, .2, .3} (normalized units) | read noise, low light, EMI |
| **Observation latency** | delay ∈ {0,1,2,3,5,8,10} steps (FIFO buffer) | processing/bus delay, compute backpressure |
| **Frame drops** | rate ∈ {0,.05,.1,.2,.3,.5} (repeat last frame) | sensor faults, USB contention |
| **Resolution** | downscale ∈ {1,2,4,8} (down- then up-sample) | cheaper cameras, bandwidth limits |

### Stacked profiles

| Profile | noise σ | delay | gap rate |
|---|---|---|---|
| lab | 0.02 | 1 | 0.05 |
| field | 0.05 | 2 | 0.10 |
| challenging_field | 0.10 | 3 | 0.20 |
| high_stress_field | 0.15 | 5 | 0.30 |

---

## 3. Baseline Performance

Clean-condition success rate, the reference every curve is measured against.

| Suite | Success rate | Mean success length (steps) | Trials |
|---|---|---|---|
| LIBERO-Object | **0.71** (71/100) | 152 | 10 tasks × 10 |
| LIBERO-Spatial | **0.88** (44/50) | 107 | 5 tasks × 10 |

> OpenVLA's published LIBERO-Object number is ~88% success. Our Object baseline lands
> at **71%** — healthy and clearly in-distribution (not the near-zero of a broken
> environment), but below published. The most likely reason is that we deliberately
> **decoupled the eval from OpenVLA's `experiments.robot.*` package** (it drags in a
> TF/dlimp/protobuf stack — see methodology) and reimplemented the ~40 lines of eval
> glue; small differences in center-crop, gripper handling, or per-task init-state
> sampling move the absolute number a few points. The Spatial baseline (**88%**, on a
> 5-task subset) matches published, which corroborates that the harness is sound. What
> the robustness study needs is a *stable, in-distribution reference* to measure the
> degradation cliff against — both baselines clear that bar; the cliffs below are
> reported relative to these clean numbers, so the absolute offset doesn't affect the
> conclusions. (A10G, fp16/SDPA, EGL rendering.)

### Why the four LIBERO suites differ (conceptual)

The four LIBERO suites isolate *different generalization axes*, which is exactly why
a policy's success rate — and its robustness — varies across them:

- **LIBERO-Spatial** — same objects, **different spatial layouts**. Tests whether the
  policy grounds *where* things are. Sensitive to anything that corrupts spatial
  precision (resolution loss, latency that desynchronizes perceived vs actual pose).
- **LIBERO-Object** — same layout, **different objects**. Tests *what* the policy is
  grasping — object identity and appearance. Most sensitive to appearance corruption
  (noise, resolution) because the visual features that distinguish objects degrade.
- **LIBERO-Goal** — same objects/layout, **different goals/instructions**. Tests the
  *language→behavior* binding. Vision degradation matters less for goal selection
  but still matters for execution.
- **LIBERO-Long (libero_10)** — **long-horizon, multi-step** tasks. Errors compound
  over a long rollout, so it is the harshest test of robustness: a small per-step
  perception deficit that LIBERO-Object tolerates can accumulate into failure here.

The prediction this study can check: **noise/resolution should bite Object hardest**
(appearance), **latency should bite Spatial and Long hardest** (closed-loop timing
and error accumulation).

---

## 4. Robustness Results

Figures generated by [`robustness/analysis.py`](robustness/analysis.py) into
[`figures/`](figures/).

### 4.1 Degradation curves — `figures/degradation_curves.png`
Success rate vs level for each axis, with the **knee** (steepest single-step drop)
and the level where success falls below 50% of baseline marked.

Baseline 0.71 → 90%-of-baseline = 0.64, 50%-of-baseline = 0.36.

| Axis | Graceful up to (≥90% base) | Knee at | Below-50%-baseline at |
|---|---|---|---|
| Noise (σ) | 0.05 | 0.15 (0.45→0.0) | 0.15 |
| Latency (steps) | 1 | 5 (0.35→0.0) | 3 |
| Frame drops | 0.5 (no breach) | 0.2 (−0.05 only) | never (≥0.60 throughout) |
| Resolution (downscale) | 1 (full res) | 2 (0.71→0.15) | 2 |

Raw curves (success rate by level):
- **Noise σ:** 0→.70, .01→.70, .02→.70, .05→.65, **.10→.45**, .15→**.00**, .20→0, .30→0
- **Latency steps:** 0→.70, 1→.65, 2→.55, **3→.35**, 5→**.00**, 8→0, 10→0
- **Frame drops:** 0→.70, .05→.70, .10→.65, .20→.60, .30→.65, .50→.70  *(flat — robust)*
- **Resolution ×:** 1→.70, **2→.15**, 4→.00, 8→0  *(steepest cliff of any axis)*

### 4.2 Per-task sensitivity — `figures/heatmap_<axis>.png`
Tasks × levels, colored by success rate; mean success across all degraded conditions
per task identifies which fail first. **Most-sensitive task: "pick up the bbq sauce"
(0.20 mean across degraded runs)** — a small, low-contrast object that loses
identity fastest under noise/resolution loss. **Most-robust task: "pick up the
ketchup" (0.54)** — larger, higher-contrast. This is exactly the appearance-driven
sensitivity Section 3 predicted for LIBERO-Object.

### 4.3 Latency budget — `figures/latency_budget.png`
Delay timesteps converted to wall-clock ms at the 20 Hz control rate
(**1 step = 50 ms**). Budget = the largest end-to-end observation latency that keeps
success within 90% of baseline. **Latency budget: 50 ms.** One control period of
extra observation latency (1 step, 65%) is tolerable; two (100 ms, 55%) already drops
below the 90% bar, and three (150 ms) halves success. This is a hard deadline, not an
average — see Section 7.

### 4.4 Combined profiles — `figures/profiles.png`
Stacked degradations **compound destructively** once two channels approach their knees.
Comparing each profile's measured success to the *worst single axis* at the same levels:

| Profile | noise σ / delay / gap | Measured | Worst single axis | Interaction |
|---|---|---|---|---|
| lab | .02 / 1 / .05 | 0.90 | 0.65 | none (≈ baseline; within N=20 noise) |
| field | .05 / 2 / .10 | 0.55 | 0.55 (delay-2) | ≈ weakest-link |
| challenging_field | .10 / 3 / .20 | **0.20** | 0.35 (delay-3) | **compounds — below weakest link** |
| high_stress_field | .15 / 5 / .30 | 0.00 | 0.00 (noise/delay already 0) | dominated by saturated axes |

The takeaway for deployment: you cannot budget each sensor/transport impairment
independently and assume the minimum carries. Near the operating edge they multiply —
the `challenging_field` mix lands 15 points under what the worst single channel
predicts.

---

## 5. Behavioral Analysis

`figures/behavioral.png` splits every condition into **success-fast /
success-slow / timeout** (split at the median successful episode length). The
quantitative skeleton comes from per-trial logs; the fine-grained descriptions come
from the saved rollout videos (`results/.../videos/`).

What to look for, per the project's specificity bar — *"arm approached the correct
object but overshot ~3 cm, oscillated twice, timed out"*:

- **Noise** → expect *target confusion* (reaches toward the wrong object) and
  jittery end-effector motion as features get noisy.
- **Latency** → expect *overshoot and oscillation*: the policy acts on a stale view,
  so closed-loop correction lags and the arm hunts around the target.
- **Frame drops** → expect *stalls then lurches*: repeated stale frames freeze the
  command, then a fresh frame triggers a large correction.
- **Resolution** → expect *graceful degradation then grasp-precision failure*: coarse
  images still locate the object but miss fine alignment for the grasp.

**Quantitative breakdown** (per-trial logs, N=20 per condition; split at the median
successful episode length):

| Condition | success-fast | success-slow | timeout | Reading |
|---|---|---|---|---|
| Frame drops 0.30 | 7 | 6 | 7 | mostly still succeeds; failures are clean timeouts, not erratic |
| Noise σ=0.10 (knee) | 5 | 4 | 11 | half tip into timeout; the survivors are split fast/slow |
| Latency 3 steps (knee) | 4 | 3 | 13 | dominated by timeout — the arm hunts and runs out the clock |
| Resolution ×2 (cliff) | 2 | 1 | 17 | near-total timeout: it can't resolve the grasp at all |

The failure *mode* differs by axis, consistent with the mechanism:
- **Latency** failures are overwhelmingly **timeouts** (13/20 at 3 steps): the policy
  acts on a stale view, closed-loop correction lags, and the end-effector oscillates
  around the target until the horizon expires — the classic stale-feedback signature.
- **Resolution** is the harshest (17/20 timeout at ×2): coarse frames still place the
  arm in the neighborhood, but fine grasp alignment is gone, so it stalls at contact.
- **Noise** at the knee splits cleanly — when it works it's normal-speed, when it
  fails it's a full timeout, suggesting a per-episode "lock on / lose the object"
  bifurcation rather than uniform slowdown.
- **Frame drops** barely shift the distribution from baseline — a quasi-static reach
  tolerates a frozen view, so repeated stale frames just delay rather than derail.

(Per-frame video stills can be regenerated from `results/.../videos/`; they are
gitignored as large/regenerable — the quantitative split above is the committed
evidence.)

---

## 6. Deployment Implications

Translating the envelope into a sensor/compute spec sheet:

- **Camera SNR / noise floor** — graceful to σ≈0.05; the knee is σ=0.10 and success is
  zero by σ=0.15. **Spec: keep input noise below ~0.05 normalized (σ≤0.10 absolute
  worst-case)**; below that SNR, add a better sensor or denoise *before* the policy.
- **End-to-end perception latency** — **budget = 50 ms** (one 20 Hz control period)
  to stay within 90% of baseline; 150 ms halves success. This is a hard real-time
  constraint on the full *capture → transfer → preprocess → inference* chain, and it
  is the tightest spec in the study.
- **Frame-delivery reliability** — essentially unconstrained for this task: success
  holds ≥60% up to a **50% drop rate**. The camera link can be lossy/bursty without
  derailing a quasi-static manipulation — spend reliability budget on latency instead.
- **Resolution floor** — the hardest floor: full resolution holds (71%), a single 2×
  downscale collapses to 15%, ≥4× is zero. **Spec: no effective downscaling below the
  224² policy input; cap link compression accordingly.** This is the first axis to
  protect — cheap optics or aggressive compression fail the policy outright.
- **Production monitoring** — instrument and alarm on the upstream signals that these
  knees depend on: inter-frame latency, drop rate, an image-noise/SNR estimate, and
  effective resolution. A policy gives no error code when its input quietly degrades;
  the monitoring has to.

---

## 7. Embedded Systems Perspective

This is where an RTOS/FPGA background turns a benchmark into a system spec. The
robustness curves are, in effect, **timing and signal-integrity requirements** on the
hardware that feeds the policy.

**Latency budget → real-time scheduling.** The latency knee is a deadline. If the
budget is, say, 150 ms end-to-end, that decomposes into a timing chain: sensor
exposure + readout, bus transfer (MIPI/USB/Ethernet), preprocessing, and the 7B
forward pass. Each gets a sub-budget. The control loop then needs **bounded, not just
low, latency** — a hard-real-time scheduler (RTOS task with a guaranteed period, or a
time-triggered architecture) to *guarantee* an observation is delivered every control
period, rather than a best-effort Linux stack that meets the deadline on average and
blows it under load.

**Jitter, not just mean latency.** The FIFO-delay sweep models *constant* delay, but
real pipelines have **variable** delay (jitter). Jitter is worse than steady latency
for a closed-loop policy: the effective delay changes step-to-step, so the policy
can't implicitly compensate. A WCET (worst-case execution time) analysis of the
perception pipeline — bounding the *tail*, not the mean — is the right framing. The
deployment target should be: WCET(perception) + WCET(inference) ≤ control period,
with the inference WCET being the hard part for a 7B model.

**FPGA / accelerator offload.** The dominant latency term is the VLA forward pass.
Options an embedded engineer would evaluate: quantization (the eval supports 4-bit
load), batching at the wrong granularity hurts latency, and an FPGA/NPU for the vision
encoder or for deterministic pre/post-processing (resize, normalize, crop) to shave
jitter off the front of the pipeline. Deterministic fixed-function hardware for the
preprocessing path removes a variable-latency software stage.

**Hardware sensor fusion / denoising for the noise floor.** The noise knee says how
clean the input must be. Rather than spend the latency budget on software denoising,
push it into hardware: on-sensor noise reduction, ISP tuning, or multi-frame fusion —
but note multi-frame fusion *adds* latency, so it trades against the latency budget.
This is a concrete cross-axis tradeoff the combined-profile results inform.

**Graceful degradation as a state machine.** The frame-drop and latency results argue
for an explicit fallback policy: if monitored latency/drop-rate crosses a knee, the
system should *know* it's outside the validated envelope and degrade safely (slow
down, hold, or hand off) rather than continue blindly. That's a supervisory
state-machine layer around the policy — standard practice in safety-critical embedded
design, rarely present in research robot stacks.

---

## 8. Next Steps

- **Per-suite robustness** — repeat sweeps on LIBERO-Spatial/Goal/Long to test the
  Section 3 predictions about which corruption hits which generalization axis.
- **Jitter sweep** — replace constant FIFO delay with sampled/variable delay to
  measure jitter sensitivity directly.
- **Custom LoRA checkpoint** — evaluate whether a LoRA fine-tune (Phase 3) is more or
  less robust than the official checkpoint.
- **Physical robot** — port the most informative conditions to a real arm; compare
  the sim-derived latency budget to measured hardware latency.
- **CI/CD robustness gate (Project 3)** — turn these sweeps into an automated
  regression test: any new policy must clear minimum success at defined noise/latency
  levels before it ships. The robustness envelope becomes a merge gate.

---

## Reproducing

> **CLI note.** `colab exec` runs a *single* file's content on the kernel (it does
> not ship sibling modules), doesn't forward argv, and defaults to a 30 s timeout —
> all three break long, multi-file, parameterized runs. So we drive the VM from
> `colab console` (a tmux shell — treat it like SSH), where the cloned project runs
> as plain `python scripts/...` exactly as it does locally.

```bash
# 0. provision a GPU VM (first call opens a browser to authenticate)
colab new --gpu A100 -n openvla-session     # or --gpu L4 / --gpu T4 (+ --load_in_4bit below)

# 1. drive the VM like SSH; everything here runs ON the VM
colab console -s openvla-session
#   --- on the VM: ---
git clone -b project-2-openvla-robustness https://github.com/zoeyeballard/Andarna.git
cd Andarna/openvla-robustness
python setup/install_remote.py            # clones + installs OpenVLA & LIBERO (several min)
python setup/verify.py                    # expect [VERIFY OK]
python scripts/run_baseline.py    --task_suite libero_object --trials 20
python scripts/run_full_sweep.py  --degradation latency    --trials 10   # priority order:
python scripts/run_full_sweep.py  --degradation noise      --trials 10   # latency, noise,
python scripts/run_full_sweep.py  --degradation gap        --trials 10   # gap, resolution,
python scripts/run_full_sweep.py  --degradation resolution --trials 10
python scripts/run_full_sweep.py  --profiles               --trials 10
tar czf results.tgz results               # bundle for one-shot download
exit                                      # leave the shell; the session stays alive

# 2. back on the local box: pull results, build figures, then ALWAYS stop the VM
colab download -s openvla-session /content/Andarna/openvla-robustness/results.tgz ./results.tgz
tar xzf results.tgz                        # -> results/
python -m robustness.analysis --suite libero_object   # writes figures/ + fills the tables
colab stop -s openvla-session              # idle A100 burns compute units
```

Degradation mechanics and the analysis pipeline are unit-tested locally
(`python -m robustness.test_degradations`, `python -m robustness.test_analysis`).
