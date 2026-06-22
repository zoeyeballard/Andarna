# Andarna 🐉

**A four-project robotics-AI portfolio — the full vertical slice of a modern robot
software stack, from data and training through robustness, CI/CD, and GPU inference
optimization. Built on an AWS EC2 A10G and an embedded-systems mindset.**

Each project maps to a layer real robotics companies (Figure, Physical Intelligence,
NVIDIA) actually ship:

| # | Project | Layer | Status | Headline result |
|---|---|---|---|---|
| **1** | LeRobot + ACT | Data → training → sim eval | ✅ merged | ACT retrained to convergence → **66%** task success (LIBERO/ALOHA cube) |
| **2** | [OpenVLA robustness](openvla-robustness/) | Failure analysis | ✅ merged | Robustness envelope of OpenVLA-7B; fragility **resolution > latency > noise ≫ drops** |
| **3** | [robotics-test-infra](robotics-test-infra/) | Test infrastructure | ✅ merged | 4-tier GitHub Actions CI/CD; P2↔P3 perturbation bridge |
| **4** | [Andarna-p4](Andarna-p4/) | GPU inference optimization | ✅ merged | OpenVLA-7B profiled + quantized; **quantization buys memory, not speed** |

> 🖥️ **[`visualizer.html`](visualizer.html)** — an interactive single-page reference covering
> all four projects phase-by-phase (live demos for action chunking, ACT/diffusion, the VLA
> stack, LoRA, the robustness cliff, the CI pyramid, the precision/roofline tradeoffs) plus a
> deep dive on the five frontier growth areas (π0 flow matching, World Action Models, and more).

---

## Project 1 — LeRobot + ACT (data → training → sim eval)

Train and evaluate an **ACT** (Action Chunking with Transformers) imitation policy entirely in
simulation with Hugging Face **LeRobot** — the full data-to-deployment loop, CPU-capable and
GPU-scalable.

| | |
|---|---|
| Framework | LeRobot · Policy: **ACT** (52M params, ResNet18 + transformer + CVAE) |
| Dataset | `lerobot/aloha_sim_transfer_cube_human` (50 episodes, bimanual ALOHA) |
| Simulator | **MuJoCo** + gym-aloha |
| Result | Retrained to convergence (100k steps, A10G) → **66% success (33/50)**; checkpoint on HF Hub `zoeyeballard/act-aloha-cube` |

```bash
uv venv --python 3.12 && uv pip install --torch-backend=cpu lerobot gym-aloha
python scripts/01_verify.py          # stack + MuJoCo sanity
python scripts/02_explore_dataset.py # inspect the LeRobotDataset
bash   scripts/03_train_act.sh       # train ACT
bash   scripts/04_eval_sim.sh        # closed-loop sim success rate
```

**Gotchas solved** (full detail in [NOTES.md](NOTES.md)): CPU-only torch via `uv`
(`pip install lerobot` pulls ~3 GB of unusable CUDA wheels); video decoding needs
`--dataset.video_backend=pyav`; headless sim needs `MUJOCO_GL=egl`; disable `push_to_hub`.

## Project 2 — OpenVLA robustness ([openvla-robustness/](openvla-robustness/))

Held OpenVLA-7B fixed and degraded only what it *sees*, measuring exactly where the policy breaks
on LIBERO (100 episodes/condition, A10G, fp16/SDPA/EGL). Baselines: Object 71%, Spatial 88%.

- **Fragility ranking:** `resolution > latency > noise ≫ frame-drops`.
- **Operating envelope:** SNR floor σ < 0.05; latency budget ~50 ms (1 step @ 20 Hz); resolution
  floor = native only (2× downsample → 70%→15%); frame drops benign even at 50%.
- **Behavioral signatures:** latency → oscillation/overshoot/timeout; noise → lock-on/lose
  bifurcation; resolution → can't resolve grasp targets. See
  [ROBUSTNESS_REPORT.md](openvla-robustness/ROBUSTNESS_REPORT.md).

## Project 3 — robotics-test-infra ([robotics-test-infra/](robotics-test-infra/))

A four-tier GitHub Actions CI/CD pipeline for a learned policy — answering both "does it work?"
and "how gracefully does it degrade?"

- **Tier 1** lint + 62 unit tests (<3 min) · **Tier 2** Docker MuJoCo sim eval · **Tier 3**
  regression gate vs stored baseline · **Tier 4** perturbation sweep — the **P2↔P3 bridge** that
  reuses P2's pure-NumPy degradation module against the ACT policy on CPU (resolution cliff at 4×).
- Success gate ≥40% (vs 66% operating point), seed-locked deterministic eval, p95 ~0.85 ms.
  See [ARCHITECTURE.md](robotics-test-infra/ARCHITECTURE.md) and
  [TESTING_PHILOSOPHY.md](robotics-test-infra/TESTING_PHILOSOPHY.md).

## Project 4 — inference optimization ([Andarna-p4/](Andarna-p4/))

Profiled and quantized the OpenVLA-7B inference pipeline on the A10G across 14 phases — where the
time goes, what precision costs, and what it means for a real control loop.

- **LLM/GEMM-bound:** LLM backbone ≈ 94% of latency (decode 59% + prefill 35%); `aten::mm` ≈ 86%
  of GPU time on Ampere tensor-core kernels.
- **Quantization buys memory, not speed:** BF16 359 ms / FP16 360 / **INT8 1365 (3.8× slower)** /
  INT4 572 (1.6×); memory 15.5 / 15.5 / 8.5 / **4.7 GB**. Reproduces the known *INT8-worse-than-INT4*
  anomaly.
- **Roofline:** decode is memory-bound (arithmetic intensity ≈ 1, ~100× below the BF16 ridge) and
  is **2% of FLOPs but 59% of time**; prefill/vision are compute-bound.
- **Deployment:** ~2.8 Hz at BF16 — latency, not memory, is the binding constraint. Run BF16/FP16
  if it fits; use INT4 only to *fit* on constrained edge memory; never INT8. Inference is
  deterministic (run-to-run CV 0.09%, leak-free over 500 iters). Full write-up + figures in
  [PROFILING_REPORT.md](Andarna-p4/PROFILING_REPORT.md).

---

## Repo layout

```
Andarna/
├── scripts/                 # P1: LeRobot/ACT pipeline (verify → train → eval)
├── colab/                   # P1: run the same pipeline on a free GPU
├── openvla-robustness/      # P2: OpenVLA robustness study + report
├── robotics-test-infra/     # P3: 4-tier CI/CD + perturbation bridge
├── Andarna-p4/              # P4: inference profiling/quantization + report + figures
├── visualizer.html          # interactive reference for all four projects
├── NOTES.md                 # P1 command log + results + interview concepts
└── README.md
```

Each project lives on its own and is independently runnable; together they trace one policy
(ACT/OpenVLA) from raw data to a profiled, deployment-ready inference budget.
