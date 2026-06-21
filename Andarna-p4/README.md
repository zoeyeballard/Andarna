# OpenVLA Inference Optimization

Profiling and inference optimization for **OpenVLA** (7B-parameter vision-language-action
model) on an NVIDIA A10G. The goal: measure where inference time goes, test precision levels
(BF16 / FP16 / INT8 / INT4), and document the speed-vs-accuracy tradeoffs that matter for
deploying a manipulation policy on a real robot.

## Why this project

Robotics control loops have a latency budget. A policy that needs to act at 10 Hz has ~100 ms
per observation-to-action cycle. This project takes an embedded-systems view of that budget:
which model component (vision encoder, projector, LLM prefill, LLM decode) consumes it, which
CUDA kernels are memory-bound vs compute-bound, and how quantization shifts the profile.

## Hardware / model

- **GPU:** AWS EC2 with NVIDIA A10G (24 GB GDDR6, Ampere `sm_86`)
  — 31.2 TFLOPS FP32, 62.5 TFLOPS FP16, 125 TOPS INT8, ~600 GB/s memory bandwidth.
- **Model:** OpenVLA-7B, four inference stages:
  1. **Vision Encoder** (SigLIP + DINOv2) — processes the camera image
  2. **MLP Projector** — maps vision features into the language embedding space
  3. **LLM Backbone Prefill** — processes instruction + visual tokens
  4. **LLM Backbone Decode** — autoregressively generates action tokens

## Repository layout

```
Andarna-p4/
├── profiling/        # Timing & measurement (torch profiler, NVTX, CUDA events, memory)
├── quantization/     # Precision sweeps and accuracy/parity checks
├── optimization/     # torch.compile, CUDA graphs, batching experiments
├── analysis/         # Parse traces, generate plots, roofline analysis
├── scripts/          # CLI entry points (incl. install_ec2.sh)
├── tests/            # Validation tests
├── results/          # Raw profiling data (gitignored — commit summaries only)
├── figures/          # Generated plots (committed)
└── PROFILING_REPORT.md
```

## Setup (EC2 A10G)

The GPU work runs on the EC2 instance. From a fresh instance:

```bash
bash scripts/install_ec2.sh
```

This verifies the NVIDIA driver / CUDA toolkit, creates a Python 3.10 venv, installs PyTorch
(CUDA build), OpenVLA, LIBERO, and bitsandbytes, then runs a GPU smoke test. See the script
header for prerequisites and flags.

> **Cost note:** GPU instances are expensive — stop the EC2 instance when you are not actively
> profiling.

## Known result worth reproducing

OpenVLA INT8 is *slower and less accurate* than INT4. INT8's quantize/dequantize overhead
isn't offset by enough memory-bandwidth savings; INT4 saves enough bandwidth to come out ahead.
Reproducing and explaining this (rather than "fixing" it) is part of the project's value.

## Workflow

- Develop locally in WSL2; SSH into EC2 to run profiling/inference.
- Run `nsys` / `ncu` on EC2, download `.nsys-rep` / `.ncu-rep`, open in the Nsight GUI locally.
- The GitHub token lives in `.env` as `personal_token` (never committed). Source it before push.
- Work happens on branch `project-4-inference-optimization`; commit after each phase.

## Profiling scripts

All run on the EC2 A10G with the venv active (`source .venv/bin/activate`):

| Script | What it produces |
|---|---|
| `scripts/run_baseline_latency.py` | End-to-end BF16 latency (mean/p50/p95/max) via CUDA events |
| `scripts/run_component_breakdown.py` | Per-stage time + % of total (vision / projector / prefill / decode) |
| `profiling/torch_profiler.py` | Chrome trace + top-20 ops by GPU time and by memory |
| `profiling/nsight_runner.py` | NVTX-annotated runner for an Nsight Systems timeline |

### Nsight Systems timeline (Phase 4)

The runner brackets steady-state steps with `cudaProfilerStart/Stop` and tags the four stages
with NVTX ranges (`VisionEncoder`, `MLPProjector`, `LLM_prefill`, `LLM_decode`). Capture with:

```bash
nsys profile \
  --trace=cuda,nvtx,osrt \
  --cuda-memory-usage=true \
  --capture-range=cudaProfilerApi --capture-range-end=stop \
  --force-overwrite=true \
  --output=traces/openvla_nsys_timeline \
  python profiling/nsight_runner.py --warmup 20 --iters 10
```

Download `traces/openvla_nsys_timeline.nsys-rep` and open it in the Nsight Systems GUI locally.
Quick text check on the box: `nsys stats --report nvtx_pushpop_sum <file>.nsys-rep`.

## Results so far (Phases 0–6)

Full write-up with per-phase detail and caveats: **[PROFILING_REPORT.md](PROFILING_REPORT.md)**.
Raw data per phase is committed under [`results/`](results/).

**Headline:** inference is LLM/GEMM-bound (LLM ≈ 94% of latency; `aten::mm` ≈ 86% of GPU time on
Ampere tensor-core kernels). Quantization buys **memory, not speed** — every quantized config was
slower than BF16, and INT8 was worse than INT4 on both axes (the known OpenVLA anomaly, reproduced).

| Phase | What | Headline result | Data |
|---|---|---|---|
| 1 | BF16 baseline latency | 358.7 ms mean / p95 363.7 / 15.5 GB / **~2.8 Hz** | [json](results/baseline/baseline_latency_bf16.json) |
| 2 | Per-stage breakdown | Vision 5.5% · Projector 0.2% · **Prefill 35% · Decode 59%** (LLM ≈ 94%) | [json](results/baseline/component_breakdown_bf16.json) |
| 3 | PyTorch Profiler | `aten::mm` **86%** of GPU time (ampere_bf16 tensor-core GEMMs); flash-attn active (1.8%) | [txt](results/baseline/torch_profiler_top20.txt) |
| 4 | Nsight timeline (NVTX) | 4 stage ranges captured; steady-state `.nsys-rep` (5.5 MB) | _(trace gitignored)_ |
| 5 | Precision sweep | BF16 359 ms / FP16 360 / **INT8 1365 (3.8×)** / INT4 572 (1.6×); mem 15.5/15.5/8.5/**4.7 GB** | [json](results/quantization/precision_sweep.json) |
| 6a | Action error vs BF16 | FP16 cos 0.997 · INT8 0.953 · **INT4 0.899** (MAE ≈ 41% of action scale) | [json](results/quantization/accuracy_validation.json) |
| 6b | Gripper-flip test | **Refuted** — error is in translation deltas (`dz` worst), not the gripper | [json](results/quantization/action_dim_breakdown.json) |
| 6c | LIBERO success (n=10) | BF16 80% · FP16 70% · INT4 60% (monotonic but within noise) | [json](results/behavioral/libero_success.json) |
| 8 | torch.compile (reduce-overhead) | **errors** on the LLM attention mask (dynamic shape) — no speedup; eager baseline 359 ms | [json](results/optimization/torch_compile.json) |
| 9 | Batch-size scaling | generation **batch-1-locked** (model assert); prefill forward scales 7.0→12.1 items/s (B1→8), saturates by B=2 | [json](results/optimization/batch_scaling.json) |
| 10 | Memory stability (500 iters) | **leak-free** — allocated/peak flat to the byte (+0.0 MB drift) | [json](results/memory/memory_stability.json) · [fig](figures/memory_stability.png) |
| 11 | Reproducibility (5 runs) | **CV 0.093%** of mean latency — far under 5%; benchmark is reproducible | [json](results/baseline/reproducibility.json) |
| 12 | Roofline analysis | ridge (BF16) 104 F/B; **decode AI≈1 (memory-bound), prefill≈250 (compute-bound)**; decode = 2% of FLOPs / 59% of time | [json](results/analysis/roofline.json) · [fig](figures/roofline.png) |
| 13 | Summary figures | 5 charts (latency/memory/accuracy/control-Hz vs precision + BF16 component split) | [figures/](figures/) |
| 14 | Final report | full write-up incl. embedded-systems perspective (Jetson / FPGA / RTOS) | [PROFILING_REPORT.md](PROFILING_REPORT.md) |

**Deployment takeaway:** run BF16/FP16 if memory allows; use INT4 only to *fit* on constrained
memory (4.7 GB), never for speed; avoid INT8. Latency (~2.8 Hz) — not memory — is the binding
constraint for a real control loop.

**Optional follow-ups (not blocking):** per-precision component decomposition (fig1 is BF16-only —
needs GPU); full-suite LIBERO eval (10×20–50 trials) for statistical significance; a fused-INT4
kernel to test decode's roofline bandwidth headroom.

## Status

**All planned phases complete (0–6, 8–14).** Scaffold, EC2 setup, baseline latency, component
breakdown, PyTorch Profiler, Nsight timeline, precision sweep, accuracy + behavioral validation,
torch.compile, batch scaling, memory stability, reproducibility, roofline, summary figures, and
the final report. See [PROFILING_REPORT.md](PROFILING_REPORT.md).
