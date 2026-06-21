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

## Status

Phase 0 — project scaffolding. See [PROFILING_REPORT.md](PROFILING_REPORT.md) for findings as
they land.
