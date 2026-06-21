# Project: OpenVLA Inference Optimization

## Environment
- OS: Windows 11 with WSL2 (Ubuntu) for local dev
- GPU: AWS EC2 instance with NVIDIA A10G (24GB, Ampere)
- GitHub: personal access token in .env as `personal_token`
- Push to branch: project-4-inference-optimization

## What This Project Is
Standard ML inference optimization for a robotics vision-language-action model (OpenVLA, 7B params).
We measure how long each part of the model takes to run, test different precision levels
(BF16, FP16, INT8, INT4), and document the speed/accuracy tradeoffs for robot deployment.

## Tech Stack
- OpenVLA (github.com/openvla/openvla) — VLA model for robot manipulation
- LIBERO — simulation benchmark for evaluation
- PyTorch Profiler — for measuring operator-level timing
- bitsandbytes — for INT8 and INT4 quantization
- torch.cuda.Event — for precise component timing
- NVIDIA Nsight Systems (nsys) — for GPU timeline visualization

## OpenVLA Architecture (4 stages)
1. Vision Encoder (SigLIP + DINOv2) — processes camera image
2. MLP Projector — maps vision features to language space
3. LLM Backbone Prefill — processes instruction + visual tokens
4. LLM Backbone Decode — generates action tokens autoregressively

## Key Known Result
OpenVLA INT8 is actually slower AND less accurate than INT4 (published in the original paper).
INT8 quantization/dequantization overhead isn't offset by bandwidth savings. INT4 saves enough
bandwidth to compensate. Reproducing and explaining this is part of the project.

## GitHub Convention
- Source .env for token, push after each phase
- Descriptive commits: feat: baseline timing complete
- .gitignore: .env, __pycache__, *.pt, *.bin, large trace files, results/

## Project Structure
~/repositories/personal/Andarna-p4/
├── profiling/        # Timing and measurement scripts
├── quantization/     # Precision experiments and accuracy checks
├── optimization/     # torch.compile, batching experiments
├── analysis/         # Parse results, generate plots
├── scripts/          # CLI entry points
├── tests/            # Validation tests
├── results/          # Raw data (gitignored)
├── figures/          # Plots (committed)
└── PROFILING_REPORT.md