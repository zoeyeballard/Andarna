#!/usr/bin/env bash
# Phase 5 (ALTERNATIVE PATH): evaluate a PRETRAINED pi0 VLA on the LIBERO
# benchmark. This is the bridge to Project 2 (OpenVLA fine-tuning on LIBERO).
#
# !!! GPU STRONGLY RECOMMENDED !!!
# pi0 is a ~3B-parameter Vision-Language-Action model. On CPU its inference is
# minutes PER STEP -> a 10-episode eval would take many hours/days. Run this on
# a CUDA GPU (Colab A100, Lambda/RunPod, or WSL+NVIDIA). Swap device=cuda.
#
# Install (one-time):
#   uv pip install "lerobot[libero]"        # pulls LIBERO + robosuite deps
#   # LIBERO also needs its task bddl assets; the lerobot libero env handles
#   # download on first use.
set -euo pipefail
cd "$(dirname "$0")/.."
source .venv/bin/activate
export HF_HUB_DISABLE_PROGRESS_BARS=1
export MUJOCO_GL=egl

DEVICE="${1:-cuda}"   # pass 'cpu' only if you really want to wait

lerobot-eval \
  --policy.path=lerobot/pi0_libero_finetuned \
  --policy.device="$DEVICE" \
  --env.type=libero \
  --env.task=libero_object \
  --eval.n_episodes=10 \
  --eval.batch_size=1 \
  --output_dir=outputs/eval/pi0_libero_object
