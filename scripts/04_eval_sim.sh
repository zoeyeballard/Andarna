#!/usr/bin/env bash
# Phase 4a: Roll out the trained ACT policy in the gym-aloha MuJoCo sim and
# measure task SUCCESS RATE over N episodes. This is the eval that actually
# matters -- low training loss does NOT guarantee task success (distribution
# shift / compounding error). The sim resets the cube to random poses each ep.
#
# MUJOCO_GL=egl  -> use EGL offscreen rendering (verified working on this box).
# The env renders the top camera each step and feeds it to the policy.
set -euo pipefail
cd "$(dirname "$0")/.."
source .venv/bin/activate
export HF_HUB_DISABLE_PROGRESS_BARS=1
export MUJOCO_GL=egl

CKPT="${1:-outputs/train/act_aloha_cube/checkpoints/last/pretrained_model}"

lerobot-eval \
  --policy.path="$CKPT" \
  --policy.device=cpu \
  --env.type=aloha \
  --env.task=AlohaTransferCube-v0 \
  --eval.n_episodes=10 \
  --eval.batch_size=1 \
  --output_dir=outputs/eval/act_aloha_cube
