#!/usr/bin/env bash
# Phase 3: Train an ACT (Action Chunking Transformer) policy on the ALOHA
# sim transfer-cube dataset. Configured for CPU (no NVIDIA GPU here).
#
# To run a FULL GPU training instead (e.g. on Colab / a rented A100), change:
#     --policy.device=cpu      ->  --policy.device=cuda
#     --steps=1000             ->  --steps=100000   (ACT paper-scale)
#     --num_workers=0          ->  --num_workers=4
# and drop --dataset.video_backend=pyav (use the default torchcodec if FFmpeg
# is installed). Everything else stays identical -- same dataset, same CLI.
set -euo pipefail
cd "$(dirname "$0")/.."
source .venv/bin/activate
export HF_HUB_DISABLE_PROGRESS_BARS=1

lerobot-train \
  --policy.type=act \
  --dataset.repo_id=lerobot/aloha_sim_transfer_cube_human \
  --dataset.video_backend=pyav \
  --policy.device=cpu \
  --policy.push_to_hub=false \
  --wandb.enable=false \
  --num_workers=0 \
  --batch_size=8 \
  --steps=400 \
  --log_freq=25 \
  --save_freq=200 \
  --eval_freq=0 \
  --seed=1000 \
  --output_dir=outputs/train/act_aloha_cube
