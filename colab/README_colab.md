# Running this pipeline on a GPU (Google Colab)

The local run is CPU-only (Intel Arc, no CUDA). To do a *real* training run on a
free/cheap NVIDIA GPU, paste these cells into a Colab notebook
(Runtime → Change runtime type → **T4 GPU** or A100). Everything else — dataset,
policy, CLI — is identical to the local run; only the device changes.

### Cell 1 — check the GPU
```python
!nvidia-smi
```

### Cell 2 — install LeRobot (Colab already has CUDA torch)
```python
!pip install -q lerobot gym-aloha
!python -c "import torch; print('CUDA:', torch.cuda.is_available())"
```

### Cell 3 — train ACT on GPU (full-scale)
```python
!lerobot-train \
  --policy.type=act \
  --dataset.repo_id=lerobot/aloha_sim_transfer_cube_human \
  --policy.device=cuda \
  --policy.push_to_hub=false \
  --wandb.enable=false \
  --batch_size=8 \
  --steps=100000 \
  --log_freq=250 --save_freq=25000 \
  --output_dir=outputs/train/act_aloha_cube_gpu
```
On a T4 this is ~hours (vs ~days on CPU); ACT typically needs ~50k–100k steps to
actually solve transfer-cube. On Colab you usually don't need
`--dataset.video_backend=pyav` because FFmpeg/torchcodec work out of the box.

### Cell 4 — evaluate (success rate in sim)
```python
import os; os.environ["MUJOCO_GL"] = "egl"
!lerobot-eval \
  --policy.path=outputs/train/act_aloha_cube_gpu/checkpoints/last/pretrained_model \
  --policy.device=cuda \
  --env.type=aloha --env.task=AlohaTransferCube-v0 \
  --eval.n_episodes=50
```

### Tip
Mount Google Drive and point `--output_dir` there so checkpoints survive Colab
session resets.
