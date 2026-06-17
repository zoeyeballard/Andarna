# Andarna — NOTES (LeRobot ACT Pipeline, Project 1)

Built on **WSL2 Ubuntu**, Python 3.12, **CPU-only** (Intel Arc iGPU, no CUDA),
LeRobot **0.5.1**, MuJoCo 3.8, gym-aloha 0.1.4. Date: 2026-06-16.

---

## 1. Exact commands and what each does

### Environment (Phase 1)
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh   # fast Python/dep manager
uv venv --python 3.12                              # standalone CPython 3.12 venv
uv pip install --torch-backend=cpu lerobot         # LeRobot + CPU torch (no CUDA)
uv pip install gym-aloha                           # MuJoCo + ALOHA sim env
python scripts/01_verify.py                        # imports + a real MuJoCo step
```
- `--torch-backend=cpu` is the key flag: the default pulls ~3 GB of unusable
  `nvidia-*` CUDA wheels (and one failed on a DNS blip). CPU wheels are ~200 MB.

### Dataset exploration (Phase 2)
```bash
python scripts/02_explore_dataset.py               # loads LeRobotDataset, prints schema
```

### Training (Phase 3)
```bash
lerobot-train \
  --policy.type=act \
  --dataset.repo_id=lerobot/aloha_sim_transfer_cube_human \
  --dataset.video_backend=pyav \      # decode video obs w/o system FFmpeg
  --policy.device=cpu \
  --policy.push_to_hub=false \        # default is TRUE -> would fail on auth
  --wandb.enable=false \
  --num_workers=0 \                   # avoid dataloader-worker issues
  --batch_size=8 --steps=400 \        # REDUCED: CPU pipeline validation only
  --log_freq=25 --save_freq=200 \
  --output_dir=outputs/train/act_aloha_cube
```
- Modern CLI is `lerobot-train` / `lerobot-eval` (hyphenated entrypoints), and
  config is dotted: `--policy.type=act`, not the old `--policy=act`.

### Evaluation (Phase 4)
```bash
# 4a. Closed-loop success rate in the MuJoCo sim:
MUJOCO_GL=egl lerobot-eval \
  --policy.path=outputs/train/act_aloha_cube/checkpoints/last/pretrained_model \
  --policy.device=cpu \
  --env.type=aloha --env.task=AlohaTransferCube-v0 \
  --eval.n_episodes=10 --eval.batch_size=1

# 4b. Open-loop action error vs expert (no sim/render needed):
python scripts/04b_inference_heldout.py
```

---

## 2. Dataset: `lerobot/aloha_sim_transfer_cube_human`

| property | value |
|---|---|
| robot_type | aloha (bimanual, two 6-DoF arms + grippers) |
| fps | 50 |
| episodes | 50 |
| total frames | 20,000 (~8 s/episode) |
| task | "Pick up the cube with the right arm and transfer it to the left arm." |
| source | **human** teleoperation → imitation-learning data |

**Features (the LeRobotDataset schema):**
```
observation.images.top   video    (480, 640, 3)   # overhead RGB camera (exteroception)
observation.state        float32  (14,)           # current joint positions (proprioception)
action                   float32  (14,)           # TARGET joint positions for next step
episode_index/frame_index/timestamp/next.done/index/task_index
```
- 14 = (6 arm joints + 1 gripper) × 2 arms.
- `action` is a *target/command*, not the achieved state — a low-level
  controller closes the gap. The policy learns `(image, state) → next action`.

---

## 3. Policy: ACT (Action Chunking with Transformers)

- **Params:** 51.6 M learnable.
- **Architecture:** ResNet18 image encoder → transformer encoder/decoder that
  outputs a *chunk* of future actions, wrapped in a **Conditional VAE** (the
  `kl_weight` term) to model human action multimodality.
- **Default hyperparameters (LeRobot ACT preset):** `chunk_size=100`,
  `n_action_steps=100`, `dim_model=512`, `n_heads=8`, optimizer AdamW,
  `lr=1e-5`, L1 action loss + β·KL.
- **This run (reduced for CPU):** `batch_size=8`, `steps=400`, `seed=1000`.

---

## 4. Results

### Training loss (CPU, 400 steps, batch 8, ~41 min wall)
| step | loss | grad-norm |
|---|---|---|
| 25  | 24.30 | 399.9 |
| 50  | 7.42  | 169.9 |
| 75  | 5.89  | 143.4 |
| 100 | 5.05  | 126.2 |
| 200 | 3.60  | 94.4  |
| 400 (final) | **2.855** | 80.2 |

Clear, fast 8.5× decrease → the imitation objective is being optimized correctly
and the data pipeline is sound. (This is **not** a success metric; see §5.)

### Evaluation
- **4a closed-loop sim (10 episodes, `gym_aloha/AlohaTransferCube-v0`):**
  - `pc_success = 0.0` → **0% task success** (as expected for 400 CPU steps).
  - `avg_max_reward = 0.9`; **4/10 episodes reached reward stage 2** — i.e. the
    policy sometimes drives the right arm to/at the cube but never completes the
    lift-and-transfer. So it learned a coarse behavior, not the full task.
  - Rollout videos written to `outputs/eval/act_aloha_cube/videos/`.
- **4b open-loop action error (held-out episode, 120 steps):**
  mean L1 = **0.62 rad**, worst joint (joint 0) = 1.18 rad.

> The honest takeaway: with a few hundred CPU steps the *pipeline* is fully
> working end-to-end and the policy shows the first glimmer of task-relevant
> behavior, but it is nowhere near competent. A full GPU run (Colab/A100,
> ~50k–100k steps) is what turns this into a policy with a real success rate.
> Reference: a fully-trained ACT reaches ~90% on this sim task.

---

## 5. Interview-ready concepts

**Imitation learning (IL) vs reinforcement learning (RL).**
IL learns from a fixed dataset of *expert demonstrations* — supervised mapping
from observations to expert actions (what ACT does here, "behavioral cloning"
with chunking). No reward function, no environment interaction needed to train.
RL learns from *reward signals* gathered by the agent acting in an environment,
optimizing long-horizon return via trial and error. IL is sample-efficient and
stable but capped by demo quality and suffers **distribution shift** (the policy
visits states the expert never did, and errors compound). RL can exceed the
demonstrator but needs a reward, lots of interaction, and is less stable. Real
robot stacks often combine them (IL pretraining → RL fine-tuning).

**ACT (Action Chunking with Transformers).**
A transformer-based IL policy that predicts a *chunk* of the next k actions per
inference instead of one. The chunk reduces compounding error and handles
non-Markovian human demos. A CVAE latent captures multimodality at train time
(z=0 at inference for decisive motion). Overlapping chunks enable temporal
ensembling for smooth deployment. `chunk_size` trades reactivity (small) vs
robustness-to-latency/smoothness (large) — directly relevant to real-time
control loops.

**VLA (Vision-Language-Action model).**
A generalist robot policy that conditions on **vision + a natural-language
instruction** and outputs actions — e.g. π0 (Physical Intelligence), OpenVLA,
RT-2, GR00T. Typically a pretrained vision-language model backbone (billions of
params) adapted to emit actions (via tokenized actions or a diffusion/flow
"action expert"). Relation to this project: ACT is a **single-task, from-scratch
specialist** (~50 M params, no language grounding beyond one fixed string). A
VLA is a **multi-task, language-conditioned generalist** pretrained on huge
cross-embodiment datasets, then fine-tuned per robot. Same `(obs)→action`
problem, vastly more scale, semantics, and generality. Project 2 (OpenVLA)
steps up to this tier.

**LeRobotDataset format & why standardization matters.**
A standardized on-disk schema: per-frame `observation.*` / `action` tensors in
Parquet, camera streams stored as video, plus metadata (fps, features, shapes,
stats, episode index ranges in `meta.episodes`). Standardization means **any**
dataset plugs into **any** policy and the same CLI — datasets are swappable,
stats/normalization are computed uniformly, and cross-embodiment training (the
thing VLAs need) becomes tractable. It's the "ImageNet moment" enabler for
robotics: a common data contract so the community can pool demonstrations.

**Sim-to-real transfer and why it's hard.**
Training in sim is cheap, safe, parallelizable; deploying on hardware exposes
the **reality gap**: physics mismatch (friction, contact, deformables),
sensor/visual differences (lighting, textures, camera noise), latency/actuation
dynamics, and unmodeled disturbances. A policy overfit to sim quirks fails on
real robots. Mitigations: **domain randomization** (randomize textures, lighting,
masses, friction so reality looks like one more variation), system
identification, real-data co-training/fine-tuning, and architectures robust to
visual shift. This project is sim-only, so it deliberately *brackets* this
problem — but it's the central challenge for companies shipping real robots.

---

## 6. Next steps → Project 2 (OpenVLA fine-tuning on LIBERO)

1. **Get a real GPU.** OpenVLA is ~7 B params — needs a 24–80 GB NVIDIA GPU.
   Use Colab (A100), Lambda/RunPod, or an NVIDIA box with WSL CUDA passthrough.
   (This exact LeRobot env runs on GPU by swapping `--policy.device=cuda`.)
2. **LIBERO benchmark.** Install LIBERO; evaluate a pretrained VLA first
   (`lerobot-eval --env.type=libero --env.task=libero_object …`) to get a
   working eval baseline, then fine-tune.
3. **Fine-tune OpenVLA** on a LIBERO task suite (LoRA to fit in memory),
   compare success rate vs the pretrained checkpoint.
4. **Contrast with this project:** specialist ACT vs generalist VLA —
   data scale, language conditioning, compute, and success-rate ceilings.

### CUDA experience without local NVIDIA hardware
- **Colab** — fastest path to a GPU `lerobot-train` run.
- **Lambda / RunPod / Vast.ai** — by-the-hour A100/H100, real Linux + nvidia-smi.
- **WSL2 + NVIDIA GPU** — native CUDA passthrough; this env would run as-is.
