# Andarna 🐉

**A complete, reproducible robot-learning pipeline — train and evaluate a
manipulation policy entirely in simulation, on a CPU, with Hugging Face
LeRobot.**

Andarna takes you through the full data-to-deployment loop that modern robotics
companies (Figure, Physical Intelligence, NVIDIA) use: load a standardized
demonstration dataset, train an **ACT** (Action Chunking with Transformers)
imitation-learning policy on a bimanual pick-and-transfer task, and evaluate it
in a **MuJoCo** simulator — no physical robot and no NVIDIA GPU required. The
same pipeline scales to a GPU unchanged (see [colab/](colab/README_colab.md)).

> Project 1 of a robotics-AI ramp. Project 2 = OpenVLA fine-tuning on LIBERO.

## What's inside

| | |
|---|---|
| Framework | Hugging Face **LeRobot 0.5.1** |
| Policy    | **ACT** — 52M params, ResNet18 + transformer + CVAE |
| Dataset   | `lerobot/aloha_sim_transfer_cube_human` (50 episodes, bimanual ALOHA) |
| Simulator | **MuJoCo 3.8** + gym-aloha 0.1.4 |
| Platform  | Windows 11 + **WSL2 Ubuntu**, Python 3.12 (via `uv`), **CPU-only** |

## Quick start

```bash
# 1. create env + install (uv resolves Python 3.12 + CPU-only torch)
uv venv --python 3.12
uv pip install --torch-backend=cpu lerobot gym-aloha

# 2. verify the stack (imports + a real MuJoCo physics step)
python scripts/01_verify.py

# 3. explore the dataset (schema, shapes, episode structure)
python scripts/02_explore_dataset.py

# 4. train ACT (CPU, reduced steps to validate the pipeline)
bash scripts/03_train_act.sh

# 5a. evaluate in sim — closed-loop task success rate
bash scripts/04_eval_sim.sh

# 5b. or offline action-error check (no sim / no render needed)
python scripts/04b_inference_heldout.py
```

## Repo layout

```
Andarna/
├── scripts/
│   ├── 01_verify.py            # Phase 1: stack + MuJoCo sanity check
│   ├── 02_explore_dataset.py   # Phase 2: inspect the LeRobotDataset
│   ├── 03_train_act.sh         # Phase 3: train ACT (CPU; GPU-ready comments)
│   ├── 04_eval_sim.sh          # Phase 4a: closed-loop sim success rate
│   ├── 04b_inference_heldout.py# Phase 4b: open-loop action error vs expert
│   └── 05_eval_libero.sh       # Phase 5: pretrained pi0 VLA on LIBERO (GPU)
├── colab/README_colab.md       # run the same pipeline on a free GPU
├── NOTES.md                    # full command log + results + interview concepts
└── README.md
```

## Key gotchas solved (full detail in [NOTES.md](NOTES.md))

- **CPU torch:** default `pip install lerobot` pulls ~3 GB of unusable CUDA
  wheels — use `uv pip install --torch-backend=cpu`.
- **Video decoding:** camera obs are stored as video; torchcodec needs system
  FFmpeg (root). Use `--dataset.video_backend=pyav` (bundles its own).
- **Headless rendering:** sim eval needs `MUJOCO_GL=egl`.
- **No hub push:** ACT defaults `--policy.push_to_hub=true`; set it `false`.

See **[NOTES.md](NOTES.md)** for results and explanations
(imitation vs RL, ACT internals, VLAs, the LeRobotDataset format, sim-to-real).
