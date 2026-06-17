"""Provision the OpenVLA + LIBERO stack on the remote Colab GPU VM (EVAL-focused).

Runs *on the VM* via:  python setup/install_remote.py   (inside `colab console`)

Reality of free Colab today: the VM is **Python 3.12** and the GPU is often a **T4
(Turing, sm75)**. That breaks the vanilla OpenVLA install two ways, both handled here:

  * OpenVLA's setup.py hard-pins ``tensorflow==2.15.0`` (no 3.12 wheel). TF is only
    needed for the RLDS *training* data pipeline — **not for eval** — so we install
    OpenVLA with ``--no-deps`` and add a curated eval-only dependency set instead.
  * flash-attn 2.x doesn't support Turing. We **skip** it (set OPENVLA_FLASH_ATTN=1
    to force it on an Ampere+ GPU). The model loader falls back to PyTorch SDPA.

Idempotent: clones are skipped if present; pip no-ops happily. Does NOT touch torch
(Colab's CUDA build stays).
"""

import os
import subprocess
import sys

WORKROOT = os.environ.get("OPENVLA_WORKROOT", "/content")
OPENVLA_DIR = os.path.join(WORKROOT, "openvla")
LIBERO_DIR = os.path.join(WORKROOT, "LIBERO")
WANT_FLASH = os.environ.get("OPENVLA_FLASH_ATTN", "0") == "1"

# Pinned core OpenVLA validated against (NOT tensorflow — see module docstring).
PINNED_CORE = ["transformers==4.40.1", "tokenizers==0.19.1", "timm==0.9.10"]

# Everything `prismatic` + the LIBERO eval path import at runtime, minus the
# TF/RLDS/dlimp data stack which eval never touches.
EVAL_DEPS = [
    "accelerate", "bitsandbytes", "draccus", "einops", "huggingface_hub",
    "jsonlines", "peft", "sentencepiece", "protobuf", "rich",
    "imageio", "imageio-ffmpeg",
]


def run(cmd, check=True):
    print(f"\n$ {' '.join(cmd)}", flush=True)
    return subprocess.run(cmd, check=check)


def pip(*args, check=True):
    run([sys.executable, "-m", "pip", "install", "-q", *args], check=check)


def clone(url, dest):
    if os.path.isdir(os.path.join(dest, ".git")):
        print(f"[skip] {dest} already cloned", flush=True)
        return
    run(["git", "clone", "--depth", "1", url, dest])


def main():
    os.makedirs(WORKROOT, exist_ok=True)

    print("=== 1/5 pinned core deps (NOT torch, NOT tensorflow) ===")
    pip(*PINNED_CORE)

    print("\n=== 2/5 clone OpenVLA + editable install WITHOUT deps (skips TF pin) ===")
    clone("https://github.com/openvla/openvla.git", OPENVLA_DIR)
    pip("-e", OPENVLA_DIR, "--no-deps")

    print("\n=== 3/5 curated eval-only deps ===")
    pip(*EVAL_DEPS)

    print("\n=== 4/5 flash-attn ===")
    if WANT_FLASH:
        pip("packaging", "ninja")
        run([sys.executable, "-m", "pip", "install", "-q",
             "flash-attn==2.5.5", "--no-build-isolation"], check=False)
    else:
        print("[skip] flash-attn (unsupported on Turing/T4; loader uses SDPA). "
              "Set OPENVLA_FLASH_ATTN=1 on an Ampere+ GPU to enable.", flush=True)

    print("\n=== 5/5 clone LIBERO + install (pulls robosuite/mujoco/bddl) ===")
    clone("https://github.com/Lifelong-Robot-Learning/LIBERO.git", LIBERO_DIR)
    pip("-e", LIBERO_DIR)
    req = os.path.join(OPENVLA_DIR, "experiments", "robot", "libero",
                       "libero_requirements.txt")
    if os.path.isfile(req):
        # don't abort the whole install if one pinned line is unhappy on 3.12
        pip("-r", req, check=False)
    else:
        print(f"[warn] {req} not found — OpenVLA layout may have changed", flush=True)

    print("\n[OK] install complete (eval-only). Run setup/verify.py next.", flush=True)


if __name__ == "__main__":
    main()
