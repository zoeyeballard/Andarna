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
    # accelerate is PINNED to transformers-4.40.1's era: accelerate 1.x calls
    # model.to() on single-GPU dispatch, which bnb-quantized models reject, and the
    # old transformers can't signal "quantized, skip .to()". 0.30.1 dispatches right.
    "accelerate==0.30.1",
    "bitsandbytes", "draccus", "einops", "huggingface_hub",
    "jsonlines", "peft", "sentencepiece", "protobuf", "rich",
    "imageio", "imageio-ffmpeg",
    # matplotlib is imported at module load by LIBERO's envs/env_wrapper.py (and by
    # robustness/analysis.py); pandas backs the analysis tables. Missing => LIBERO
    # import fails with ModuleNotFoundError on a clean box.
    "matplotlib", "pandas",
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


def _seed_libero_config():
    """Write ~/.libero/config.yaml so LIBERO's first import doesn't block on an
    interactive input() prompt (it EOFs in a non-interactive run). Points every path
    at the bundled package files. Idempotent: leaves an existing config untouched."""
    cfg_dir = os.environ.get("LIBERO_CONFIG_PATH",
                             os.path.expanduser("~/.libero"))
    cfg_file = os.path.join(cfg_dir, "config.yaml")
    if os.path.isfile(cfg_file):
        print(f"[skip] LIBERO config already at {cfg_file}", flush=True)
        return
    root = os.path.join(LIBERO_DIR, "libero", "libero")
    paths = {
        "benchmark_root": root,
        "bddl_files": os.path.join(root, "./bddl_files"),
        "init_states": os.path.join(root, "./init_files"),
        "datasets": os.path.join(root, "../datasets"),
        "assets": os.path.join(root, "./assets"),
    }
    os.makedirs(cfg_dir, exist_ok=True)
    import yaml
    with open(cfg_file, "w") as f:
        yaml.dump(paths, f)
    print(f"[ok] seeded LIBERO config -> {cfg_file}", flush=True)


def main():
    os.makedirs(WORKROOT, exist_ok=True)

    # Colab runs as root; a bare GPU box (EC2 etc.) runs as a normal user and needs
    # sudo for apt. libegl1/libgles2 are for the EGL backend (preferred on a real
    # GPU); libosmesa6 is the CPU-GL fallback. Best-effort: skip if neither apt nor
    # sudo is available (libs may already be present).
    sudo = "" if os.geteuid() == 0 else "sudo "
    print("=== system render libs (EGL on GPU; OSMesa CPU-GL fallback) ===")
    run(["bash", "-lc",
         f"{sudo}apt-get update -y >/dev/null 2>&1; "
         f"{sudo}apt-get install -y libosmesa6 libgl1-mesa-glx libglfw3 "
         "libegl1 libgles2 2>&1 | tail -4"],
        check=False)

    print("\n=== 1/6 pinned core deps (NOT torch, NOT tensorflow) ===")
    pip(*PINNED_CORE)

    print("\n=== 2/6 clone OpenVLA + editable install WITHOUT deps (skips TF pin) ===")
    clone("https://github.com/openvla/openvla.git", OPENVLA_DIR)
    pip("-e", OPENVLA_DIR, "--no-deps")

    print("\n=== 3/6 curated eval-only deps ===")
    pip(*EVAL_DEPS)
    # openvla_utils.py imports dlimp at module load even on the eval path, so it must
    # be importable. Install moojink's fork WITHOUT deps so it doesn't drag in the old
    # tensorflow==2.15 pin (TF is already present on the VM and dlimp works with it).
    run([sys.executable, "-m", "pip", "install", "-q", "--no-deps",
         "git+https://github.com/moojink/dlimp_openvla"], check=False)

    print("\n=== 4/6 flash-attn ===")
    if WANT_FLASH:
        pip("packaging", "ninja")
        run([sys.executable, "-m", "pip", "install", "-q",
             "flash-attn==2.5.5", "--no-build-isolation"], check=False)
    else:
        print("[skip] flash-attn (unsupported on Turing/T4; loader uses SDPA). "
              "Set OPENVLA_FLASH_ATTN=1 on an Ampere+ GPU to enable.", flush=True)

    print("\n=== 5/6 clone LIBERO + install (pulls robosuite/mujoco/bddl) ===")
    clone("https://github.com/Lifelong-Robot-Learning/LIBERO.git", LIBERO_DIR)
    # editable_mode=compat: LIBERO's flat layout + modern setuptools' strict PEP660
    # editable install produces an EMPTY import MAPPING (import libero -> ModuleNotFound).
    # The legacy/compat mode just puts the project root on a .pth, which resolves it.
    pip("-e", LIBERO_DIR, "--config-settings", "editable_mode=compat")
    req = os.path.join(OPENVLA_DIR, "experiments", "robot", "libero",
                       "libero_requirements.txt")
    if os.path.isfile(req):
        # don't abort the whole install if one pinned line is unhappy on 3.12
        pip("-r", req, check=False)
    else:
        print(f"[warn] {req} not found — OpenVLA layout may have changed", flush=True)
    _seed_libero_config()

    # MUST be last: robosuite/mujoco C-extensions segfault under NumPy 2.x, and the
    # steps above can pull NumPy 2.x back in. Force it down as the final action.
    print("\n=== 6/6 pin NumPy < 2 (robosuite/mujoco segfault on NumPy 2.x) ===")
    pip("numpy==1.26.4")

    print("\n[OK] install complete (eval-only). Run setup/verify.py next.", flush=True)


if __name__ == "__main__":
    main()
