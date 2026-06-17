"""Provision the OpenVLA + LIBERO stack on the remote Colab GPU VM.

Runs *on the VM* via:  colab exec -s openvla-session -f setup/install_remote.py

Colab already ships a CUDA-enabled PyTorch, so we do NOT reinstall torch (doing so
routinely breaks the CUDA build). We pin the exact versions OpenVLA's LIBERO
experiments were validated against, clone + editable-install both repos, then apply
the two fixes that bite everyone (tensorflow-datasets pin, dlimp from git).

Idempotent: re-running skips clones that already exist and pip is happy to no-op.
Kernel state persists across `colab exec` calls, but this script shells out, so it
also works from a fresh kernel.
"""

import os
import subprocess
import sys

# Where we keep the cloned repos on the VM. /content is Colab's working root.
WORKROOT = os.environ.get("OPENVLA_WORKROOT", "/content")
OPENVLA_DIR = os.path.join(WORKROOT, "openvla")
LIBERO_DIR = os.path.join(WORKROOT, "LIBERO")

# Versions OpenVLA's LIBERO eval was validated against (see openvla README).
PINNED = [
    "transformers==4.40.1",
    "tokenizers==0.19.1",
    "timm==0.9.10",
    "tensorflow-datasets==4.9.3",  # >4.9.3 breaks RLDS dataset loading
]


def run(cmd, **kw):
    """Echo + run a command, streaming output, abort on failure."""
    printable = cmd if isinstance(cmd, str) else " ".join(cmd)
    print(f"\n$ {printable}", flush=True)
    subprocess.run(cmd, check=True, shell=isinstance(cmd, str), **kw)


def pip(*args):
    run([sys.executable, "-m", "pip", "install", "-q", *args])


def clone(url, dest):
    if os.path.isdir(os.path.join(dest, ".git")):
        print(f"[skip] {dest} already cloned", flush=True)
        return
    run(["git", "clone", url, dest])


def main():
    os.makedirs(WORKROOT, exist_ok=True)

    print("=== 1/6 pinned core deps (NOT torch — Colab's CUDA torch stays) ===")
    pip(*PINNED)

    print("\n=== 2/6 clone OpenVLA + editable install ===")
    clone("https://github.com/openvla/openvla.git", OPENVLA_DIR)
    pip("-e", OPENVLA_DIR)

    print("\n=== 3/6 flash-attn (needs torch already present; --no-build-isolation) ===")
    # flash-attn must see the installed torch at build time.
    pip("packaging", "ninja")
    run([sys.executable, "-m", "pip", "install", "-q",
         "flash-attn==2.5.5", "--no-build-isolation"])

    print("\n=== 4/6 clone LIBERO + editable install ===")
    clone("https://github.com/Lifelong-Robot-Learning/LIBERO.git", LIBERO_DIR)
    pip("-e", LIBERO_DIR)

    print("\n=== 5/6 LIBERO experiment requirements ===")
    req = os.path.join(OPENVLA_DIR, "experiments", "robot", "libero",
                       "libero_requirements.txt")
    if os.path.isfile(req):
        pip("-r", req)
    else:
        print(f"[warn] {req} not found — OpenVLA layout may have changed", flush=True)

    print("\n=== 6/6 known-good fixes (dlimp from git) ===")
    # dlimp on PyPI is stale; OpenVLA needs moojink's fork.
    run([sys.executable, "-m", "pip", "install", "-q", "--no-deps",
         "--force-reinstall", "git+https://github.com/moojink/dlimp_openvla"])

    print("\n[OK] install_remote complete. Run setup/verify.py next.", flush=True)


if __name__ == "__main__":
    main()
