"""Sanity-check the remote VM after install_remote.py.

Runs *on the VM* via:  colab exec -s openvla-session -f setup/verify.py

Confirms CUDA is live, the GPU is what we asked for, the pinned versions stuck,
and every import the eval path depends on actually loads. Exits non-zero on any
failure so an orchestrating script can branch on it.
"""

import importlib
import os
import sys

# `experiments.*` lives at the OpenVLA repo root (not an installed package).
OPENVLA_DIR = os.environ.get("OPENVLA_DIR", "/content/openvla")
if os.path.isdir(OPENVLA_DIR) and OPENVLA_DIR not in sys.path:
    sys.path.insert(0, OPENVLA_DIR)

EXPECTED_TRANSFORMERS = "4.40.1"

# Imports the (TF-free) eval harness genuinely needs (failure => eval can't run).
# NOTE: we deliberately do NOT import OpenVLA's experiments.robot.* — it pulls
# dlimp->tensorflow and a fragile protobuf stack the eval doesn't need. The model
# is loaded directly via transformers AutoModelForVision2Seq + trust_remote_code.
CRITICAL_IMPORTS = [
    ("torch", "PyTorch"),
    ("transformers", "transformers"),
    ("timm", "timm"),
    ("bitsandbytes", "bitsandbytes (4/8-bit)"),
    ("PIL", "Pillow"),
    ("imageio", "imageio"),
    ("libero.libero", "LIBERO"),
]

# Not needed by the decoupled eval path (flash-attn is Ampere+ only; TF/dlimp are
# the training data stack we removed from the import chain). Missing => fine.
OPTIONAL_IMPORTS = [
    ("flash_attn", "flash-attn (Ampere+ only)"),
    ("tensorflow_datasets", "tensorflow-datasets (training only)"),
    ("dlimp", "dlimp (training only)"),
]


def main():
    failures = []

    # --- GPU / CUDA ---
    try:
        import torch
        cuda = torch.cuda.is_available()
        print(f"torch                 {torch.__version__}")
        print(f"cuda available        {cuda}")
        if cuda:
            print(f"gpu                   {torch.cuda.get_device_name(0)}")
            free, total = torch.cuda.mem_get_info()
            print(f"gpu memory            {total / 1e9:.1f} GB total, "
                  f"{free / 1e9:.1f} GB free")
        else:
            failures.append("CUDA not available — eval will be unusably slow")
    except Exception as e:  # noqa: BLE001
        failures.append(f"torch import failed: {e}")

    # --- NumPy must be < 2 (robosuite/mujoco C-exts segfault on 2.x) ---
    try:
        import numpy as np
        major = int(np.__version__.split(".")[0])
        print(f"numpy                 {np.__version__} "
              f"({'ok' if major < 2 else 'TOO NEW — rendering will segfault'})")
        if major >= 2:
            failures.append(f"numpy {np.__version__} >= 2 — run: pip install numpy==1.26.4")
    except Exception as e:  # noqa: BLE001
        failures.append(f"numpy import failed: {e}")

    # --- version pin ---
    try:
        import transformers
        v = transformers.__version__
        ok = v == EXPECTED_TRANSFORMERS
        print(f"transformers          {v} ({'ok' if ok else 'EXPECTED ' + EXPECTED_TRANSFORMERS})")
        if not ok:
            failures.append(f"transformers {v} != {EXPECTED_TRANSFORMERS}")
    except Exception as e:  # noqa: BLE001
        failures.append(f"transformers import failed: {e}")

    # --- critical imports ---
    print("\nrequired imports:")
    for mod, name in CRITICAL_IMPORTS:
        try:
            importlib.import_module(mod)
            print(f"  [ok]   {name}")
        except Exception as e:  # noqa: BLE001
            print(f"  [FAIL] {name}: {e}")
            failures.append(f"import {mod}: {e}")

    # functional check: the exact classes the eval path constructs
    print("\neval-path classes:")
    try:
        from transformers import AutoModelForVision2Seq, AutoProcessor  # noqa: F401
        print("  [ok]   transformers AutoModelForVision2Seq / AutoProcessor")
    except Exception as e:  # noqa: BLE001
        print(f"  [FAIL] transformers auto-classes: {e}")
        failures.append(f"transformers auto-classes: {e}")
    try:
        from libero.libero.envs import OffScreenRenderEnv  # noqa: F401
        print("  [ok]   LIBERO OffScreenRenderEnv (headless sim)")
    except Exception as e:  # noqa: BLE001
        print(f"  [FAIL] OffScreenRenderEnv: {e}")
        failures.append(f"OffScreenRenderEnv: {e}")

    print("\noptional imports (fine to be missing for eval):")
    for mod, name in OPTIONAL_IMPORTS:
        try:
            importlib.import_module(mod)
            print(f"  [ok]   {name}")
        except Exception:  # noqa: BLE001
            print(f"  [--]   {name} not installed (ok)")

    print()
    if failures:
        print(f"[VERIFY FAILED] {len(failures)} problem(s):")
        for f in failures:
            print(f"  - {f}")
        sys.exit(1)
    print("[VERIFY OK] environment is ready for baseline + sweeps.")


if __name__ == "__main__":
    main()
