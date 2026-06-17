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

# Imports the eval harness genuinely needs (failure => eval can't run).
CRITICAL_IMPORTS = [
    ("torch", "PyTorch"),
    ("transformers", "transformers"),
    ("timm", "timm"),
    ("bitsandbytes", "bitsandbytes (4/8-bit)"),
    ("libero.libero", "LIBERO"),
    ("experiments.robot.openvla_utils", "openvla eval utils"),
]

# Nice-to-have but NOT needed for eval (flash-attn unsupported on T4; TF/dlimp are
# only for the RLDS training pipeline). Missing => warn, don't fail.
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
