#!/usr/bin/env python
"""Pre-flight: validate the checkpoint (5d) and environment health (5e) before eval.

Exits non-zero if either fails — so CI catches a corrupt checkpoint or broken sim in
seconds rather than after a full (wasted) evaluation. Run standalone or as the first
step of the sim-validation tier.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import EvalConfig  # noqa: E402
from src.validation import health_check_env, validate_checkpoint  # noqa: E402


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Pre-flight checkpoint + env validation.")
    ap.add_argument("--checkpoint_path", default=None)
    ap.add_argument("--skip_env", action="store_true", help="checkpoint checks only")
    args = ap.parse_args(argv)

    overrides = {"checkpoint_path": args.checkpoint_path} if args.checkpoint_path else {}
    cfg = EvalConfig.from_env(**overrides)

    # resolve a local checkpoint path if possible (Hub ids pass through)
    ckpt = cfg.checkpoint_path
    try:
        from src.evaluator import resolve_checkpoint

        ckpt = resolve_checkpoint(cfg.checkpoint_path)
    except Exception:
        pass

    ok = True
    ckpt_report = validate_checkpoint(ckpt)
    print(ckpt_report.render())
    ok = ok and ckpt_report.ok

    if not args.skip_env:
        env_report = health_check_env(cfg.env_type, cfg.task, cfg.render_backend)
        print("\n" + env_report.render())
        ok = ok and env_report.ok

    print("\nPREFLIGHT", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
