"""Phase 4b: evaluate OpenVLA under ONE degradation condition.

Single point on a sweep — a degradation axis at one level, or a named stacked
profile. The model is identical to baseline; only the observation pipeline changes.

    # one axis at one level
    colab exec -s openvla-session -f scripts/run_degradation.py -- \
        --degradation noise --level 0.05 --task_suite libero_object --trials 10

    # a stacked deployment profile
    colab exec -s openvla-session -f scripts/run_degradation.py -- \
        --profile field --task_suite libero_object --trials 10
"""

from __future__ import annotations

import argparse

from libero_eval import EvalConfig, run_eval

from robustness.degradations import (
    SWEEPS,
    ObservationDegrader,
    make_degrader,
    profile_degrader,
)
from run_baseline import CHECKPOINTS


def build_degrader(args) -> tuple[ObservationDegrader, str]:
    if args.profile:
        deg = profile_degrader(args.profile)
        return deg, f"profile_{args.profile}"
    if args.degradation is None or args.level is None:
        raise SystemExit("provide either --profile NAME or (--degradation KIND --level L)")
    if args.degradation not in SWEEPS:
        raise SystemExit(f"--degradation must be one of {list(SWEEPS)}")
    deg = make_degrader(args.degradation, args.level)
    lvl = args.level
    lvl_str = f"{lvl:g}".replace(".", "p")
    return deg, f"{args.degradation}_{lvl_str}"


def main():
    ap = argparse.ArgumentParser(description="OpenVLA LIBERO eval under degradation")
    ap.add_argument("--degradation", choices=list(SWEEPS),
                    help="noise | delay | gap | resolution")
    ap.add_argument("--level", type=float, help="sweep level for --degradation")
    ap.add_argument("--profile", help="stacked profile: lab|field|challenging_field|high_stress_field")
    ap.add_argument("--task_suite", default="libero_object", choices=list(CHECKPOINTS))
    ap.add_argument("--checkpoint", default=None)
    ap.add_argument("--trials", type=int, default=10)
    ap.add_argument("--num_tasks", type=int, default=0)
    ap.add_argument("--no_center_crop", action="store_true")
    ap.add_argument("--load_in_4bit", action="store_true")
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()

    degrader, run_name = build_degrader(args)
    cfg = EvalConfig(
        pretrained_checkpoint=args.checkpoint or CHECKPOINTS[args.task_suite],
        task_suite_name=args.task_suite,
        center_crop=not args.no_center_crop,
        load_in_4bit=args.load_in_4bit,
        num_trials_per_task=args.trials,
        num_tasks=args.num_tasks,
        seed=args.seed,
        run_name=f"{args.task_suite}_{run_name}",
        results_root="results/robustness",
    )
    run_eval(cfg, degrader=degrader)


if __name__ == "__main__":
    main()
