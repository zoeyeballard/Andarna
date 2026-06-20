"""Phase 2: clean-condition baseline eval of OpenVLA on a LIBERO suite.

Establishes the un-degraded success rate every robustness curve is measured against.
Runs on the Colab VM:

    colab exec -s openvla-session -f scripts/run_baseline.py -- \
        --task_suite libero_object --trials 20

Then pull results locally:
    colab download -s openvla-session results/baseline -o ./results/baseline
"""

from __future__ import annotations

import argparse

from libero_eval import EvalConfig, run_eval

# Official OpenVLA finetuned checkpoints, one per suite.
CHECKPOINTS = {
    "libero_object": "openvla/openvla-7b-finetuned-libero-object",
    "libero_spatial": "openvla/openvla-7b-finetuned-libero-spatial",
    "libero_goal": "openvla/openvla-7b-finetuned-libero-goal",
    "libero_10": "openvla/openvla-7b-finetuned-libero-10",
}


def main():
    ap = argparse.ArgumentParser(description="OpenVLA LIBERO baseline eval")
    ap.add_argument("--task_suite", default="libero_object", choices=list(CHECKPOINTS))
    ap.add_argument("--checkpoint", default=None,
                    help="override; defaults to the official finetune for the suite")
    ap.add_argument("--trials", type=int, default=20)
    ap.add_argument("--num_tasks", type=int, default=0, help="0 == all tasks")
    ap.add_argument("--no_center_crop", action="store_true")
    ap.add_argument("--load_in_4bit", action="store_true",
                    help="halve VRAM if the GPU is small (slower, ~same accuracy)")
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()

    cfg = EvalConfig(
        pretrained_checkpoint=args.checkpoint or CHECKPOINTS[args.task_suite],
        task_suite_name=args.task_suite,
        center_crop=not args.no_center_crop,
        load_in_4bit=args.load_in_4bit,
        num_trials_per_task=args.trials,
        num_tasks=args.num_tasks,
        seed=args.seed,
        run_name=f"baseline_{args.task_suite}",
        results_root="results/baseline",
    )
    run_eval(cfg, degrader=None)


if __name__ == "__main__":
    main()
