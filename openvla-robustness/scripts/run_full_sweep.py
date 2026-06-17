"""Phase 4c: sweep one degradation axis (or all stacked profiles) end-to-end.

Loads the 7B model ONCE, then evaluates every level of the chosen axis, writing a
per-level summary plus an aggregated ``sweep_<axis>.csv`` the analysis step consumes.

    colab exec -s openvla-session -f scripts/run_full_sweep.py -- \
        --degradation latency --task_suite libero_object --trials 10
    colab exec -s openvla-session -f scripts/run_full_sweep.py -- \
        --profiles --task_suite libero_object --trials 10

Priority order if GPU time is tight: latency, noise, gap, resolution, profiles.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from libero_eval import EvalConfig, prepare, run_eval

from robustness.degradations import (
    PROFILES,
    SWEEPS,
    ObservationDegrader,
    make_degrader,
    profile_degrader,
)
from run_baseline import CHECKPOINTS

# "latency" is the embedded-systems-facing alias for the delay axis.
AXIS_ALIASES = {"latency": "delay"}


def _level_tag(v) -> str:
    return f"{v:g}".replace(".", "p")


def _conditions(args) -> list[tuple[str, ObservationDegrader, dict]]:
    """Return (run_suffix, degrader, point_metadata) for each condition to run."""
    out = []
    if args.profiles:
        for name in ("clean", "lab", "field", "challenging_field", "high_stress_field"):
            deg = profile_degrader(name)
            out.append((f"profile_{name}", deg,
                        {"axis": "profile", "level": name, **deg.cfg.as_metadata()}))
        return out

    axis = AXIS_ALIASES.get(args.degradation, args.degradation)
    if axis not in SWEEPS:
        raise SystemExit(f"--degradation must be one of {list(SWEEPS)} (or 'latency')")
    for level in SWEEPS[axis]:
        deg = make_degrader(axis, level)
        out.append((f"{axis}_{_level_tag(level)}", deg,
                    {"axis": axis, "level": level, **deg.cfg.as_metadata()}))
    return out


def main():
    ap = argparse.ArgumentParser(description="OpenVLA LIBERO degradation sweep")
    ap.add_argument("--degradation", help="noise | latency/delay | gap | resolution")
    ap.add_argument("--profiles", action="store_true",
                    help="run the stacked deployment profiles instead of one axis")
    ap.add_argument("--task_suite", default="libero_object", choices=list(CHECKPOINTS))
    ap.add_argument("--checkpoint", default=None)
    ap.add_argument("--trials", type=int, default=10)
    ap.add_argument("--num_tasks", type=int, default=0)
    ap.add_argument("--no_center_crop", action="store_true")
    ap.add_argument("--load_in_4bit", action="store_true")
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()
    if not args.profiles and not args.degradation:
        raise SystemExit("pass --degradation AXIS or --profiles")

    conditions = _conditions(args)
    axis_name = "profiles" if args.profiles else AXIS_ALIASES.get(
        args.degradation, args.degradation)
    results_root = Path("results/robustness")

    # Load the model ONCE; reuse across every level.
    base_cfg = EvalConfig(
        pretrained_checkpoint=args.checkpoint or CHECKPOINTS[args.task_suite],
        task_suite_name=args.task_suite,
        center_crop=not args.no_center_crop,
        load_in_4bit=args.load_in_4bit,
    )
    prepared = prepare(base_cfg)

    rows = []
    for suffix, degrader, point in conditions:
        cfg = EvalConfig(
            pretrained_checkpoint=base_cfg.pretrained_checkpoint,
            task_suite_name=args.task_suite,
            center_crop=not args.no_center_crop,
            load_in_4bit=args.load_in_4bit,
            num_trials_per_task=args.trials,
            num_tasks=args.num_tasks,
            seed=args.seed,
            run_name=f"{args.task_suite}_{suffix}",
            results_root=str(results_root),
        )
        print(f"\n===== sweep point: {point} =====", flush=True)
        summary = run_eval(cfg, degrader=degrader, prepared=prepared)
        rows.append({
            "axis": point["axis"],
            "level": point["level"],
            "success_rate": summary["success_rate"],
            "n_success": summary["n_success"],
            "n_trials": summary["n_trials"],
            "mean_success_length": summary["mean_success_length"],
            "run_name": cfg.run_name,
            **{k: point.get(k) for k in
               ("noise_sigma", "downscale", "gap_rate", "delay")},
        })

    # Aggregated, analysis-ready outputs.
    out_csv = results_root / f"sweep_{args.task_suite}_{axis_name}.csv"
    out_json = results_root / f"sweep_{args.task_suite}_{axis_name}.json"
    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    with open(out_json, "w") as f:
        json.dump({"task_suite": args.task_suite, "axis": axis_name, "points": rows},
                  f, indent=2)
    print(f"\n[sweep done] {len(rows)} points -> {out_csv}", flush=True)
    for r in rows:
        print(f"  {r['axis']}={r['level']!s:>16}  "
              f"success={r['success_rate']:.1%}  ({r['n_success']}/{r['n_trials']})")


if __name__ == "__main__":
    main()
