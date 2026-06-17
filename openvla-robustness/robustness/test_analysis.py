"""Local smoke test for analysis.py — synthesizes plausible results, runs the full
figure pipeline, asserts the PNGs and budget come out. No GPU / no OpenVLA needed.

Run:  python -m robustness.test_analysis   (from openvla-robustness/)
"""

import csv
import json
import tempfile
from pathlib import Path

import numpy as np

from robustness import analysis
from robustness.degradations import SWEEPS

RNG = np.random.default_rng(0)
TASKS = [f"task_{i}: pick up the object {i}" for i in range(5)]


def _decay(level, scale, floor=0.0):
    """A monotone-ish degradation curve with a knee, in [floor, 0.95]."""
    return float(np.clip(0.95 * np.exp(-level / scale) + floor, 0, 0.95))


def _write_sweep(results, suite, axis, levels, scale):
    rob = results / "robustness"
    rob.mkdir(parents=True, exist_ok=True)
    rows = []
    for lv in levels:
        rate = _decay(lv, scale)
        run_name = f"{suite}_{axis}_{('%g' % lv).replace('.', 'p')}"
        # per-trial records (10 trials/task) so heatmap + behavioral have data
        run_dir = rob / run_name
        run_dir.mkdir(parents=True, exist_ok=True)
        n_succ = 0
        with open(run_dir / "trials.jsonl", "w") as f:
            for ti, task in enumerate(TASKS):
                for ep in range(10):
                    succ = RNG.random() < rate
                    n_succ += succ
                    f.write(json.dumps({
                        "task_id": ti, "task": task, "episode": ep,
                        "success": bool(succ),
                        "episode_length": int(RNG.integers(40, 220)),
                        "timed_out": not succ, "max_steps": 280,
                        "degradation": {"axis": axis, "level": lv}, "seed": ep,
                    }) + "\n")
        rows.append({
            "axis": axis, "level": lv, "success_rate": round(rate, 3),
            "n_success": n_succ, "n_trials": len(TASKS) * 10,
            "mean_success_length": 120.0, "run_name": run_name,
            "noise_sigma": lv if axis == "noise" else 0,
            "downscale": lv if axis == "resolution" else 1,
            "gap_rate": lv if axis == "gap" else 0,
            "delay": lv if axis == "delay" else 0,
        })
    with open(rob / f"sweep_{suite}_{axis}.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


def _write_baseline(results, suite):
    d = results / "baseline" / f"baseline_{suite}"
    d.mkdir(parents=True, exist_ok=True)
    json.dump({"success_rate": 0.92, "task_suite": suite},
              open(d / "summary.json", "w"))


def main():
    suite = "libero_object"
    with tempfile.TemporaryDirectory() as tmp:
        results = Path(tmp) / "results"
        figures = Path(tmp) / "figures"
        _write_baseline(results, suite)
        _write_sweep(results, suite, "noise", SWEEPS["noise"], scale=0.08)
        _write_sweep(results, suite, "delay", SWEEPS["delay"], scale=4.0)
        _write_sweep(results, suite, "gap", SWEEPS["gap"], scale=0.25)

        # exercise the individual pieces
        base = analysis.baseline_rate(results, suite)
        assert base == 0.92

        sweeps = {a: analysis.load_sweep_csv(p)
                  for a, p in analysis.find_sweeps(results).items()}
        assert set(sweeps) == {"noise", "delay", "gap"}, sweeps.keys()

        lvls = [r["level"] for r in sorted(sweeps["delay"], key=lambda r: r["level"])]
        rates = [r["success_rate"] for r in sorted(sweeps["delay"],
                                                   key=lambda r: r["level"])]
        knee_lvl, knee_drop = analysis.find_knee(lvls, rates)
        assert knee_drop > 0, "should detect a drop"

        # run the whole driver via its CLI entry behavior
        import sys
        argv = sys.argv
        sys.argv = ["analysis", "--results", str(results),
                    "--figures", str(figures), "--suite", suite]
        try:
            analysis.main()
        finally:
            sys.argv = argv

        produced = sorted(p.name for p in figures.glob("*.png"))
        for expected in ("degradation_curves.png", "latency_budget.png",
                         "heatmap_noise.png", "behavioral.png"):
            assert expected in produced, f"missing figure {expected}; got {produced}"

        # latency budget should be a real number of ms
        budget = analysis.plot_latency_budget(
            sorted(sweeps["delay"], key=lambda r: r["level"]),
            figures / "latency_budget2.png", baseline=base)
        assert budget is not None and budget >= 0
        print(f"\n[PASS] analysis pipeline produced {len(produced)} figures; "
              f"latency budget ~{budget:.0f} ms")


if __name__ == "__main__":
    main()
