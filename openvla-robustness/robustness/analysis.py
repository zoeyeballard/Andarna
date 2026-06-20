"""Phase 4d: turn raw sweep results into the report's figures (local, no GPU).

Consumes the artifacts the runners write under ``results/``:
  * ``results/robustness/sweep_<suite>_<axis>.csv`` — one row per sweep point
  * ``results/robustness/<run>/trials.jsonl``        — per-trial records
  * ``results/baseline/<run>/summary.json``          — clean-condition reference

Produces into ``figures/``:
  1. degradation curves (success vs level, per axis, with the knee marked)
  2. per-task sensitivity heatmap (tasks x levels)
  3. latency budget (delay timesteps -> wall-clock ms at the control rate)
  4. stacked-profile bar chart
  5. behavioral breakdown (success-fast / success-slow / timeout)

Everything is defensive: a missing input is logged and skipped, so this runs the
moment the first sweep lands and fills in as more arrive.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless: write PNGs, never open a window
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

# LIBERO's OSC controller runs at 20 Hz, so one policy timestep == 50 ms wall-clock.
CONTROL_HZ = 20.0
MS_PER_STEP = 1000.0 / CONTROL_HZ

AXIS_XLABEL = {
    "noise": "Gaussian noise sigma (normalized)",
    "delay": "Observation delay (timesteps)",
    "gap": "Frame-drop probability",
    "resolution": "Downscale factor",
}


# --- loading ----------------------------------------------------------------------
def load_sweep_csv(path: Path) -> list[dict]:
    """Load a sweep CSV into a list of row dicts (numeric fields coerced)."""
    import csv

    rows = []
    with open(path) as f:
        for r in csv.DictReader(f):
            for k in ("level", "success_rate", "n_success", "n_trials",
                      "noise_sigma", "downscale", "gap_rate", "delay",
                      "mean_success_length"):
                if r.get(k) not in (None, "", "None"):
                    try:
                        r[k] = float(r[k])
                    except ValueError:
                        pass  # 'level' for profiles is a name string — leave it
            rows.append(r)
    return rows


def load_trials(run_dir: Path) -> list[dict]:
    path = run_dir / "trials.jsonl"
    if not path.exists():
        return []
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def find_sweeps(results_root: Path) -> dict[str, Path]:
    """Map axis name -> sweep CSV path found under results/robustness."""
    out = {}
    rob = results_root / "robustness"
    if not rob.exists():
        return out
    for p in sorted(rob.glob("sweep_*_*.csv")):
        axis = p.stem.split("_")[-1]  # sweep_<suite>_<axis>
        out[axis] = p
    return out


# --- knee / threshold detection ---------------------------------------------------
def find_knee(levels: list[float], rates: list[float]) -> tuple[float, float]:
    """Return (level, drop) of the steepest single-step drop in success rate."""
    if len(levels) < 2:
        return (levels[0] if levels else 0.0, 0.0)
    drops = [(levels[i + 1], rates[i] - rates[i + 1]) for i in range(len(levels) - 1)]
    return max(drops, key=lambda x: x[1])


def threshold_level(levels, rates, baseline, frac=0.5):
    """First level whose success rate falls below ``frac`` of the baseline rate."""
    cutoff = baseline * frac
    for lv, r in zip(levels, rates):
        if r < cutoff:
            return lv
    return None


# --- figures ----------------------------------------------------------------------
def plot_degradation_curves(sweeps: dict[str, list[dict]], out: Path,
                            baseline: float | None = None):
    axes = [a for a in ("noise", "delay", "gap", "resolution") if a in sweeps]
    if not axes:
        print("[skip] no axis sweeps for degradation curves")
        return
    fig, axarr = plt.subplots(1, len(axes), figsize=(4.2 * len(axes), 3.6),
                              squeeze=False)
    for ax, axis in zip(axarr[0], axes):
        rows = sorted(sweeps[axis], key=lambda r: r["level"])
        x = [r["level"] for r in rows]
        y = [r["success_rate"] for r in rows]
        ax.plot(x, y, "o-", color="#1f77b4", lw=2)
        base = baseline if baseline is not None else (y[0] if y else 1.0)
        knee_lvl, knee_drop = find_knee(x, y)
        if knee_drop > 0.05:
            ax.axvline(knee_lvl, ls="--", color="#d62728", alpha=0.7,
                       label=f"knee @ {knee_lvl:g}")
        thr = threshold_level(x, y, base, 0.5)
        if thr is not None:
            ax.axvline(thr, ls=":", color="#ff7f0e", alpha=0.7,
                       label=f"50%-baseline @ {thr:g}")
        ax.set_title(axis)
        ax.set_xlabel(AXIS_XLABEL.get(axis, axis))
        ax.set_ylim(-0.02, 1.02)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8, loc="lower left")
    axarr[0][0].set_ylabel("success rate")
    fig.suptitle("OpenVLA robustness: success rate vs degradation level")
    fig.tight_layout()
    fig.savefig(out, dpi=130)
    plt.close(fig)
    print(f"[fig] {out}")


def plot_task_heatmap(run_dirs_by_level: list[tuple[float, Path]], axis: str,
                      out: Path):
    """tasks (rows) x sweep levels (cols), colored by per-task success rate."""
    levels, per_task_per_level, task_names = [], [], None
    for level, rd in sorted(run_dirs_by_level, key=lambda x: x[0]):
        trials = load_trials(rd)
        if not trials:
            continue
        agg = defaultdict(lambda: [0, 0])
        for t in trials:
            agg[t["task"]][0] += int(t["success"])
            agg[t["task"]][1] += 1
        names = sorted(agg)
        task_names = task_names or names
        per_task_per_level.append([agg[n][0] / agg[n][1] for n in task_names])
        levels.append(level)
    if not per_task_per_level:
        print(f"[skip] no per-trial data for {axis} heatmap")
        return
    mat = np.array(per_task_per_level).T  # tasks x levels
    fig, ax = plt.subplots(figsize=(1.1 * len(levels) + 3, 0.4 * len(task_names) + 2))
    im = ax.imshow(mat, aspect="auto", cmap="RdYlGn", vmin=0, vmax=1)
    ax.set_xticks(range(len(levels)), [f"{lv:g}" for lv in levels])
    ax.set_yticks(range(len(task_names)),
                  [n[:34] for n in task_names], fontsize=7)
    ax.set_xlabel(f"{axis} level")
    ax.set_title(f"Per-task sensitivity to {axis}")
    fig.colorbar(im, ax=ax, label="success rate", fraction=0.046)
    fig.tight_layout()
    fig.savefig(out, dpi=130)
    plt.close(fig)
    print(f"[fig] {out}")


def plot_latency_budget(delay_rows: list[dict], out: Path,
                        baseline: float | None = None, keep_frac: float = 0.9):
    rows = sorted(delay_rows, key=lambda r: r["level"])
    steps = [r["level"] for r in rows]
    ms = [s * MS_PER_STEP for s in steps]
    y = [r["success_rate"] for r in rows]
    base = baseline if baseline is not None else (y[0] if y else 1.0)
    budget_ms = None
    for s_ms, r in zip(ms, y):
        if r >= base * keep_frac:
            budget_ms = s_ms
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(ms, y, "o-", color="#2ca02c", lw=2)
    ax.axhline(base * keep_frac, ls=":", color="grey",
               label=f"{keep_frac:.0%} of baseline")
    if budget_ms is not None:
        ax.axvline(budget_ms, ls="--", color="#d62728",
                   label=f"latency budget ~{budget_ms:.0f} ms")
    ax.set_xlabel(f"observation latency (ms @ {CONTROL_HZ:g} Hz control)")
    ax.set_ylabel("success rate")
    ax.set_ylim(-0.02, 1.02)
    ax.set_title("Latency budget: how stale an observation can be")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(out, dpi=130)
    plt.close(fig)
    print(f"[fig] {out}  (budget ~{budget_ms} ms)")
    return budget_ms


def plot_profiles(profile_rows: list[dict], out: Path):
    order = ["clean", "lab", "field", "challenging_field", "high_stress_field"]
    rows = {str(r["level"]): r for r in profile_rows}
    names = [n for n in order if n in rows]
    if not names:
        print("[skip] no profile data")
        return
    y = [rows[n]["success_rate"] for n in names]
    fig, ax = plt.subplots(figsize=(7, 4))
    colors = plt.cm.RdYlGn(np.linspace(0.85, 0.15, len(names)))
    ax.bar(names, y, color=colors)
    for i, v in enumerate(y):
        ax.text(i, v + 0.02, f"{v:.0%}", ha="center", fontsize=9)
    ax.set_ylabel("success rate")
    ax.set_ylim(0, 1.05)
    ax.set_title("Stacked deployment profiles (mild -> edge-case)")
    plt.setp(ax.get_xticklabels(), rotation=20, ha="right")
    fig.tight_layout()
    fig.savefig(out, dpi=130)
    plt.close(fig)
    print(f"[fig] {out}")


def behavioral_breakdown(trials: list[dict]) -> dict:
    """Coarse outcome taxonomy from per-trial fields.

    success-fast / success-slow (split at the median successful length) / timeout.
    The fine-grained "overshot ~3cm, oscillated twice" notes come from the saved
    rollout videos; this gives the quantitative skeleton.
    """
    succ_lengths = [t["episode_length"] for t in trials if t["success"]]
    median = float(np.median(succ_lengths)) if succ_lengths else 0.0
    cats = {"success_fast": 0, "success_slow": 0, "timeout": 0}
    for t in trials:
        if not t["success"]:
            cats["timeout"] += 1
        elif t["episode_length"] <= median:
            cats["success_fast"] += 1
        else:
            cats["success_slow"] += 1
    cats["median_success_length"] = median
    cats["n"] = len(trials)
    return cats


def plot_behavioral(breakdowns: dict[str, dict], out: Path):
    labels = list(breakdowns)
    if not labels:
        print("[skip] no behavioral data")
        return
    fast = [breakdowns[l]["success_fast"] for l in labels]
    slow = [breakdowns[l]["success_slow"] for l in labels]
    to = [breakdowns[l]["timeout"] for l in labels]
    fig, ax = plt.subplots(figsize=(1.0 * len(labels) + 3, 4))
    ax.bar(labels, fast, label="success (fast)", color="#2ca02c")
    ax.bar(labels, slow, bottom=fast, label="success (slow)", color="#ffdf6b")
    ax.bar(labels, to, bottom=np.array(fast) + np.array(slow),
           label="timeout / fail", color="#d62728")
    ax.set_ylabel("trials")
    ax.set_title("Behavioral breakdown by condition")
    plt.setp(ax.get_xticklabels(), rotation=25, ha="right", fontsize=8)
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(out, dpi=130)
    plt.close(fig)
    print(f"[fig] {out}")


# --- driver -----------------------------------------------------------------------
def baseline_rate(results_root: Path, suite: str) -> float | None:
    p = results_root / "baseline" / f"baseline_{suite}" / "summary.json"
    if p.exists():
        return json.load(open(p))["success_rate"]
    return None


def main():
    ap = argparse.ArgumentParser(description="Generate robustness figures")
    ap.add_argument("--results", default="results")
    ap.add_argument("--figures", default="figures")
    ap.add_argument("--suite", default="libero_object")
    args = ap.parse_args()

    results_root = Path(args.results)
    fig_dir = Path(args.figures)
    fig_dir.mkdir(parents=True, exist_ok=True)
    base = baseline_rate(results_root, args.suite)
    print(f"[baseline] {args.suite} success rate = {base}")

    sweep_paths = find_sweeps(results_root)
    sweeps = {axis: load_sweep_csv(p) for axis, p in sweep_paths.items()}

    # 1) degradation curves
    axis_sweeps = {a: v for a, v in sweeps.items() if a != "profiles"}
    if axis_sweeps:
        plot_degradation_curves(axis_sweeps, fig_dir / "degradation_curves.png", base)

    # 2) per-task heatmap for each axis we have run dirs for
    for axis, rows in axis_sweeps.items():
        run_dirs = [(r["level"], results_root / "robustness" / r["run_name"])
                    for r in rows if r.get("run_name")]
        plot_task_heatmap(run_dirs, axis, fig_dir / f"heatmap_{axis}.png")

    # 3) latency budget
    if "delay" in axis_sweeps:
        plot_latency_budget(axis_sweeps["delay"], fig_dir / "latency_budget.png", base)

    # 4) profiles
    if "profiles" in sweeps:
        plot_profiles(sweeps["profiles"], fig_dir / "profiles.png")

    # 5) behavioral breakdown across whatever runs exist
    breakdowns = {}
    rob = results_root / "robustness"
    if rob.exists():
        for rd in sorted(rob.iterdir()):
            if rd.is_dir():
                trials = load_trials(rd)
                if trials:
                    breakdowns[rd.name.replace(f"{args.suite}_", "")] = \
                        behavioral_breakdown(trials)
    if breakdowns:
        plot_behavioral(breakdowns, fig_dir / "behavioral.png")
        json.dump(breakdowns, open(fig_dir / "behavioral_summary.json", "w"), indent=2)

    print("[done] analysis complete")


if __name__ == "__main__":
    main()
