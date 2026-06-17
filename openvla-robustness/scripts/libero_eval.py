"""Shared LIBERO evaluation engine for OpenVLA, with an observation-degradation hook.

This is the one rollout loop that every experiment shares. ``run_baseline.py`` calls
it with no degrader; ``run_degradation.py`` / ``run_full_sweep.py`` call it with an
:class:`~robustness.degradations.ObservationDegrader`. The model is never touched —
the degrader is applied to the policy-input image, *after* LIBERO produces the frame
and *before* OpenVLA's own preprocessing, which is exactly "what the sensor delivers
to the policy."

Runs **on the Colab GPU VM** (needs torch+CUDA, the `openvla` repo, and `LIBERO`).
The harness adds the OpenVLA repo to sys.path and reuses its validated utilities so
the action de-normalization / gripper handling matches the official eval exactly.

Authored locally; executed via:
    colab exec -s openvla-session -f scripts/run_baseline.py
"""

from __future__ import annotations

import dataclasses
import json
import os
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

# --- make this file runnable both as a module and as a bare `colab exec` script ---
_HERE = Path(__file__).resolve().parent
_PROJECT = _HERE.parent
for p in (_PROJECT, _HERE):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from robustness.degradations import ObservationDegrader  # noqa: E402

# OpenVLA repo root on the VM (install_remote.py clones it here).
OPENVLA_DIR = os.environ.get("OPENVLA_DIR", "/content/openvla")
RESIZE_SIZE = 224  # OpenVLA's LIBERO eval policy-input size

# Per-suite episode caps OpenVLA uses (longer suites get more steps).
MAX_STEPS = {
    "libero_spatial": 220,
    "libero_object": 280,
    "libero_goal": 300,
    "libero_10": 520,
    "libero_90": 400,
}
NUM_STEPS_WAIT = 10  # sim settle steps before the policy takes over


@dataclass
class EvalConfig:
    """Mirrors the fields of OpenVLA's GenerateConfig that the eval path reads."""

    pretrained_checkpoint: str = "openvla/openvla-7b-finetuned-libero-object"
    task_suite_name: str = "libero_object"
    model_family: str = "openvla"
    unnorm_key: str = ""               # defaults to task_suite_name if empty
    center_crop: bool = True
    load_in_8bit: bool = False
    load_in_4bit: bool = False
    num_trials_per_task: int = 20
    num_tasks: int = 0                 # 0 == all tasks in the suite
    seed: int = 7
    save_videos: int = 2               # save this many rollout videos per task
    run_name: str = "baseline"
    results_root: str = str(_PROJECT / "results" / "baseline")

    def __post_init__(self):
        if not self.unnorm_key:
            self.unnorm_key = self.task_suite_name


def _import_openvla():
    """Import OpenVLA's eval utilities (only available on the VM). Returns a bundle."""
    if OPENVLA_DIR not in sys.path:
        sys.path.insert(0, OPENVLA_DIR)
    try:
        from experiments.robot.libero.libero_utils import (
            get_libero_dummy_action,
            get_libero_env,
            get_libero_image,
            quat2axisangle,
            save_rollout_video,
        )
        from experiments.robot.openvla_utils import get_processor
        from experiments.robot.robot_utils import (
            get_action,
            get_model,
            invert_gripper_action,
            normalize_gripper_action,
            set_seed_everywhere,
        )
        from libero.libero import benchmark
    except ImportError as e:  # pragma: no cover - only hits off-VM
        raise RuntimeError(
            "OpenVLA/LIBERO not importable. This engine runs on the Colab VM after "
            f"setup/install_remote.py. (OPENVLA_DIR={OPENVLA_DIR}); import error: {e}"
        ) from e
    return dict(
        get_libero_dummy_action=get_libero_dummy_action,
        get_libero_env=get_libero_env,
        get_libero_image=get_libero_image,
        quat2axisangle=quat2axisangle,
        save_rollout_video=save_rollout_video,
        get_processor=get_processor,
        get_action=get_action,
        get_model=get_model,
        invert_gripper_action=invert_gripper_action,
        normalize_gripper_action=normalize_gripper_action,
        set_seed_everywhere=set_seed_everywhere,
        benchmark=benchmark,
    )


def load_policy(cfg: EvalConfig, ov):
    """Load the OpenVLA model + processor once; reused across all trials."""
    print(f"[load] {cfg.pretrained_checkpoint} (8bit={cfg.load_in_8bit} "
          f"4bit={cfg.load_in_4bit})", flush=True)
    model = ov["get_model"](cfg)
    processor = ov["get_processor"](cfg) if cfg.model_family == "openvla" else None
    return model, processor


def _state_vector(obs, quat2axisangle):
    return np.concatenate([
        obs["robot0_eef_pos"],
        quat2axisangle(obs["robot0_eef_quat"]),
        obs["robot0_gripper_qpos"],
    ])


def prepare(cfg: EvalConfig):
    """Import OpenVLA and load the model+processor ONCE. Return a reusable bundle.

    A full sweep (many degradation levels, same checkpoint) should call this once and
    pass the result to every ``run_eval`` via ``prepared=`` — loading the 7B model is
    the slow part, and it is identical across levels.
    """
    ov = _import_openvla()
    model, processor = load_policy(cfg, ov)
    return {"ov": ov, "model": model, "processor": processor}


def run_eval(cfg: EvalConfig, degrader: ObservationDegrader | None = None,
             prepared: dict | None = None) -> dict:
    """Run the full suite eval (optionally degraded). Returns the summary dict and
    writes ``trials.jsonl`` + ``summary.json`` under ``results_root/run_name``.

    Pass ``prepared`` (from :func:`prepare`) to reuse an already-loaded model.
    """
    if prepared is None:
        prepared = prepare(cfg)
    ov, model, processor = prepared["ov"], prepared["model"], prepared["processor"]
    ov["set_seed_everywhere"](cfg.seed)

    run_dir = Path(cfg.results_root) / cfg.run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    trials_path = run_dir / "trials.jsonl"
    video_dir = run_dir / "videos"

    suite = ov["benchmark"].get_benchmark_dict()[cfg.task_suite_name]()
    n_tasks = suite.n_tasks if cfg.num_tasks == 0 else min(cfg.num_tasks, suite.n_tasks)
    max_steps = MAX_STEPS.get(cfg.task_suite_name, 300)

    deg_meta = degrader.cfg.as_metadata() if degrader else {"profile": "clean"}
    print(f"[run] suite={cfg.task_suite_name} tasks={n_tasks} "
          f"trials/task={cfg.num_trials_per_task} degradation={deg_meta}", flush=True)

    trials = []
    t_start = time.time()
    with open(trials_path, "w") as tf:
        for task_id in range(n_tasks):
            task = suite.get_task(task_id)
            init_states = suite.get_task_init_states(task_id)
            env, task_description = ov["get_libero_env"](
                task, cfg.model_family, resolution=256)

            for ep in range(cfg.num_trials_per_task):
                episode_seed = cfg.seed + task_id * 1000 + ep
                if degrader is not None:
                    degrader.reset(seed=episode_seed)

                env.reset()
                obs = env.set_init_state(init_states[ep % len(init_states)])
                frames, success, steps = [], False, 0

                for t in range(max_steps + NUM_STEPS_WAIT):
                    if t < NUM_STEPS_WAIT:
                        obs, _, done, _ = env.step(
                            ov["get_libero_dummy_action"](cfg.model_family))
                        continue

                    img = ov["get_libero_image"](obs, RESIZE_SIZE)  # H,W,3 uint8
                    if degrader is not None:
                        img = degrader.process(img)
                    if len(frames) < 400:
                        frames.append(img)

                    observation = {
                        "full_image": img,
                        "state": _state_vector(obs, ov["quat2axisangle"]),
                    }
                    action = ov["get_action"](
                        cfg, model, observation, task_description, processor=processor)
                    action = ov["normalize_gripper_action"](action, binarize=True)
                    action = ov["invert_gripper_action"](action)

                    obs, _, done, _ = env.step(action.tolist())
                    steps += 1
                    if done:
                        success = True
                        break

                record = {
                    "task_id": task_id,
                    "task": task_description,
                    "episode": ep,
                    "success": bool(success),
                    "episode_length": steps,
                    "timed_out": (not success),
                    "max_steps": max_steps,
                    "degradation": deg_meta,
                    "degrader_stats": degrader.stats if degrader else {},
                    "seed": episode_seed,
                }
                trials.append(record)
                tf.write(json.dumps(record) + "\n")
                tf.flush()

                if ep < cfg.save_videos:
                    try:
                        os.makedirs(video_dir, exist_ok=True)
                        ov["save_rollout_video"](
                            frames, f"{cfg.run_name}_t{task_id}_e{ep}",
                            success=success, task_description=task_description)
                    except Exception as e:  # noqa: BLE001 - video is best-effort
                        print(f"[warn] video save failed: {e}", flush=True)

                tag = "OK " if success else "FAIL"
                print(f"  [{tag}] task {task_id} '{task_description[:40]}' "
                      f"ep {ep}: {steps} steps", flush=True)

    summary = summarize(trials, cfg, deg_meta, elapsed_s=time.time() - t_start)
    with open(run_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n[done] success_rate={summary['success_rate']:.3f} "
          f"({summary['n_success']}/{summary['n_trials']}) "
          f"-> {run_dir}", flush=True)
    _print_table(summary)
    return summary


def summarize(trials, cfg: EvalConfig, deg_meta: dict, elapsed_s: float) -> dict:
    n = len(trials)
    n_succ = sum(t["success"] for t in trials)
    succ_lengths = [t["episode_length"] for t in trials if t["success"]]
    # per-task breakdown
    per_task = {}
    for t in trials:
        d = per_task.setdefault(t["task"], {"n": 0, "success": 0})
        d["n"] += 1
        d["success"] += int(t["success"])
    per_task_rate = {k: v["success"] / v["n"] for k, v in per_task.items()}
    return {
        "run_name": cfg.run_name,
        "task_suite": cfg.task_suite_name,
        "checkpoint": cfg.pretrained_checkpoint,
        "degradation": deg_meta,
        "n_trials": n,
        "n_success": int(n_succ),
        "success_rate": (n_succ / n) if n else 0.0,
        "mean_success_length": float(np.mean(succ_lengths)) if succ_lengths else None,
        "per_task_success_rate": per_task_rate,
        "elapsed_s": round(elapsed_s, 1),
        "config": dataclasses.asdict(cfg),
    }


def _print_table(summary: dict):
    print("\n  per-task success rate")
    print("  " + "-" * 52)
    for task, rate in summary["per_task_success_rate"].items():
        bar = "#" * int(rate * 20)
        print(f"  {rate:5.0%} |{bar:<20}| {task[:30]}")
    print("  " + "-" * 52)
    print(f"  OVERALL {summary['success_rate']:.1%} "
          f"({summary['n_success']}/{summary['n_trials']})")
