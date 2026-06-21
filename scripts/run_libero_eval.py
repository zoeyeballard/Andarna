#!/usr/bin/env python
"""
run_libero_eval.py — Phase 6 (behavioral): LIBERO task success rate per precision.

Numerical accuracy (Phase 6) showed quantization perturbs the action vectors, worst on the
translation deltas. This asks the question that actually matters: does that perturbation change
whether the robot *succeeds at the task*? We run LIBERO-Object rollouts at each precision
(BF16/FP16/INT8/INT4) and report success rate — quantization as a behavioral perturbation,
connecting back to the Project 2 robustness study.

Reuses Project 2's validated rollout glue (env setup, OpenVLA center-crop, gripper convention,
agentview extraction, success detection) from openvla-robustness/scripts/libero_eval.py, but
drives model loading through our Phase-5 precision loader so every precision is controlled and
labeled exactly (the Project 2 loader auto-picks dtype; here we pin it).

Each precision runs in its own subprocess: clean CUDA context, and the EGL render context is
primed BEFORE CUDA inits (priming after segfaults — see the Project 2 harness notes).

Usage:
    # quick end-to-end smoke (1 task, 1 trial, capped steps):
    python scripts/run_libero_eval.py --precision bf16 --num-tasks 1 --trials 1 --max-steps 25 \
        --fragment-out /tmp/smoke.json
    # full sweep:
    python scripts/run_libero_eval.py --num-tasks 2 --trials 3 --only bf16 fp16 int4
"""
from __future__ import annotations

import os

# Headless MuJoCo on the A10G — must be set before robosuite/libero import.
os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("PYOPENGL_PLATFORM", os.environ["MUJOCO_GL"])

import argparse
import json
import platform
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

# --- reuse Project 2's rollout glue --------------------------------------------------
HARNESS_DIR = Path.home() / "repositories/personal/Andarna/openvla-robustness/scripts"
sys.path.insert(0, str(HARNESS_DIR))
sys.path.insert(0, str(HARNESS_DIR.parent))  # so `robustness` package imports
import libero_eval as H  # noqa: E402

# --- our precision-controlled loader -------------------------------------------------
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from quantization.precision_runner import INPUT_DTYPE, PRECISIONS, load_model  # noqa: E402

DEFAULT_MODEL = "openvla/openvla-7b-finetuned-libero-object"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="OpenVLA LIBERO behavioral eval per precision.")
    p.add_argument("--model-id", default=DEFAULT_MODEL)
    p.add_argument("--task-suite", default="libero_object")
    p.add_argument("--unnorm-key", default="")
    p.add_argument("--num-tasks", type=int, default=2, help="0 = all tasks in suite.")
    p.add_argument("--trials", type=int, default=3, help="Trials per task.")
    p.add_argument("--max-steps", type=int, default=0, help="0 = suite default cap (else override).")
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--only", nargs="+", choices=PRECISIONS, default=PRECISIONS)
    p.add_argument("--output", default="results/behavioral/libero_success.json")
    # worker-mode:
    p.add_argument("--precision", choices=PRECISIONS)
    p.add_argument("--fragment-out")
    return p.parse_args()


def resolve_unnorm_key(model, suite_name: str, override: str) -> str:
    ns = getattr(model, "norm_stats", None)
    keys = list(ns.keys()) if ns else []
    if override and override in keys:
        return override
    if len(keys) == 1:
        return keys[0]
    if suite_name in keys:
        return suite_name
    return override or (keys[0] if keys else "")


# --------------------------------------------------------------------------- worker
def run_worker(args) -> dict:
    lb = H._import_libero()
    # Prime the EGL render context BEFORE the model inits CUDA (ordering matters).
    H._warmup_render(lb, H.EvalConfig(task_suite_name=args.task_suite))

    processor, model = load_model(args.model_id, args.precision, args.device)
    dtype = INPUT_DTYPE[args.precision]
    unnorm_key = resolve_unnorm_key(model, args.task_suite, args.unnorm_key)
    H._set_seed(args.seed)

    suite = lb["benchmark"].get_benchmark_dict()[args.task_suite]()
    n_tasks = suite.n_tasks if args.num_tasks == 0 else min(args.num_tasks, suite.n_tasks)
    max_steps = args.max_steps or H.MAX_STEPS.get(args.task_suite, 300)

    trials, step_latencies_ms = [], []
    t0 = time.time()
    for task_id in range(n_tasks):
        task = suite.get_task(task_id)
        init_states = suite.get_task_init_states(task_id)
        env, task_description = H.make_libero_env(lb, task, resolution=256)
        for ep in range(args.trials):
            env.reset()
            obs = env.set_init_state(init_states[ep % len(init_states)])
            success, steps = False, 0
            with torch.inference_mode():
                for t in range(max_steps + H.NUM_STEPS_WAIT):
                    if t < H.NUM_STEPS_WAIT:
                        obs, _, done, _ = env.step(H.DUMMY_ACTION)
                        continue
                    img = H.extract_agentview(obs, H.RESIZE_SIZE)
                    ev0 = torch.cuda.Event(enable_timing=True)
                    ev1 = torch.cuda.Event(enable_timing=True)
                    ev0.record()
                    action = H.predict_action(model, processor, dtype, img,
                                              task_description, unnorm_key, center_crop=True)
                    ev1.record(); torch.cuda.synchronize()
                    step_latencies_ms.append(ev0.elapsed_time(ev1))
                    action = H.normalize_gripper_action(action, binarize=True)
                    action = H.invert_gripper_action(action)
                    obs, _, done, _ = env.step(action.tolist())
                    steps += 1
                    if done:
                        success = True
                        break
            trials.append({"task_id": task_id, "task": task_description, "episode": ep,
                           "success": bool(success), "episode_length": steps})
            print(f"  [{ 'OK ' if success else 'FAIL'}] task {task_id} ep {ep}: {steps} steps",
                  flush=True)
        env.close()

    n = len(trials)
    n_succ = sum(t["success"] for t in trials)
    lat = np.asarray(step_latencies_ms, dtype=np.float64)
    return {
        "status": "ok", "precision": args.precision,
        "n_trials": n, "n_success": int(n_succ),
        "success_rate": (n_succ / n) if n else 0.0,
        "mean_step_latency_ms": round(float(lat.mean()), 2) if lat.size else None,
        "control_freq_hz": round(1000.0 / float(lat.mean()), 2) if lat.size else None,
        "max_steps": max_steps, "elapsed_s": round(time.time() - t0, 1),
        "trials": trials,
    }


# --------------------------------------------------------------------------- orchestrator
def run_worker_subprocess(precision: str, args, frag: Path) -> dict:
    cmd = [
        sys.executable, os.path.abspath(__file__),
        "--precision", precision, "--fragment-out", str(frag),
        "--model-id", args.model_id, "--task-suite", args.task_suite,
        "--unnorm-key", args.unnorm_key, "--num-tasks", str(args.num_tasks),
        "--trials", str(args.trials), "--max-steps", str(args.max_steps),
        "--seed", str(args.seed), "--device", args.device,
    ]
    print(f"\n[libero] === {precision.upper()} === ({args.num_tasks} tasks x {args.trials} trials)")
    proc = subprocess.run(cmd, text=True)
    if proc.returncode != 0 or not frag.exists():
        print(f"[libero] {precision} FAILED (rc={proc.returncode})")
        return {"status": "failed", "precision": precision, "returncode": proc.returncode}
    r = json.loads(frag.read_text())
    print(f"[libero] {precision}: success {r['n_success']}/{r['n_trials']} "
          f"({r['success_rate']:.0%}) | {r['mean_step_latency_ms']} ms/step | {r['elapsed_s']}s")
    return r


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is not available — this requires a GPU.")

    if args.precision:  # worker mode
        result = run_worker(args)
        Path(args.fragment_out).write_text(json.dumps(result))
        return

    frag_dir = Path(args.output).parent / "_libero_fragments"
    frag_dir.mkdir(parents=True, exist_ok=True)
    results = {}
    for precision in args.only:
        frag = frag_dir / f"{precision}.json"
        if frag.exists():
            frag.unlink()
        results[precision] = run_worker_subprocess(precision, args, frag)

    summary = {
        "metadata": {
            "model_id": args.model_id, "task_suite": args.task_suite,
            "num_tasks": args.num_tasks, "trials_per_task": args.trials,
            "device_name": torch.cuda.get_device_name(args.device),
            "torch_version": torch.__version__, "python_version": platform.python_version(),
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        },
        "results": results,
    }
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, indent=2))

    print(f"\n=== LIBERO {args.task_suite} success rate by precision ===")
    print(f"  {'prec':6}{'success':>12}{'rate':>8}{'ms/step':>10}{'Hz':>7}")
    print("  " + "-" * 41)
    for p in PRECISIONS:
        r = results.get(p)
        if not r or r.get("status") != "ok":
            continue
        print(f"  {p:6}{r['n_success']:>5}/{r['n_trials']:<6}{r['success_rate']:>7.0%}"
              f"{r['mean_step_latency_ms']:>10}{r['control_freq_hz']:>7}")
    print(f"\nSaved -> {out}")


if __name__ == "__main__":
    main()
