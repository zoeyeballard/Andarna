"""Shared LIBERO evaluation engine for OpenVLA, with an observation-degradation hook.

This is the one rollout loop that every experiment shares. ``run_baseline.py`` calls
it with no degrader; ``run_degradation.py`` / ``run_full_sweep.py`` call it with an
:class:`~robustness.degradations.ObservationDegrader`. The model is never touched —
the degrader is applied to the policy-input image, *after* LIBERO produces the frame
and *before* the model's own preprocessing, which is exactly "what the sensor
delivers to the policy."

Runs **on the Colab GPU VM** (needs torch+CUDA + the `libero` package).

Why we don't import OpenVLA's ``experiments.robot.*`` helpers: that package imports
``dlimp -> tensorflow_datasets -> tensorflow``, which pulls a fragile TF/protobuf
stack we don't need for eval. So we reimplement the ~40 lines of eval glue here
(prompt format, 0.9 center-crop, gripper normalization, LIBERO env setup) to match
OpenVLA's official ``run_libero_eval.py`` behavior, and talk to the model directly
via ``AutoModelForVision2Seq`` + ``model.predict_action`` (the documented HF API).

Authored locally; executed via `colab console` -> `python scripts/run_baseline.py ...`.
"""

from __future__ import annotations

import dataclasses
import json
import math
import os
import random
import sys
import time
from dataclasses import dataclass
from pathlib import Path

# Headless MuJoCo rendering on the Colab VM (must be set before robosuite imports).
# Default to OSMesa (CPU software GL): creating an EGL context on the GPU AFTER torch
# has initialized CUDA segfaults on Colab. OSMesa never touches the GPU's GL, so it
# can't conflict — and since rendering is tiny next to T4 inference, it's ~free.
# Override with `MUJOCO_GL=egl python ...` on hardware where EGL is known-good.
os.environ.setdefault("MUJOCO_GL", "osmesa")
os.environ.setdefault("PYOPENGL_PLATFORM", os.environ["MUJOCO_GL"])

import numpy as np

# --- make this file runnable both as a module and as a bare script ---------------
_HERE = Path(__file__).resolve().parent
_PROJECT = _HERE.parent
for p in (_PROJECT, _HERE):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from robustness.degradations import ObservationDegrader  # noqa: E402

RESIZE_SIZE = 224  # OpenVLA's LIBERO eval policy-input size
CENTER_CROP_SCALE = 0.9  # OpenVLA center-crops to 90% area at eval (matches train aug)

# Per-suite episode caps OpenVLA uses (longer suites get more steps).
MAX_STEPS = {
    "libero_spatial": 220,
    "libero_object": 280,
    "libero_goal": 300,
    "libero_10": 520,
    "libero_90": 400,
}
NUM_STEPS_WAIT = 10  # sim settle steps before the policy takes over
DUMMY_ACTION = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0]  # no motion, gripper open


@dataclass
class EvalConfig:
    """Eval configuration (the subset of OpenVLA's GenerateConfig the eval path uses)."""

    pretrained_checkpoint: str = "openvla/openvla-7b-finetuned-libero-object"
    task_suite_name: str = "libero_object"
    model_family: str = "openvla"
    unnorm_key: str = ""               # "" => auto-resolve from the model's norm_stats
    center_crop: bool = True
    load_in_8bit: bool = False
    load_in_4bit: bool = False
    num_trials_per_task: int = 20
    num_tasks: int = 0                 # 0 == all tasks in the suite
    seed: int = 7
    save_videos: int = 2               # save this many rollout videos per task
    run_name: str = "baseline"
    results_root: str = str(_PROJECT / "results" / "baseline")


# --- imports that only exist on the VM --------------------------------------------
def _patch_torch_load():
    """torch>=2.6 defaults torch.load(weights_only=True), which refuses to unpickle
    LIBERO's numpy-array init-state files. They're trusted local sim assets, so make
    weights_only=False the default. Idempotent."""
    import torch
    if getattr(torch.load, "_libero_compat", False):
        return
    _orig = torch.load

    def _patched(*args, **kwargs):
        kwargs.setdefault("weights_only", False)
        return _orig(*args, **kwargs)

    _patched._libero_compat = True
    torch.load = _patched


def _import_libero():
    """Import the LIBERO package (TF-free). Raises a helpful error off-VM."""
    _patch_torch_load()
    try:
        from libero.libero import benchmark, get_libero_path
        from libero.libero.envs import OffScreenRenderEnv
    except ImportError as e:  # pragma: no cover - only hits off-VM
        raise RuntimeError(
            "LIBERO not importable. This engine runs on the Colab VM after "
            f"setup/install_remote.py. import error: {e}"
        ) from e
    return {"benchmark": benchmark, "get_libero_path": get_libero_path,
            "OffScreenRenderEnv": OffScreenRenderEnv}


def _gpu_supports_flash_attn() -> bool:
    """flash-attn 2.x needs Ampere+ (sm80). T4/V100 (Turing/Volta) do not qualify."""
    try:
        import torch
        if not torch.cuda.is_available():
            return False
        major, _ = torch.cuda.get_device_capability(0)
        return major >= 8
    except Exception:  # noqa: BLE001
        return False


# --- model loading (direct; no OpenVLA experiments package) -----------------------
def load_policy(cfg: EvalConfig):
    """Load the OpenVLA model + processor once. Returns (model, processor, dtype).

    flash-attn on Ampere+ (bf16); SDPA + fp16 on a T4. unnorm_key is auto-resolved
    from the checkpoint's own norm_stats so we never guess the dataset key.
    """
    import torch
    from transformers import (AutoModelForVision2Seq, AutoProcessor,
                              BitsAndBytesConfig)

    flash = _gpu_supports_flash_attn()
    dtype = torch.bfloat16 if flash else torch.float16
    print(f"[load] {cfg.pretrained_checkpoint} "
          f"(4bit={cfg.load_in_4bit} 8bit={cfg.load_in_8bit} "
          f"attn={'flash_attention_2' if flash else 'sdpa'} dtype={dtype})", flush=True)

    kwargs = dict(
        attn_implementation="flash_attention_2" if flash else "sdpa",
        torch_dtype=dtype,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
    )
    # Quantized models MUST be placed via device_map (modern transformers/accelerate
    # forbid a post-hoc .to() on bnb models). Pin the whole model to GPU 0.
    if cfg.load_in_4bit:
        kwargs["device_map"] = {"": 0}
        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=dtype,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        )
    elif cfg.load_in_8bit:
        kwargs["device_map"] = {"": 0}
        kwargs["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True)

    model = AutoModelForVision2Seq.from_pretrained(cfg.pretrained_checkpoint, **kwargs)
    if not (cfg.load_in_4bit or cfg.load_in_8bit):
        model = model.to("cuda")
    model.eval()
    processor = AutoProcessor.from_pretrained(
        cfg.pretrained_checkpoint, trust_remote_code=True)

    # Resolve the action un-normalization key from the model itself.
    norm_stats = getattr(model, "norm_stats", None)
    if norm_stats:
        keys = list(norm_stats.keys())
        if cfg.unnorm_key and cfg.unnorm_key in norm_stats:
            pass
        elif len(keys) == 1:
            print(f"[load] unnorm_key -> '{keys[0]}' (only key in checkpoint)", flush=True)
            cfg.unnorm_key = keys[0]
        elif cfg.task_suite_name in norm_stats:
            cfg.unnorm_key = cfg.task_suite_name
        else:
            raise ValueError(
                f"unnorm_key '{cfg.unnorm_key}' not in checkpoint norm_stats {keys}; "
                "pass --checkpoint's correct key.")
    print(f"[load] using unnorm_key='{cfg.unnorm_key}'", flush=True)
    return model, processor, dtype


# --- eval-glue reimplemented to match OpenVLA's run_libero_eval.py -----------------
def _center_crop(image, scale: float = CENTER_CROP_SCALE):
    """Center-crop to ``scale`` *area* then resize back — OpenVLA's eval-time crop."""
    from PIL import Image
    w, h = image.size
    side = math.sqrt(scale)
    cw, ch = int(round(w * side)), int(round(h * side))
    left, top = (w - cw) // 2, (h - ch) // 2
    return image.crop((left, top, left + cw, top + ch)).resize((w, h), Image.BILINEAR)


def _build_prompt(instruction: str) -> str:
    return f"In: What action should the robot take to {instruction.lower()}?\nOut:"


def predict_action(model, processor, dtype, img_uint8: np.ndarray,
                   instruction: str, unnorm_key: str, center_crop: bool) -> np.ndarray:
    """One OpenVLA action from one (already-degraded) RGB frame."""
    from PIL import Image
    image = Image.fromarray(img_uint8).convert("RGB")
    if center_crop:
        image = _center_crop(image)
    inputs = processor(_build_prompt(instruction), image).to("cuda", dtype=dtype)
    action = model.predict_action(**inputs, unnorm_key=unnorm_key, do_sample=False)
    return np.asarray(action, dtype=np.float32).reshape(-1)


def normalize_gripper_action(action: np.ndarray, binarize: bool = True) -> np.ndarray:
    """Map gripper dim from [0,1] -> [-1,1] (binarized), matching OpenVLA."""
    action = action.copy()
    action[-1] = 2.0 * (action[-1] - 0.0) / (1.0 - 0.0) - 1.0
    if binarize:
        action[-1] = float(np.sign(action[-1]))
    return action


def invert_gripper_action(action: np.ndarray) -> np.ndarray:
    """LIBERO's gripper convention is inverted vs the model's output."""
    action = action.copy()
    action[-1] *= -1.0
    return action


def make_libero_env(lb, task, resolution: int = 256):
    bddl = os.path.join(lb["get_libero_path"]("bddl_files"),
                        task.problem_folder, task.bddl_file)
    env = lb["OffScreenRenderEnv"](
        bddl_file_name=bddl, camera_heights=resolution, camera_widths=resolution)
    env.seed(0)
    return env, task.language


def extract_agentview(obs, resize_size: int = RESIZE_SIZE) -> np.ndarray:
    """LIBERO agentview frame -> the policy-input RGB (flip 180, resize)."""
    from PIL import Image
    img = obs["agentview_image"][::-1, ::-1]  # LIBERO renders 180-deg rotated
    img = Image.fromarray(img).resize((resize_size, resize_size), Image.LANCZOS)
    return np.asarray(img, dtype=np.uint8)


def _set_seed(seed: int):
    import torch
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _save_video(frames, path: Path, fps: int = 10):
    import imageio
    imageio.mimwrite(str(path), [np.asarray(f, np.uint8) for f in frames],
                     fps=fps, macro_block_size=None)


# --- orchestration ----------------------------------------------------------------
def prepare(cfg: EvalConfig):
    """Import LIBERO and load the model ONCE; reuse across all sweep levels."""
    lb = _import_libero()
    model, processor, dtype = load_policy(cfg)
    return {"lb": lb, "model": model, "processor": processor, "dtype": dtype}


def run_eval(cfg: EvalConfig, degrader: ObservationDegrader | None = None,
             prepared: dict | None = None) -> dict:
    """Run the full suite eval (optionally degraded). Writes trials.jsonl + summary.json."""
    if prepared is None:
        prepared = prepare(cfg)
    lb, model, processor, dtype = (prepared["lb"], prepared["model"],
                                   prepared["processor"], prepared["dtype"])
    _set_seed(cfg.seed)

    run_dir = Path(cfg.results_root) / cfg.run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    trials_path = run_dir / "trials.jsonl"
    video_dir = run_dir / "videos"

    suite = lb["benchmark"].get_benchmark_dict()[cfg.task_suite_name]()
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
            env, task_description = make_libero_env(lb, task, resolution=256)

            for ep in range(cfg.num_trials_per_task):
                episode_seed = cfg.seed + task_id * 1000 + ep
                if degrader is not None:
                    degrader.reset(seed=episode_seed)

                env.reset()
                obs = env.set_init_state(init_states[ep % len(init_states)])
                frames, success, steps = [], False, 0

                for t in range(max_steps + NUM_STEPS_WAIT):
                    if t < NUM_STEPS_WAIT:
                        obs, _, done, _ = env.step(DUMMY_ACTION)
                        continue

                    img = extract_agentview(obs, RESIZE_SIZE)  # H,W,3 uint8
                    if degrader is not None:
                        img = degrader.process(img)
                    if len(frames) < 400:
                        frames.append(img)

                    action = predict_action(model, processor, dtype, img,
                                            task_description, cfg.unnorm_key,
                                            cfg.center_crop)
                    action = normalize_gripper_action(action, binarize=True)
                    action = invert_gripper_action(action)

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

                if ep < cfg.save_videos and frames:
                    try:
                        video_dir.mkdir(parents=True, exist_ok=True)
                        _save_video(frames, video_dir /
                                    f"{cfg.run_name}_t{task_id}_e{ep}"
                                    f"_{'ok' if success else 'fail'}.mp4")
                    except Exception as e:  # noqa: BLE001 - video is best-effort
                        print(f"[warn] video save failed: {e}", flush=True)

                tag = "OK " if success else "FAIL"
                print(f"  [{tag}] task {task_id} '{task_description[:40]}' "
                      f"ep {ep}: {steps} steps", flush=True)

    summary = summarize(trials, cfg, deg_meta, elapsed_s=time.time() - t_start)
    with open(run_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n[done] success_rate={summary['success_rate']:.3f} "
          f"({summary['n_success']}/{summary['n_trials']}) -> {run_dir}", flush=True)
    _print_table(summary)
    return summary


def summarize(trials, cfg: EvalConfig, deg_meta: dict, elapsed_s: float) -> dict:
    n = len(trials)
    n_succ = sum(t["success"] for t in trials)
    succ_lengths = [t["episode_length"] for t in trials if t["success"]]
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
