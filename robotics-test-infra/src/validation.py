"""Pre-flight validation: checkpoint integrity (5d) and environment health (5e).

These are the "bring-up" checks that run *before* a full evaluation — catching a
corrupted checkpoint or a broken sim environment in seconds instead of wasting a
15-minute eval (or worse, reporting garbage metrics). This mirrors hardware bring-up
validation: confirm the system is functional before you trust its measurements.

Both heavy dependencies (safetensors / lerobot / mujoco) are imported lazily so the
report dataclasses and the orchestration import without the sim stack.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class Check:
    name: str
    passed: bool
    detail: str = ""


@dataclass
class ValidationReport:
    title: str
    checks: list[Check] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def add(self, name: str, passed: bool, detail: str = "") -> None:
        self.checks.append(Check(name, passed, detail))

    @property
    def ok(self) -> bool:
        return all(c.passed for c in self.checks)

    def failures(self) -> list[Check]:
        return [c for c in self.checks if not c.passed]

    def render(self) -> str:
        lines = [f"{self.title}: {'PASS' if self.ok else 'FAIL'}"]
        for c in self.checks:
            mark = "✓" if c.passed else "✗"
            lines.append(f"  {mark} {c.name}" + (f" — {c.detail}" if c.detail else ""))
        return "\n".join(lines)


# --------------------------------------------------------------------------- #
# 5d: checkpoint validation
# --------------------------------------------------------------------------- #
def validate_checkpoint(checkpoint_path: str) -> ValidationReport:
    """Validate a policy checkpoint before evaluation.

    Checks: directory + config.json + model.safetensors exist; config declares the
    expected ACT architecture with input/output feature shapes; weights are finite
    (no NaN/Inf); training-step metadata is readable. A Hub repo id can't be checked
    offline, so that case passes with a note and defers to the loader.
    """
    report = ValidationReport(title=f"Checkpoint validation [{checkpoint_path}]")

    p = Path(checkpoint_path)
    if not p.exists():
        if "/" in checkpoint_path and not checkpoint_path.startswith((".", "/")):
            report.add("local_path", True, "remote Hub id — offline validation skipped")
            return report
        report.add("exists", False, "path does not exist")
        return report

    config_json = p / "config.json"
    model_file = p / "model.safetensors"
    report.add("config.json present", config_json.exists())
    report.add("model.safetensors present", model_file.exists())
    if not config_json.exists() or not model_file.exists():
        return report

    # architecture / metadata from config.json
    try:
        cfg = json.loads(config_json.read_text())
        ptype = cfg.get("type")
        report.add("architecture is ACT", ptype == "act", f"type={ptype}")
        has_io = bool(cfg.get("input_features")) and bool(cfg.get("output_features"))
        report.add("input/output features declared", has_io)
        report.metadata["chunk_size"] = cfg.get("chunk_size")
        report.metadata["n_action_steps"] = cfg.get("n_action_steps")
    except Exception as e:
        report.add("config.json parses", False, str(e))
        return report

    # training step from the checkpoint directory name (…/checkpoints/000400/…)
    step = None
    for part in p.parts:
        if part.isdigit():
            step = int(part)
    report.metadata["training_step"] = step
    # Informational, not an integrity gate: a Hub/non-standard path may have no step.
    report.add("metadata readable", True, f"step={step}, chunk_size={report.metadata['chunk_size']}")

    # weights finite — open with the numpy backend so we don't need torch
    try:
        import numpy as np
        from safetensors import safe_open

        bad = []
        with safe_open(str(model_file), framework="numpy") as f:
            for key in f.keys():  # noqa: SIM118 (safetensors handle, not a dict)
                arr = f.get_tensor(key)
                if arr.dtype.kind == "f" and not np.isfinite(arr).all():
                    bad.append(key)
        report.add("weights finite (no NaN/Inf)", not bad,
                   "all finite" if not bad else f"{len(bad)} bad tensors: {bad[:3]}")
    except ImportError:
        report.add("weights finite (no NaN/Inf)", True, "safetensors unavailable — skipped")
    except Exception as e:
        report.add("weights finite (no NaN/Inf)", False, f"could not read weights: {e}")

    return report


# --------------------------------------------------------------------------- #
# 5e: environment health check
# --------------------------------------------------------------------------- #
def health_check_env(
    env_type: str = "aloha",
    task: str = "AlohaTransferCube-v0",
    render_backend: str = "egl",
) -> ValidationReport:
    """Verify the simulation environment is functional before running tests.

    Checks: MuJoCo + the task MJCF initialize; one step executes without crashing;
    offscreen render returns a frame; action/observation spaces have sane shapes.
    """
    import os

    os.environ["MUJOCO_GL"] = render_backend
    os.environ["PYOPENGL_PLATFORM"] = render_backend

    report = ValidationReport(title=f"Env health [{env_type}/{task}, {render_backend}]")
    try:
        from lerobot.envs.factory import make_env, make_env_config
    except ImportError as e:
        report.add("sim stack importable", False, str(e))
        return report
    report.add("sim stack importable", True)

    try:
        env_cfg = make_env_config(env_type, task=task)
        envs = make_env(env_cfg, n_envs=1, use_async_envs=False)
        group = next(iter(envs))
        tid = next(iter(envs[group]))
        env = envs[group][tid]
        report.add("env + task MJCF initialize", True)
    except Exception as e:
        report.add("env + task MJCF initialize", False, str(e))
        return report

    try:
        obs, _ = env.reset(seed=0)
        report.add("reset() works", True, f"obs keys: {sorted(obs.keys())}")
    except Exception as e:
        report.add("reset() works", False, str(e))
        return report

    try:
        action = env.action_space.sample()
        env.step(action)
        report.add("single step() works", True)
    except Exception as e:
        report.add("single step() works", False, str(e))

    try:
        ashape = getattr(env.action_space, "shape", None)
        ndims = len(ashape) if ashape else 0
        report.add("action space shape sane", ndims >= 1, f"shape={ashape}")
        report.metadata["action_shape"] = list(ashape) if ashape else None
        report.metadata["max_episode_steps"] = int(env.call("_max_episode_steps")[0])
    except Exception as e:
        report.add("action space shape sane", False, str(e))

    try:
        frame = env.envs[0].render()
        ok = frame is not None and getattr(frame, "ndim", 0) == 3
        report.add("offscreen render works", ok, f"frame shape: {getattr(frame, 'shape', None)}")
    except Exception as e:
        report.add("offscreen render works", False, str(e))

    try:
        env.close()
    except Exception:
        pass
    return report
