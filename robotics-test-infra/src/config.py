"""Test configuration and thresholds.

Pure-Python (no heavy deps) so Tier-1 CI can validate it without MuJoCo. All fields
are overridable from environment variables (``RTI_*``) so a CI workflow can tune a run
— number of episodes, seed, thresholds, render backend — without editing code.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, fields
from pathlib import Path

# Default checkpoint: the Project 1 ACT policy. Relative to the repo root so it
# resolves whether the evaluator is run from robotics-test-infra/ or the repo root.
# Can be a local path OR a Hugging Face Hub repo id (CI pulls the real ckpt from HF).
DEFAULT_CHECKPOINT = "../outputs/train/act_aloha_cube/checkpoints/last/pretrained_model"

ENV_PREFIX = "RTI_"


class ConfigError(ValueError):
    """Raised when a configuration is internally inconsistent or out of range."""


@dataclass
class EvalConfig:
    """Configuration for one policy-evaluation run.

    Thresholds default to values calibrated against the undertrained Project 1 policy;
    the *real* pass/fail bar is the committed baseline (see ``baselines/``). They are
    deliberately lenient defaults — the regression check, not these thresholds, is the
    primary gate.
    """

    # --- what to evaluate ---
    checkpoint_path: str = DEFAULT_CHECKPOINT
    env_type: str = "aloha"
    task: str = "AlohaTransferCube-v0"

    # --- how much to evaluate ---
    num_episodes: int = 10
    max_episode_steps: int = 300  # per-episode timeout (env hard cap is 400)
    seed: int = 100_000  # start seed; episode i uses seed + i

    # --- pass/fail thresholds ---
    # The Project-1 checkpoint is undertrained and empirically scores 0% on the cube
    # transfer (measured: 0/10 episodes at the full 400-step horizon). So the success
    # gate is 0.0 by design — we do NOT yet gate on task success. The binding, *passing*
    # gates are the timing budget (p95 latency) and the regression check vs. baseline;
    # raise this once a better checkpoint earns a real success rate. See README.
    success_rate_threshold: float = 0.0  # minimum acceptable success fraction
    max_avg_episode_length: float = 300.0  # latency budget, in steps
    inference_latency_ceiling_ms: float = 50.0  # ceiling on p95 per-step inference

    # --- runtime knobs ---
    device: str = "cpu"
    render_backend: str = "egl"  # MuJoCo GL backend; "osmesa" (software) for CI
    video_capture_on_nonsuccess: bool = True
    fps: int = 15  # video frame rate / downscale target handled in video_capture

    def __post_init__(self) -> None:
        self.validate()

    # ------------------------------------------------------------------ #
    def validate(self) -> None:
        """Raise ``ConfigError`` if any field is out of range or inconsistent."""
        if self.num_episodes < 1:
            raise ConfigError(f"num_episodes must be >= 1, got {self.num_episodes}")
        if self.max_episode_steps < 1:
            raise ConfigError(f"max_episode_steps must be >= 1, got {self.max_episode_steps}")
        if not (0.0 <= self.success_rate_threshold <= 1.0):
            raise ConfigError(
                f"success_rate_threshold must be in [0, 1], got {self.success_rate_threshold}"
            )
        if self.max_avg_episode_length <= 0:
            raise ConfigError(
                f"max_avg_episode_length must be > 0, got {self.max_avg_episode_length}"
            )
        if self.inference_latency_ceiling_ms <= 0:
            raise ConfigError(
                f"inference_latency_ceiling_ms must be > 0, got {self.inference_latency_ceiling_ms}"
            )
        if self.fps < 1:
            raise ConfigError(f"fps must be >= 1, got {self.fps}")
        if self.render_backend not in ("egl", "osmesa", "glfw"):
            raise ConfigError(
                f"render_backend must be one of egl/osmesa/glfw, got {self.render_backend!r}"
            )

    def checkpoint_exists(self) -> bool:
        """True if the checkpoint resolves to a local directory containing a model.

        Returns True for anything that *looks* like a Hub repo id (``org/name``, no
        path separators that resolve locally) so config validation doesn't reject a
        Hub-hosted checkpoint; the evaluator does the real existence check at load.
        """
        p = Path(self.checkpoint_path)
        if p.exists():
            return (p / "config.json").exists() or (p / "model.safetensors").exists()
        # Looks like a Hub repo id (e.g. "zoey/act-aloha-cube"): can't check offline.
        return "/" in self.checkpoint_path and not self.checkpoint_path.startswith((".", "/"))

    # ------------------------------------------------------------------ #
    @classmethod
    def from_env(cls, **overrides) -> EvalConfig:
        """Build a config from defaults, overlaying ``RTI_*`` env vars, then explicit
        keyword overrides (highest precedence). Used by CI to parameterize a run."""
        kwargs: dict = {}
        type_by_name = {f.name: f.type for f in fields(cls)}
        for f in fields(cls):
            env_key = f"{ENV_PREFIX}{f.name.upper()}"
            if env_key in os.environ:
                kwargs[f.name] = _coerce(os.environ[env_key], type_by_name[f.name])
        kwargs.update(overrides)
        return cls(**kwargs)


def _coerce(raw: str, type_hint) -> object:
    """Coerce an env-var string to the field's type (str hints arrive as strings)."""
    hint = str(type_hint)
    if "bool" in hint:
        return raw.strip().lower() in ("1", "true", "yes", "on")
    if "int" in hint:
        return int(raw)
    if "float" in hint:
        return float(raw)
    return raw
