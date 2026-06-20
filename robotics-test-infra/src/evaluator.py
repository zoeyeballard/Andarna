"""Runs policy evaluation episodes in MuJoCo and records instrumented results.

This wraps LeRobot's evaluation pipeline (policy loading + the per-step rollout glue)
but adds the instrumentation a test framework needs: per-step inference latency, a
clean per-episode timeout, final object pose, and frame capture for video.

Design notes:
- The heavy simulation stack (lerobot / torch / mujoco / gymnasium) is imported
  *lazily* inside ``load()`` so this module imports — and mocks — without it. The
  rollout loop in ``_run_episode`` references only ``self.*`` attributes, so a unit
  test can inject fakes for the env/policy/processors and exercise the loop with no
  checkpoint and no MuJoCo (see tests/unit/test_evaluator.py).
- ``MUJOCO_GL`` is set from the config *before* any mujoco import — egl locally,
  osmesa (software GL) inside the CI container which has no display/GPU.
"""

from __future__ import annotations

import contextlib
import os
import time
from pathlib import Path
from typing import Any

import numpy as np

from src.config import EvalConfig
from src.metrics import EpisodeResult, EvalResult

_PKG_ROOT = Path(__file__).resolve().parents[1]  # robotics-test-infra/
_REPO_ROOT = _PKG_ROOT.parent  # the Andarna repo root


class CheckpointError(FileNotFoundError):
    """Raised when the policy checkpoint can't be found or loaded."""


class SimEnvironmentError(RuntimeError):
    """Raised when the simulation environment fails to initialize or step."""


def resolve_checkpoint(checkpoint_path: str) -> str:
    """Resolve a checkpoint reference to a concrete local path or a Hub repo id.

    Tries the path as-given, then relative to the package root, then the repo root.
    Returns the first that exists. If none exist but the string looks like a Hub repo
    id (``org/name``), returns it unchanged for the loader to fetch. Otherwise raises.
    """
    p = Path(checkpoint_path)
    if p.is_absolute() and p.exists():
        return str(p)
    candidates = [p, _PKG_ROOT / checkpoint_path, _REPO_ROOT / checkpoint_path]
    for cand in candidates:
        if cand.exists():
            return str(cand.resolve())
    # Hub repo id: "org/name", not a local-looking relative path
    if "/" in checkpoint_path and not checkpoint_path.startswith((".", "/")):
        return checkpoint_path
    raise CheckpointError(
        f"Checkpoint not found: {checkpoint_path!r}. Looked in: "
        + ", ".join(str(c) for c in candidates)
    )


class PolicyEvaluator:
    """Loads a policy and runs N instrumented evaluation episodes in MuJoCo."""

    def __init__(self, config: EvalConfig):
        self.config = config
        # Pipeline components — populated by load(), or injected directly in tests.
        # Typed Any: concrete types live in the (optional) sim stack, and tests inject
        # fakes; this keeps mypy useful elsewhere without stubbing all of lerobot.
        self.env: Any = None
        self.policy: Any = None
        self.preprocessor: Any = None
        self.postprocessor: Any = None
        self.env_preprocessor: Any = None
        self.env_postprocessor: Any = None
        self._preprocess_observation: Any = None
        self._add_envs_task: Any = None
        self._ACTION: str = "action"
        # No-op by default so the rollout loop (and its unit tests) need no torch;
        # load() swaps in torch.inference_mode once the sim stack is imported.
        self._inference_ctx: Any = contextlib.nullcontext
        self._resolved_checkpoint: str | None = None
        self._loaded = False

    # ------------------------------------------------------------------ #
    def load(self) -> None:
        """Load the env, policy, and processors. Idempotent."""
        if self._loaded:
            return

        # Render backend must be chosen before mujoco is imported anywhere.
        os.environ["MUJOCO_GL"] = self.config.render_backend
        os.environ["PYOPENGL_PLATFORM"] = self.config.render_backend
        os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")

        try:
            import torch
            from lerobot.configs.policies import PreTrainedConfig
            from lerobot.envs.factory import (
                make_env,
                make_env_config,
                make_env_pre_post_processors,
            )
            from lerobot.envs.utils import add_envs_task, preprocess_observation
            from lerobot.policies.factory import make_policy, make_pre_post_processors
            from lerobot.utils.constants import ACTION
            from lerobot.utils.random_utils import set_seed
        except ImportError as e:  # pragma: no cover - exercised only on lean runners
            raise SimEnvironmentError(
                "Simulation dependencies not installed. Install the 'sim' extra "
                "(pip install -e '.[sim]') or run inside the Docker image."
            ) from e

        set_seed(self.config.seed)
        self._ACTION = ACTION
        self._preprocess_observation = preprocess_observation
        self._add_envs_task = add_envs_task
        self._inference_ctx = torch.inference_mode

        # --- environment ---
        try:
            env_cfg = make_env_config(self.config.env_type, task=self.config.task)
            envs = make_env(env_cfg, n_envs=1, use_async_envs=False)
        except Exception as e:
            raise SimEnvironmentError(
                f"Failed to create env {self.config.env_type}/{self.config.task}: {e}"
            ) from e
        # make_env returns {group: {task_id: vec_env}}; unwrap the single vec env.
        group = next(iter(envs))
        task_id = next(iter(envs[group]))
        self.env = envs[group][task_id]

        # --- policy + processors ---
        ckpt = resolve_checkpoint(self.config.checkpoint_path)
        self._resolved_checkpoint = ckpt
        try:
            policy_cfg = PreTrainedConfig.from_pretrained(ckpt)
            policy_cfg.pretrained_path = ckpt
            # The eval device, not the checkpoint's training device, governs where the
            # policy runs. A checkpoint trained on cuda records device="cuda" in its
            # config; loading it as-is puts the model on cuda while the preprocessor
            # (overridden below) feeds cpu tensors -> "tensors on different devices".
            # CI is cpu-only, so a GPU-trained checkpoint must still eval on cpu here.
            policy_cfg.device = self.config.device
            self.policy = make_policy(cfg=policy_cfg, env_cfg=env_cfg)
            self.policy.eval()
            self.preprocessor, self.postprocessor = make_pre_post_processors(
                policy_cfg=policy_cfg,
                pretrained_path=ckpt,
                preprocessor_overrides={"device_processor": {"device": self.config.device}},
            )
            self.env_preprocessor, self.env_postprocessor = make_env_pre_post_processors(
                env_cfg=env_cfg, policy_cfg=policy_cfg
            )
        except CheckpointError:
            raise
        except Exception as e:
            raise CheckpointError(f"Failed to load policy from {ckpt!r}: {e}") from e

        self._loaded = True

    # ------------------------------------------------------------------ #
    def evaluate(self, video_dir: str | Path | None = None) -> EvalResult:
        """Run ``config.num_episodes`` episodes and return structured results.

        If ``video_dir`` is given and ``config.video_capture_on_nonsuccess`` is set,
        each non-success episode is written there as an MP4.
        """
        if not self._loaded:
            self.load()

        capture = self.config.video_capture_on_nonsuccess and video_dir is not None
        episodes: list[EpisodeResult] = []
        start = time.perf_counter()
        for i in range(self.config.num_episodes):
            seed = self.config.seed + i
            ep, frames = self._run_episode(seed, episode_index=i, capture_frames=capture)
            episodes.append(ep)
            if capture and not ep.success and frames:
                assert video_dir is not None  # capture implies a target dir
                self._write_video(frames, ep, video_dir)
        wall = time.perf_counter() - start

        return EvalResult(
            episodes=episodes,
            wall_time_s=wall,
            checkpoint=self._resolved_checkpoint or self.config.checkpoint_path,
            env_type=self.config.env_type,
            task=self.config.task,
            num_episodes_requested=self.config.num_episodes,
            render_backend=self.config.render_backend,
        )

    # ------------------------------------------------------------------ #
    def _run_episode(
        self,
        seed: int,
        episode_index: int,
        capture_frames: bool = False,
        action_trace: list | None = None,
        step_times_ms: list | None = None,
    ) -> tuple[EpisodeResult, list]:
        """Run one instrumented rollout. Returns (result, captured_frames).

        Optional out-params (used by the determinism and benchmark tools, so they share
        this single rollout loop): ``action_trace`` collects each applied action array;
        ``step_times_ms`` collects each ``env.step`` wall time in ms.
        """
        policy, env = self.policy, self.env
        policy.reset()
        obs, _info = env.reset(seed=[seed])

        frames: list = []
        if capture_frames:
            frames.append(env.envs[0].render())

        latencies_ms: list[float] = []
        success = False
        sum_reward = 0.0
        max_reward = float("-inf")
        num_envs = getattr(env, "num_envs", 1)
        done = np.array([False] * num_envs)

        env_max = int(env.call("_max_episode_steps")[0])
        max_steps = min(self.config.max_episode_steps, env_max)
        executed = 0

        for _step in range(max_steps):
            o = self._preprocess_observation(obs)
            o = self._add_envs_task(env, o)
            o = self.env_preprocessor(o)
            o = self.preprocessor(o)

            t0 = time.perf_counter()
            with self._inference_ctx():
                action = policy.select_action(o)
            latencies_ms.append((time.perf_counter() - t0) * 1000.0)

            action = self.postprocessor(action)
            action = self.env_postprocessor({self._ACTION: action})[self._ACTION]
            action_np = action.to("cpu").numpy()
            if action_trace is not None:
                action_trace.append(np.array(action_np, copy=True))

            t_step = time.perf_counter()
            obs, reward, terminated, truncated, info = env.step(action_np)
            if step_times_ms is not None:
                step_times_ms.append((time.perf_counter() - t_step) * 1000.0)
            executed += 1
            if capture_frames:
                frames.append(env.envs[0].render())

            r = float(np.asarray(reward).reshape(-1)[0])
            sum_reward += r
            max_reward = max(max_reward, r)
            if isinstance(info, dict) and "final_info" in info and isinstance(info["final_info"], dict):
                success = success or bool(np.asarray(info["final_info"]["is_success"]).reshape(-1)[0])

            done = np.asarray(terminated) | np.asarray(truncated) | done
            if np.all(done):
                break

        timed_out = not bool(np.all(done))  # hit our step cap without natural termination
        result = EpisodeResult(
            episode_index=episode_index,
            seed=seed,
            success=success,
            length=executed,
            timed_out=timed_out,
            inference_times_ms=latencies_ms,
            sum_reward=sum_reward,
            max_reward=(max_reward if max_reward != float("-inf") else 0.0),
            final_object_position=self._read_object_position(),
        )
        return result, frames

    # ------------------------------------------------------------------ #
    def _read_object_position(self) -> list[float] | None:
        """Read the cube free-joint pose [x,y,z,qw,qx,qy,qz] from the physics, if reachable."""
        try:
            physics = self.env.envs[0].unwrapped._env.physics
            return [float(v) for v in physics.data.qpos[-7:]]
        except Exception:
            return None

    def _write_video(self, frames: list, ep: EpisodeResult, video_dir: str | Path) -> None:
        from src.video_capture import write_episode_video

        out = Path(video_dir) / f"episode_{ep.episode_index:02d}_seed{ep.seed}_FAIL.mp4"
        caption = "TIMEOUT" if ep.timed_out else "no success"
        write_episode_video(
            frames,
            out,
            inference_times_ms=ep.inference_times_ms,
            fps=self.config.fps,
            caption=caption,
        )

    def close(self) -> None:
        if self.env is not None:
            try:
                self.env.close()
            except Exception:
                pass
