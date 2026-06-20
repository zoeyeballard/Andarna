"""Unit tests for src.evaluator — checkpoint resolution + a mock-based rollout.

These run with NO simulation stack: the env/policy/processors are injected fakes, so
we verify the rollout loop's control flow (policy call count, timeout vs. natural
termination, result structure) without MuJoCo, torch, lerobot, or a real checkpoint.
"""

from unittest import mock

import numpy as np
import pytest

from src.config import EvalConfig
from src.evaluator import CheckpointError, PolicyEvaluator, resolve_checkpoint


# --- checkpoint resolution ---------------------------------------------- #
def test_resolve_checkpoint_existing_local(tmp_path):
    (tmp_path / "config.json").write_text("{}")
    assert resolve_checkpoint(str(tmp_path)) == str(tmp_path.resolve())


def test_resolve_checkpoint_hub_id_passthrough():
    assert resolve_checkpoint("lerobot/act_aloha_sim") == "lerobot/act_aloha_sim"


def test_resolve_checkpoint_missing_raises():
    with pytest.raises(CheckpointError):
        resolve_checkpoint("./definitely/not/here")


# --- fakes for the rollout loop ----------------------------------------- #
class FakeAction:
    """Stands in for a policy action tensor: supports .to(device).numpy()."""

    def to(self, _device):
        return self

    def numpy(self):
        return np.zeros((1, 14), dtype=np.float32)


class FakeSubEnv:
    def __init__(self, qpos):
        self._qpos = qpos

    def render(self):
        return np.zeros((48, 64, 3), dtype=np.uint8)

    @property
    def unwrapped(self):
        phys = mock.MagicMock()
        phys.data.qpos = self._qpos
        env = mock.MagicMock()
        env.physics = phys
        wrapper = mock.MagicMock()
        wrapper._env = env
        return wrapper


class FakeVecEnv:
    """Single-env vector env that succeeds at ``terminate_at`` steps, else never."""

    def __init__(self, terminate_at=None, max_steps=400):
        self.num_envs = 1
        self._terminate_at = terminate_at
        self._max_steps = max_steps
        self._steps = 0
        self.envs = [FakeSubEnv(np.arange(23, dtype=float))]
        self.closed = False

    def reset(self, seed=None):
        self._steps = 0
        return {"obs": np.zeros((1, 4))}, {}

    def step(self, action):
        self._steps += 1
        terminated = np.array([self._terminate_at is not None and self._steps >= self._terminate_at])
        truncated = np.array([False])
        info = {}
        if terminated[0]:
            info = {"final_info": {"is_success": np.array([True])}}
        return {"obs": np.zeros((1, 4))}, np.array([1.0]), terminated, truncated, info

    def call(self, name):
        if name == "_max_episode_steps":
            return [self._max_steps]
        raise KeyError(name)

    def close(self):
        self.closed = True


def _wire(evaluator, env):
    """Inject identity processors + the fake env so _run_episode can run pure-Python."""
    evaluator.env = env
    evaluator.policy = mock.MagicMock()
    evaluator.policy.select_action.return_value = FakeAction()
    identity = lambda x: x  # noqa: E731
    evaluator.preprocessor = identity
    evaluator.postprocessor = identity
    evaluator.env_preprocessor = identity
    evaluator.env_postprocessor = identity
    evaluator._preprocess_observation = identity
    evaluator._add_envs_task = lambda _env, o: o
    evaluator._ACTION = "action"
    evaluator._loaded = True


def test_rollout_natural_termination_success():
    cfg = EvalConfig(num_episodes=1, max_episode_steps=50, video_capture_on_nonsuccess=False)
    ev = PolicyEvaluator(cfg)
    _wire(ev, FakeVecEnv(terminate_at=6))
    result, frames = ev._run_episode(seed=42, episode_index=0, capture_frames=False)

    assert result.success is True
    assert result.length == 6
    assert result.timed_out is False
    assert ev.policy.select_action.call_count == 6  # one inference per executed step
    assert len(result.inference_times_ms) == 6
    assert result.final_object_position == list(np.arange(23, dtype=float)[-7:])


def test_rollout_respects_config_timeout():
    cfg = EvalConfig(num_episodes=1, max_episode_steps=20, video_capture_on_nonsuccess=False)
    ev = PolicyEvaluator(cfg)
    _wire(ev, FakeVecEnv(terminate_at=None, max_steps=400))  # never terminates
    result, _ = ev._run_episode(seed=1, episode_index=0, capture_frames=False)

    assert result.length == 20  # capped by config, not env
    assert result.timed_out is True
    assert result.success is False
    assert ev.policy.select_action.call_count == 20


def test_rollout_respects_env_cap_below_config():
    cfg = EvalConfig(num_episodes=1, max_episode_steps=500, video_capture_on_nonsuccess=False)
    ev = PolicyEvaluator(cfg)
    _wire(ev, FakeVecEnv(terminate_at=None, max_steps=30))  # env cap < config cap
    result, _ = ev._run_episode(seed=1, episode_index=0, capture_frames=False)

    assert result.length == 30  # min(config, env) wins
    assert result.timed_out is True


def test_evaluate_runs_n_episodes_and_aggregates():
    cfg = EvalConfig(num_episodes=3, max_episode_steps=10, video_capture_on_nonsuccess=False)
    ev = PolicyEvaluator(cfg)
    _wire(ev, FakeVecEnv(terminate_at=4))
    result = ev.evaluate(video_dir=None)

    assert len(result.episodes) == 3
    assert all(e.success for e in result.episodes)
    assert [e.seed for e in result.episodes] == [cfg.seed, cfg.seed + 1, cfg.seed + 2]
    assert result.num_episodes_requested == 3


def test_capture_frames_collected():
    cfg = EvalConfig(num_episodes=1, max_episode_steps=5, video_capture_on_nonsuccess=False)
    ev = PolicyEvaluator(cfg)
    _wire(ev, FakeVecEnv(terminate_at=None, max_steps=5))
    _, frames = ev._run_episode(seed=1, episode_index=0, capture_frames=True)
    # one frame after reset + one per executed step
    assert len(frames) == 6
