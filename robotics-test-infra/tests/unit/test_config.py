"""Unit tests for src.config — defaults, validation ranges, env-var overrides."""

import pytest

from src.config import DEFAULT_CHECKPOINT, ConfigError, EvalConfig


def test_defaults_are_valid():
    cfg = EvalConfig()
    assert cfg.num_episodes == 10
    assert cfg.task == "AlohaTransferCube-v0"
    assert cfg.checkpoint_path == DEFAULT_CHECKPOINT
    assert 0.0 <= cfg.success_rate_threshold <= 1.0
    # __post_init__ ran validate() without raising
    cfg.validate()


@pytest.mark.parametrize(
    "kwargs",
    [
        {"num_episodes": 0},
        {"max_episode_steps": 0},
        {"success_rate_threshold": 1.5},
        {"success_rate_threshold": -0.1},
        {"max_avg_episode_length": 0},
        {"inference_latency_ceiling_ms": 0},
        {"fps": 0},
        {"render_backend": "vulkan"},
    ],
)
def test_invalid_configs_rejected(kwargs):
    with pytest.raises(ConfigError):
        EvalConfig(**kwargs)


def test_valid_threshold_boundaries_accepted():
    EvalConfig(success_rate_threshold=0.0)
    EvalConfig(success_rate_threshold=1.0)


def test_from_env_overrides(monkeypatch):
    monkeypatch.setenv("RTI_NUM_EPISODES", "25")
    monkeypatch.setenv("RTI_SUCCESS_RATE_THRESHOLD", "0.4")
    monkeypatch.setenv("RTI_RENDER_BACKEND", "osmesa")
    monkeypatch.setenv("RTI_VIDEO_CAPTURE_ON_NONSUCCESS", "false")
    cfg = EvalConfig.from_env()
    assert cfg.num_episodes == 25
    assert cfg.success_rate_threshold == 0.4
    assert cfg.render_backend == "osmesa"
    assert cfg.video_capture_on_nonsuccess is False


def test_explicit_overrides_beat_env(monkeypatch):
    monkeypatch.setenv("RTI_NUM_EPISODES", "25")
    cfg = EvalConfig.from_env(num_episodes=3)
    assert cfg.num_episodes == 3  # kwarg wins over env var


def test_from_env_validates(monkeypatch):
    monkeypatch.setenv("RTI_SUCCESS_RATE_THRESHOLD", "2.0")
    with pytest.raises(ConfigError):
        EvalConfig.from_env()


def test_checkpoint_exists_hub_id_passthrough():
    # A Hub repo id can't be checked offline, but must not be rejected as "missing".
    assert EvalConfig(checkpoint_path="zoey/act-aloha-cube").checkpoint_exists() is True


def test_checkpoint_exists_missing_local(tmp_path):
    assert EvalConfig(checkpoint_path=str(tmp_path / "nope")).checkpoint_exists() is False


def test_checkpoint_exists_local_with_model(tmp_path):
    (tmp_path / "config.json").write_text("{}")
    assert EvalConfig(checkpoint_path=str(tmp_path)).checkpoint_exists() is True
