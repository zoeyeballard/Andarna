"""Integration test (5a): seeded determinism of the rollout.

Same seed → identical action trajectory. Marked sim/slow; skipped when the sim stack
or checkpoint is missing. Short (40 steps) to keep it quick.
"""

import os

import pytest

from src.config import EvalConfig

pytestmark = [pytest.mark.sim, pytest.mark.slow]

pytest.importorskip("lerobot", reason="simulation stack not installed")
pytest.importorskip("gym_aloha", reason="gym-aloha not installed")


def test_same_seed_identical_trajectory():
    from scripts.check_determinism import compare_traces, run_twice
    from src.evaluator import CheckpointError, resolve_checkpoint

    cfg = EvalConfig(
        max_episode_steps=40,
        render_backend=os.environ.get("MUJOCO_GL", "egl"),
        video_capture_on_nonsuccess=False,
    )
    try:
        resolve_checkpoint(cfg.checkpoint_path)
    except CheckpointError:
        pytest.skip("Project-1 checkpoint not available")

    a, b = run_twice(cfg, seed=100_000, max_steps=40)
    ok, detail = compare_traces(a, b)
    assert ok, detail
    assert len(a) == 40
