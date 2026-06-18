"""Integration test: a real (tiny) evaluation through the full sim stack.

Marked ``sim`` + ``slow`` — skipped automatically when the simulation stack or the
Project-1 checkpoint isn't available (e.g. Tier-1 runners). Tier 2/3 run this inside
the Docker image. It is intentionally minimal (1 episode, short cap) — it asserts the
*plumbing* end-to-end (load → rollout → metrics → report), not policy quality.
"""

import os

import pytest

from src.config import EvalConfig
from src.metrics import compute_metrics, passes_threshold
from src.reporter import render_validation_comment

pytestmark = [pytest.mark.sim, pytest.mark.slow]

# skip the whole module if the sim stack isn't importable
pytest.importorskip("lerobot", reason="simulation stack not installed")
pytest.importorskip("gym_aloha", reason="gym-aloha not installed")


@pytest.fixture(scope="module")
def evaluated():
    from src.evaluator import CheckpointError, PolicyEvaluator, resolve_checkpoint

    cfg = EvalConfig(
        num_episodes=1,
        max_episode_steps=60,
        seed=100_000,
        render_backend=os.environ.get("MUJOCO_GL", "egl"),
        video_capture_on_nonsuccess=False,
    )
    try:
        resolve_checkpoint(cfg.checkpoint_path)
    except CheckpointError:
        pytest.skip("Project-1 checkpoint not available")

    ev = PolicyEvaluator(cfg)
    ev.load()
    result = ev.evaluate(video_dir=None)
    ev.close()
    return cfg, result


def test_pipeline_produces_valid_results(evaluated):
    cfg, result = evaluated
    assert len(result.episodes) == 1
    ep = result.episodes[0]
    assert 0 < ep.length <= cfg.max_episode_steps
    assert len(ep.inference_times_ms) == ep.length
    assert ep.final_object_position is None or len(ep.final_object_position) == 7


def test_pipeline_metrics_and_report(evaluated):
    cfg, result = evaluated
    metrics = compute_metrics(result)
    assert metrics.num_episodes == 1
    assert metrics.latency.n_samples > 0
    assert metrics.latency.max_ms >= metrics.latency.p95_ms  # bimodal: max is the chunk spike

    tr = passes_threshold(metrics, cfg)
    comment = render_validation_comment(metrics, tr, cfg)
    assert "Simulation Validation Results" in comment
    assert "Inference Latency" in comment
