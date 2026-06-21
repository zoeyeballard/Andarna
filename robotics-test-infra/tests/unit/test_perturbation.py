"""Tier-1 unit tests for the Project 2 ↔ Project 3 perturbation bridge.

Deps-light: exercises the degradation import, the cliff detector, and the evaluator's
observation-perturbation hook on a fake observation — no torch, no MuJoCo, no policy
load. (The full perturbed rollout is a Tier-4 / scheduled job; see
scripts/run_perturbation_sweep.py.)
"""

from __future__ import annotations

import numpy as np
import pytest

from src.config import EvalConfig
from src.evaluator import PolicyEvaluator
from src.perturbation_tests import (
    PERTURBATION_TESTS_IMPLEMENTED,
    _import_degradations,
    find_cliff,
)


def test_bridge_is_implemented():
    assert PERTURBATION_TESTS_IMPLEMENTED is True


def test_project2_degradations_importable():
    deg = _import_degradations()
    # Project 2's public API the bridge relies on.
    assert set(deg.SWEEPS) == {"noise", "delay", "gap", "resolution"}
    assert callable(deg.make_degrader)


def test_find_cliff():
    # success collapses below 50% of the clean (0.8) rate at level 4.
    pts = [
        {"level": 1, "success_rate": 0.8},
        {"level": 2, "success_rate": 0.7},
        {"level": 4, "success_rate": 0.2},
        {"level": 8, "success_rate": 0.0},
    ]
    assert find_cliff(pts, frac=0.5) == 4
    # never crosses -> None
    flat = [{"level": 0, "success_rate": 0.8}, {"level": 1, "success_rate": 0.75}]
    assert find_cliff(flat, frac=0.5) is None


def _fake_obs():
    # mirrors the lerobot aloha vec-env shape: (1, H, W, 3) uint8 under pixels/<cam>.
    rng = np.random.default_rng(0)
    img = rng.integers(0, 256, size=(1, 32, 48, 3), dtype=np.uint8)
    return {"agent_pos": np.zeros((1, 14)), "pixels": {"top": img}}


def test_perturb_observation_applies_degrader():
    deg = _import_degradations()
    ev = PolicyEvaluator(EvalConfig())  # no load() — hook works without the sim stack
    # noise degrader must change the frame; shape/dtype preserved.
    degrader = deg.make_degrader("noise", 0.2)
    degrader.reset(seed=0)
    ev._image_degrader = degrader
    obs = _fake_obs()
    original = obs["pixels"]["top"].copy()
    out = ev._perturb_observation(obs)
    assert out["pixels"]["top"].shape == original.shape
    assert out["pixels"]["top"].dtype == np.uint8
    assert not np.array_equal(out["pixels"]["top"], original), "noise should perturb the frame"


def test_perturb_observation_noop_without_degrader():
    ev = PolicyEvaluator(EvalConfig())
    obs = _fake_obs()
    ref = obs["pixels"]["top"].copy()
    out = ev._perturb_observation(obs)
    assert np.array_equal(out["pixels"]["top"], ref)


def test_resolution_degrader_preserves_shape():
    # resolution reduction down/up-samples via Pillow, which is a sim-stack dep, not a
    # Tier-1 one; skip on lean runners (the noise hook test above is numpy-only).
    pytest.importorskip("PIL")
    deg = _import_degradations()
    ev = PolicyEvaluator(EvalConfig())
    degrader = deg.make_degrader("resolution", 4)
    degrader.reset(seed=0)
    ev._image_degrader = degrader
    obs = _fake_obs()
    out = ev._perturb_observation(obs)
    # down- then up-sampled back to the same shape (a coarser image, same dims).
    assert out["pixels"]["top"].shape == (1, 32, 48, 3)
