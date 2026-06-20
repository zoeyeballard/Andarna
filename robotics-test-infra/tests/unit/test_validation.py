"""Unit tests for src.validation — checkpoint integrity checks (no sim needed)."""

import json

import numpy as np
import pytest

from src.validation import validate_checkpoint


def _write_config(d, ptype="act", with_io=True):
    cfg = {"type": ptype, "chunk_size": 100, "n_action_steps": 100}
    if with_io:
        cfg["input_features"] = {"observation.state": {"type": "STATE", "shape": [14]}}
        cfg["output_features"] = {"action": {"type": "ACTION", "shape": [14]}}
    (d / "config.json").write_text(json.dumps(cfg))


def test_missing_local_checkpoint():
    report = validate_checkpoint("./nope/missing")
    assert not report.ok
    assert any(c.name == "exists" and not c.passed for c in report.checks)


def test_hub_id_passes_with_note():
    report = validate_checkpoint("zoey/act-aloha-cube")
    assert report.ok
    assert any("Hub" in c.detail for c in report.checks)


def test_missing_model_file(tmp_path):
    _write_config(tmp_path)
    report = validate_checkpoint(str(tmp_path))  # no model.safetensors
    assert not report.ok
    assert any(c.name == "model.safetensors present" and not c.passed for c in report.checks)


def test_wrong_architecture_flagged(tmp_path):
    _write_config(tmp_path, ptype="diffusion")
    (tmp_path / "model.safetensors").write_bytes(b"")  # presence only
    report = validate_checkpoint(str(tmp_path))
    assert any(c.name == "architecture is ACT" and not c.passed for c in report.checks)


def test_training_step_from_path(tmp_path):
    ckpt = tmp_path / "checkpoints" / "000400" / "pretrained_model"
    ckpt.mkdir(parents=True)
    _write_config(ckpt)
    (ckpt / "model.safetensors").write_bytes(b"")
    report = validate_checkpoint(str(ckpt))
    assert report.metadata.get("training_step") == 400


def test_weights_finite_detects_nan(tmp_path):
    sf = pytest.importorskip("safetensors.numpy")
    _write_config(tmp_path)
    good = np.ones((4, 4), dtype=np.float32)
    bad = np.array([[np.nan, 1.0]], dtype=np.float32)
    sf.save_file({"good": good, "bad": bad}, str(tmp_path / "model.safetensors"))
    report = validate_checkpoint(str(tmp_path))
    finite = [c for c in report.checks if c.name == "weights finite (no NaN/Inf)"]
    assert finite and not finite[0].passed


def test_weights_finite_passes_clean(tmp_path):
    sf = pytest.importorskip("safetensors.numpy")
    _write_config(tmp_path)
    sf.save_file({"w": np.ones((4, 4), dtype=np.float32)}, str(tmp_path / "model.safetensors"))
    report = validate_checkpoint(str(tmp_path))
    assert report.ok
