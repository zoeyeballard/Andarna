"""Unit tests for src.video_capture pure helpers (no ffmpeg/PIL needed)."""

import numpy as np
import pytest

from src.video_capture import downscale_frame


def test_downscale_reduces_longest_edge():
    out = downscale_frame(np.zeros((480, 640, 3), dtype=np.uint8), max_edge=256)
    assert max(out.shape[:2]) == 256
    assert out.shape == (192, 256, 3)  # aspect preserved


def test_downscale_noop_when_small():
    frame = np.zeros((100, 120, 3), dtype=np.uint8)
    out = downscale_frame(frame, max_edge=256)
    assert out.shape == frame.shape


def test_downscale_rejects_non_rgb():
    with pytest.raises(ValueError):
        downscale_frame(np.zeros((10, 10), dtype=np.uint8))
