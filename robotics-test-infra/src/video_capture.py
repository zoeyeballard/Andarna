"""Record evaluation episodes as MP4 video.

Used to capture *non-success* episodes for debugging — the artifact a reviewer pulls
from a CI run to see *how* the policy failed (overshoot, drop, timeout). Frames come
from MuJoCo offscreen rendering (the evaluator collects them); this module overlays a
step counter + inference latency, downscales for size, and encodes to MP4.

Heavy/optional deps (``imageio``, ``PIL``) are imported lazily so importing this
module — and unit-testing the pure helpers below — needs neither installed.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np


def downscale_frame(frame: np.ndarray, max_edge: int = 256) -> np.ndarray:
    """Downscale an RGB frame so its longest edge is ``max_edge`` px, preserving aspect.

    Pure-numpy nearest-neighbour resample (no SciPy/PIL needed) — keeps file size and
    dependencies small. Returns the frame unchanged if it's already small enough.
    """
    if frame.ndim != 3 or frame.shape[2] != 3:
        raise ValueError(f"expected HxWx3 RGB frame, got shape {frame.shape}")
    h, w = frame.shape[:2]
    longest = max(h, w)
    if longest <= max_edge:
        return frame
    scale = max_edge / longest
    new_h, new_w = max(1, int(round(h * scale))), max(1, int(round(w * scale)))
    ys = (np.arange(new_h) * (h / new_h)).astype(int).clip(0, h - 1)
    xs = (np.arange(new_w) * (w / new_w)).astype(int).clip(0, w - 1)
    return frame[ys][:, xs]


def _overlay_text(frame: np.ndarray, lines: list[str]) -> np.ndarray:
    """Draw text lines onto a frame using PIL if available; no-op fallback otherwise."""
    try:
        from PIL import Image, ImageDraw  # lazy: only needed when actually writing video
    except ImportError:
        return frame
    img = Image.fromarray(frame)
    draw = ImageDraw.Draw(img)
    y = 2
    for line in lines:
        # cheap drop-shadow for legibility over light/dark backgrounds
        draw.text((3, y + 1), line, fill=(0, 0, 0))
        draw.text((2, y), line, fill=(255, 255, 0))
        y += 11
    return np.asarray(img)


def write_episode_video(
    frames: list[np.ndarray],
    out_path: str | Path,
    *,
    inference_times_ms: list[float] | None = None,
    fps: int = 15,
    max_edge: int = 256,
    caption: str = "",
    overlay: bool = True,
) -> Path:
    """Encode ``frames`` to an MP4 at ``out_path`` with optional per-frame overlays.

    Returns the written path. Raises ``RuntimeError`` if given no frames.
    """
    if not frames:
        raise RuntimeError("write_episode_video called with no frames")

    import imageio.v2 as imageio  # lazy: pulls imageio-ffmpeg

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    lat = inference_times_ms or []

    processed: list = []
    for i, frame in enumerate(frames):
        f = downscale_frame(np.ascontiguousarray(frame), max_edge=max_edge)
        if overlay:
            lines = [f"step {i:>3d}"]
            if i < len(lat):
                lines.append(f"infer {lat[i]:6.1f}ms")
            if caption:
                lines.append(caption)
            f = _overlay_text(f, lines)
        processed.append(f)

    # macro_block_size=1 avoids ffmpeg silently resizing odd dimensions
    imageio.mimwrite(out_path, processed, fps=fps, macro_block_size=1)
    return out_path
