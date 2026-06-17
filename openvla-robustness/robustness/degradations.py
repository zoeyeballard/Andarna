"""Observation-pipeline degradations for the OpenVLA robustness study.

We hold the **model fixed** and corrupt only **what it sees**. Every degradation
here sits between the simulated camera (LIBERO returns an ``H x W x 3`` ``uint8``
RGB frame) and OpenVLA's own preprocessing (center-crop + resize to 224). That
placement is deliberate: it mirrors a real robot, where corruption happens in the
sensor and on the wire, *upstream* of the policy.

Four independent failure axes, plus stacked "deployment profiles":

  * **noise**       Gaussian sensor noise. ``sigma`` is in normalized [0,1] units
                    (so 0.05 == 5% of full scale), matching read-noise / low-light.
  * **resolution**  Downscale-then-upscale: a cheaper sensor or a compressed link
                    throws away spatial detail that no upscaler restores.
  * **gap**         Intermittent availability: with probability ``gap_rate`` the
                    frame never arrives and the last delivered frame is repeated.
  * **delay**       A FIFO buffer ``delay`` frames deep: at step *t* the policy sees
                    the frame captured at *t - delay* (processing / bus latency).

Two of these are **stateful across a rollout** — ``gap`` (needs the last delivered
frame) and ``delay`` (needs a FIFO history). So the public surface is a small
class, :class:`ObservationDegrader`, that you ``reset()`` per episode and call
``process(frame)`` on at every control step. Pure-function helpers are exposed too
for unit testing.

Ordering inside ``process`` follows the physical pipeline:
    capture corruption (resolution -> noise)  ->  transport (gap -> delay)
The gap repeats the last *delivered* (already-noisy) frame, and the delay buffer
delays that post-gap stream, because that is the order a real system sees them.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field

import numpy as np

try:  # Pillow is ubiquitous (torchvision/transformers depend on it) but guard anyway.
    from PIL import Image
except ImportError:  # pragma: no cover
    Image = None


# --- Canonical sweep grids (single source of truth; see Claude.md Phase 4a) -------
SWEEPS: dict[str, list[float]] = {
    "noise": [0.0, 0.01, 0.02, 0.05, 0.1, 0.15, 0.2, 0.3],
    "delay": [0, 1, 2, 3, 5, 8, 10],
    "gap": [0.0, 0.05, 0.1, 0.2, 0.3, 0.5],
    "resolution": [1, 2, 4, 8],
}

# Plain-English unit + label per axis, reused by the runners and analysis.
AXIS_META: dict[str, dict[str, str]] = {
    "noise": {"unit": "sigma (normalized)", "label": "Gaussian sensor noise"},
    "delay": {"unit": "timesteps", "label": "Observation latency"},
    "gap": {"unit": "drop probability", "label": "Frame drops"},
    "resolution": {"unit": "downscale factor", "label": "Resolution reduction"},
}


@dataclass
class DegradationConfig:
    """One point in the (noise, resolution, gap, delay) corruption space.

    Defaults are the identity (a clean pipeline), so ``DegradationConfig()`` is a
    no-op and equals the baseline.
    """

    noise_sigma: float = 0.0          # std-dev of additive Gaussian noise, [0,1] units
    downscale: int = 1                # 1 == full resolution; 2/4/8 == coarser sensor
    gap_rate: float = 0.0             # P(frame unavailable) per step, [0,1]
    delay: int = 0                    # FIFO depth in timesteps

    name: str = "clean"               # human label, used in result metadata

    def is_clean(self) -> bool:
        return (self.noise_sigma == 0.0 and self.downscale == 1
                and self.gap_rate == 0.0 and self.delay == 0)

    def as_metadata(self) -> dict:
        return {
            "profile": self.name,
            "noise_sigma": self.noise_sigma,
            "downscale": self.downscale,
            "gap_rate": self.gap_rate,
            "delay": self.delay,
        }


# --- Named stacked profiles ("what a real deployment looks like") -----------------
PROFILES: dict[str, DegradationConfig] = {
    "clean": DegradationConfig(name="clean"),
    "lab": DegradationConfig(noise_sigma=0.02, delay=1, gap_rate=0.05, name="lab"),
    "field": DegradationConfig(noise_sigma=0.05, delay=2, gap_rate=0.1, name="field"),
    "challenging_field": DegradationConfig(
        noise_sigma=0.1, delay=3, gap_rate=0.2, name="challenging_field"),
    "high_stress_field": DegradationConfig(
        noise_sigma=0.15, delay=5, gap_rate=0.3, name="high_stress_field"),
}


# --- Pure-function helpers (stateless; directly unit-testable) --------------------
def add_gaussian_noise(frame: np.ndarray, sigma: float,
                       rng: np.random.Generator) -> np.ndarray:
    """Add Gaussian noise with std ``sigma`` (in [0,1] units) to a uint8 RGB frame.

    Operates in normalized float space — ``sigma=0.05`` perturbs each channel by
    ~5% of full scale — then clips and returns uint8 so the rest of the pipeline
    stays in the camera's native dtype.
    """
    if sigma <= 0.0:
        return frame
    f = frame.astype(np.float32) / 255.0
    f = f + rng.normal(0.0, sigma, size=f.shape).astype(np.float32)
    return (np.clip(f, 0.0, 1.0) * 255.0).round().astype(np.uint8)


def reduce_resolution(frame: np.ndarray, factor: int) -> np.ndarray:
    """Downscale by ``factor`` then upscale back to the original size (bilinear).

    The round trip destroys high-frequency detail and cannot recover it — exactly
    what a cheaper sensor or a bandwidth-limited link does. ``factor=1`` is identity.
    """
    if factor <= 1:
        return frame
    if Image is None:  # pragma: no cover
        raise RuntimeError("Pillow required for resolution reduction")
    h, w = frame.shape[:2]
    small = (max(1, w // factor), max(1, h // factor))
    img = Image.fromarray(frame)
    img = img.resize(small, Image.BILINEAR).resize((w, h), Image.BILINEAR)
    return np.asarray(img, dtype=np.uint8)


# --- Stateful per-rollout degrader ------------------------------------------------
@dataclass
class ObservationDegrader:
    """Apply a :class:`DegradationConfig` to a stream of frames across one rollout.

    Usage::

        deg = ObservationDegrader(cfg)
        deg.reset(seed=episode_idx)           # at the start of every episode
        for t in range(T):
            obs_img = deg.process(env_frame)  # at every control step

    ``reset`` is mandatory before each episode: it reseeds the RNG (so episode *k*
    is reproducible) and clears the latency FIFO and last-delivered-frame memory.
    """

    cfg: DegradationConfig
    _rng: np.random.Generator = field(default=None, repr=False)
    _fifo: deque = field(default=None, repr=False)
    _last_delivered: np.ndarray = field(default=None, repr=False)
    _dropped: int = 0
    _steps: int = 0

    def reset(self, seed: int = 0) -> None:
        self._rng = np.random.default_rng(seed)
        # FIFO holds the last `delay` frames; maxlen+1 so we can read the oldest.
        self._fifo = deque(maxlen=max(1, self.cfg.delay + 1))
        self._last_delivered = None
        self._dropped = 0
        self._steps = 0

    def process(self, frame: np.ndarray) -> np.ndarray:
        if self._rng is None:
            raise RuntimeError("call reset() before process()")
        if frame.ndim != 3 or frame.shape[2] != 3:
            raise ValueError(f"expected H x W x 3 RGB frame, got {frame.shape}")
        self._steps += 1
        cfg = self.cfg

        # 1) capture-side corruption: resolution then sensor noise
        out = reduce_resolution(frame, cfg.downscale)
        out = add_gaussian_noise(out, cfg.noise_sigma, self._rng)

        # 2) transport: frame gap -> repeat last delivered frame
        if cfg.gap_rate > 0.0 and self._last_delivered is not None \
                and self._rng.random() < cfg.gap_rate:
            out = self._last_delivered
            self._dropped += 1
        self._last_delivered = out

        # 3) transport: FIFO latency buffer -> deliver frame from t-delay.
        # maxlen is delay+1, so fifo[0] is the frame from t-delay once full, and the
        # oldest available frame (a stand-in) before the buffer has filled.
        if cfg.delay > 0:
            self._fifo.append(out)
            out = self._fifo[0]
        return out

    @property
    def stats(self) -> dict:
        """Realized degradation stats for this rollout (for result metadata)."""
        return {
            "steps": self._steps,
            "frames_dropped": self._dropped,
            "realized_gap_rate": (self._dropped / self._steps) if self._steps else 0.0,
        }


# --- Factories --------------------------------------------------------------------
def make_config(kind: str, level: float) -> DegradationConfig:
    """Build a single-axis :class:`DegradationConfig` for one sweep point."""
    if kind == "noise":
        return DegradationConfig(noise_sigma=float(level), name=f"noise={level}")
    if kind == "delay" or kind == "latency":
        return DegradationConfig(delay=int(level), name=f"delay={int(level)}")
    if kind == "gap":
        return DegradationConfig(gap_rate=float(level), name=f"gap={level}")
    if kind == "resolution":
        return DegradationConfig(downscale=int(level), name=f"downscale={int(level)}")
    raise ValueError(f"unknown degradation kind: {kind!r}")


def make_degrader(kind: str, level: float) -> ObservationDegrader:
    return ObservationDegrader(make_config(kind, level))


def profile_degrader(name: str) -> ObservationDegrader:
    if name not in PROFILES:
        raise ValueError(f"unknown profile {name!r}; have {list(PROFILES)}")
    return ObservationDegrader(PROFILES[name])
