"""Local CPU tests for the degradation module — no GPU, no OpenVLA needed.

Run:  python -m robustness.test_degradations   (from openvla-robustness/)
These verify the *mechanics* of each corruption so we trust them before they ever
touch a GPU rollout. Plain asserts, no pytest dependency.
"""

import numpy as np

from robustness.degradations import (
    SWEEPS,
    DegradationConfig,
    ObservationDegrader,
    add_gaussian_noise,
    make_degrader,
    profile_degrader,
    reduce_resolution,
)


def _frame(seed=0):
    rng = np.random.default_rng(seed)
    return rng.integers(0, 256, size=(64, 64, 3), dtype=np.uint8)


def test_clean_is_identity():
    f = _frame()
    deg = ObservationDegrader(DegradationConfig())
    deg.reset(0)
    out = deg.process(f)
    assert np.array_equal(out, f), "clean config must be a no-op"


def test_noise_scales_with_sigma():
    f = _frame()
    rng = np.random.default_rng(1)
    d_small = add_gaussian_noise(f, 0.02, np.random.default_rng(1))
    d_big = add_gaussian_noise(f, 0.2, np.random.default_rng(1))
    err_small = np.abs(d_small.astype(int) - f.astype(int)).mean()
    err_big = np.abs(d_big.astype(int) - f.astype(int)).mean()
    assert err_small < err_big, "bigger sigma must perturb more"
    assert add_gaussian_noise(f, 0.0, rng) is f, "sigma=0 must be identity"
    # output stays a valid uint8 image
    assert d_big.dtype == np.uint8 and d_big.shape == f.shape


def test_resolution_roundtrip_loses_detail():
    f = _frame()
    assert np.array_equal(reduce_resolution(f, 1), f), "factor 1 == identity"
    out = reduce_resolution(f, 4)
    assert out.shape == f.shape and out.dtype == np.uint8
    # downscale-upscale must blur => total variation (neighbor differences) drops
    tv = lambda x: np.abs(np.diff(x.astype(int), axis=0)).mean()
    assert tv(out) < tv(f), "downscaled frame should be smoother"


def test_delay_fifo_shifts_stream():
    # Feed a sequence of constant-valued frames; with delay=2 the output at step t
    # must equal the input from step t-2 (and repeat the first frame early).
    delay = 2
    deg = make_degrader("delay", delay)
    deg.reset(0)
    frames = [np.full((8, 8, 3), v, dtype=np.uint8) for v in (10, 20, 30, 40, 50)]
    outs = [int(deg.process(f)[0, 0, 0]) for f in frames]
    # step0->10 (only frame), step1->10 (t-2 missing, repeat oldest), step2->10,
    # step3->20, step4->30
    assert outs == [10, 10, 10, 20, 30], f"unexpected delayed stream: {outs}"


def test_gap_repeats_last_and_respects_rate():
    deg = make_degrader("gap", 0.5)
    deg.reset(123)
    # Distinct frames; any output must equal some *previously delivered* frame.
    prev = []
    n_repeat = 0
    for v in range(40):
        f = np.full((4, 4, 3), v, dtype=np.uint8)
        out = int(deg.process(f)[0, 0, 0])
        if out != v:
            assert out in prev, "a dropped frame must repeat an earlier delivered one"
            n_repeat += 1
        prev.append(out)
    # ~50% drop expected; allow a wide band, just confirm it actually drops some.
    assert 5 < n_repeat < 35, f"gap_rate=0.5 produced {n_repeat}/40 drops"
    assert deg.stats["frames_dropped"] == n_repeat


def test_reset_makes_episodes_reproducible():
    deg = make_degrader("noise", 0.1)
    frames = [_frame(i) for i in range(5)]
    deg.reset(7)
    run1 = [deg.process(f).copy() for f in frames]
    deg.reset(7)
    run2 = [deg.process(f).copy() for f in frames]
    assert all(np.array_equal(a, b) for a, b in zip(run1, run2)), \
        "same seed must reproduce the episode exactly"


def test_profiles_and_sweeps_wellformed():
    for name in ("lab", "field", "challenging_field", "high_stress_field"):
        deg = profile_degrader(name)
        deg.reset(0)
        out = deg.process(_frame())
        assert out.shape == (64, 64, 3)
    assert SWEEPS["noise"][0] == 0.0 and SWEEPS["delay"][0] == 0
    # clean baseline is the first point of every numeric sweep where it makes sense
    assert make_degrader("noise", 0.0).cfg.is_clean()


def main():
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
        print(f"  [ok] {t.__name__}")
    print(f"\n[PASS] {len(tests)} degradation tests")


if __name__ == "__main__":
    main()
