"""
component_timer.py — Per-stage CUDA-event timing for OpenVLA's four inference stages.

OpenVLA's `predict_action` calls `generate()` internally, which fuses the whole pipeline:
the image is encoded once, then the LLM runs a prefill over the full prompt followed by an
autoregressive decode loop that emits the action tokens. To attribute time to each stage we
attach forward hooks to the relevant submodules and bracket each call with a CUDA event pair:

    1. Vision Encoder  -> `vision_backbone`  (SigLIP + DINOv2; called once per inference)
    2. MLP Projector   -> `projector`        (called once per inference)
    3. LLM Prefill     -> `language_model`, first call  (full prompt + visual tokens)
    4. LLM Decode      -> `language_model`, calls 2..N  (one new action token each; summed)

Events are recorded *on the CUDA stream* with no mid-stream synchronize, so the timed run is
not serialized stage-by-stage. We synchronize exactly once per inference and then read the
elapsed times — accurate GPU-side per-stage durations with minimal perturbation.

This is the core Phase-2 measurement: knowing decode is (e.g.) 60% of the budget while the
vision encoder is 25% tells you exactly where optimization effort pays off.
"""
from __future__ import annotations

import torch

STAGES = ("vision", "projector", "prefill", "decode")


def locate_submodules(model):
    """Return the (vision_backbone, projector, language_model) modules for an OpenVLA model."""
    missing = [a for a in ("vision_backbone", "projector", "language_model") if not hasattr(model, a)]
    if missing:
        raise AttributeError(
            f"Model {type(model).__name__} is missing expected submodules {missing}; "
            "the hook targets need updating for this architecture."
        )
    return model.vision_backbone, model.projector, model.language_model


class ComponentTimer:
    """Attach forward hooks that time OpenVLA's four stages with torch.cuda.Event.

    Usage:
        timer = ComponentTimer(model)
        with timer.attached():
            for _ in range(iters):
                timer.start_iter()
                model.predict_action(...)
                timer.end_iter()          # one synchronize; records this iter's per-stage ms
        summary = timer.summarize()       # mean ms + % of total per stage, across all iters
    """

    def __init__(self, model):
        self.vision, self.projector, self.lm = locate_submodules(model)
        self._handles: list = []
        # Per-iteration scratch:
        self._pending: dict[int, list] = {}   # module id -> stack of (label, start_event)
        self._recorded: list[tuple[str, object, object]] = []  # (label, start, end)
        self._lm_call_idx = 0
        # Results across iterations: stage -> list of per-iter total ms
        self.per_iter: dict[str, list[float]] = {s: [] for s in STAGES}
        self.decode_steps: list[int] = []

    # -- hook plumbing ------------------------------------------------------------------
    def _make_pre_hook(self, label_fn):
        def pre_hook(module, args, kwargs):
            start = torch.cuda.Event(enable_timing=True)
            start.record()
            self._pending.setdefault(id(module), []).append((label_fn(), start))
        return pre_hook

    def _make_post_hook(self):
        def post_hook(module, args, kwargs, output):
            end = torch.cuda.Event(enable_timing=True)
            end.record()
            label, start = self._pending[id(module)].pop()
            self._recorded.append((label, start, end))
            return output
        return post_hook

    def _lm_label(self) -> str:
        # First language_model call of the iteration is the prefill; the rest are decode steps.
        self._lm_call_idx += 1
        return "prefill" if self._lm_call_idx == 1 else "decode"

    def attach(self):
        specs = [
            (self.vision,    lambda: "vision"),
            (self.projector, lambda: "projector"),
            (self.lm,        self._lm_label),
        ]
        for module, label_fn in specs:
            self._handles.append(
                module.register_forward_pre_hook(self._make_pre_hook(label_fn), with_kwargs=True)
            )
            self._handles.append(
                module.register_forward_hook(self._make_post_hook(), with_kwargs=True)
            )
        return self

    def detach(self):
        for h in self._handles:
            h.remove()
        self._handles.clear()

    def attached(self):
        timer = self

        class _Ctx:
            def __enter__(self_):
                timer.attach()
                return timer

            def __exit__(self_, *exc):
                timer.detach()
                return False
        return _Ctx()

    # -- per-iteration lifecycle --------------------------------------------------------
    def start_iter(self):
        self._recorded = []
        self._lm_call_idx = 0

    def end_iter(self):
        """Synchronize once, then attribute elapsed GPU time to each stage for this iteration."""
        torch.cuda.synchronize()
        totals = {s: 0.0 for s in STAGES}
        n_decode = 0
        for label, start, end in self._recorded:
            ms = start.elapsed_time(end)
            totals[label] += ms
            if label == "decode":
                n_decode += 1
        for s in STAGES:
            self.per_iter[s].append(totals[s])
        self.decode_steps.append(n_decode)

    # -- aggregation --------------------------------------------------------------------
    def summarize(self) -> dict:
        import numpy as np

        n = len(self.per_iter["vision"])
        per_iter_total = [
            sum(self.per_iter[s][i] for s in STAGES) for i in range(n)
        ]
        grand_mean_total = float(np.mean(per_iter_total)) if n else 0.0

        stages = {}
        for s in STAGES:
            arr = np.asarray(self.per_iter[s], dtype=np.float64)
            stages[s] = {
                "mean_ms": float(arr.mean()) if n else 0.0,
                "p50_ms": float(np.percentile(arr, 50)) if n else 0.0,
                "p95_ms": float(np.percentile(arr, 95)) if n else 0.0,
                "pct_of_total": float(arr.mean() / grand_mean_total * 100.0) if grand_mean_total else 0.0,
            }
        return {
            "iterations": n,
            "mean_decode_steps": float(np.mean(self.decode_steps)) if n else 0.0,
            "summed_stage_mean_ms": grand_mean_total,
            "stages": stages,
        }
