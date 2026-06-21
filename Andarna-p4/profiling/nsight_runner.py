#!/usr/bin/env python
"""
nsight_runner.py — Phase 4: clean OpenVLA inference harness with NVTX ranges, for Nsight Systems.

No PyTorch Profiler here — just NVTX range annotations around the four model stages so they
appear as named bands on the nsys timeline:

    VisionEncoder   -> vision_backbone   (SigLIP + DINOv2)
    MLPProjector    -> projector
    LLM_prefill     -> language_model, first call  (full prompt + visual tokens)
    LLM_decode      -> language_model, calls 2..N  (one action token each)

Each profiled iteration is wrapped in an `inference_step` range. Warmup runs *before*
`cudaProfilerStart()`, and capture is bounded by `cudaProfilerStart/Stop`, so when you launch
nsys with `--capture-range=cudaProfilerApi` the report contains only steady-state steps.

Run it under nsys (see the command printed at the end of this docstring / in the README):

    nsys profile \\
      --trace=cuda,nvtx,osrt \\
      --cuda-memory-usage=true \\
      --capture-range=cudaProfilerApi --capture-range-end=stop \\
      --force-overwrite=true \\
      --output=traces/openvla_nsys_timeline \\
      python profiling/nsight_runner.py --warmup 20 --iters 10

Then download traces/openvla_nsys_timeline.nsys-rep and open it in the Nsight Systems GUI.
The script also runs standalone (NVTX calls are no-ops without a profiler) for a quick check.
"""
from __future__ import annotations

import argparse

import numpy as np
import torch
from PIL import Image

DEFAULT_MODEL = "openvla/openvla-7b-finetuned-libero-object"
DEFAULT_UNNORM_KEY = "libero_object"
PROMPT_TEMPLATE = "In: What action should the robot take to {instruction}?\nOut:"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="NVTX-annotated OpenVLA inference for nsys.")
    p.add_argument("--model-id", default=DEFAULT_MODEL)
    p.add_argument("--unnorm-key", default=DEFAULT_UNNORM_KEY)
    p.add_argument("--warmup", type=int, default=20, help="Untimed warmup steps (before capture).")
    p.add_argument("--iters", type=int, default=10, help="Steps captured on the timeline.")
    p.add_argument("--instruction", default="pick up the object and place it in the basket")
    p.add_argument("--attn-impl", default="sdpa", choices=["flash_attention_2", "sdpa", "eager"])
    p.add_argument("--device", default="cuda:0")
    return p.parse_args()


def load_model(model_id: str, attn_impl: str, device: str):
    from transformers import AutoModelForVision2Seq, AutoProcessor

    processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)

    def _load(impl):
        return AutoModelForVision2Seq.from_pretrained(
            model_id, attn_implementation=impl, torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True, trust_remote_code=True,
        ).to(device)

    try:
        model, used = _load(attn_impl), attn_impl
    except (ImportError, RuntimeError, ValueError) as e:
        if attn_impl == "flash_attention_2":
            print(f"[warn] flash_attention_2 unavailable ({e}); falling back to sdpa.")
            model, used = _load("sdpa"), "sdpa"
        else:
            raise
    model.eval()
    return processor, model, used


class NVTXStageAnnotator:
    """Forward hooks that push/pop NVTX ranges around OpenVLA's four stages."""

    def __init__(self, model):
        for attr in ("vision_backbone", "projector", "language_model"):
            if not hasattr(model, attr):
                raise AttributeError(f"Model is missing submodule '{attr}'.")
        self.vision = model.vision_backbone
        self.projector = model.projector
        self.lm = model.language_model
        self._handles: list = []
        self._lm_idx = 0

    def _lm_label(self) -> str:
        self._lm_idx += 1
        return "LLM_prefill" if self._lm_idx == 1 else "LLM_decode"

    def reset_step(self):
        self._lm_idx = 0

    def attach(self):
        def pre(label_fn):
            def hook(module, args, kwargs):
                torch.cuda.nvtx.range_push(label_fn())
            return hook

        def post(module, args, kwargs, output):
            torch.cuda.nvtx.range_pop()
            return output

        specs = [
            (self.vision, lambda: "VisionEncoder"),
            (self.projector, lambda: "MLPProjector"),
            (self.lm, self._lm_label),
        ]
        for module, label_fn in specs:
            self._handles.append(module.register_forward_pre_hook(pre(label_fn), with_kwargs=True))
            self._handles.append(module.register_forward_hook(post, with_kwargs=True))
        return self

    def detach(self):
        for h in self._handles:
            h.remove()
        self._handles.clear()


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is not available — this harness requires a GPU.")
    device = args.device
    torch.cuda.set_device(device)

    print(f"[load] {args.model_id} (BF16) on {torch.cuda.get_device_name(device)} ...")
    processor, model, used_impl = load_model(args.model_id, args.attn_impl, device)
    print(f"[load] done (attn={used_impl})")

    rng = np.random.default_rng(0)
    image = Image.fromarray(rng.integers(0, 256, (224, 224, 3), dtype=np.uint8), "RGB")
    prompt = PROMPT_TEMPLATE.format(instruction=args.instruction)
    unnorm_key = args.unnorm_key or None

    def run_step():
        inputs = processor(prompt, image).to(device, dtype=torch.bfloat16)
        return model.predict_action(**inputs, unnorm_key=unnorm_key, do_sample=False)

    # --- Warmup OUTSIDE the captured region -------------------------------------------
    print(f"[warmup] {args.warmup} steps (not captured) ...")
    with torch.inference_mode():
        for _ in range(args.warmup):
            run_step()
    torch.cuda.synchronize(device)

    annotator = NVTXStageAnnotator(model).attach()

    # --- Captured region: nsys --capture-range=cudaProfilerApi keys off these ----------
    print(f"[capture] {args.iters} steps with NVTX ranges ...")
    torch.cuda.profiler.start()
    try:
        with torch.inference_mode():
            for i in range(args.iters):
                annotator.reset_step()
                torch.cuda.nvtx.range_push(f"inference_step_{i}")
                run_step()
                torch.cuda.nvtx.range_pop()
        torch.cuda.synchronize(device)
    finally:
        torch.cuda.profiler.stop()
        annotator.detach()

    print("[done] capture complete. If run under nsys, open the .nsys-rep in Nsight Systems.")


if __name__ == "__main__":
    main()
