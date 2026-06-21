#!/usr/bin/env python
"""
nsight_runner.py — Phase 4: clean NVTX-annotated OpenVLA inference for Nsight Systems.

No PyTorch Profiler here — just NVTX range markers around the four stages so they show up
as named, colored bands on the nsys timeline:

    inference_step
      ├─ vision      (vision_backbone: SigLIP + DINOv2)
      ├─ projector   (MLP projector)
      ├─ prefill     (language_model, first call — full prompt + visual tokens)
      └─ decode      (language_model, calls 2..N — one action token each)

The script warms up, then opens a cudaProfiler capture window (cudaProfilerStart/Stop) around
a small steady-state burst. Run it under nsys with `--capture-range=cudaProfilerApi` so the
.nsys-rep contains only those steady-state steps — small enough to download and open locally.

----------------------------------------------------------------------------------------------
nsys command (run on the EC2 A10G, from the repo root, with the venv active):

    nsys profile \
      --trace=cuda,nvtx,osrt,cublas,cudnn \
      --capture-range=cudaProfilerApi \
      --capture-range-end=stop \
      --cuda-memory-usage=true \
      --force-overwrite=true \
      --output=traces/openvla_nvtx_timeline \
      python profiling/nsight_runner.py --warmup 20 --capture-steps 10

Then download traces/openvla_nvtx_timeline.nsys-rep and open it in the Nsight Systems GUI.
----------------------------------------------------------------------------------------------
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from profiling.component_timer import locate_submodules  # noqa: E402

DEFAULT_MODEL = "openvla/openvla-7b-finetuned-libero-object"
DEFAULT_UNNORM_KEY = "libero_object"
PROMPT_TEMPLATE = "In: What action should the robot take to {instruction}?\nOut:"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="NVTX-annotated OpenVLA runner for nsys.")
    p.add_argument("--model-id", default=DEFAULT_MODEL)
    p.add_argument("--unnorm-key", default=DEFAULT_UNNORM_KEY)
    p.add_argument("--warmup", type=int, default=20, help="Untimed warmup steps (outside capture).")
    p.add_argument("--capture-steps", type=int, default=10, help="Steps inside the nsys capture window.")
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


class NvtxStageAnnotator:
    """Forward hooks that wrap each OpenVLA stage in an NVTX range for the nsys timeline."""

    def __init__(self, model):
        self.vision, self.projector, self.lm = locate_submodules(model)
        self._handles: list = []
        self._lm_call_idx = 0

    def _push(self, label_fn):
        def pre_hook(module, args, kwargs):
            torch.cuda.nvtx.range_push(label_fn())
        return pre_hook

    def _pop(self):
        def post_hook(module, args, kwargs, output):
            torch.cuda.nvtx.range_pop()
            return output
        return post_hook

    def _lm_label(self) -> str:
        self._lm_call_idx += 1
        return "prefill" if self._lm_call_idx == 1 else "decode"

    def reset_step(self):
        self._lm_call_idx = 0

    def attach(self):
        for module, label_fn in [
            (self.vision, lambda: "vision"),
            (self.projector, lambda: "projector"),
            (self.lm, self._lm_label),
        ]:
            self._handles.append(module.register_forward_pre_hook(self._push(label_fn), with_kwargs=True))
            self._handles.append(module.register_forward_hook(self._pop(), with_kwargs=True))
        return self

    def detach(self):
        for h in self._handles:
            h.remove()
        self._handles.clear()


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is not available — this runner requires a GPU.")
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

    annotator = NvtxStageAnnotator(model).attach()

    # Warm up OUTSIDE the capture window so the trace shows steady-state only.
    print(f"[warmup] {args.warmup} steps ...")
    with torch.inference_mode():
        for _ in range(args.warmup):
            annotator.reset_step()
            run_step()
    torch.cuda.synchronize(device)

    # Steady-state capture: nsys --capture-range=cudaProfilerApi keys off these calls.
    print(f"[capture] {args.capture_steps} steps inside cudaProfiler window ...")
    torch.cuda.profiler.start()
    with torch.inference_mode():
        for i in range(args.capture_steps):
            annotator.reset_step()
            torch.cuda.nvtx.range_push(f"inference_step_{i}")
            run_step()
            torch.cuda.nvtx.range_pop()
    torch.cuda.synchronize(device)
    torch.cuda.profiler.stop()

    annotator.detach()
    print("[done] capture complete. If run under nsys, the .nsys-rep is now written.")


if __name__ == "__main__":
    main()
