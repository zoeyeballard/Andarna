# OpenVLA Inference Profiling Report

Profiling and quantization study of **OpenVLA-7B** (`openvla/openvla-7b-finetuned-libero-object`)
on an **NVIDIA A10G** (24 GB, Ampere `sm_86`). All numbers are measured on the A10G with
`torch 2.2.0+cu121`, `transformers 4.40.1`, SDPA attention (flash-attn not built). Raw data for
every phase is committed under [`results/`](results/).

## Executive summary

- **Inference is LLM-bound and GEMM-bound.** The LLM backbone is ~94% of latency (decode 59% +
  prefill 35%); the vision encoder is 5.5% and the projector is negligible. ~86% of GPU time is
  `aten::mm`, running on Ampere BF16 tensor-core GEMM kernels that are *already* fast.
- **Quantization here buys memory, not speed.** Every quantized config was *slower* than BF16.
  INT8 was the worst (3.8× slower); INT4 (1.6× slower) beat INT8 on both speed and memory —
  reproducing the known OpenVLA "INT8 is worse than INT4" anomaly.
- **FP16 is a free swap** for BF16 (identical latency, memory, and actions). **INT4 is for
  *fitting* the model** (0.31× memory → 4.7 GB), accepting a latency penalty and a measurable
  behavioral risk concentrated in end-effector positioning — never for speed.
- At BF16 the policy runs at **~2.8 Hz** on the A10G — well under a 5–10 Hz robot control budget,
  which is the headline deployment constraint.

| Precision | Latency (mean) | vs BF16 | Peak GPU mem | Action fidelity (cosine vs BF16) | LIBERO success (n=10) |
|---|---:|---:|---:|---:|---:|
| BF16 | 358.9 ms | 1.00× | 15.5 GB | 1.000 (ref) | 80% |
| FP16 | 359.9 ms | 1.00× | 15.5 GB | 0.997 | 70% |
| INT8 | 1365.1 ms | 3.80× | 8.5 GB | 0.953 | _(skipped — dominated)_ |
| INT4 | 572.1 ms | 1.59× | 4.7 GB | 0.899 | 60% |

---

## Phase 0 — Setup
Project scaffold in `~/repositories/personal/Andarna-p4/`, EC2 install script
([`scripts/install_ec2.sh`](scripts/install_ec2.sh)) for the A10G (PyTorch cu121, OpenVLA, LIBERO,
pinned `accelerate==0.30.1` / `bitsandbytes==0.43.1`). Pushes to `zoeyeballard/Andarna` on branch
`project-4-inference-optimization`.

## Phase 1 — BF16 baseline latency
*Script:* [`scripts/run_baseline_latency.py`](scripts/run_baseline_latency.py) · *Data:*
[`results/baseline/baseline_latency_bf16.json`](results/baseline/baseline_latency_bf16.json)

End-to-end `predict_action` latency, 20 warmup + 100 timed steps via `torch.cuda.Event`:

| mean | p50 | p95 | max | std | peak mem |
|---:|---:|---:|---:|---:|---:|
| 358.7 ms | 358.0 ms | 363.7 ms | 368.4 ms | 2.1 ms | 15,476 MB |

**Read:** ~2.79 Hz achievable control frequency. Distribution is very tight (CV 0.6%) — latency
is highly deterministic, good for WCET-style reasoning. The 20-step warmup fully absorbs the
JIT/allocator warm-up tail.

## Phase 2 — Per-stage component breakdown
*Scripts:* [`profiling/component_timer.py`](profiling/component_timer.py),
[`scripts/run_component_breakdown.py`](scripts/run_component_breakdown.py) · *Data:*
[`results/baseline/component_breakdown_bf16.json`](results/baseline/component_breakdown_bf16.json)

CUDA-event forward hooks on the four stages, 100 iters (prefill vs decode split by LLM call order):

| Stage | mean ms | % of total |
|---|---:|---:|
| Vision Encoder (SigLIP+DINOv2) | 19.44 | 5.5% |
| MLP Projector | 0.77 | 0.2% |
| LLM Prefill | 123.89 | 34.9% |
| **LLM Decode** | **210.58** | **59.4%** |
| overhead (tokenize/merge/sample/de-norm) | 5.17 | 1.4% |
| end-to-end | 359.85 | 100% |

**Read:** LLM ≈ 94% of latency → optimize the LLM, not the vision tower. Decode is ~35 ms/token
across 6 tokens — memory-bandwidth-bound (full 7B weight read per token). Prefill is one larger,
more compute-bound pass. This split is what makes the decode stage the prime target for the
bandwidth savings of low-precision weights.

## Phase 3 — PyTorch Profiler (operator/kernel level)
*Script:* [`profiling/torch_profiler.py`](profiling/torch_profiler.py) · *Data:*
[`results/baseline/torch_profiler_top20.txt`](results/baseline/torch_profiler_top20.txt)
(chrome trace gitignored, 808 MB)

Top by GPU time (self CUDA total 6.66 s / 20 steps):

| Op / kernel | % self CUDA |
|---|---:|
| `aten::mm` (all GEMMs) | 85.97% |
| ↳ `ampere_bf16_s16816gemm` tensor-core kernels (64×64 / 128×128 / 256×128 tiles) | — |
| `cudaLaunchKernel` overhead | 7.66% |
| flash attention (`_flash_attention_forward`, via SDPA) | 1.80% |

Top by GPU memory: `aten::linear` (20.25 GB cumulative), `aten::cat` (19.75 GB — KV-cache concat),
`aten::mul` (18.72 GB).

**Read:** It's a GEMM machine already using tensor cores — there's no "fast path left on the
table." Flash attention is active (via SDPA) and attention is cheap (1.8%). The big-tile GEMMs
are prefill (compute-bound); the 19,200× `64×64` skinny GEMMs are decode (where launch overhead
and KV-cache `cat` traffic live).

## Phase 4 — Nsight Systems timeline (NVTX)
*Script:* [`profiling/nsight_runner.py`](profiling/nsight_runner.py) · *Output:*
`traces/openvla_nsys_timeline.nsys-rep` (5.5 MB, gitignored — download for the Nsight GUI)

Clean NVTX-annotated runner (`VisionEncoder` / `MLPProjector` / `LLM_prefill` / `LLM_decode`),
warmup outside a `cudaProfilerStart/Stop` window so the capture is steady-state only. nsys command
is in the [README](README.md#nsight-systems-timeline-phase-4). All four NVTX ranges confirmed
present via `nsys stats`.

**Caveat captured:** the NVTX push/pop summary is *CPU-side* range time; async kernel launches make
prefill look shorter than its true GPU time. The GUI is where you correlate NVTX bands (CPU row)
with CUDA kernels (GPU row) and read inter-kernel gaps in the decode loop.

## Phase 5 — Precision sweep (BF16 / FP16 / INT8 / INT4)
*Script:* [`quantization/precision_runner.py`](quantization/precision_runner.py) · *Data:*
[`results/quantization/precision_sweep.json`](results/quantization/precision_sweep.json)
(subprocess per precision for clean CUDA context → accurate load time & peak memory)

| Precision | load s | mean ms | p95 ms | peak MB | vs BF16 latency | vs BF16 mem |
|---|---:|---:|---:|---:|---:|---:|
| BF16 | 10.7 | 358.9 | 360.6 | 15,476 | 1.00× | 1.00× |
| FP16 | 22.9 | 359.9 | 367.1 | 15,476 | 1.00× | 1.00× |
| INT8 | 36.1 | **1365.1** | 1397.9 | 8,456 | **3.80× slower** | 0.55× |
| INT4 | 12.0 | 572.1 | 586.2 | 4,730 | 1.59× slower | **0.31×** |

**Read (the key anomaly):** INT8 (1365 ms) ≫ INT4 (572 ms) ≫ BF16 (359 ms). Quantization slowed
everything down. INT8's LLM.int8() mixed-precision decomposition (INT8 path + FP16 outlier path +
quant/dequant per op) costs more than the bandwidth it saves; INT4 (NF4) has lighter per-op
overhead and recovers more of the bandwidth win, but still loses to native BF16 tensor-core GEMMs.
**Quantization here buys memory, not speed.**

## Phase 6 — Accuracy / behavioral validation
*Scripts:* [`quantization/accuracy_validator.py`](quantization/accuracy_validator.py),
[`analysis/action_dim_breakdown.py`](analysis/action_dim_breakdown.py),
[`scripts/run_libero_eval.py`](scripts/run_libero_eval.py)

### 6a. Numerical action error vs BF16 (5 fixed observations, greedy)
*Data:* [`results/quantization/accuracy_validation.json`](results/quantization/accuracy_validation.json)
(baseline mean |action| = 0.235)

| Precision | mean MAE | max abs err | mean cosine | min cosine |
|---|---:|---:|---:|---:|
| FP16 | 0.0077 | 0.143 | 0.9969 | 0.9846 |
| INT8 | 0.0675 | 0.398 | 0.9528 | 0.9222 |
| INT4 | 0.0969 | 0.831 | 0.8985 | 0.7428 |

FP16 is behaviorally identical; INT4 drifts most (mean error ≈ 41% of action scale; worst-case
direction cosine 0.74).

### 6b. Per-dimension breakdown — the gripper-flip test
*Data:* [`results/quantization/action_dim_breakdown.json`](results/quantization/action_dim_breakdown.json)

Hypothesis (INT4's max error is a flipped gripper bit) **refuted**: the gripper dimension has
**0/5 sign flips at every precision** and contributes 0% of the error. Error concentrates in the
**translation deltas — `dz` worst** (INT4: dz 0.26, dy 0.20, dx 0.18); rotation dims barely move.
→ Quantization degrades *spatial precision (where to move)*, not *discrete decisions
(grasp/release)*. Caveat: synthetic inputs saturate the gripper logit, so real-frame gripper
sensitivity is untested here.

### 6c. Behavioral LIBERO-Object success rate (2 tasks × 5 trials)
*Data:* [`results/behavioral/libero_success.json`](results/behavioral/libero_success.json)

| Precision | success | rate | ms/step | control Hz |
|---|---:|---:|---:|---:|
| BF16 | 8/10 | 80% | 360 | 2.77 |
| FP16 | 7/10 | 70% | 361 | 2.77 |
| INT4 | 6/10 | 60% | 576 | 1.74 |

**Read:** success degrades monotonically with precision, consistent with INT4's spatial drift —
but **at n=10 the confidence intervals (±~25 pp) overlap**, so this is *suggestive, not
significant*. FP16 scoring 5/5 on task0 (above BF16) confirms we're near the noise floor. A
full-suite run (10 tasks × 20–50 trials) would be needed to establish a real INT4 penalty. INT8
rollouts were skipped (strictly dominated by INT4). Sim stack: robosuite 1.4.0 + mujoco 3.9.0 +
bddl 1.0.1, headless EGL on the A10G, reusing Project 2's validated rollout glue.

## Phase 8 — torch.compile (reduce-overhead)
*Script:* [`optimization/torch_compile_test.py`](optimization/torch_compile_test.py) · *Data:*
[`results/optimization/torch_compile.json`](results/optimization/torch_compile.json)

`torch.compile(model.forward, mode="reduce-overhead")` on the full OpenVLA pipeline:

| | result |
|---|---|
| Uncompiled baseline (eager) | 359.3 ms |
| Compiled | **errored on first call — no speedup obtained** |

**It errors during tracing**, before producing a single compiled graph (0 graphs, 0 graph breaks):

```
TorchRuntimeError: Failed running call_function scaled_dot_product_attention(... attn_mask
size (1,1,281,280) ...): expand: attempting to expand a dimension of length 280!
  in modeling_llama.py ... scaled_dot_product_attention   (called from OpenVLA forward)
```

**Read:** this confirms the VLM-generation caveat. The LLM backbone's attention uses a 4-D causal
mask whose shape depends on the prefill sequence length vs. the growing KV cache (query len 281 vs
key len 280). Inductor's fake-tensor tracing can't reconcile that dynamic mask shape and aborts —
so `reduce-overhead` never even reaches CUDA-graph capture. Out of the box, torch.compile does
**not** accelerate this pipeline. Paths that *could* work (not pursued here): compile only the
**vision encoder** (static 224×224 input → no dynamic shapes), upgrade to a transformers/torch
version with `StaticCache` + compile support for generation, or `suppress_errors=True` to fall
back to eager (which yields no speedup). Given inference is GEMM-bound on already-fast tensor-core
kernels (Phase 3), compile upside would be limited regardless.

## Phase 9 — Batch-size scaling
*Script:* [`optimization/batch_scaling_test.py`](optimization/batch_scaling_test.py) · *Data:*
[`results/optimization/batch_scaling.json`](results/optimization/batch_scaling.json)

**Finding A — end-to-end action generation is batch-1-locked.** OpenVLA's modeling code asserts
`input_ids.shape[0] == 1` in the cached-decode path; a batched `predict_action` errors out. So a
single deployed policy *cannot* batch its inference out of the box.

**Finding B — the prefill forward (vision + projector + LLM prefill) does batch.** Sweeping the
batchable forward (the B=1 number, 143.5 ms, matches Phase 2's vision+projector+prefill = 144 ms):

| Batch | ms/call | ms/item | throughput (items/s) | peak mem |
|---:|---:|---:|---:|---:|
| 1 | 143.5 | 143.5 | 6.97 | 15,204 MB |
| 2 | 183.0 | 91.5 | 10.93 | 15,266 MB |
| 4 | 338.5 | 84.6 | 11.82 | 15,390 MB |
| 8 | 661.5 | 82.7 | 12.09 | 15,641 MB |

**Read:** per-item latency drops 143→83 ms (throughput 7.0→12.1 items/s, ~1.7×) but **saturates
by B=2** — almost all the gain is B=1→2. The prefill GEMMs are already near compute-bound at
batch 1 (≈280-token sequence), so batching only amortizes fixed launch/overhead, then flatlines.
Memory barely moves (+440 MB at B=8) — the 7B weights dominate, so memory is *not* the batch limit;
compute saturation is. **Net deployment reality:** since decode (59% of latency, Phase 2) is
batch-1-locked, real end-to-end throughput does **not** scale with batch — it stays pinned at the
batch-1 rate (~2.8 inferences/s). Batching would only help a hypothetical multi-robot/parallel-sim
server *and* only after the model gained batched-decode support.

## Phase 10 — Memory stability (leak check)
*Script:* [`profiling/memory_profiler.py`](profiling/memory_profiler.py) · *Data:*
[`results/memory/memory_stability.json`](results/memory/memory_stability.json) · *Figure:*
[`figures/memory_stability.png`](figures/memory_stability.png)

500 inference iterations, GPU memory logged every 50 steps:

| | value (constant across all 500 iters) |
|---|---|
| allocated | 15,138.1 MB |
| reserved | 15,531.5 MB |
| peak allocated | 15,476.0 MB |
| **allocated drift over run** | **+0.0 MB** |
| **peak growth after warmup** | **+0.0 MB** |

**Read:** memory is byte-identical at every checkpoint — **leak-free, peak fully plateaued**. The
CUDA caching allocator reuses the same blocks each step and the KV cache is released after every
`predict_action`, so a long-running control session won't creep toward OOM. This is the
determinism property that matters for an embedded deployment (predictable, bounded memory →
no mid-task OOM). See the figure for the flat memory-over-time curves.

## Phase 11 — Cross-run reproducibility
*Script:* [`scripts/run_reproducibility.py`](scripts/run_reproducibility.py) · *Data:*
[`results/baseline/reproducibility.json`](results/baseline/reproducibility.json)

The latency benchmark (100 iters) run 5× as separate fresh processes:

| run | mean ms |
|---|---:|
| 1 | 359.31 |
| 2 | 359.03 |
| 3 | 358.54 |
| 4 | 359.03 |
| 5 | 359.39 |

| metric | value |
|---|---:|
| across-run mean | 359.06 ms |
| across-run std | 0.335 ms |
| **CV (mean latency)** | **0.093%** |
| CV (p95 latency) | 0.560% |

**Read:** CV of **0.093%** — ~50× under the 5% threshold, so no variance investigation was needed.
Run-to-run spread is sub-millisecond across fresh CUDA contexts and model reloads. This validates
that every single-run number in this report is trustworthy, and (with Phases 1 + 10) confirms
OpenVLA inference on the A10G is deterministic in **time, tail, and memory** — the reproducibility
property a CI latency-regression gate (or a WCET argument) would rely on.

## Phase 12 — Roofline analysis
*Script:* [`analysis/roofline.py`](analysis/roofline.py) · *Data:*
[`results/analysis/roofline.json`](results/analysis/roofline.json) · *Figure:*
[`figures/roofline.png`](figures/roofline.png)

**Ridge points** (arithmetic intensity where the A10G flips memory- → compute-bound = peak ÷ 600 GB/s):

| Precision | Peak | Ridge point |
|---|---:|---:|
| FP32 | 31.2 TFLOPS | 52.0 FLOP/byte |
| FP16/BF16 | 62.5 TFLOPS | 104.2 FLOP/byte |
| INT8 | 125 TOPS | 208.3 OP/byte |

**Kernel arithmetic intensity** (AI = 2·M·K·N ÷ bytes of HBM traffic; classified at the BF16 ridge 104.2):

| Kernel | AI (FLOP/byte) | vs ridge | Bound |
|---|---:|---:|---|
| LLM **decode** · attn QKV/O | 1.0 | 0.01× | **memory-bound** |
| LLM **decode** · MLP gate/up | 1.0 | 0.01× | **memory-bound** |
| LLM **decode** · MLP down | 1.0 | 0.01× | **memory-bound** |
| LLM **decode** · LM head | 1.0 | 0.01× | **memory-bound** |
| LLM prefill · attn QKV/O | 247.1 | 2.37× | compute-bound |
| LLM prefill · MLP gate/up | 256.8 | 2.47× | compute-bound |
| LLM prefill · MLP down | 256.8 | 2.47× | compute-bound |
| Vision · ViT MLP (256 patch) | 195.1 | 1.87× | compute-bound |

**The FLOPs-vs-time paradox (the headline):**

| Stage | FLOPs/step | share of FLOPs | share of *time* (Phase 2) |
|---|---:|---:|---:|
| Prefill | 3.71 TFLOP | **97.9%** | ~35% |
| Decode | 0.079 TFLOP | **2.1%** | **~59%** |

Decode does **2% of the arithmetic but takes 59% of the wall-clock** — the textbook memory-bound
signature. Each of the 7B weights is read from HBM to do a single MAC per token (AI ≈ 1), so the
tensor cores sit ~99% idle during decode, starved on bandwidth. Prefill, batching ~281 tokens
through the same weights, reaches AI ≈ 250 and runs compute-bound near the BF16 ceiling.

**Cross-check:** analytic GEMM FLOPs = **3.79 TFLOP/step**, exactly matching the Phase-3 profiler's
measured `aten::mm` (3.79 TFLOP/step, ratio 1.00) — the model is validated against measured data.

**Why this ties the whole project together:** decode sits ~100× below the ridge, with enormous
*bandwidth* headroom and zero *compute* headroom. That is precisely why low-precision **weights**
are the theoretically correct lever for decode (fewer bytes/token moved → AI rises toward the
ridge), and why INT4's 4-bit weights *should* help — the Phase-5 result that they don't in
practice is a bitsandbytes dequant-overhead problem, not a roofline one. Conversely, prefill and
vision are compute-bound, so they'd benefit from tensor-core throughput, not quantization.

---

## Deployment recommendations

- **If memory allows, run BF16 (or FP16 — identical).** It's the fastest and most faithful.
- **Use INT4 only to *fit* the model** on a memory-constrained edge device (4.7 GB), accepting
  ~1.6× latency and a behavioral risk in end-effector positioning. Do not quantize for speed.
- **Never use INT8 here** — slower and larger than INT4 with no compensating benefit.
- **Latency, not memory, is the binding constraint on the A10G:** ~2.8 Hz at BF16 is below a
  typical 5–10 Hz manipulation loop. Precision alone won't close that gap; speculative decoding,
  fewer action tokens, or batching would be the next levers (see roofline analysis, pending).

## Status / next
Phases 0–6 complete. **Pending:** roofline analysis (map the hot GEMM/decode kernels onto the
A10G's compute-vs-bandwidth roofline) and full-suite behavioral eval for statistical significance.
