# OpenVLA Inference Profiling & Optimization Report

Profiling and quantization study of **OpenVLA-7B** (`openvla/openvla-7b-finetuned-libero-object`)
on an **NVIDIA A10G**. All numbers measured on the A10G with `torch 2.2.0+cu121`,
`transformers 4.40.1`, SDPA attention (flash-attn not built). Raw data per phase is committed
under [`results/`](results/); figures under [`figures/`](figures/).

---

## 1. Executive summary

- **Inference is LLM-bound and GEMM-bound.** The LLM backbone is ~94% of latency (decode 59% +
  prefill 35%); the vision encoder is 5.5% and the projector negligible. ~86% of GPU time is
  `aten::mm` on Ampere BF16 tensor-core kernels that are *already* fast.
- **Quantization buys memory, not speed.** Every quantized config was *slower* than BF16. INT8 was
  worst (3.8×); INT4 (1.6×) beat INT8 on both speed and memory — reproducing the known OpenVLA
  "INT8 worse than INT4" anomaly.
- **FP16 is a free swap** for BF16 (identical latency, memory, actions, and — within noise —
  task success). **INT4 is for *fitting* the model** (0.31× memory → 4.7 GB), accepting a latency
  penalty and a behavioral risk concentrated in end-effector positioning.
- **Latency, not memory, is the binding constraint.** At BF16 the policy runs **~2.8 Hz** —
  far below a 5–10 Hz manipulation loop. The roofline shows why: the decode stage is
  memory-bandwidth-bound at ~100× below the GPU's ridge point.
- **Inference is highly deterministic** (run-to-run CV 0.09%, leak-free over 500 iters) — the
  prerequisite for treating it as a schedulable real-time task.

| Precision | Latency (mean) | vs BF16 | Peak mem | Cosine vs BF16 | LIBERO success (n=10) |
|---|---:|---:|---:|---:|---:|
| BF16 | 358.9 ms | 1.00× | 15.5 GB | 1.000 (ref) | 80% |
| FP16 | 359.9 ms | 1.00× | 15.5 GB | 0.997 | 70% |
| INT8 | 1365.1 ms | 3.80× | 8.5 GB | 0.953 | _(skipped — dominated)_ |
| INT4 | 572.1 ms | 1.59× | 4.7 GB | 0.899 | 60% |

---

## 2. GPU and model specifications

**NVIDIA A10G** (Ampere `sm_86`, 24 GB GDDR6):

| | FP32 | FP16/BF16 | INT8 | Mem BW | Ridge (BF16) |
|---|---:|---:|---:|---:|---:|
| peak | 31.2 TFLOPS | 62.5 TFLOPS | 125 TOPS | 600 GB/s | 104.2 FLOP/byte |

Ridge points (peak ÷ bandwidth): **FP32 52.0**, **BF16 104.2**, **INT8 208.3** FLOP/byte — the
arithmetic intensity above which a kernel is compute-bound rather than memory-bound.

**OpenVLA-7B** — Llama-2-7B backbone (hidden 4096, FFN 11008, 32 layers, vocab 32064 with action
tokens) + dual SigLIP/DINOv2 vision encoder + MLP projector. Four inference stages:
1. **Vision Encoder** (SigLIP + DINOv2) — camera image → patch features
2. **MLP Projector** — vision features → language embedding space
3. **LLM Prefill** — instruction + ~256 visual tokens (seq ≈ 281)
4. **LLM Decode** — autoregressive generation of ~7 action tokens

---

## 3. Baseline timing results
*Phase 1 · [`run_baseline_latency.py`](scripts/run_baseline_latency.py) · [data](results/baseline/baseline_latency_bf16.json)*

End-to-end `predict_action`, 20 warmup + 100 timed steps (`torch.cuda.Event`):

| mean | p50 | p95 | max | std | peak mem |
|---:|---:|---:|---:|---:|---:|
| 358.7 ms | 358.0 ms | 363.7 ms | 368.4 ms | 2.1 ms | 15,476 MB |

**Determinism (Phases 10–11).** Across 5 fresh-process runs the mean-latency **CV is 0.093%**
([data](results/baseline/reproducibility.json)); over 500 iterations GPU memory is byte-identical
at every checkpoint (**+0.0 MB drift, leak-free** — [data](results/memory/memory_stability.json),
[fig](figures/memory_stability.png)). Within-run CV 0.6%, jitter (max−p50) ~10 ms. Inference is
deterministic in **time, tail, and memory** — see §8 for why that matters.

---

## 4. Component breakdown
*Phases 2–4, 12 · [`component_timer.py`](profiling/component_timer.py) · [data](results/baseline/component_breakdown_bf16.json)*

| Stage | mean ms | % of total |
|---|---:|---:|
| Vision Encoder | 19.44 | 5.5% |
| MLP Projector | 0.77 | 0.2% |
| LLM Prefill | 123.89 | 34.9% |
| **LLM Decode** | **210.58** | **59.4%** |
| overhead | 5.17 | 1.4% |

**Operator level (Phase 3 · [data](results/baseline/torch_profiler_top20.txt)):** `aten::mm` is
**86%** of self-CUDA time, all on `ampere_bf16_s16816gemm` tensor-core kernels; flash attention is
active via SDPA (1.8%). Top memory consumers: `aten::linear` (20.25 GB cumulative), `aten::cat`
(19.75 GB — KV-cache concat). Nsight timeline (Phase 4) confirms the four NVTX stage bands.

**Roofline (Phase 12 · [data](results/analysis/roofline.json) · [fig](figures/roofline.png)):**

| Kernel | AI (FLOP/byte) | bound |
|---|---:|---|
| LLM **decode** (QKV/O, MLP, LM head) | ~1.0 (0.01× ridge) | **memory-bound** |
| LLM prefill (attn / MLP) | 247 / 257 | compute-bound |
| Vision ViT MLP | 195 | compute-bound |

**The FLOPs-vs-time paradox:** decode is **2.1% of FLOPs but ~59% of time**; prefill is **97.9% of
FLOPs but ~35% of time**. Decode reads each 7B weight from HBM to do a *single* MAC per token
(AI ≈ 1), so tensor cores idle ~99% of decode — the textbook memory-bound signature. Analytic GEMM
FLOPs (3.79 TFLOP/step) match the profiler's `aten::mm` exactly (ratio 1.00).

---

## 5. Quantization results
*Phases 5–6 · [`precision_runner.py`](quantization/precision_runner.py),
[`accuracy_validator.py`](quantization/accuracy_validator.py),
[`run_libero_eval.py`](scripts/run_libero_eval.py)*

**Latency + memory ([data](results/quantization/precision_sweep.json), [fig](figures/fig2_latency_vs_precision.png) · [fig](figures/fig3_memory_vs_precision.png)):**

| Precision | load s | mean ms | p95 ms | peak mem | vs BF16 lat | vs BF16 mem |
|---|---:|---:|---:|---:|---:|---:|
| BF16 | 10.7 | 358.9 | 360.6 | 15,476 MB | 1.00× | 1.00× |
| FP16 | 22.9 | 359.9 | 367.1 | 15,476 MB | 1.00× | 1.00× |
| INT8 | 36.1 | **1365.1** | 1397.9 | 8,456 MB | **3.80×** | 0.55× |
| INT4 | 12.0 | 572.1 | 586.2 | 4,730 MB | 1.59× | **0.31×** |

INT8's LLM.int8() mixed-precision decomposition (INT8 + FP16 outlier path + per-op quant/dequant)
costs more than the bandwidth it saves; INT4 (NF4) has lighter overhead and wins, but neither
beats native BF16 tensor-core GEMMs. **Quantization here buys memory, not speed.**

**Accuracy ([data](results/quantization/accuracy_validation.json), [fig](figures/fig4_accuracy_vs_precision.png)):**
action-vector comparison vs BF16 (5 fixed obs, baseline mean |action| = 0.235):

| Precision | mean MAE | max abs err | mean cosine | min cosine |
|---|---:|---:|---:|---:|
| FP16 | 0.0077 | 0.143 | 0.9969 | 0.9846 |
| INT8 | 0.0675 | 0.398 | 0.9528 | 0.9222 |
| INT4 | 0.0969 | 0.831 | 0.8985 | 0.7428 |

Per-dimension breakdown ([data](results/quantization/action_dim_breakdown.json)) **refutes a
gripper-flip**: the gripper is stable (0/5 sign flips at every precision); error concentrates in
the **translation deltas — `dz` worst**. Quantization degrades *spatial precision (where to move)*,
not *discrete decisions (grasp/release)*. (Caveat: synthetic inputs saturate the gripper logit.)

**Behavioral ([data](results/behavioral/libero_success.json)):** LIBERO-Object, 10 episodes each —
BF16 **80%** / FP16 **70%** / INT4 **60%**. Monotonic with precision and consistent with the
spatial drift, but **at n=10 the CIs (~±25 pp) overlap** — suggestive, not significant. INT4 also
ran slower (1.74 vs 2.77 Hz) with longer successful episodes (178 vs 160 steps).

---

## 6. Optimization experiments

**torch.compile (Phase 8 · [`torch_compile_test.py`](optimization/torch_compile_test.py) ·
[data](results/optimization/torch_compile.json)):** `torch.compile(model.forward,
mode="reduce-overhead")` **errors during tracing** — `scaled_dot_product_attention` can't expand
the dynamic 4-D causal mask (query 281 vs KV 280) inside the Llama backbone. No graph is produced,
no speedup. This is the autoregressive-VLM caveat in action; paths that could work (compile the
vision encoder only; `StaticCache` + a newer stack) weren't pursued, and given inference is GEMM-
bound on already-fast kernels, the upside would be limited regardless.

**Batch scaling (Phase 9 · [`batch_scaling_test.py`](optimization/batch_scaling_test.py) ·
[data](results/optimization/batch_scaling.json)):** end-to-end generation is **batch-1-locked**
(OpenVLA asserts `batch == 1` in the decode path). The batchable *prefill forward* scales
7.0 → 12.1 items/s (B=1→8) but **saturates by B=2** (already compute-bound); memory grows only
+440 MB (weights dominate). **Net: real throughput does not scale with batch** — decode is locked,
so it stays pinned at ~2.8 inferences/s.

---

## 7. Deployment recommendations — which precision for a real robot?

| Scenario | Choice | Why |
|---|---|---|
| Memory available (≥16 GB) | **BF16 or FP16** | Fastest *and* most faithful; FP16 = BF16 here on every axis |
| Memory-constrained (must fit) | **INT4** | 4.7 GB fits where 15.5 GB can't — but accept ~1.6× latency + spatial-precision risk |
| Any | **never INT8** | Slower *and* larger than INT4 with no compensating benefit |

- **Quantize to *fit*, not to *go faster*.** Nothing here makes OpenVLA faster than BF16.
- **Latency is the real problem, not precision.** ~2.8 Hz at BF16 is below a usable manipulation
  loop. Precision alone can't fix it — the levers are a smaller/distilled VLA, fewer action
  tokens, speculative decoding, or a proper fused-INT4 kernel (TensorRT-LLM / Marlin) that captures
  the decode bandwidth win that bitsandbytes squanders.
- If the policy must run on a fixed latency budget, run it as a **slow deliberative task under a
  fast inner servo loop** (see §8).

---

## 8. Embedded systems perspective

This is where the GPU systems data meets real robot deployment. Three angles, all anchored to the
measured roofline and timing.

### 8.1 Edge deployment on Jetson — bandwidth is destiny
The decode stage is **memory-bandwidth-bound** (§4), so edge performance is set by memory
bandwidth, not TOPS. Scaling the A10G's ~2.8 Hz by the bandwidth ratio:

| Platform | Mem BW | Capacity | BF16 fit? | Projected decode-bound rate* |
|---|---:|---:|---|---:|
| A10G (server) | 600 GB/s | 24 GB | yes | ~2.8 Hz (measured) |
| Orin AGX 64 GB | 204 GB/s | 64 GB | yes | ~0.95 Hz |
| Orin Nano 8 GB | 68 GB/s | 8 GB | **no — needs INT4** | ~0.3 Hz (BF16-equiv) |

\*first-order scaling of the memory-bound term; real numbers also lose to weaker compute on prefill.

Two hard conclusions: (1) **OpenVLA-7B at BF16 does not fit on an Orin Nano** (15.5 GB > 8 GB) —
INT4 (4.7 GB) is *mandatory just to load it*, which is exactly the "quantize to fit" case from §7;
(2) even where it fits, the ~9× bandwidth deficit makes a 7B VLA **fundamentally too slow for
real-time control at the edge**. The deployable path is a smaller VLA, distillation, or a
hardware INT4 datapath — not this model as-is.

### 8.2 FPGA acceleration opportunities
The roofline tells you where custom silicon helps and where it doesn't:
- **Decode (memory-bound) is *not* an FPGA win by itself.** 7B INT4 weights = ~3.5 GB ≫ on-chip
  SRAM (tens of MB), so an FPGA can't cache the model; it's still gated by external memory
  bandwidth. An HBM-class FPGA (Alveo U280) buys bandwidth but not a fundamental change.
- **The real FPGA/ASIC win is a fused dequant+GEMM datapath.** The Phase-5 anomaly (INT8 slower
  than INT4) is a *software* problem — bitsandbytes' outlier-path and per-op quant/dequant
  overhead. Custom hardware that fuses INT4 dequant into the matmul pipeline captures decode's
  bandwidth savings with none of that overhead — the single highest-leverage accelerator target.
- **The vision encoder is the clean FPGA candidate:** compute-bound, static 224×224 shapes, only
  5.5% of latency but a fixed feed-forward pipeline ideal for a streaming dataflow accelerator —
  and it offloads the GPU to focus on the LLM.

### 8.3 RTOS timing guarantees for inference scheduling
The determinism results (§3) make inference schedulable as a real-time task:
- **Soft-real-time WCET budget ≈ 400 ms** (p95 363.7, max 368.4, +margin). The tight distribution
  (CV 0.6%, jitter ~10 ms) means WCET ≈ mean — friendly to rate-monotonic analysis.
- **Hierarchical control is mandatory.** A 360 ms policy step → ~2.8 Hz deliberative task; safety
  and stability require a **fast inner servo loop (100–1000 Hz)** on the CPU/MCU that holds or
  interpolates the last action between policy updates. The VLA sets intent; the servo guarantees
  the deadline.
- **GPU non-preemptibility is the real-time hazard.** A CUDA inference call occupies the GPU for
  ~360 ms with coarse preemption, so a shared Jetson GPU **must not** host both the policy and a
  safety-critical task — keep the hard-real-time loop on a separate CPU core / MCU. And note the
  honest limit: GPU latency is *statistically* stable but **not hard-real-time guaranteed**
  (thermal throttling, driver, allocator can spike) — true WCET guarantees need the deterministic
  FPGA/ASIC path of §8.2 or generous margins.
- **Memory is bounded and leak-free** (§3), so the inference task has a fixed, analyzable footprint
  — no runtime growth to blow a memory budget mid-mission.

---

## 9. Status & next steps
Phases 0–6, 8–14 complete. Open items: per-precision component decomposition (fig1 is BF16-only —
needs GPU re-runs of the component timer at FP16/INT8/INT4); full-suite LIBERO behavioral eval
(10 tasks × 20–50 trials) to make the INT4 success gap statistically conclusive; and a fused-INT4
kernel (TensorRT-LLM / Marlin) to test whether decode's roofline bandwidth headroom is recoverable.
