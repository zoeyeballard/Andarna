# OpenVLA Inference Profiling Report

> **Status:** scaffold. Sections fill in as each phase completes.

## 1. Executive Summary
_What was profiled, the key findings, and the headline optimization recommendations._

## 2. GPU and Model Specifications
_A10G specs; OpenVLA architecture (ViT → Projector → LLM); parameter counts per component._

## 3. Baseline Profile
_End-to-end latency distribution, component time breakdown, top CUDA kernels, memory, CPU vs GPU._

## 4. Quantization Results
_BF16 / FP16 / INT8 / INT4 latency + memory; the INT8-slower-than-INT4 anomaly; per-module sensitivity; accuracy impact on LIBERO._

## 5. Optimization Results
_torch.compile, Flash Attention verification, CUDA graphs, batch-size scaling._

## 6. Roofline Analysis
_A10G roofline with key kernels plotted; memory-bound vs compute-bound; optimization opportunity map._

## 7. Deployment Latency Analysis
_Control frequency achievable per precision; tie-back to latency budget for a real robot._

## 8. Embedded Systems Perspective
_Edge deployment (Jetson Orin Nano), FPGA acceleration candidates, deterministic-timing / WCET framing._

## 9. Recommendations
_Best latency-accuracy precision for production; what to optimize first; target hardware._

## 10. Next Steps
_Production inference pipeline, TensorRT conversion, portfolio connections (Projects 1–4)._
