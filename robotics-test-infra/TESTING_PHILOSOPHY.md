# Testing Philosophy

Why this project exists, and how it connects embedded-systems testing discipline to
robot-learning software.

## The thesis: the harness is the product

A policy that scores well in a notebook tells you almost nothing about whether it will
still work after the next commit, on a different seed, under a tighter latency budget,
or six months from now when a dependency bumps. What makes a robot *shippable* is the
machinery around the model: deterministic evaluation, explicit budgets, regression
gates, reproducible environments, and an unambiguous pass/fail signal on every change.

That machinery — not the policy — is the deliverable here. The Project-1 ACT policy is
deliberately undertrained and scores 0% on the task; that's fine, even useful. A
policy with a non-trivial, noisy behaviour is a *better* test subject than a perfect
one, because it gives the infrastructure something real to measure and guard.

## From embedded validation to SIL

I come from embedded systems — RTOS, FPGAs, firmware — where this discipline is
routine and the vocabulary is precise. The mapping to simulation testing is direct:

| Embedded / real-time concept | This framework's analogue |
|---|---|
| **Bring-up / power-on self-test** | `validation.py`: checkpoint integrity + env health checks run *before* trusting any measurement |
| **WCET (worst-case execution time)** | `inference_latency_max_ms` — the periodic chunk-boundary spike |
| **Steady-state / typical latency** | `inference_latency_p95_ms` — the gated budget |
| **Jitter** | spread between p50 and p95/p99 inference latency |
| **Deadline monitoring** | `passes_threshold`: p95 latency must stay under the ceiling |
| **Determinism / reproducibility** | `check_determinism.py`: same seed → identical trajectory |
| **Regression testing** | Tier 3 vs. a committed baseline |
| **HIL test rig** | the MuJoCo env today; a real-robot driver tomorrow (same pyramid) |

## WCET, jitter, and why we gate p95 (not max)

The ACT policy uses **action chunking**: it runs a real forward pass, emits 100
actions, then replays them one per step before inferring again. So per-step inference
latency is strongly **bimodal** — and the benchmark suite measures exactly this:

- steady-state (queue replay): **p50 ≈ 0.8 ms, p95 ≈ 2.9 ms**
- chunk-boundary inference (every 100 steps): a **~1.3 s spike** (`max`)

This is precisely the shape a real-time engineer reasons about: a light periodic task
with a heavy periodic event every N ticks. We **gate on p95** — the steady-state
budget the system lives in almost all the time — and **track max** as the WCET that a
scheduler would have to absorb (with a buffer, a deadline, or by spreading the chunk
computation). Gating on `max` would be the equivalent of sizing every cycle for the
worst case; tracking it without gating is the equivalent of a documented WCET that
informs the schedule. Both numbers matter; they answer different questions.

A second finding from the benchmark reinforces the point: **policy inference is only
~4.6% of the loop**. The MuJoCo step — which re-renders the camera every tick —
dominates at ~180 ms/step. In a real system that ratio flips (real cameras and buses,
not software rendering), but the lesson is the same one embedded work teaches: profile
before you optimize, and budget the part that actually dominates. Here the "sensor +
transport" layer dominates, which is exactly where Project 2's perturbation study
(latency, frame-drops, noise) targets its analysis.

## Why determinism is non-negotiable

If the same seed produced different trajectories, every regression delta would be
indistinguishable from run-to-run noise, and the regression tolerances would have to
be so wide they'd catch nothing. Determinism is the foundation that lets the gates be
tight enough to be useful. It's the same reason embedded tests pin clocks, seed PRNGs,
and control for nondeterministic DMA ordering: a test you can't reproduce isn't a test,
it's an anecdote.

## Tolerances are a budget, not a guess

The regression gate uses `delta > max(baseline × frac, abs_floor)` per metric. The
absolute floor encodes a real engineering judgment: at a ~2.6 ms p95 baseline, a 25%
fractional move is 0.7 ms — well inside shared-runner jitter — so a `+5 ms` floor says
"don't cry wolf below 5 ms." This is the same reasoning as a timing margin in a
hardware spec: the threshold reflects what the system can actually tolerate, derived
from how the metric behaves, not a number pulled from the air. When the environment
gets quieter (a dedicated runner) or the stakes get higher (HIL), you tighten the
budget — deliberately, with the data in front of you.

## Fail fast, fail cheap, fail loud

The three tiers are ordered by cost. The cheapest, most common breakage (code) is
caught in seconds and blocks everything downstream; the expensive sim run only happens
once the cheap checks pass; the regression gate only runs once the sim is green. A
degenerate run (zero episodes, NaN metrics) **fails** rather than slipping through a
NaN comparison — a test framework that silently passes on garbage is worse than no
framework, because it manufactures false confidence. The whole point is a signal you
can trust enough to merge on.
