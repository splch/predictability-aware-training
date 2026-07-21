# Research: Predictability-Aware Training for MoE Expert Prefetching

Date: 2026-07-21

## The proposal

Train a small activation-prediction model **jointly** with an MoE LLM, folding
prediction accuracy into the LLM's training loss, so the large model is optimized
not just for output quality but for being *predictable*. The predictor forecasts
which experts will fire ahead of time; because predictability is baked into
training, forecasts are accurate enough to prefetch expert weights from disk
before they're needed — hiding disk latency for disk-resident MoE inference.

## Target runtime: Colibri

- https://github.com/JustVugg/colibri (~17k stars): pure-C engine, runs GLM-5.2
  (744B MoE, int4) on ~25 GB RAM. Dense backbone (~17B params) resident
  (~9.9 GB); 19,456 routed experts (75 layers x 256, ~19 MB each) on disk
  (~370 GB), streamed via single `pread`s with per-layer LRU cache, learned
  pinned hot-store, async I/O pool, batch-union reads.
- **Already has the reactive/heuristic version of the idea**: router-lookahead
  thread (`PILOT=1`) prefetches next-layer experts; reports routing is
  **measurably 71.6% predictable one layer ahead**. This is our baseline to beat.
- Colibri also supports OLMoE (`c/olmoe.c`) — a natural target for a
  transfer/fine-tune experiment.

## Prior art: expert prediction (all train predictors against a *frozen* model)

| Work | Venue | Approach | Reported accuracy / effect |
|---|---|---|---|
| **Pre-gated MoE** (arXiv 2308.12066) | ISCA'24, Microsoft | Algorithm-system co-design; small "pre-gate" module per layer selects next layer's experts for one-layer-ahead prefetch. Co-trained end-to-end with the backbone during fine-tuning (so the backbone DOES co-adapt to the predictor via the task loss), but there is no explicit predictor-accuracy loss and no regularization of the router toward predictability. Closest co-design precedent. (Note: arXiv 2205.10034 is SE-MoE, a different paper — earlier draft of this doc mis-cited it.) | Prefetch hides most offloading latency |
| **Fate** (arXiv 2502.12224) | 2025 | Cross-layer gate: adjacent layers' gate inputs predict each other; shallow-favoring caching; quantization for cache/IO | 99% hit rate; ~4x prefill/decode speedups over load-on-demand |
| **Pre-attention expert prediction** (arXiv 2511.10676) | Nov 2025 | Key insight: some LLM functions are *ranking-preserving*, so 2 linear layers on pre-attention activations + **ranking-aware loss** suffice; also covers layer 1 | 93.0% (DeepSeek-V2-Lite), 94.7% (Qwen3-30B), 97.6% (Phi-mini-MoE); ~15 pts over prior SOTA |
| **MoE-Beyond** (arXiv 2508.17137) | Aug 2025 | Lightweight transformer trained on 66M activation traces (DeepSeek-V2-Lite) as multi-label sequence prediction | 97.5% acc / 86.6 F1; cache hit 17%→72% at 10% cache budget |
| **PreScope** (arXiv 2509.23638) | ICS 2026 | Learnable layer-aware predictor (LLaPor) + globally optimal prefetch scheduling + async I/O | +141% throughput, -74.6% latency vs SOTA |
| **LLM in a Flash** (arXiv 2312.11514) | Apple | Windowing (reuse active params across tokens) + row-column bundling for flash-resident sparse inference | Runs 2x RAM-size models on iPhone |
| **PowerInfer-2** (arXiv 2406.06282) | SJTU | Neuron/expert activation prediction for smartphones (Mixtral 47B) | 11.7 tok/s on phone |
| **MoE-Infinity** | ATC'25 | Activation-aware expert caching from traces on personal machines | Better hit rates than LRU at small budgets |

## Training-time prior art (added after novelty red-team, 2026-07-21)

The original framing ("all prior work trains predictors post-hoc against frozen
models") is FALSE as stated. Training-time work on inference-friendly routing
exists and must be differentiated:

| Work | What it does | How it differs from this project |
|---|---|---|
| **StickyMoE** (arXiv 2607.08780, Jul 2026) | Differentiable routing-consistency loss (l2 between consecutive tokens' gate distributions) in pretraining; -59% switch rate, 3.92x fewer cache misses, Pareto-dominates post-hoc router fine-tuning | Most dangerous prior art. Optimizes *temporal stickiness* (token t vs t-1), not *predictability from earlier layers*; no predictor, no lookahead horizon. Sticky routing can't help when topic genuinely changes. **Mandatory training-time baseline; test stacking with our loss.** |
| **Oracle-MoE** (2025) | From-scratch training with locality-preserving routing in an attention-derived "oracle space" | Architectural redesign; locality emergent, no explicit predictability objective |
| **ReMoE** (2026) | Router-only fine-tune with locality-aware gate regularizer for expert reuse in serving | Post-hoc, router-only (experts/backbone frozen); no predictor/lookahead. Cheap strong control. |
| **Halfway Speculative Decoding** (2026) | Joint drafter+target training optimizing acceptance rate directly | Same co-design *pattern* (train big model to be predictable by small one) in token space, not expert space. Cite proactively. |
| Routing regularization (StableMoE 2204.08396, ERC 2512.23447, cross-layer reg 2602.14159) | Router losses for quality/stability/specialization | Never target inference-time predictability |

No industry precedent found (DeepSeek-V3, Qwen3, Mistral tech reports) for
training-for-prefetchable routing.

## Re-scoped novelty claim

Prior systems work trains predictors post-hoc against frozen models; prior
training-time work (StickyMoE, Oracle-MoE, ReMoE) optimizes routing *temporal
locality*; Pre-gated MoE co-trains a lookahead gate that *replaces* routing.
**No prior work adds the accuracy of an independent expert-activation
predictor as an explicit term in the MoE's own loss**, regularizing the
backbone/router to *be predictable* at multi-layer lookahead horizons while
the predictor simultaneously adapts to the model. Verdict: claim stands but
narrowed; the works above must be cited and StickyMoE implemented as a
baseline.

## Design considerations surfaced by the literature

1. **Loss shaping**: ranking-aware losses beat regression — only top-k order
   matters (2511.10676). Prediction = multi-label CE over experts per layer.
2. **Horizon trade-off**: 1-layer-ahead is easy (71.6% free, per Colibri) but
   NVMe latency may need 2+ layers of lead; accuracy drops with horizon — this
   is exactly where training-for-predictability should pay.
3. **Failure modes to watch**: predictability pressure can distort routing
   (expert collapse, load imbalance, degraded specialization). Co-train with the
   standard load-balancing loss; track perplexity + downstream quality deltas.
4. **Cheap experimental path**: fine-tune an open MoE (Qwen3-30B-A3B, OLMoE,
   DeepSeek-V2-Lite) with the joint loss rather than pretraining; measure top-k
   hit-rate gain vs the frozen-model predictor baseline.
5. **Eval protocol**: per-layer top-k hit rate → simulated cache hit rate under
   memory budget → end-to-end tok/s & TTFT vs Colibri `PILOT=1` baseline, plus
   quality (PPL + downstream) to prove predictability didn't cost capability.
