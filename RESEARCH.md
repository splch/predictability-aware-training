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
| **Pre-gated MoE** (arXiv 2205.10034) | ISCA'24, Microsoft | Algorithm-system co-design; small "pre-gate" module per layer predicts next layer's expert choice for one-layer-ahead prefetch. Closest in spirit, but pre-gate adapts to the model, not vice versa. | Prefetch hides most offloading latency |
| **Fate** (arXiv 2502.12224) | 2025 | Cross-layer gate: adjacent layers' gate inputs predict each other; shallow-favoring caching; quantization for cache/IO | 99% hit rate; ~4x prefill/decode speedups over load-on-demand |
| **Pre-attention expert prediction** (arXiv 2511.10676) | Nov 2025 | Key insight: some LLM functions are *ranking-preserving*, so 2 linear layers on pre-attention activations + **ranking-aware loss** suffice; also covers layer 1 | 93.0% (DeepSeek-V2-Lite), 94.7% (Qwen3-30B), 97.6% (Phi-mini-MoE); ~15 pts over prior SOTA |
| **MoE-Beyond** (arXiv 2508.17137) | Aug 2025 | Lightweight transformer trained on 66M activation traces (DeepSeek-V2-Lite) as multi-label sequence prediction | 97.5% acc / 86.6 F1; cache hit 17%→72% at 10% cache budget |
| **PreScope** (arXiv 2509.23638) | ICS 2026 | Learnable layer-aware predictor (LLaPor) + globally optimal prefetch scheduling + async I/O | +141% throughput, -74.6% latency vs SOTA |
| **LLM in a Flash** (arXiv 2312.11514) | Apple | Windowing (reuse active params across tokens) + row-column bundling for flash-resident sparse inference | Runs 2x RAM-size models on iPhone |
| **PowerInfer-2** (arXiv 2406.06282) | SJTU | Neuron/expert activation prediction for smartphones (Mixtral 47B) | 11.7 tok/s on phone |
| **MoE-Infinity** | ATC'25 | Activation-aware expert caching from traces on personal machines | Better hit rates than LRU at small budgets |

## The gap this project fills

No found work folds predictor accuracy into the LLM's **own training loss** so
the big model is optimized to *be predictable*. All existing work treats the
model as fixed and makes the predictor chase it. Existing router losses
(load-balancing aux loss, expert-router coupling) target quality/balance, not
temporal predictability of routing decisions. The novel contribution is the
**bidirectional co-design**: predictor learns the router; router is regularized
toward predictability.

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
