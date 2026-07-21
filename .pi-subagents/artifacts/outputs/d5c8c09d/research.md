# Research: Red-team of the "predictability-aware training" novelty claim

Date: 2026-07-21 (adversarial prior-art review)

## Summary

**Verdict: WEAKENED, not dead.** The narrow mechanism — *an auxiliary predictor whose top-k accuracy on future expert activations is an explicit term in the MoE's own training loss, with gradients shaping the backbone/router to be predictable* — still appears unclaimed. But the broader framing in RESEARCH.md ("no prior work trains the model to be inference-friendly; all prior work trains predictors post-hoc against frozen models") is now **false as stated**: StickyMoE (2026), Oracle-MoE (2025), and ReMoE (2026) all modify training to make routing cache/prefetch-friendly, and Pre-gated MoE fine-tunes the backbone *with* its pre-gate in the loop (so the model does co-adapt to the predictor). The claim must be re-scoped to "predictor-accuracy-as-loss, bidirectional co-design" and must cite/differentiate these works. Also: RESEARCH.md mis-cites Pre-gated MoE as arXiv 2205.10034 — that ID is **SE-MoE**; Pre-gated MoE is **arXiv 2308.12066** (ISCA'24).

## Findings

1. **Pre-gated MoE is co-trained, not post-hoc — but has no predictability objective on the backbone.** Reading the full ISCA'24 paper: the pre-gate is a small MLP added per MoE block that *replaces* routing for the next block (its mask deterministically selects which experts execute). It is "incrementally trained during the fine-tuning stage," end-to-end with the whole model on the task loss, starting from pretrained SwitchTransformer weights. So gradients from the task DO shape the backbone to work with the pre-gate's choices (and vice versa) — the RESEARCH.md characterization "pre-gate adapts to the model, not vice versa" is only half right. However, there is **no explicit predictor-accuracy loss and no regularization of the router toward predictability**; the pre-gate learns to route *well*, not to predict a frozen router. Claim survives on the loss-function point, weakens on the "post-hoc/frozen" framing. [Pre-gated MoE, arXiv 2308.12066](https://arxiv.org/html/2308.12066v3)

2. **StickyMoE (arXiv 2607.08780, Jul 2026) is the most dangerous prior art.** It adds a differentiable *routing consistency loss* (ℓ2 between consecutive tokens' gate distributions, plus a window-anchor variant) to pretraining, explicitly to reduce expert switching and cache misses on memory-constrained devices. It is training-time, architecture-agnostic, co-trains expert representations and routing "from the first training step," reports up to 59% lower switch rate and 3.92× fewer cache misses while *improving* perplexity, and Pareto-dominates post-hoc router fine-tuning. This kills the sentence "all prior work treats the model as fixed." It differs from our proposal: no predictor module, no lookahead horizon, no ranking-aware top-k objective; it optimizes *temporal stickiness* (token t vs t−1), not *predictability from earlier layers*. Note: sticky routing likely *helps* 1-step-ahead prediction — a reviewer will ask why a consistency loss isn't a cheaper way to get most of the benefit. We must implement it as a training-time baseline. [StickyMoE](https://arxiv.org/html/2607.08780)

3. **Oracle-MoE (Zhou et al., 2025) trains from scratch for locality-preserving routing.** Redesigns routing to operate in an attention-derived "oracle space" that is semantically stable across tokens, explicitly targeting memory-constrained inference; trained from scratch, SOTA edge-inference speed. Training-time, inference-motivated, but the locality is *structural* (choice of routing input), not an explicit predictability loss. Must cite. [via StickyMoE related work](https://arxiv.org/html/2607.08780)

4. **ReMoE (Zhu et al., 2026) — post-hoc router fine-tuning for expert reuse.** Fine-tunes only the router of a pretrained MoE with LM loss + locality-aware gate regularizer to boost short-horizon expert reuse and cut cache misses in vLLM serving. This is *not* post-hoc predictor training; it is post-hoc *model* modification for the same systems goal — another framing threat and a cheap strong baseline (StickyMoE reports it as their main post-hoc comparator and could not reproduce meaningful switch-rate reductions from it at small scale). [via StickyMoE](https://arxiv.org/html/2607.08780)

5. **Speculative-decoding analog exists: Halfway Speculative Decoding does joint drafter+target training with direct acceptance-rate optimization** — i.e., exactly our co-design pattern (train the big model to be predictable by the small one) but in token space, not expert-routing space. Jakiro/MoE-Spec/MoESD/EVICT apply SD to MoE but treat routing as fixed or as a budgeting problem. This precedent doesn't invalidate the claim but gives reviewers an "isn't this just X for experts?" line; cite it proactively and note expert-activation prediction differs (per-layer sets, no acceptance/rejection correction — mispredictions cost latency, not correctness). [Halfway SD](https://sergiu-nistor.com/assets/publications/Halfway_Speculative_Decoding.pdf)

6. **No industry precedent found for training-for-prefetchable routing.** DeepSeek-V3 (aux-loss-free load balancing via bias correction, node-limited routing at inference), Qwen3, and Mistral tech reports contain nothing about optimizing routing for predictability/prefetch. DeepSeek's inference-friendliness is architectural (MLA, fine-grained experts), not a training objective. [DeepSeek-V3 TR](https://arxiv.org/pdf/2412.19437)

7. **Routing regularization literature (StableMoE 2204.08396, ERC loss 2512.23447, cross-layer regularization 2602.14159, uncertainty-aware routing ACL'26) targets quality/stability/specialization, never inference-time predictability.** StableMoE's two-stage distillation fixes routing *fluctuation across training*, a different axis from predictability across layers/tokens. Safe to cite as "regularizes routing but not for predictability." [StableMoE](https://ar5iv.labs.arxiv.org/html/2204.08396)

8. **SOTA-control check: 2511.10676 is a strong but not unambiguously strongest post-hoc protocol.** Its headline numbers (93.0/94.7/97.6% top-k on DSv2-Lite/Qwen3-30B/Phi-mini-MoE) are comparable to or below **MoE-Beyond** (arXiv 2508.17137: 97.5% acc / 86.6 F1 on DSv2-Lite, cache hit 17%→72% at 10% budget) and it lacks the end-to-end scheduling evaluation of **PreScope** (arXiv 2509.23638, ICS'26: +141% throughput). Its ranking-aware loss and pre-attention input make it the best *methodological* match for our predictor, but we should benchmark against all three post-hoc predictors **and** the two training-time baselines (StickyMoE consistency loss, ReMoE router fine-tune). [MoE-Beyond](https://arxiv.org/abs/2508.17137), [PreScope](https://arxiv.org/abs/2509.23638)

## Closest works and how they differ

1. **StickyMoE (2607.08780)** — Training-time loss on the model for inference-friendly routing: identical philosophy, different objective. It penalizes routing *change between adjacent tokens*; we optimize *predictability of future layers' routing from earlier activations*. Sticky gives temporal reuse (cache hits from repetition); ours gives lookahead prefetch accuracy (hits from forecasting), which composes with rather than duplicates stickiness — sticky routing can't help when the topic genuinely changes, and does nothing for cross-layer horizon. Differentiator to prove empirically: our loss should beat/consistently complement StickyMoE on prefetch hit-rate at 2+ layer horizon, and we should show the two losses stack.
2. **Pre-gated MoE (2308.12066)** — Co-trained lookahead module whose output *is* the routing decision (mask consumed by next block), fine-tuned end-to-end with the backbone. Differs: no separate router whose behavior is predicted (routing is moved, not predicted); no predictor-accuracy term in the loss; backbone shaping is incidental via task loss, not an explicit predictability regularizer; evaluated on encoder-decoder Switch-Base, not modern 100+-expert decoder LLMs. Ours keeps the true router and adds a bidirectional objective.
3. **Oracle-MoE (2025)** — Training-time, from-scratch, inference-motivated routing redesign. Differs: architectural (attention-derived routing input) with no explicit predictability loss; locality is emergent. Ours is an objective, applicable to any existing MoE via fine-tuning.
4. **ReMoE (2026)** — Router-only fine-tuning for expert reuse. Differs: post-hoc, router-only (experts/backbone frozen, which StickyMoE shows limits gains); no predictor, no lookahead.
5. **Halfway Speculative Decoding (2026)** — Same co-design *pattern* (train target to be draftable) in token space. Differs: domain (tokens vs per-layer expert sets), failure semantics (SD is lossless via rejection; prefetch misprediction is a latency cost), and no MoE.

## Stronger baselines we should implement

- **StickyMoE consistency loss** (single hyperparameter λ, trivial to implement) — mandatory training-time ablation; also test *stacking* it with our predictor-accuracy loss.
- **ReMoE-style router-only fine-tune** — cheap post-hoc control isolating "router-only vs full-backbone" shaping.
- Keep **2511.10676** as the frozen-predictor control but add **MoE-Beyond**'s protocol (66M-trace sequence predictor, F1 + cache-hit-at-budget metrics) since its numbers are arguably stronger; report both top-k accuracy and cache-hit/end-to-end metrics like PreScope.

## Recommended re-scoped claim

"Prior systems-side work trains predictors post-hoc against frozen models; prior training-time work (StickyMoE, Oracle-MoE, ReMoE) optimizes routing *temporal locality*; Pre-gated MoE co-trains a lookahead gate that *replaces* routing. No prior work adds the accuracy of an independent expert-activation predictor as an explicit term in the MoE's own loss, regularizing the backbone/router to *be predictable* at multi-layer lookahead horizons while the predictor simultaneously adapts to the model."

## Sources

- Kept: [Pre-gated MoE, ISCA'24 (arXiv 2308.12066)](https://arxiv.org/html/2308.12066v3) — full-text read; settles the pre-gate training question; also fixes the citation error in RESEARCH.md.
- Kept: [StickyMoE (arXiv 2607.08780)](https://arxiv.org/html/2607.08780) — closest prior art; primary threat and new mandatory baseline.
- Kept: [StableMoE (arXiv 2204.08396)](https://ar5iv.labs.arxiv.org/html/2204.08396) — routing-stability regularization, different axis.
- Kept: [Halfway Speculative Decoding](https://sergiu-nistor.com/assets/publications/Halfway_Speculative_Decoding.pdf) — token-space analog of the co-design.
- Kept: [DeepSeek-V3 Technical Report](https://arxiv.org/pdf/2412.19437) — confirms no industry training-for-prefetch precedent.
- Kept: [SE-MoE (arXiv 2205.10034)](https://ar5iv.labs.arxiv.org/html/2205.10034) — resolves the mis-citation; it is a different paper (prefetch-all system + distillation).
- Dropped: MoE-Spec / EVICT / MoESD / Jakiro — MoE speculative decoding; routing treated as fixed or budgeted, no training-for-predictability.
- Dropped: ERC loss (2512.23447), cross-layer reg (2602.14159), uncertainty-aware routing (ACL'26) — routing losses for quality, not predictability; cite-in-one-sentence material only.

## Gaps

- Oracle-MoE and ReMoE were read only through StickyMoE's related-work section; pull their PDFs before writing the paper's related work (exact numbers, venues).
- MoE-Beyond vs 2511.10676 head-to-head on identical models/metrics is unclear (different trace sets); resolving "strongest post-hoc protocol" requires running both — flagged as implementation work.
- StickyMoE is days/weeks old (Jul 2026) and evaluated only on small WikiText-2 MoEs; its claims may not hold at Qwen3-30B scale — which is itself a usable differentiator if our fine-tune-scale experiments beat it.
- Did not exhaustively search Chinese-lab tech reports beyond DeepSeek/Qwen (e.g., GLM-5.2's own report, given Colibri targets it) — worth one follow-up pass.
