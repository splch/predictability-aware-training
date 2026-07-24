# Predictability is Trainable: Optimizing MoE LLMs to be Forecastable for Disk-Resident Inference

**Workshop draft (6–8 pp equivalent). Evidence base: this repository, Experiments 1–11.**

## Abstract

Inference engines such as Colibri run 744B-parameter Mixture-of-Experts (MoE)
models on consumer hardware by keeping the dense backbone resident and
streaming expert weights from disk on demand — a reactive design whose
throughput is bounded by fetch latency. Post-hoc predictors can forecast
expert activations from earlier hidden states, but they chase a model never
asked to be predictable. We ask whether predictability is *trainable*: we
fold the accuracy of a small expert-activation predictor into the MoE's own
training loss, so the model is optimized jointly for quality and for being
forecastable. At 341M-parameter scale we show that (i) predictability is a
trainable, transferable **backbone property** — a fresh post-hoc predictor on
the frozen jointly-trained backbone recovers the full gain (+6–7 pts top-k
hit rate over a linear control, +3–4 pts over a ranking-aware control, 3
seeds); (ii) the gain is **structural, not sharpening** — entropy penalties
at any strength cannot reach it at matched entropy; (iii) it **grows with
training** while its quality cost vanishes; and (iv) it converts to systems
value (+3.2% tok/s, −31% misprefetch waste in a disk-queue-accurate
simulator; +5.0% mean tok/s in a purpose-built real engine). Finally we map
the boundary: a LoRA fine-tune on pretrained OLMoE-1B-7B leaves backbone
predictability unchanged, indicating the method is **pretraining-time only**.

## 1. Introduction

Sparse MoE LLMs activate only a small fraction of parameters per token, which
makes their *weights* — not their compute — the binding resource on consumer
hardware. Engines like Colibri exploit this: the dense backbone stays
resident while tens of thousands of routed experts are streamed from disk,
cached, and pinned by learned hot-stores. Every such engine is reactive: the
router fires, and only then does the fetch begin. Colibri's own mitigation is
a router-lookahead thread (PILOT) that prefetches next-layer experts,
exploiting that routing is ~72% predictable one layer ahead on a production
model.

A literature of post-hoc predictors (§2) shows expert activations can be
forecast from earlier hidden states with 93–97% accuracy. But all of it
treats the model as fixed: the predictor chases the model. We invert the
relationship — **the accuracy of an independent expert-activation predictor
becomes an explicit term in the MoE's own training loss** — so the backbone
is regularized toward being forecastable at multi-layer lookahead horizons
while the predictor simultaneously adapts to the model.

Contributions, each matched to an exhibit:

- **C1 — Predictability is a trainable, transferable backbone property.**
  Fresh post-hoc predictors trained on the frozen jointly-trained backbone
  recover the entire gain; an engine needs no co-training to exploit it
  (Table 1).
- **C2 — The gain is structural, not sharpening.** Entropy-penalty controls
  at any strength cannot reach the jointly-trained predictability at matched
  entropy; ~+2–2.4 pts of the advantage is pure structure (Figure 1).
- **C3 — The effect is robust and grows with training.** It survives init +
  data-order variance (±0.3 pts) and *increases* at 4× tokens, where the
  quality cost flips to a small gain (Table 2).
- **C4 — The accuracy converts to systems value, with a mapped boundary.**
  +3.2% tok/s and −31% misprefetch waste over the post-hoc control at
  Colibri-like geometry (+1.5%/−25% at ranking-control accuracies), gated by
  fetch_time ≤ compute_window; +5.0% mean tok/s in a real O_DIRECT engine;
  and the effect does not retrofit via LoRA onto a pretrained model (Table 3).

## 2. Related Work

**Post-hoc expert prediction (frozen model).** Pre-attention linear
predictors with ranking-aware losses reach 93–97.6% top-k accuracy
(arXiv 2511.10676); cross-layer gate inputs enable edge offloading with 99%
cache hits (Fate, 2502.12224); trace-trained sequence predictors raise cache
hit rates from 17% to 72% at 10% budget (MoE-Beyond, 2508.17137); PreScope
(2509.23638) adds globally scheduled prefetching. Apple’s LLM-in-a-Flash
(2312.11514), PowerInfer-2 (2406.06282), and MoE-Infinity address adjacent
offloading/caching problems. All of these train predictors *against* a fixed
model; none change the model.

**Training-time routing modification.** StickyMoE (2607.08780) adds a
differentiable routing-consistency loss for temporal reuse — the closest
philosophy to ours, but it optimizes *stickiness* (token t vs t−1), not
*forecastability from earlier layers*; we show in §4.6 that it does not buy
lookahead predictability in our reimplementation. Oracle-MoE redesigns the
routing input for emergent locality (no explicit objective). ReMoE
fine-tunes only the router post-hoc. Pre-gated MoE (ISCA'24, 2308.12066) is
the closest co-design: a pre-gate module that *replaces* next-layer routing,
co-trained end-to-end with the backbone — but with no predictor-accuracy
objective and no realized-router to evaluate forecasts against. Halfway
Speculative Decoding applies the same co-design pattern (train the target to
be draftable) in token space. No prior work, to our knowledge, adds the
accuracy of an *independent* expert-activation predictor as an explicit term
in the MoE's own loss at multi-layer lookahead; our stronger claim is
empirical: that predictability is a transferable backbone property (C1) with
a structural signature distinguishable from entropy reduction (C2).

**Our novelty statement, precisely scoped.** Prior systems work trains
predictors post-hoc against frozen models; prior training-time work optimizes
routing temporal locality; Pre-gated MoE co-trains a lookahead gate that
*replaces* routing. We are aware of no work that (a) makes an independent
predictor's accuracy an explicit training objective of the MoE itself, and
(b) demonstrates the resulting predictability is a recoverable property of
the backbone's representations. Where this paper's ranking-aware control
reimplements arXiv 2511.10676's protocol, we say so; all comparisons use our
implementation, not their published numbers.

## 3. Method

**Model.** GPT-style decoder (d=512, 12 layers, 16 experts, top-2, 341M
total / 99M active parameters), trained on FineWeb-edu (GPT-2 BPE,
document-shuffled stream with EOS, ctx 512, bf16 autocast).

**Predictor.** Per-(layer, horizon) heads from the pre-MoE hidden state at
layer l to the router decision at layer l+h, h ∈ {1, 2, 4}. Two families:
*linear* (single projection) and *ranking-MLP* (2-layer MLP with a margin
loss: the weakest true top-k logit must exceed the strongest intruder by a
margin), the latter reimplementing the protocol of arXiv 2511.10676.

**Joint loss.** L = L_LM + 0.01·L_balance + λ·L_pred (+ optional entropy or
sticky terms for controls). L_pred is a soft cross-entropy of predictor
logits against the realized top-k set (uniform 1/k mass). Gradients from
L_pred reach the backbone only through hidden representations (realized
top-k targets are argmax, non-differentiable) — the loss shapes
representations, not router weights directly.

**Isolation protocol (a first-class methodological contribution).** To test
whether predictability lives in the *backbone* rather than in predictor
co-adaptation, we freeze a trained backbone and train a *fresh* predictor on
it (2000–3000 steps), comparing against the same fresh-predictor protocol on
the baseline backbone. This mirrors deployment: an inference engine trains
its own predictor against a shipped model.

**Metrics.** hit@k per horizon (top-k set overlap |pred ∩ true| / k);
router entropy; adjacent-layer routing persistence; utilization max; val LM
loss on a fixed 16-batch held-out set drawn once before training and shared
by every run. Unless stated, numbers are n=3 seeds (init + data order) with
between-seed std.

## 4. Experiments

### 4.1 Setup and main result (Table 1)

Baseline (λ=0) vs joint (λ=0.3), 25M tokens, horizons {1,2,4}. Controls are
fresh post-hoc predictors on the frozen baseline backbone; the joint row is
the fresh-predictor-on-joint-backbone isolation configuration.

**Table 1. hit@k (between-seed std), fresh linear post-hoc predictor.**

| backbone | h=1 | h=2 | h=4 | val LM |
|---|---|---|---|---|
| baseline | 0.826 (0.003) | 0.797 (0.002) | 0.732 (0.004) | 5.767 (0.006) |
| **joint (λ=0.3)** | **0.888 (0.002)** | **0.865 (0.002)** | **0.796 (0.004)** | 5.790 (0.006) |

Under the stronger ranking-MLP control (single seed): baseline 0.903 /
0.863 / 0.794 vs joint **0.930 / 0.903 / 0.838** — the advantage compresses
to **+2.7 / +4.0 / +4.4 pts** but survives intact under an identical
predictor. We report +6–7 (linear) and +3–4 (ranking) as dual honest effect
sizes throughout. Paired quality cost: +0.023 ± 0.009 nats (Exp 6 protocol);
+0.033 ± 0.016 with data-order variance included.

### 4.2 The isolation test (C1)

At Tier A, a fresh post-hoc predictor on the frozen joint backbone recovers
the co-trained predictor's accuracy *exactly* (0.890/0.866/0.796 vs
0.887/0.863/0.793). The predictability is a property of the weights — the
deployable artifact.

### 4.3 Structure, not sharpening (C2, Figure 1)

An entropy-penalty ladder (λ_ent ∈ {0.005, 0.01, 0.05}) plus the baseline
and joint points, all evaluated with the ranking-MLP probe:

hit@1 vs entropy: baseline (2.26, 0.903) < λ0.005 (1.62, 0.912) <
λ0.01 (1.17, 0.916) < **joint (2.10, 0.930)**; near-collapse (0.28) reaches
only 0.883 under the linear probe. Interpolating to the joint model's own
entropy, sharpening alone yields ~0.906–0.910: **~+2.0–2.4 pts of the joint
advantage is pure structure** — routing that is simultaneously informative
and forecastable, not merely peaked.

### 4.4 Robustness and growth with training (C3, Table 2)

| | +hit@k vs control (h=1 / h=4) | quality cost |
|---|---|---|
| 25M tokens, init-only variance | +6.2 ± 0.4 / +6.4 ± 0.6 | +0.023 ± 0.009 nats |
| 25M tokens, init+data variance | +6.2 ± 0.3 / +6.8 ± 0.3 | +0.033 ± 0.016 nats |
| **100M tokens (4×)** | **+6.1 / +7.9** | **−0.019 nats (free)** |

At 4× training the advantage *grows* (h4 +7.9) and the quality cost
disappears; baseline routers sharpen naturally with training (entropy
2.26 → 1.98) and the joint model stays ahead.

### 4.5 Degeneracy checks

At λ=0.3: entropy −9% (no collapse), adjacent-layer persistence ≈ chance
(no temporal copying), utilization unchanged, token-ID→expert lookup probe
equal in both backbones (no memorization), marginal expert usage near-uniform
(popularity trivial predictor hits 0.15). At λ=1.0 the degenerate mode
appears measurably: entropy −28% and +0.62 nats quality cost — the frontier
turn, visible in our diagnostics.

### 4.6 Training-time baseline: StickyMoE (Table 3a)

A routing-consistency loss (our reimplementation of StickyMoE's objective)
*monotonically reduces* lookahead hit@k (posthoc probe: 0.81 → 0.77 across
λ_sticky ∈ {0.3, 1, 3}) while raising router entropy, and does not stack
with our loss. Scope: we tested only lookahead predictability; StickyMoE's
actual target — temporal-reuse cache locality — is orthogonal and untested
here.

## 5. Systems Analysis

**Simulator.** A trace-driven engine with an honest single-disk FIFO queue
(demand-priority; prefetch issued only into idle compute windows that fit
the fetch; bandwidth conservation is exact). Synthetic Colibri geometry:
75×256 experts, top-8, 19 MB int4 experts, 4 ms/layer dense compute,
C=2000-expert cache, Zipf+locality routing, predictor accuracies measured
in §4.

**Figure 2 (tok/s, prefetch h=1).** demand 0.876 → base-control 1.051 →
joint 1.085 → oracle 1.161 (linear-control accuracies): **+3.2% joint vs
base, −31% misprefetch waste (24.0 vs 34.6 MB/tok)**, ~31% of the
base-to-oracle headroom captured. At ranking-control accuracies (0.903 vs
0.930): **+1.5% and −25% waste** — the conversion paired with the honest
effect size. TTFT (cold cache): −11% from prediction generally, −1–3% joint
vs base — not the differentiator.

**Gating condition.** Across a 3×3 latency/bandwidth grid the joint-vs-base
uplift is stable (+2.9–3.2%) whenever prefetch fires; below ~5 GB/s (19 MB
expert), fetch time exceeds the compute window and *no* prefetch is possible:
**bandwidth, not latency, gates the regime** — a design constraint for
engines exploiting trained predictability.

**Real engine (Exp 11).** A purpose-built O_DIRECT disk-resident engine for
the toy model (192 experts × 2.88 MB, demand-priority disk queue, posthoc
predictor driving a prefetch thread, cache budget C=64): joint vs base
backbone, deployable configuration, 6 decode sequences: **+5.0% mean tok/s
(range −7% to +14.5%)**, consistent with the simulator's direction and
magnitude; per-sequence variance is large at toy scale, so we treat this as
qualitative confirmation and keep the quantitative claim on the simulator.

## 6. Boundary: the method is pretraining-time (Table 3b)

OLMoE-1B-7B (64 experts top-8, 3T pretraining tokens), LoRA r16 on
attention+experts plus trainable routers, λ=0.1, 25M tokens:

| post-hoc probe | on baseline backbone | on joint backbone |
|---|---|---|
| linear | 0.799 / 0.781 / 0.752 | 0.800 / 0.783 / 0.755 |
| ranking-MLP (177M) | 0.845 / 0.801 / 0.739 | 0.845 / 0.801 / 0.741 |

(h=1 / h=2 / h=4). A tie under both a weak and a strong probe: the
fine-tune did not move backbone predictability (router entropy 3.745 vs
3.748). We read this as **boundary mapping, not failure**: representations
must be shaped while plastic; the entire training-time prior-art family
(StickyMoE, Oracle-MoE) shares the property. Scope of the negative: one
seed, one λ, one LoRA rank; heavier fine-tuning (unfrozen blocks, full FT)
is untested. Note also the baseline itself: a post-hoc predictor on stock
OLMoE reaches 0.845 h=1 — post-hoc predictors are strong on real models,
and that is the bar any pretraining application must clear.

## 7. Limitations

Positive evidence is one toy from-scratch model (341M, 16 experts,
FineWeb-edu only, ≤100M tokens); systems numbers are simulator + toy-engine,
not a production engine; quality is val-LM only (no downstream tasks); n=3;
the ranking control is our reimplementation of 2511.10676's protocol;
StickyMoE was tested on our metric only; the Tier C negative is a single
configuration. At toy B=8/C=64 the joint engine row slightly exceeds the
oracle (97.9 vs 97.4 tok/s) — a queue-interaction effect consistent with the
waste ordering (oracle issues more prefetch traffic), noted for
completeness.

## 8. Conclusion

Predictability is trainable: an MoE optimized jointly for output quality and
for the accuracy of an independent expert-activation predictor becomes
measurably, structurally, and transferably more forecastable — and the gain
converts to throughput and waste reductions wherever an engine has disk
slack. The effect must be baked in during pretraining. For pretraining labs
targeting disk-resident deployment, predictability pressure is a cheap
(λ ≈ 0.3), quality-neutral auxiliary objective; for engine authors, the
result defines what a shipped model can make recoverable. Data, code, and
the full audit trail (including two red-team rounds and all negative
results) are available in the repository.
