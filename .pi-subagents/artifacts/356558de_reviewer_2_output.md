I have everything I need: README.md, RESEARCH.md, RESULTS.md (all 10 experiments + 2 red-team rounds), PLAN.md, plus git history and a spot-check of `train.py` flags. No files will be modified (read-only task).

## Review

### Correct (verified with evidence)

- **The evidence arc is real and internally consistent.** Exp 4 isolation test (0.890 vs 0.829 at h=1), Exp 6 3-seed replication (+6.2–6.8 pts, ~15× seed noise, +0.023±0.009 nats), Exp 7a ranking control (+2.7–4.4 pts under identical strong predictor), Exp 7b StickyMoE defused, Exp 9a sharpening frontier, Exp 9b 4×-training confound killed, Exp 10 boundary. Cross-checked numbers: Exp 5's "31% of base-to-oracle headroom" = (1.085−1.051)/(1.161−1.051) = 0.034/0.110 ✓; Tier A chance 0.125 = 2/16 ✓.
- **The remediation discipline is genuinely paper-grade.** Both round-2 blockers were fixed and re-measured (commit fbaa4c2); the headline was revised downward when the ranking control was added (+6–7 → +3–4 honest effect); "zero quality cost" was retracted to "+0.023–0.033 nats, small but real." Git history contains no overclaims.
- **README discloses both control numbers** (linear +6.2–6.8, ranking +2.7–4.4, "the honest effect size") — this is the right framing and must survive into the paper.

---

### (1) Strongest honest claim set for a workshop paper

Four claims survive; each maps to one figure/table:

| # | Claim | Carrying exhibit |
|---|---|---|
| C1 | **Predictability is trainable into the backbone as a transferable property** — a fresh post-hoc predictor on the frozen joint backbone recovers the full gain (+6–7 pts vs the same predictor on the baseline backbone), so an inference engine can exploit it with zero co-training. | **Table 1**: Exp 4 isolation rows + Exp 6/7c 3-seed error bars (posthoc-on-base vs posthoc-on-joint, h∈{1,2,4}, ±std). |
| C2 | **The gain is structural, not sharpening.** Entropy penalties at any strength cannot reach the joint backbone's predictability at matched entropy (~0.906–0.910 vs 0.930 under the ranking predictor); ~+2–2.4 pts is pure structure. | **Figure 1 (money figure)**: hit@k vs router-entropy scatter — the Exp 9a entropy-penalty ladder traces a frontier, the joint point sits above it, and even near-collapse (entropy 0.275) stays below the ranking-control number. |
| C3 | **The effect is robust and grows with training**: survives init+data-order variance (±0.3 pts), and at 4× tokens the advantage *grows* (+7.9 at h=4) while the quality cost flips to a small gain (−0.019 nats). | **Table 2**: paired per-seed effects (Exp 6/7c) + Exp 9b 100M-token row. |
| C4 | **The accuracy converts to system value, with a mapped boundary**: +3.2% tok/s and −31% misprefetch waste over the strongest control at Colibri-like geometry, gated by fetch_time ≤ compute_window; and the effect is pretraining-time only (LoRA on OLMoE leaves backbone predictability unchanged: 0.800 vs 0.799). | **Figure 2**: sim tok/s + waste bars (Exp 5 corrected, C=2000); **Table 3**: Tier C isolation flat-line. |

Note the honest headline is **+3–4 pts over the strongest post-hoc predictor** (+6–7 over the linear control, reported secondarily). Do not lead with +6–7.

### (2) Top 3 reviewer objections after remediation

1. **Scale/generality (highest risk).** All positive evidence is one toy from-scratch model (341M, 16 experts, 25M–100M tokens, FineWeb-edu only), and the single real-model test is *negative*. Real targets (GLM-5.2, OLMoE) have 64–256 experts and 71–98% baseline predictability. **Fix: careful writing** — frame as a *mechanism paper* ("proof that predictability is a trainable backbone property, with a systems-economic analysis"), limitations section prominent. A main-track version needs a ≥1B-token pretraining-scale arm; that's a new experiment, out of workshop scope.
2. **Sim-to-real gap.** The +3.2% tok/s comes from a synthetic simulator (Zipf+locality routing, synthetic geometry), and at 3.5 GB/s prefetch never fires — the regime is bandwidth-gated. **Fix: mostly writing** — report the gating condition explicitly as a contribution ("when does trained predictability pay?"), present the sensitivity grid (Exp 8: +2.9–3.2% stable whenever prefetch fires). The killer upgrade (a real Colibri integration) is an experiment; not required for a workshop if the sim is labeled as such.
3. **Quality accounting.** +0.02–0.03 nats is "small but real," measured only as val LM loss — no downstream-task eval, and the 9% router-entropy drop is unexplored for capability impact. **Fix: one cheap new experiment is prudent** (any small downstream probe or held-out-domain eval on the Exp 6 checkpoints); absent that, write the limitation honestly and lean on Exp 9b's quality-*gain* result at 4× tokens.

### (3) Tier C negative result: strength, with precise framing

**Strength — frame it as boundary mapping, not failure.** Rationale: (a) it converts the paper from "a trick that works on toys" to "a characterization of *when* predictability is trainable" — the isolation test gives a crisp mechanistic statement (backbone representations must be shaped while plastic; 25M tokens of LoRA-scale gradient on a 3T-token model moves routers by nothing: entropy 3.745 vs 3.748); (b) it preempts the most obvious reviewer question ("can I retrofit this onto OLMoE/Qwen3?") with data instead of silence; (c) the whole training-time prior art shares the same boundary — StickyMoE and Oracle-MoE are also pretraining-only, so this is a property of the *research direction*, not a defect of this method. Defuse the hostile reading ("method doesn't apply to any existing model") in one paragraph: the deployable artifact is the *fresh-posthoc-recovers-it* result (Exp 4) applied at a pretraining lab, plus the explicit open question (unfrozen-last-k blocks / full FT) as future work — do not hide that it was feasible-but-not-run.

### (4) Novelty claim stress test

The re-scoped claim ("no prior work adds the accuracy of an independent expert-activation predictor as an explicit term in the MoE's own loss, regularizing the backbone toward multi-horizon predictability while the predictor co-adapts") survives StickyMoE (empirically defused, Exp 7b), ReMoE (post-hoc, router-only), Oracle-MoE (architectural, no explicit objective), and Halfway SD (token space, cited proactively). **Two residual vulnerabilities:**

- **Pre-gated MoE (2308.12066) — the real remaining threat.** It co-trains a lookahead gate end-to-end with the backbone, so the backbone *does* receive co-adaptation pressure through the task loss. The current differentiator ("no explicit predictor-accuracy loss") is defensible but thin — a reviewer can argue implicit regularization via task loss is the same mechanism. Recommended hardening: lead with the cleaner distinction (the pre-gate *replaces* next-layer routing; our predictor is independent of and evaluated against the *realized* router), and **shift the novelty weight from the loss to the empirical findings** — the isolation result (predictability as a transferable backbone property) and the structure-vs-sharpening frontier are more novel and harder to contest than the loss term itself.
- **Oracle-MoE — moderate.** Its emergent locality may already yield high post-hoc predictability; untested here. One sentence acknowledging it as an alternative route to the same property (and that measuring predictor recovery on Oracle-MoE is future work) closes the hole.
- Also keep the Exp 7b caveat verbatim in the paper: StickyMoE was tested only on *lookahead predictability*, not its actual target (temporal-reuse cache locality), and via our reimplementation of its loss. Claiming to "beat" StickyMoE would be the one self-inflicted wound.

### (5) Paper vs repo

**In the paper:** Exps 4, 6, 7a/b/c, 9a/b, 5 (corrected), 8 (condensed), 10; the revised dual-control headline (+6–7 linear / +3–4 ranking); the retracted "zero cost" replaced by the honest +0.02–0.03 nats.

**Stays in the repo (omit from paper):** the two Exp 5 sim blockers (trace-dump axis scrambling; synthetic predictor misaligned by one layer) and the train_olmoe.py label double-shift bug. These were bugs in intermediate versions, caught and fixed *before* the reported numbers were produced; standard practice is to report final correct methods and results only. The repo's git history and RESULTS.md already preserve the full audit trail — that is the right disclosure surface. One nuance: because Exp 10 reports "final loss ~2.11, zero-shot ~1.83," ensure those numbers in the paper come exclusively from the post-fix run (they do, per commit 08c1a13).

**Additional findings:**
- **Note — README.md "Status" section is stale**: it still says "Early validation at 94M scale. Next: Tier A (450M…), cache-hit simulation, stronger SOTA control" — all three are now done (Tier A, Exp 5 sim, Exp 7a ranking control) and Tier C is complete. Must be updated before the repo goes public alongside a paper.
- **Note — Exp 5 anomaly needs an explanation before submission**: at toy geometry B=8/C=64, joint (97.9) *exceeds* oracle (97.4). Reviewers will catch a treatment beating the oracle; likely a queue-interaction effect (oracle issues more prefetch fetches that delay demand fetches), but it is currently unexplained in RESULTS.md.
- **Note — Exp 7b stacking comparison is muddled** ("0.864 < 0.884 joint-only posthoc… matched-stream joint posthoc = V_s0 co-trained 0.884"): the parenthetical mixes posthoc-on-stacking with co-trained numbers; rewrite as a clean table row for the paper.
- **Note — the ranking control is an in-house reimplementation** of arXiv 2511.10676's protocol, not their published numbers; say so explicitly in the paper.

### Paper skeleton (workshop, 6–8 pp)

1. **Introduction** — disk-resident MoE inference (Colibri); reactive fetch bottleneck; post-hoc predictors have a ceiling; question: is predictability *trainable*? Four contributions (C1–C4 above).
2. **Related Work** — two tables from RESEARCH.md (post-hoc predictors; training-time routing work) + the precise novelty statement with the Pre-gated MoE distinction made explicit.
3. **Method** — joint loss, horizons {1,2,4}, hit@k/entropy/val-LM metrics, and the **isolation protocol** (fresh post-hoc predictor on frozen backbones) as a first-class methodological contribution.
4. **Experiments** — 4.1 setup (Tier A, fixed eval, 3 seeds, shuffled stream); 4.2 main result (Table 1); 4.3 backbone-property isolation; 4.4 structure-not-sharpening (Figure 1); 4.5 robustness & scaling-with-training (Table 2); 4.6 StickyMoE baseline (Table 3).
5. **Systems Analysis** — cache sim, waste figure (Figure 2), bandwidth gating condition, TTFT in one paragraph.
6. **Boundary** — Tier C (Table 3b): pretraining-time only; framed per (3) above.
7. **Limitations** — toy scale, single domain, sim-not-real-engine, n=3, downstream quality unmeasured, StickyMoE tested on our metric only, ranking control is a reimplementation.
8. **Conclusion** — predictability is a trainable, structural, transferable backbone property; a call to pretraining labs.