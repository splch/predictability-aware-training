# Results

## Experiment 1: Tier B smoke A/B, horizon=1 (2026-07-21)

Setup: Tier B (94M total / 56M active, 8 experts top-2), FineWeb-edu stream,
6000 steps x 8 x 512 = ~25M tokens, bf16 on Radeon 8060S (TheRock rocm7.13),
~19.7k tok/s. Control = baseline backbone (lambda_pred=0) + posthoc-trained
predictor on frozen weights (2000 steps, plateaued by step 500).

| Run | hit@k, 1 layer ahead | val LM loss |
|---|---|---|
| Baseline + posthoc predictor (SOTA control) | 0.849 | 5.942 |
| Joint predictability training (lambda_pred=0.1) | **0.892** | 5.937 |

- Joint training buys **+4.3 pts** of one-layer-ahead expert hit rate at
  **zero LM-quality cost** (5.937 vs 5.942, within noise).
- Direction supports the proposal; scale is small and horizon=1 is the
  easiest case (pre-attention activations are already highly informative).

Next: horizons {1,2,4} — disk latency needs 2+ layers of lead, and prior work
shows frozen-model predictability drops sharply with horizon; the joint
advantage should grow there. Also lambda_pred ablation.

Artifacts: ckpt_tierB_{base,joint,posthoc}.pt; run_tierB_{base,joint,posthoc}.log

## Experiment 2: Tier B, horizons {1,2,4} (2026-07-21)

Same setup as Exp 1, predictor now predicts 1, 2, and 4 layers ahead.

| hit@k | h=1 | h=2 | h=4 | val LM loss |
|---|---|---|---|---|
| Baseline + posthoc predictor | 0.847 | 0.815 | 0.735 | 6.285 |
| Joint (lambda_pred=0.1) | **0.873** | **0.843** | **0.760** | 6.284 |

- Joint training wins at every horizon (+2.5 to +2.8 pts), again at zero
  LM-quality cost (6.284 vs 6.285).
- Both conditions degrade with horizon (chance = 0.25); the joint advantage is
  roughly constant, not growing — at this scale the frozen model is already
  fairly predictable, so headroom is limited.
- Note: joint evals fluctuated mid-run (h4 peaked at 0.802 @ step 4500), so
  same-step final evals are used for the comparison.
- Ops note: ROCm teardown aborts the interpreter AFTER checkpoint save — chain
  scripts must tolerate non-zero exits (no `set -e`).

Implications for next experiments:
1. lambda_pred ablation ({0.03, 0.1, 0.3, 1.0}) — is there more headroom, and
   where does quality start paying for predictability?
2. Tier A (450M params, 16 experts) — more experts = harder prediction =
   potentially larger joint advantage (chance level drops to 0.125).
3. Stronger control: posthoc predictor with ranking-aware loss + longer
   training, to match arXiv 2511.10676's SOTA protocol.
4. Downstream: feed hit rates into the cache-hit simulation to convert
   +3 pts into tok/s under realistic NVMe latency.

Artifacts: ckpt_h124_{base,joint,posthoc}.pt; run_h124*.log

## Experiment 3: lambda_pred ablation {0, 0.03, 0.1, 0.3, 1.0} (2026-07-21)

Same Tier B setup, horizons {1,2,4}, 6000 steps each.

| lambda_pred | h=1 | h=2 | h=4 | val LM loss |
|---|---|---|---|---|
| 0 (baseline) + posthoc predictor | 0.847 | 0.815 | 0.735 | 6.285 |
| 0.03 | 0.856 | 0.822 | 0.738 | 6.286 |
| 0.1 | 0.873 | 0.843 | 0.760 | 6.284 |
| 0.3 | 0.900 | 0.877 | 0.800 | 6.299 |
| 1.0 | 0.916 | 0.895 | 0.848 | 6.521 |

Findings:
- Clean dose-response: predictability rises monotonically with lambda at every
  horizon. h=4 gains the most in absolute terms (+11.3 pts from control at
  lambda=1.0).
- **Quality is free up to lambda=0.3** (6.299 vs 6.285 baseline, ~noise); at
  lambda=1.0 the model pays +0.24 nats — the predictability/quality frontier
  turns. Sweet spot at this scale: lambda ~= 0.3.
- Joint lambda=0.03 roughly matches the posthoc control; everything above that
  is genuine training-induced predictability.

Artifacts: ckpt_lam{0.03,0.3,1.0}_joint.pt; run_lambda.log

## Red-team (2026-07-21, three subagents: code / methodology / novelty)

**Code review**: no critical bugs. MoE dispatch verified exact vs dense
reference (1e-18); hit@k = exact top-k set overlap; posthoc freeze verified;
joint-mode predictor gradients reach the backbone (through pre-MoE hidden
states only — realized top-k targets are argmax, non-differentiable, so the
loss shapes *representations*, not router weights directly). Minor findings
FIXED in code: predictor weight-decayed during baseline runs; posthoc mode
could inherit co-trained predictor heads from a joint ckpt (now re-initialized);
LR schedule could resurrect past args.steps; load-balancing loss used top-1
fraction with top-2 dispatch (now top-k fraction); horizons validated at
startup; eval now uses a FIXED held-out 16-batch set drawn once before
training (identical across runs, never trained on) + routing diagnostics
(entropy, adjacent-layer persistence, util_max) at every eval.

**Methodology red-team — strongest attacks and remediation**:
1. CRITICAL: "zero quality cost" deltas (~0.005-0.014 nats) were below the
   eval noise floor — two identical baseline runs measured 5.942 vs 6.285
   (different stream chunks + evals consuming the training stream). FIXED via
   fixed held-out eval set; multi-seed still TODO before strong claims.
2. CRITICAL: co-trained-predictor hit@k doesn't isolate backbone
   predictability from predictor co-adaptation. FIX (queued in Tier A chain):
   posthoc fresh-predictor run on the frozen JOINT backbone vs on the frozen
   BASELINE backbone.
3. HIGH: degenerate predictability not ruled out (router could get simpler,
   not more informed). FIX: entropy/persist/util_max diagnostics now logged;
   watch for collapse at high lambda.
4. HIGH: hit@k is not the prefetch objective (cache budget, misprefetch
   bandwidth cost, LRU/hot-store baseline). TODO: cache simulator (PLAN step 5).
5. HIGH: posthoc control undermatched (linear+softCE vs 2511.10676's 2-linear
   +ranking loss; 2000 vs 6000 steps; stream offset). Partially addressed by
   fixed eval; ranking-loss control TODO.
6. MEDIUM: scale transfer (8 experts, 25M tokens, 75x undertrained vs
   Chinchilla) — Tier A tests E=16; Tier C (OLMoE) anchors to real scale.

**Novelty red-team**: claim WEAKENED not dead — see RESEARCH.md "Training-time
prior art" + "Re-scoped novelty claim". StickyMoE (2607.08780) is the closest
prior art and a mandatory baseline: added as --lambda-sticky (routing
consistency loss), with a sticky-only arm in the Tier A chain. Pre-gated MoE
citation fixed (2308.12066, not 2205.10034 = SE-MoE).

**Consequence for prior results**: Experiments 1-3 used the old eval protocol
(stream-chunk evals); treat their LM-loss deltas as provisional. Hit@k
directions are consistent across runs but exact values will be re-measured
under the fixed eval set at Tier A.

## Experiment 4: Tier A, 16 experts, fixed-eval protocol, six arms (2026-07-21)

Tier A (341M total / 99M active, d=512, 12 layers, 16 experts top-2),
horizons {1,2,4}, 6000 steps (~25M tokens), fixed 16-batch held-out eval set.
Chance hit@k = 0.125.

| arm | h=1 | h=2 | h=4 | val LM | router entropy |
|---|---|---|---|---|---|
| baseline (lambda=0) | 0.122 | 0.125 | 0.123 | 5.761 | 2.261 |
| posthoc on baseline (SOTA control) | 0.829 | 0.798 | 0.735 | — | 2.261 |
| **joint lambda_pred=0.3 (co-trained)** | **0.887** | **0.863** | **0.793** | 5.793 | 2.065 |
| **posthoc on frozen JOINT backbone** | **0.890** | **0.866** | **0.796** | — | 2.065 |
| joint lambda_pred=1.0 | 0.913 | 0.893 | 0.868 | 6.381 | 1.627 |
| sticky-only lambda_sticky=0.1 | (untrained pred) | | | 5.772 | 2.392 |

Findings:
1. **Backbone-predictability isolation test PASSES** (red-team critical attack
   #2): a fresh post-hoc predictor on the frozen joint backbone recovers the
   co-trained predictor's accuracy exactly (0.890/0.866/0.796 vs
   0.887/0.863/0.793), beating posthoc-on-baseline by ~6 pts at every horizon.
   The predictability is a property of the MODEL — deployable: an inference
   engine can recover it with its own cheap predictor.
2. **Joint advantage grew with expert count**: +5.8-6.5 pts at 16 experts vs
   +2.5-2.8 pts at 8 experts (Tier B). Consistent with the method mattering
   more as routing gets higher-dimensional.
3. **Quality frontier**: lambda=0.3 costs +0.032 nats (5.793 vs 5.761);
   lambda=1.0 costs +0.62 nats (vs +0.24 at Tier B) — the quality-free window
   narrows with scale. Sweet spot remains lambda ~= 0.3.
4. **Degeneracy is measurable**: lambda=1.0 collapses router entropy -28%
   (1.627 vs 2.261) with NO rise in adjacent-layer persistence (0.111) —
   predictability via stereotyped routing, not temporal copying. lambda=0.3:
   only -9% entropy, persistence/utilization unchanged. Operational
   definition: good predictability = high hit@k at near-baseline entropy.
5. StickyMoE arm inconclusive: lambda_sticky=0.1 too weak (entropy rose,
   nothing moved), and its hit@k needs a posthoc-on-sticky run to evaluate.
   TODO: stronger lambda_sticky sweep + posthoc evaluation + stacking test.

Open items before scaling investment:
- multi-seed (3x) at lambda=0.3 to put error bars on the +0.032-nat cost and
  the +6pt gains
- ranking-aware SOTA control (2511.10676 protocol, 2-layer head)
- cache simulator: hit@k -> tok/s under NVMe latency and cache budgets,
  vs LRU/hot-store baselines (PLAN step 5)
- sticky baseline done properly (sweep + posthoc eval)

Artifacts: ckpt_A_{base,lam0.3,lam1.0,sticky,posthoc,posthoc_on_joint}.pt;
run_tierA2.log
