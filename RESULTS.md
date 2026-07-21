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
