All checks complete. Every published table was traced to the logs or reproduced by re-running the simulator. Here is the full-arc audit.

## Review

### Correct (verified with evidence)
- **All published numbers match the logs.** Exp 1–3 (run_tierB_*.log, run_h124*.log, run_lambda.log), Exp 4 (run_tierA2.log: posthoc-on-joint 0.890/0.866/0.796 vs co-trained 0.887/0.863/0.793), Exp 6 (run_seeds.log; seed-0 values correctly reused from Exp 4: e.g. joint h1 = mean(0.887, 0.890, 0.888) = 0.888 ✓), Exp 7a/7b/7c (run_remediation.log, all ~25 table entries match, including recomputed stds: 7c quality +0.033±0.016 ✓), Exp 9a/9b (run_ent_rank.log, run_undertrain2.log), Exp 10 (run_tierC2.log: 0.799/0.800/0.842, ent 3.748/3.745 ✓).
- **Exp 5 and Exp 8 reproduced exactly** by re-running `cache_sim.py --synthetic` and `--ttft` on CPU: 1.051→1.085 (+3.2%), waste 34.6→24.0 (-31%), TTFT 2398/2156/2133/2110ms — all match.
- **Cross-protocol defused where it matters most (attack 2, partial):** the Exp 7a ranking numbers (0.903/0.930) were *re-measured on the new shuffled+EOS eval set* (eval lm 5.914/5.961 in run_remediation.log:158,319 vs Exp 4's 5.761/5.793 on the same ckpts), so the 7a/9a hit@k comparisons share one eval distribution. LM losses are correctly declared non-comparable (RESULTS.md Exp 7 header).

### Blocker
- None that invalidate published tables. Two HIGH items below should be resolved *in the writeup text* (one may need one cheap run).

### Findings, ranked

**1. HIGH — Tier C negative is underpowered for the "pretraining-time only" claim (attack 1).** RESULTS.md Exp 10 and README boundary bullet rest on one seed × lambda=0.1 × LoRA r16 × 25M tokens × one model. Evidence it's stacked toward a null:
- Exp 3 dose-response: lambda=0.1 delivers less than half the lambda=0.3 effect (h1 +2.6 vs +5.3 over control); the single Tier C shot used the *weak* dose against a router at 3.745/4.16 nats (90% of max entropy).
- Checkpoint diff (my probe): routers *were* trainable and received gradient (max |Δgate.weight| = 0.020 across the 16 layers; max |ΔLoRA| = 0.038) yet every behavioral metric is flat — the intervention was applied but behaviorally inert at 25M tokens (= 0.0008% of the model's 3T pretraining). This supports "intervention too weak" as strongly as "pretraining-time only".
- Loss-scale check: lambda·pred = 0.27 vs LM 2.11 at Tier C is actually a *larger* relative push than Tier A's 0.36/5.79, so the null is more about plasticity/curvature than loss balance — but that argument is currently absent from RESULTS.
- Reviewer-demanded controls: lambda sweep {0.3, 1.0}, LoRA rank {64, 256} or unfrozen-last-k-blocks (RESULTS itself flags this as feasible), ≥2 seeds. Minimum fix: reword claim to "a 25M-token LoRA-r16 fine-tune at lambda_pred=0.1 does not transfer predictability" and drop the unqualified "pretraining-time only" from the README bullet.

**2. HIGH — systems headline (f) is anchored to the discredited linear control (attacks 2+4).** Exp 5/8 feed Exp 6 *linear-control* accuracies (0.826 vs 0.888) into the sim. Exp 7a showed the honest gap under the ranking control is +2.7pt (h=1), not +6.2pt — roughly halving the accuracy delta that drives "+3.2% tok/s / -31% waste". A reviewer pairing claim (a)'s "+3-4 pts vs ranking control" with claim (f) will ask for the conversion at ranking-control accuracies (0.903 vs 0.930). Fix: one `cache_sim.py --synthetic` run with those two accuracies (minutes on CPU) and report both conversions.

**3. MEDIUM — Tier C isolation test cannot distinguish "backbone unchanged" from "predictor ceiling" (attack 5).** Only the *linear* posthoc head was run at Tier C; both curves plateau at ~0.800 (step 2500→3000: +0.001, run_tierC2.log:1541,1567). The decisive, cheap measurement: run the Exp 7a ranking-MLP posthoc on **both** ckpt_C_base and ckpt_C_lam0.1 — Tier A precedent (0.903 vs 0.930) shows the stronger head lifts both backbones and *preserves* a real gap, so a persistent tie at Tier C would be strong evidence for "unchanged", while joint > base would expose a ceiling artifact. Without it, the negative rests on the weakest probe in the repo.

**4. MEDIUM — "co-adaptation gap is not deployable value" is overclaimed (attack 3).** The +4.3pt gap (0.842 vs 0.800) is real co-adaptation, not predictor undertraining (at matched 3000 steps: co-trained 0.828 vs posthoc 0.799, run_tierC2.log:776,1404). But the co-trained predictor is saved inside ckpt_C_lam0.1.pt (41 predictor keys) and an engine *could* deploy that exact predictor — the 0.842 is deployable if the predictor ships with the model and matches the serving stack. Correct framing: the gap is "not a backbone property and does not survive predictor re-training", which is what matters for the paper's thesis; "not deployable" should be deleted or qualified.

**5. MEDIUM — Exp 9a table mixes backbones and mislabels a predictor (attack 2).** RESULTS.md 9a row "joint lambda_pred=0.3 (V_s0)": entropy 2.102 and "0.884 (linear)" come from V_s0_lam0.3 (new stream), but "0.930 (ranking)" comes from ckpt_A_lam0.3 (old stream, different seed). Also "0.884 (linear)" is the *co-trained* predictor — no linear posthoc-on-joint run exists for any V backbone (checked run_remediation.log; only posthoc-on-V-baselines). Same issue in 7b: "matched-stream joint posthoc = V_s0 co-trained 0.884" cites a run that doesn't exist; the stacking conclusion (0.864 < 0.884) compares posthoc-on-stack to co-trained-joint. Direction is safe (Tier A: posthoc-on-joint ≥ co-trained), the citation is wrong. Fix: label co-trained as co-trained, and either run posthoc-on-V_s0-lam0.3 or remove the pseudo-citation.

**6. LOW — wording/provenance nits.**
- Exp 10: "LoRA converged by step 1000" — the LM loss did (2.116→2.113); the co-trained predictor kept improving 0.765→0.842 through step 9000 (run_tierC2.log:674,1082). Reword.
- Exp 9b: posthoc control ran 4000 steps vs the 2000-step standard everywhere else (run_undertrain2.log:3516 is the source of the published 0.839/0.797/0.708); conservative direction, but disclose.
- README pairs the Exp 6 table (init-only seeds, +0.023 nats implied) with the Exp 7c quality cost (+0.033±0.016, init+data); both are cited correctly but a reader will pair table-with-bullet and find two costs.
- Exp 10 "zero-shot ~1.83" is train-stream loss (run_tierC2.log step 8260 lm 1.835), not a held-out zero-shot eval.
- Exp 9b "U baseline 0.125 (chance)" vs logged 0.121 — labeled as chance, acceptable.

### Notes
- git log is clean (conventional commits, red-team fixes documented in-place; no history of overclaims).
- The double-shift bug narrative is corroborated (run_tierC.log step 0 lm 9.457 vs run_tierC2.log 2.231; commit 08c1a13).