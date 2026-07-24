## Review

### Correct (verified by re-execution)

All Exp 5/8 numbers I re-ran reproduce **exactly**:

- `cache_sim.py --synthetic` (default lat/bw): demand 0.876, base-h1 1.051 (+20%), joint-h1 1.085 (+24%), oracle 1.161 (+33%), waste 34.6 → 24.0 MB/tok — matches RESULTS.md Exp 5 table verbatim.
- Derived claims check out arithmetically: joint-vs-base = 1.085/1.051 = **+3.24%** ("+3.2%" ✓); waste = 1 − 24.0/34.6 = **−30.6%** ("−31%" ✓); headroom capture = 0.034/0.110 = **30.9%** ("~31%" ✓); joint-vs-demand = **+23.9%** ("+24%" ✓).
- `--ttft`: 2398/2156/2133/2110 ms TTFT, p50 1142/947/920/862 ms — matches Exp 8 exactly; TTFT −11.1%, p50 −17.1%/−24.5% (base/oracle vs demand), joint-vs-base −1.1% TTFT / −2.9% p50 ("~1%/~−3%" ✓).
- Toy-geometry traces (B=8): 89.3/89.8, 97.3, 97.9, 216.2/216.4, 260.0, 262.3, 267.0 — all match.
- Sensitivity corners I spot-checked: `--bw 3500` → all policies collapse to demand 0.670 (fits-in-window guard blocks all prefetch, as claimed, `cache_sim.py:167`); `--bw 7000 --lat 0.2` → joint-vs-base +2.9%, joint-vs-demand +21.4%, inside the claimed +2.9–3.2% / +21–24% bands.

### Findings, ranked

**1. HIGH — The sim's headline delta rests on accuracy inputs that don't transfer to the stated geometry.** `cache_sim.py:152` injects Tier-A *16-expert, top-2* accuracies (0.826/0.888) as per-expert hit probabilities into a *256-expert, top-8* Colibri geometry. Chance level differs 4× (0.125 vs 0.031), and Colibri's own reported one-layer-ahead predictability on GLM-5.2 is **71.6%** — well below the sim's "base" 0.826. So the sim's *absolute* tok/s numbers are optimistic relative to a real PILOT baseline; the +3.2% joint-vs-base *delta* survives only if the 6.2-pt accuracy gap survives the geometry change, which nothing here measures. This is a modeling assumption, not a bug, but it is the weakest link in the quantitative claim.

**2. HIGH (for the demo plan, not the sim) — Exp 10 kills the OLMoE demo's ability to show the thesis.** RESULTS.md Exp 10: posthoc-on-joint == posthoc-on-base on OLMoE (0.800 vs 0.799) — the LoRA fine-tune did **not** make the backbone more predictable. Wiring a trained predictor into Colibri's `c/olmoe.c` can therefore only demonstrate *predictor-vs-PILOT* (which Colibri already claims at 71.6% and which arXiv 2511.10676 demonstrates better), never *trained-backbone-vs-stock-backbone*. The single most important number of the project — the +3–4 pt structural predictability gain converting to tok/s — is **unmeasurable on OLMoE**.

**3. MEDIUM — Sim policy vs real PILOT: sign prediction.** The sim is simultaneously optimistic and pessimistic vs Colibri's PILOT:
   - *Optimistic*: perfect idle-window knowledge (`prefetch()` bails if disk not fully idle, `cache_sim.py:79-82`), free demand-priority queue-jumping (`cache_sim.py:120-124`), i.i.d. mispredicts (`synth_pred` draws uniform-random wrong experts; real errors correlate with popular experts and with each other, inflating real waste).
   - *Pessimistic*: one fetch per idle window (`cache_sim.py:93`) and no prefetch while the disk is busy; Colibri's async I/O pool can keep multiple reads in flight. (Partially excused: at 19 MB / 5 GB/s = 3.9 ms ≈ the 4 ms window, one fetch per window is roughly the bandwidth limit anyway — the cap is nearly free *for this geometry*.)
   - Net: the *prediction-general* effect vs PILOT likely comes out **LARGER** in the demo (trained predictor ~0.83–0.89 vs router-lookahead ~0.72 is a bigger gap than the sim's base-vs-joint 6 pts). The *trained-backbone* effect (the actual contribution) will be **SMALLER-to-nonexistent** on OLMoE per finding 2.

**4. MEDIUM — Demo misleading-good hazards (question 4), concretely for this box:**
   - **Page cache**: OLMoE-1B-7B int4 is ~4 GB; this box has ~102 GB free RAM. After one warm-up pass every "disk fetch" is a page-cache hit and prefetch looks useless-or-amazing depending on ordering. Must use `O_DIRECT`/drop_caches + cold-start protocol, or the number is meaningless.
   - **Batch-union**: any B>1 amortizes unique-expert fetches (`cache_sim.py:113-116` models this); demo must fix B=1 or report B.
   - **Measurement window & distribution**: steady-state-only timing hides TTFT; evaluating on FineWeb-edu-like text (the predictor's training distribution) inflates hit rate vs OOD prompts, while PILOT's router lookahead is distribution-free — an unfair comparator unless prompts are held out.
   - **Thread contention**: a prefetch thread sharing Zen-5 cores with compute can shift tok/s either way; pin cores.

**5. MEDIUM — Top-3 technical risks for the planned OLMoE demo (question 3):**
   1. **Hidden-state access in the C engine**: the predictor heads consume pre-MoE hidden states per layer; `c/olmoe.c` almost certainly doesn't expose per-layer activations. Tapping them means invasive edits to the forward pass plus exporting the exact input-layernorm statistics the heads were trained with — the highest implementation risk.
   2. **Comparator fairness**: PILOT's lookahead is free (reuses router logits); the trained predictor needs hidden-state taps + per-layer matmuls + was trained on a specific distribution. A fair fight requires measuring PILOT's actual hit@8 on OLMoE *first* (it's 71.6% on GLM, unknown on OLMoE) and equalizing prompt sets.
   3. **Weight conversion**: HF → `c/olmoe.c` expert packing/quant layout; any subtle mismatch silently degrades quality and confounds "predictability cost" claims. Needs a PPL parity check against HF reference before any systems measurement.

**6. LOW — Sim internals are otherwise sound.** Batch-union dedup, FIFO disk model, demand-priority, waste accounting (evicted-unused + in-flight-at-end) all read correctly; no double-counting found. Minor: `synth_pred`'s uniform-random mispredicts and the C=64 toy-table oracle row being from the base trace (joint-trace oracle is 98.0/266.7) are presentational, not errors.

### Concrete design recommendation (cheapest path to ONE defensible end-to-end number)

Given finding 2, **drop OLMoE-on-Colibri as the thesis demo** and use the Tier A toy model with a purpose-built minimal engine — this is also what PLAN.md's "End-to-end demo" originally specified:

1. **Engine**: ~200 lines (C or Python+mmap is fine at 2.88 MB/expert scale): `O_DIRECT` pread, hard cache budget C=64 of 192 experts (matches the sim's toy geometry), one prefetch thread, demand-priority. No Colibri port needed.
2. **Arms** (identical engine, identical fresh linear predictor protocol trained on frozen backbones — the deployable configuration per Exp 4): (a) baseline backbone + posthoc predictor, (b) joint λ=0.3 backbone + posthoc predictor. Optionally (c) demand-only floor.
3. **Protocol**: cold cache (drop_caches + O_DIRECT), fixed held-out prompt set, B=1, report steady-state tok/s over ≥2k tokens.
4. **The number**: tok/s(b) / tok/s(a). The sim predicts ~+3% at C=64/128-equivalent geometry; a real-engine confirmation within noise of that converts the entire hit@k chain into one defensible end-to-end claim. Estimated cost: days, not the weeks the Colibri port (hidden-state taps, weight conversion, PILOT-fairness audit) would need to produce a number that can't show the thesis anyway.

Keep OLMoE/Colibri only as an optional *post-hoc predictors beat router-lookahead* demo, clearly labeled as not measuring training-induced predictability.

### Residual risks
- Sim's synthetic accuracies (16-expert → 256-expert transplant) remain unvalidated against any real 256-expert model (finding 1).
- Tier A demo inherits the scale-transfer caveat (toy model, 100M tokens) — defensible as a mechanism proof, not a production claim.
- I could not inspect Colibri source (`/tmp/pi-github-repos/JustVugg` is empty); PILOT analysis is based on the README summary in RESEARCH.md.