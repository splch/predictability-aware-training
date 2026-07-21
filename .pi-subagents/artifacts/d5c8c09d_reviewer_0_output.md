All key suspicions verified. Compiling the review.

## Review

**Scope:** `model.py`, `train.py` (read-only; GPU busy). Verified claims numerically on CPU with `.venv/bin/python` (dense-reference equivalence tests, gradient-flow probes, optimizer probe).

### Correct (verified with evidence)

- **MoE dispatch math is exact.** The per-expert gather/scatter loop, top-k renormalization, and `w.sum(dim=-1)` weighting match a dense per-token reference to 1.1e-18 (`model.py:62-74`). Each token selects an expert at most once, so `w.sum(-1)` correctly picks that slot's renormalized weight.
- **hit@k is exactly top-k set overlap.** `(pred_topk.unsqueeze(-1) == true.unsqueeze(-2)).any(-1).float().mean(-1)` (`model.py:138-140`) equals `|pred ∩ true| / k` to 0.0 error vs a set-based reference. Since both sets have size k, precision = recall; README's "fraction of tokens where top-k matches" is accurate.
- **Horizons indexing is correct.** `Predictor` heads `h{h}_l{l}` consume `layer_hiddens[l]`; loss pairs them with `topks[l + h]` (`model.py:97-101, 130-134`). `enumerate(pred_logits[h])` aligns l correctly.
- **Posthoc mode truly freezes the backbone.** Probe: after `requires_grad = n.startswith("predictor.")` (`train.py:72-74`), zero trainable non-predictor params exist and every predictor param receives grad. Frozen backbone activations carry no graph (embedding output has `requires_grad=False`), so no leakage.
- **Joint-mode gradient flow works as intended.** `loss_pred.backward()` puts nonzero grad on `tok.weight` (3.2e-3) — the predictability loss does shape the backbone.
- **No train/eval contamination.** `fineweb_stream` yields each chunk exactly once; `evaluate()` consumes chunks via the same generator, so eval batches are data training will *never* see (they are skipped from training, not reused). Eval is genuinely unseen data.
- **Checkpoint ordering is safe:** `Model(cfg)` init → `load_state_dict` → freeze (`train.py:68-74`). The predictor is *not* re-initialized after loading. Horizon mismatches fail loudly via `strict=True`.
- fp32 softmax for routing/top-k (`model.py:66`); soft-target scatter is collision-free since top-k indices are unique per row (`model.py:132-133`); grad clipping skips `None` grads of frozen params.

### Fixed
- None (read-only task).

### Blocker
- None. The core scientific machinery (dispatch, bal loss as documented, predictor loss, hit@k, freeze, horizons) is correct.

### Findings

1. **Minor — load-balancing loss uses top-1 dispatch fraction while dispatch is top-2.** `model.py:76-78`: `f_i` is a one-hot on `argmax`, so an expert that is consistently everyone's *second* choice registers zero load despite carrying half its tokens. Measured: `bal(top1 f)=1.04` vs `bal(top2 f)=2.14` on identical inputs — different scale *and* gradient direction. The comment acknowledges the top-1 formulation (and Switch itself is top-1), and `lambda_bal=0.01` makes the practical impact small. Fix if desired: `f = F.one_hot(topk_idx, E).float().sum(1).mean(0) / top_k` then `bal = E * (probs.mean(0) * f).sum()`.

2. **Minor — eval is unseen but not a fixed held-out set.** `train.py:36-46, 104-106`: eval batches are the *next* chunks of the same unshuffled stream, adjacent in the corpus to training data (optimistic as "val LM loss"), differ between eval points (adds eval noise), and are silently dropped from training. There is also no EOS between documents (`train.py:33`, `buf.extend(enc.encode(row["text"]))`), so contexts span document boundaries with no separator, and a literal `<|endoftext|>` in web text would make tiktoken raise. Fix: hold out a fixed slice (e.g. first N shards or a second stream from a different shard offset) for eval; append EOT per document.

3. **Minor — baseline (`lambda_pred=0`) still weight-decays the predictor.** In joint mode the optimizer includes predictor params; with zero loss gradient, AdamW's decoupled decay shrinks them anyway (verified: head norm 0.2364 → 0.2350 over 200 steps; ≈8–9% shrink over a 6000-step run). Consequences: (a) posthoc control starts from a shrunken-random predictor rather than fresh init (likely harmless — near-uniform logits — but not the clean control it appears); (b) baseline-mode eval hit@k is measured with a decaying random predictor. Fix: filter predictor params out of the optimizer when `lambda_pred == 0`, or re-init `predictor.heads` after loading in posthoc mode.

4. **Minor — LR schedule resurrects past `args.steps`.** `train.py:80-81`: `0.5*(1+cos(pi*s/steps))` climbs from 0 back toward 1 for `s ∈ (steps, 2·steps)` if a run overshoots; and for `steps < 100` warmup never completes. Fix: `min(s / args.steps, 1.0)` inside the cosine.

5. **Minor — predictor loss cannot shape routing directly (mechanism caveat, not a bug).** Verified: `loss_pred` grad on `blocks[2].moe.router.weight` is `None` for the `h1_l0` term — gradients reach the backbone only through the pre-MoE hidden `h`, i.e. upstream representation shaping; realized top-k targets are argmax indices (non-differentiable), so no term pulls router decisions toward predictions. The README's "makes routing more predictable" is achieved only indirectly. Worth stating explicitly in RESULTS.md.

6. **Minor — posthoc control is one wrong flag away from leakage.** If `--ckpt` points at a *joint* checkpoint, the predictor loads already-trained (`train.py:69-70` loads the full state dict including `predictor.*`), silently invalidating the "post-hoc SOTA control." Fix: either re-initialize predictor heads after load in posthoc mode (also fixes finding 3), or assert the ckpt was trained with `lambda_pred=0`.

7. **Minor — bf16 tie-break inconsistency in bal loss.** `top1` uses `logits.argmax` on bf16 autocast logits (`model.py:77`) while dispatch uses fp32 `probs.topk`; near-ties can disagree. Negligible magnitude; fix: `probs.argmax(dim=-1)`.

8. **Minor — `--horizons` ≥ `n_layers` crashes at eval, not at startup.** `range(n_layers - h)` yields no heads; `evaluate()` then divides by `len(v)==0` (`train.py:46`). Fix: validate horizons in `main()`.