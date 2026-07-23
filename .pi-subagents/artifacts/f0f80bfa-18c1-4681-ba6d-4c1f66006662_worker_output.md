Implemented `train_olmoe.py` (new file, no other source files touched; the parent swept it into commit `545f206` mid-flight and the committed copy includes all final fixes — verified).

## What was built

**LoRA** (`LoRALinear` on attn q/k/v/o, `LoRAExperts` on fused expert weights) + trainable routers + per-(layer,horizon) linear **Predictor**, with joint/baseline/posthoc modes mirroring `train.py`'s semantics (soft-CE predictability loss, hit@k, entropy/persist/util_max diagnostics, fixed 16-batch eval, shuffled+EOS FineWeb stream parameterized by OLMoE's tokenizer, AdamW/cosine/clip, predictor-agnostic ckpt loading with re-init in posthoc).

## OLMoE-specific gotchas (transformers 5.14.1)
- `model.layers.{i}.mlp` = `OlmoeSparseMoeBlock`; hook `mlp` (pre-hook) for the predictor's input hidden state.
- `mlp.gate` = `OlmoeTopKRouter`; its forward returns **`(router_logits, router_scores, router_indices)`** — hook output gives both entropy logits and realized top-k targets.
- `mlp.experts` = `OlmoeExperts` with **fused 3-D params** (`gate_up_proj [E, 2I, H]`, `down_proj [E, H, I]`) and a per-expert Python loop — no `nn.Linear` to wrap, so `LoRAExperts` re-implements the loop faithfully with `W + B@A` deltas.
- Config attrs: `num_local_experts`, `num_experts_per_tok`; attention has `q_norm`/`k_norm` (left frozen); `tie_word_embeddings=False`.
- Download of `allenai/OLMoE-1B-7B-0924` finished during the task (`olmoe_download.log`: "OLMoE downloaded").

## Acceptance evidence

**Joint smoke** (tiny random-weight config, 20 steps): `hit@k [h1: 0.257 | h2: 0.258] | ent 2.057` — chance level (2/8) and ln(8) entropy on random data, checkpoint saved/loaded.

**Gradient probe, joint mode** — `loss_pred.backward()`:
```
lora_B grad=0.26979  gate.weight grad=0.05168  predictor grad=6.07239
frozen attn base grad=None  embed grad=None
```

**Posthoc probe** — full-loss backward: `nonzero non-predictor grads: []`, `predictor params with grad: 5/5`.