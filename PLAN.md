# Plan: Validating Predictability-Aware Training

Date: 2026-07-21

## Hardware (this box)

| Component | Spec | Implication |
|---|---|---|
| CPU | AMD Ryzen AI MAX+ 395 "Strix Halo", 16C/32T Zen 5, AVX-512 + avx512_bf16 | ~2-4 TFLOPS bf16 (PyTorch/oneDNN) |
| GPU | Radeon 8060S iGPU (RDNA 3.5, 40 CU), unified memory, 512 MB carveout + GTT | ~15+ TFLOPS bf16 *if* ROCm works (gfx1151, ROCm >= 6.3, unofficial) |
| RAM | 128 GB unified (~102 GB free), ~256 GB/s | Bandwidth, not capacity, is the constraint |
| Disk | 6.4 TB free NVMe | Datasets + expert-streaming experiments |

No NVIDIA GPU, no ROCm/PyTorch preinstalled. Full fine-tuning of a real 30B MoE
is out (60 GB weights + ~480 GB optimizer states). From-scratch small MoE is in.

## Method: controlled small-scale ablation

Causal claim to test: joint predictability loss -> more predictable routing ->
higher prefetch hit rate at equal-or-better quality.

1. **Baseline**: tiny MoE trained normally (LM loss + load-balancing aux loss);
   then train a post-hoc linear/ranking predictor on frozen activations
   (replicates SOTA control, cf. arXiv 2511.10676).
2. **Treatment**: identical architecture/data/seed, predictor trained jointly,
   prediction accuracy folded into the LM loss; load-balancing loss retained as
   guardrail.
3. **Compare**: top-k hit rate at 1/2/4-layer horizons; simulated expert-cache
   hit rate under memory budgets; validation loss (quality must not degrade).

## Model tiers (sized to this box)

| Tier | Config | Total / Active | FLOPs/token (6NT) | Wall time / 1B tokens |
|---|---|---|---|---|
| A (recommended) | d=512, 12 layers, 16 experts, top-2 | ~341M / ~99M | ~6e8 | ~80 min CPU-off/GPU |
| B (fast iteration) | d=384, 8 layers, 8 experts, top-2 | ~94M / ~56M | ~2.4e8 | ~20 min GPU |
| C (stretch/transfer) | OLMoE-1B-7B fine-tune: LoRA on attn/MLP + trainable routers + predictor (NOT router-only — that collapses into ReMoE territory and can't test the backbone mechanism) | 7B frozen + adapters | fwd+bwd through 1.3B active | ~3.6h per 25M tokens at 15 TFLOPS eff |

Training memory for Tier A: 450M x 16 B (bf16 weights + fp32 AdamW master/m/v)
~= 7 GB — trivial.

## Data

FineWeb-edu (10BT sample), ~1-2B tokens per run (~20 tokens/active-param,
Chinchilla-ish). Streamed via HF `datasets`.

## End-to-end demo

After training: small mmap/pread inference harness (Colibri-style LRU cache +
predictor-driven prefetch horizon). NVMe streams a Tier-A expert (~2 MB) in
sub-ms. Measure real tok/s vs reactive baseline; convert hit-rate into the
throughput claim.

## Infra steps (in order)

1. [x] Install PyTorch: torch 2.13.0+cpu via uv venv (5.2 TFLOPS bf16) AND
       GPU path working: torch 2.11.0+rocm7.13 TheRock gfx1151 wheels from
       rocm.nightlies.amd.com/v2/gfx1151 (.venv-rocm) → 31 TFLOPS bf16 on the
       Radeon 8060S, ~18k tok/s Tier B vs 5.7k CPU. pytorch.org rocm wheels
       segfault on gfx1151 (VGPR bug, TheRock#2991). Must run under
       `sg render` (this session predates render-group membership) with
       TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL=1 and NO HSA override.
2. [x] Training harness: `model.py` (GPT-MoE + router + Switch bal loss +
       per-(layer,horizon) linear predictor heads + soft-CE predictability
       loss) and `train.py` (joint/posthoc modes, FineWeb-edu streaming,
       bf16 autocast). Smoke-tested: Tier B ~5,700 tok/s CPU; random-token
       hit@k = 0.25 = chance level (sanity pass); joint-mode grad flow OK.
3. [ ] Tier B smoke run: verify predictability loss moves routing entropy /
       one-layer-ahead hit rate.
4. [ ] Tier A A/B runs (baseline vs treatment), 1-2B tokens each.
5. [ ] Streaming inference benchmark + writeup.

## Joint loss sketch

L = L_LM + a * L_balance - b * L_predictability

where L_predictability is the predictor's top-k routing agreement (or its
negative cross-entropy against realized routing), computed per layer at chosen
horizon(s). Gradients from the predictor flow into the backbone so the model
itself becomes easier to predict. Ablate b over {0 (baseline), small, medium}.
