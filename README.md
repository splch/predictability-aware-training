# Predictability-Aware Training for MoE Expert Prefetching

Can an MoE LLM be trained to be *predictable* — so that expert weights can be
prefetched from disk before the router asks for them?

Inference engines like [Colibri](https://github.com/JustVugg/colibri) already
run 744B-parameter MoE models on consumer hardware by keeping the dense
backbone resident and streaming each expert's weights from disk when the
router activates it. This is reactive: throughput is bottlenecked by fetch
latency. This project trains a small activation-prediction model **jointly**
with the LLM and folds its prediction accuracy into the LLM's training loss,
so the large model is optimized not just for output quality but for being
predictable.

## Headline results (Tier A: 341M-param MoE, 16 experts, 3 seeds)

hit@k = fraction of tokens where the predictor's top-2 matches the router's
realized top-2, measured h layers ahead (chance = 0.125). Between-seed std
in parens; post-hoc control = fresh linear predictor on the frozen baseline
backbone (the SOTA deployment pattern).

| | h=1 | h=2 | h=4 | val LM loss |
|---|---|---|---|---|
| baseline + post-hoc control | 0.826 (0.003) | 0.797 (0.002) | 0.732 (0.004) | 5.767 (0.006) |
| **joint predictability training (lambda=0.3)** | **0.888 (0.002)** | **0.865 (0.002)** | **0.796 (0.004)** | 5.790 (0.006) |

- **+6.2-6.8 pts of expert-predictability at every horizon** vs the linear
  post-hoc control (~15x seed noise); **+2.7-4.4 pts vs the stronger
  ranking-aware control** (MLP + margin loss, arXiv 2511.10676-style) — the
  honest effect size. Quality cost: **+0.033 +- 0.016 nats** (3 seeds,
  init+data variance).
- A StickyMoE-style temporal-consistency loss (the closest training-time
  prior art) does NOT buy lookahead predictability: it monotonically reduces
  hit@k and raises router entropy in our formulation (Exp 7b).
- Predictability is a **property of the backbone**: fresh post-hoc
  predictors on the frozen joint backbone recover it fully (linear 0.890;
  ranking-MLP 0.930 vs 0.903 on baseline) — an inference engine can exploit
  it without any co-training.
- Trace-driven cache simulation (honest disk-queue economics): where disk
  slack exists, this accuracy converts to **+3.2% tok/s and -31% misprefetch
  waste** over the post-hoc control at Colibri-like geometry (Exp 5,
  corrected after red-team round 2).
- **Boundary result (Tier C)**: the effect is pretraining-time only. A LoRA
  fine-tune on pretrained OLMoE-1B-7B raised the co-trained predictor but
  left the backbone's intrinsic predictability unchanged (isolation test:
  0.800 vs 0.799) — the method must be applied during pretraining, not
  retrofitted (Exp 10).
- Full details in [RESULTS.md](RESULTS.md) (7 experiments, 2 red-team
  rounds); literature landscape in [RESEARCH.md](RESEARCH.md); method in
  [PLAN.md](PLAN.md).

## Reproduce

```bash
uv venv .venv && uv pip install --python .venv/bin/python \
  --index-url https://rocm.nightlies.amd.com/v2/gfx1151/ --pre torch numpy
uv pip install --python .venv/bin/python datasets tiktoken

# joint training (treatment)
python train.py --tier B --horizons 1,2,4 --lambda-pred 0.3 --steps 6000 \
  --device cuda --save ckpt_joint.pt
# baseline + post-hoc predictor (control)
python train.py --tier B --horizons 1,2,4 --lambda-pred 0.0 --steps 6000 \
  --device cuda --save ckpt_base.pt
python train.py --tier B --horizons 1,2,4 --mode posthoc --ckpt ckpt_base.pt \
  --steps 2000 --device cuda
```

Developed on an AMD Ryzen AI Max+ 395 (Strix Halo, Radeon 8060S iGPU, gfx1151).
For CPU-only: install the CPU torch wheel and use `--device cpu`.

## Status

Early validation at 94M scale. Next: Tier A (450M params, 16 experts),
cache-hit to tok/s simulation, stronger SOTA control. See RESULTS.md.
