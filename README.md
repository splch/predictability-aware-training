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

## Headline results (Tier B: 94M-param MoE, 25M tokens, FineWeb-edu)

hit@k = fraction of tokens where the predictor's top-k matches the router's
realized top-k, measured h layers ahead (chance = 0.25).

| lambda_pred | h=1 | h=2 | h=4 | val LM loss |
|---|---|---|---|---|
| 0 (baseline) + post-hoc predictor (SOTA control) | 0.847 | 0.815 | 0.735 | 6.285 |
| 0.03 | 0.856 | 0.822 | 0.738 | 6.286 |
| 0.1 | 0.873 | 0.843 | 0.760 | 6.284 |
| 0.3 | 0.900 | 0.877 | 0.800 | 6.299 |
| 1.0 | 0.916 | 0.895 | 0.848 | 6.521 |

- Predictability rises monotonically with lambda at every horizon.
- **Quality is free up to lambda ~= 0.3**; the frontier turns at lambda = 1.0.
- Full details in [RESULTS.md](RESULTS.md); literature landscape in
  [RESEARCH.md](RESEARCH.md); method in [PLAN.md](PLAN.md).

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
