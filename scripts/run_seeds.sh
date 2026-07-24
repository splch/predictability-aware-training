#!/bin/bash
# 3-seed protocol at Tier A: baseline + joint(0.3) + posthoc-on-baseline per seed.
# Seed 0 already complete (ckpt_A_{base,lam0.3,posthoc}.pt) -> run seeds 1,2.
export TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL=1
PY=.venv-rocm/bin/python
for S in 1 2; do
  $PY -u train.py --tier A --horizons 1,2,4 --seed $S --lambda-pred 0.0 --steps 6000 \
    --batch 8 --eval-every 500 --device cuda --save ckpt_A_s${S}_base.pt || true
  $PY -u train.py --tier A --horizons 1,2,4 --seed $S --lambda-pred 0.3 --steps 6000 \
    --batch 8 --eval-every 500 --device cuda --save ckpt_A_s${S}_lam0.3.pt || true
  $PY -u train.py --tier A --horizons 1,2,4 --seed $S --mode posthoc \
    --ckpt ckpt_A_s${S}_base.pt --steps 2000 --batch 8 --eval-every 500 \
    --device cuda --save ckpt_A_s${S}_posthoc.pt || true
done
