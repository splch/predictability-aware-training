#!/bin/bash
# Exp 14: multi-seed ranking-aware SOTA control (Exp 7a was single-seed).
# Fresh ranking-MLP posthoc predictors on the seed-1/seed-2 backbone pairs;
# seed-0 numbers come from Exp 7a (same protocol, default seed).
export TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL=1
PY=.venv-rocm/bin/python
for s in 1 2; do
  $PY -u train.py --tier A --horizons 1,2,4 --mode posthoc --pred-arch mlp --pred-loss ranking \
    --ckpt ckpt_A_s${s}_base.pt --steps 3000 --batch 8 --eval-every 500 --device cuda \
    --seed $s --save ckpt_A_s${s}_posthoc_rank.pt || true
  $PY -u train.py --tier A --horizons 1,2,4 --mode posthoc --pred-arch mlp --pred-loss ranking \
    --ckpt ckpt_A_s${s}_lam0.3.pt --steps 3000 --batch 8 --eval-every 500 --device cuda \
    --seed $s --save ckpt_A_s${s}_posthoc_rank_on_joint.pt || true
done
echo CHAIN_DONE
