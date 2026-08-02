#!/bin/bash
cd /home/spencer/Repositories/predictability-aware-training
export TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL=1
PY=.venv-rocm/bin/python
# strong-intervention arm: 10x the Exp 10 predictability pressure
$PY -u train_olmoe.py --mode joint --lambda-pred 1.0 --horizons 1,2,4 \
  --steps 12000 --batch 4 --eval-every 1000 --device cuda --save ckpt_C_lam1.0.pt || true
# isolation test: fresh linear predictor on the frozen lam1.0 backbone
$PY -u train_olmoe.py --mode posthoc --ckpt ckpt_C_lam1.0.pt --horizons 1,2,4 \
  --steps 3000 --batch 4 --eval-every 500 --device cuda --save ckpt_C_lam1.0_posthoc.pt || true
# strong-probe control: ranking-MLP posthoc on the lam1.0 backbone
$PY -u train_olmoe.py --mode posthoc --pred-arch mlp --pred-loss ranking \
  --ckpt ckpt_C_lam1.0.pt --horizons 1,2,4 \
  --steps 3000 --batch 4 --eval-every 500 --device cuda --save ckpt_C_lam1.0_posthoc_rank.pt || true
echo CHAIN_DONE
