#!/bin/bash
export TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL=1
PY=.venv-rocm/bin/python
$PY -u train.py --tier A --horizons 1,2,4 --lambda-pred 0.0 --steps 6000 --batch 8 \
  --eval-every 500 --device cuda --save ckpt_A_base.pt || true
$PY -u train.py --tier A --horizons 1,2,4 --lambda-pred 0.3 --steps 6000 --batch 8 \
  --eval-every 500 --device cuda --save ckpt_A_lam0.3.pt || true
$PY -u train.py --tier A --horizons 1,2,4 --lambda-pred 1.0 --steps 6000 --batch 8 \
  --eval-every 500 --device cuda --save ckpt_A_lam1.0.pt || true
$PY -u train.py --tier A --horizons 1,2,4 --lambda-sticky 0.1 --steps 6000 --batch 8 \
  --eval-every 500 --device cuda --save ckpt_A_sticky.pt || true
# control: fresh predictor on frozen BASELINE backbone
$PY -u train.py --tier A --horizons 1,2,4 --mode posthoc --ckpt ckpt_A_base.pt \
  --steps 2000 --batch 8 --eval-every 500 --device cuda --save ckpt_A_posthoc.pt || true
# decisive: fresh predictor on frozen JOINT backbone (backbone-predictability isolation)
$PY -u train.py --tier A --horizons 1,2,4 --mode posthoc --ckpt ckpt_A_lam0.3.pt \
  --steps 2000 --batch 8 --eval-every 500 --device cuda --save ckpt_A_posthoc_on_joint.pt || true
