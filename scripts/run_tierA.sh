#!/bin/bash
export TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL=1
PY=.venv-rocm/bin/python
$PY -u train.py --tier A --horizons 1,2,4 --lambda-pred 0.0 --steps 6000 --batch 8 \
  --eval-every 500 --device cuda --save ckpt_tierA_base.pt || true
$PY -u train.py --tier A --horizons 1,2,4 --lambda-pred 0.3 --steps 6000 --batch 8 \
  --eval-every 500 --device cuda --save ckpt_tierA_lam0.3.pt || true
$PY -u train.py --tier A --horizons 1,2,4 --lambda-pred 1.0 --steps 6000 --batch 8 \
  --eval-every 500 --device cuda --save ckpt_tierA_lam1.0.pt || true
$PY -u train.py --tier A --horizons 1,2,4 --mode posthoc --ckpt ckpt_tierA_base.pt \
  --steps 2000 --batch 8 --eval-every 500 --device cuda --save ckpt_tierA_posthoc.pt || true
