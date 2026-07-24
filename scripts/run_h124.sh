#!/bin/bash
set -e
export TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL=1
PY=.venv-rocm/bin/python
$PY -u train.py --tier B --horizons 1,2,4 --lambda-pred 0.1 --steps 6000 --batch 8 --eval-every 500 --device cuda --save ckpt_h124_joint.pt
$PY -u train.py --tier B --horizons 1,2,4 --lambda-pred 0.0 --steps 6000 --batch 8 --eval-every 500 --device cuda --save ckpt_h124_base.pt
$PY -u train.py --tier B --horizons 1,2,4 --mode posthoc --ckpt ckpt_h124_base.pt --steps 2000 --batch 8 --eval-every 500 --device cuda --save ckpt_h124_posthoc.pt
