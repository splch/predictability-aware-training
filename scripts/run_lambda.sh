#!/bin/bash
export TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL=1
PY=.venv-rocm/bin/python
for L in 0.03 0.3 1.0; do
  $PY -u train.py --tier B --horizons 1,2,4 --lambda-pred $L --steps 6000 --batch 8 \
    --eval-every 500 --device cuda --save ckpt_lam${L}_joint.pt || true
done
