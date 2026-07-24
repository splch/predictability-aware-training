#!/bin/bash
# Corrected entropy-matched controls + 100M-token undertraining 3-arm run
export TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL=1
PY=.venv-rocm/bin/python
for E in 0.005 0.01; do
  $PY -u train.py --tier A --horizons 1,2,4 --lambda-ent $E --steps 6000 --batch 8 \
    --eval-every 500 --device cuda --save ckpt_A_ent$E.pt || true
  $PY -u train.py --tier A --horizons 1,2,4 --mode posthoc --ckpt ckpt_A_ent$E.pt \
    --steps 2000 --batch 8 --eval-every 500 --device cuda --save ckpt_A_ent${E}_posthoc.pt || true
done
$PY -u train.py --tier A --horizons 1,2,4 --lambda-pred 0.0 --steps 24000 --batch 8 \
  --eval-every 2000 --device cuda --save ckpt_U_base.pt || true
$PY -u train.py --tier A --horizons 1,2,4 --lambda-pred 0.3 --steps 24000 --batch 8 \
  --eval-every 2000 --device cuda --save ckpt_U_lam0.3.pt || true
$PY -u train.py --tier A --horizons 1,2,4 --mode posthoc --ckpt ckpt_U_base.pt \
  --steps 4000 --batch 8 --eval-every 500 --device cuda --save ckpt_U_posthoc.pt || true
