#!/bin/bash
# Tier C: OLMoE-1B-7B LoRA fine-tune A/B (baseline vs joint) + isolation tests
export TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL=1
PY=.venv-rocm/bin/python
$PY -u train_olmoe.py --mode joint --lambda-pred 0.0 --horizons 1,2,4 \
  --steps 12000 --batch 4 --eval-every 1000 --device cuda --save ckpt_C_base.pt || true
$PY -u train_olmoe.py --mode joint --lambda-pred 0.1 --horizons 1,2,4 \
  --steps 12000 --batch 4 --eval-every 1000 --device cuda --save ckpt_C_lam0.1.pt || true
# controls: fresh posthoc predictors on each frozen backbone (isolation test)
$PY -u train_olmoe.py --mode posthoc --ckpt ckpt_C_base.pt --horizons 1,2,4 \
  --steps 3000 --batch 4 --eval-every 500 --device cuda --save ckpt_C_posthoc.pt || true
$PY -u train_olmoe.py --mode posthoc --ckpt ckpt_C_lam0.1.pt --horizons 1,2,4 \
  --steps 3000 --batch 4 --eval-every 500 --device cuda --save ckpt_C_posthoc_on_joint.pt || true
