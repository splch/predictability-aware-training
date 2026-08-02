#!/bin/bash
# Exp 12: downstream zero-shot quality eval on Tier A checkpoints
# (HellaSwag / ARC-Easy / ARC-Challenge / PIQA, n=1000 each).
export TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL=1
PY=.venv-rocm/bin/python
for CK in ckpt_A_base ckpt_A_lam0.3 ckpt_A_s1_base ckpt_A_s1_lam0.3 \
          ckpt_A_s2_base ckpt_A_s2_lam0.3 ckpt_A_lam1.0 ckpt_U_base ckpt_U_lam0.3; do
  echo "=== $CK ==="
  $PY -u eval_downstream.py --ckpt $CK.pt --device cuda --n 1000 \
    --out downstream_$CK.json || true
done
