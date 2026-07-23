#!/bin/bash
# Remediation queue items 1-3 (red-team round 2):
#   1. ranking-aware SOTA control (mlp + ranking loss) on base & joint backbones
#   2. StickyMoE sweep + posthoc-on-sticky + stacking arm
#   3. data-order variance: 3 seeds x {base, joint, posthoc} on shuffled stream
export TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL=1
PY=.venv-rocm/bin/python

# 1. ranking-aware SOTA control (matched capacity/protocol, arXiv 2511.10676-style)
$PY -u train.py --tier A --horizons 1,2,4 --mode posthoc --pred-arch mlp --pred-loss ranking \
  --ckpt ckpt_A_base.pt --steps 3000 --batch 8 --eval-every 500 --device cuda \
  --save ckpt_A_posthoc_rank.pt || true
$PY -u train.py --tier A --horizons 1,2,4 --mode posthoc --pred-arch mlp --pred-loss ranking \
  --ckpt ckpt_A_lam0.3.pt --steps 3000 --batch 8 --eval-every 500 --device cuda \
  --save ckpt_A_posthoc_rank_on_joint.pt || true

# 2. StickyMoE sweep
for L in 0.3 1.0 3.0; do
  $PY -u train.py --tier A --horizons 1,2,4 --lambda-sticky $L --steps 6000 --batch 8 \
    --eval-every 500 --device cuda --save ckpt_A_sticky$L.pt || true
  $PY -u train.py --tier A --horizons 1,2,4 --mode posthoc --ckpt ckpt_A_sticky$L.pt \
    --steps 2000 --batch 8 --eval-every 500 --device cuda --save ckpt_A_sticky${L}_posthoc.pt || true
done
# stacking arm: sticky + predictability
$PY -u train.py --tier A --horizons 1,2,4 --lambda-sticky 1.0 --lambda-pred 0.3 --steps 6000 \
  --batch 8 --eval-every 500 --device cuda --save ckpt_A_stack.pt || true
$PY -u train.py --tier A --horizons 1,2,4 --mode posthoc --ckpt ckpt_A_stack.pt \
  --steps 2000 --batch 8 --eval-every 500 --device cuda --save ckpt_A_stack_posthoc.pt || true

# 3. data-order variance: 3 seeds on shuffled stream
for S in 0 1 2; do
  $PY -u train.py --tier A --horizons 1,2,4 --seed $S --lambda-pred 0.0 --steps 6000 \
    --batch 8 --eval-every 500 --device cuda --save ckpt_V_s${S}_base.pt || true
  $PY -u train.py --tier A --horizons 1,2,4 --seed $S --lambda-pred 0.3 --steps 6000 \
    --batch 8 --eval-every 500 --device cuda --save ckpt_V_s${S}_lam0.3.pt || true
  $PY -u train.py --tier A --horizons 1,2,4 --seed $S --mode posthoc --ckpt ckpt_V_s${S}_base.pt \
    --steps 2000 --batch 8 --eval-every 500 --device cuda --save ckpt_V_s${S}_posthoc.pt || true
done
