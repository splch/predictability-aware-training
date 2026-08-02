#!/bin/bash
# Exps 5/8: routing traces -> cache simulator (toy geometry, Colibri-geometry
# synthetic, TTFT, hardware sensitivity grid).
# Requires ckpt_A_posthoc.pt and ckpt_A_posthoc_on_joint.pt (run_tierA2.sh).
export TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL=1
PY=.venv-rocm/bin/python
$PY -u dump_traces.py --ckpt ckpt_A_posthoc.pt --out traces_base.npz \
  --batches 8 --device cuda || true
$PY -u dump_traces.py --ckpt ckpt_A_posthoc_on_joint.pt --out traces_joint.npz \
  --batches 8 --device cuda || true
# toy geometry, measured predictors (writes results_cache_sim.csv)
.venv/bin/python cache_sim.py \
  --traces base=traces_base.npz joint=traces_joint.npz \
  --budgets 16,32,64,128,192 --batches 1,8
# synthetic Colibri geometry + TTFT + sensitivity grid
.venv/bin/python cache_sim.py --synthetic
.venv/bin/python cache_sim.py --ttft
for LAT in 0.05 0.1 0.2; do
  for BW in 3500 5000 7000; do
    echo "--- lat=$LAT bw=$BW ---"
    .venv/bin/python cache_sim.py --synthetic --lat $LAT --bw $BW
  done
done
