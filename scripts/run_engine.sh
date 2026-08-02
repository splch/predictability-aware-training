#!/bin/bash
# Exp 11: real-engine end-to-end demo. Exports the deployable-configuration
# checkpoints (fresh posthoc predictor on each backbone) to engine format and
# measures steady-state decode tok/s, demand vs prefetch, 6 prompts.
# Requires ckpt_A_posthoc.pt and ckpt_A_posthoc_on_joint.pt (run_tierA2.sh).
PY=.venv/bin/python
$PY export_engine_model.py --ckpt ckpt_A_posthoc.pt --out engine_base
$PY export_engine_model.py --ckpt ckpt_A_posthoc_on_joint.pt --out engine_joint
for MODEL in engine_base engine_joint; do
  for POL in demand prefetch; do
    for PROMPT in 13 100 1000 42 500 5000; do
      $PY toy_engine.py --model $MODEL --policy $POL --budget 64 \
        --tokens 300 --prompt $PROMPT
    done
  done
done
