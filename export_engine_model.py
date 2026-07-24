"""Export a Tier A checkpoint into engine format:
  - backbone.pt: dense weights (embed, attn, lns, head) + router weights
  - experts.bin: raw bf16 expert FFN weights, w1||w2 per expert, at fixed offsets
  - predictor.pt: posthoc predictor heads (per-layer linear h->E)
"""
import argparse

import numpy as np
import torch

from model import Model, TIER_A

p = argparse.ArgumentParser()
p.add_argument("--ckpt", required=True)
p.add_argument("--out", required=True, help="output prefix")
args = p.parse_args()

cfg = TIER_A
cfg.horizons = (1, 2, 4)
model = Model(cfg)
sd = torch.load(args.ckpt, weights_only=True, map_location="cpu")
model.load_state_dict(sd)

backbone, predictor = {}, {}
for k, v in sd.items():
    if k.startswith("predictor."):
        predictor[k] = v
    elif ".moe.w1" not in k and ".moe.w2" not in k:
        backbone[k] = v

# experts.bin: layout eid = l*E+e, each expert = w1[e] (d x dff) then w2[e] (dff x d)
E, D, I = cfg.n_experts, cfg.d_model, cfg.d_ff
per_expert = (D * I + I * D) * 2  # bf16 bytes
blob = np.memmap(args.out + ".experts.bin", dtype=np.uint8, mode="w+",
                 shape=(cfg.n_layers * E * per_expert,))
off = 0
for l in range(cfg.n_layers):
    w1 = sd[f"blocks.{l}.moe.w1"].to(torch.bfloat16)  # [E, D, I]
    w2 = sd[f"blocks.{l}.moe.w2"].to(torch.bfloat16)  # [E, I, D]
    for e in range(E):
        w1b = w1[e].contiguous().view(torch.uint8).numpy().tobytes()
        w2b = w2[e].contiguous().view(torch.uint8).numpy().tobytes()
        blob[off:off + D * I * 2] = np.frombuffer(w1b, dtype=np.uint8)
        blob[off + D * I * 2:off + per_expert] = np.frombuffer(w2b, dtype=np.uint8)
        off += per_expert
blob.flush()

torch.save(backbone, args.out + ".backbone.pt")
torch.save(predictor, args.out + ".predictor.pt")
print(f"exported {args.out}: {cfg.n_layers * E} experts x {per_expert} bytes, "
      f"{len(backbone)} backbone tensors, {len(predictor)} predictor heads")
