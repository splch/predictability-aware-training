"""Dump per-token expert routing traces (realized + predicted top-k) for the
cache simulator. Uses the first batches of the FineWeb stream = the fixed
held-out eval region (never trained on).
"""
import argparse

import numpy as np
import torch

from model import Model, TIER_A
from train import fineweb_stream


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--batches", type=int, default=8)
    p.add_argument("--device", default="cuda")
    args = p.parse_args()

    cfg = TIER_A
    cfg.horizons = (1, 2, 4)
    model = Model(cfg).to(args.device)
    model.load_state_dict(torch.load(args.ckpt, weights_only=True))
    model.eval()

    stream = iter(fineweb_stream(cfg, 8, args.device))
    true_topk, pred_topk = [], {h: [] for h in cfg.horizons}
    with torch.no_grad():
        for _ in range(args.batches):
            x, y = next(stream)
            with torch.autocast(args.device, dtype=torch.bfloat16):
                out = model(x, y, diag=True)
            # [B,T,L,k] -> [B*T, L, k]  (stack along T, NOT dim=1)
            true_topk.append(torch.stack(out["topks"], dim=2).reshape(-1, cfg.n_layers, cfg.top_k))
            for h in cfg.horizons:
                # pad missing tail layers with -1 (no prediction possible)
                L = cfg.n_layers
                pads = [torch.full_like(out["topks"][0], -1)] * h
                pt = torch.stack(out["pred_topk"][h] + pads, dim=2).reshape(-1, L, cfg.top_k)
                pred_topk[h].append(pt)

    np.savez(args.out,
             true_topk=torch.cat(true_topk).numpy(),
             **{f"pred_h{h}": torch.cat(v).numpy() for h, v in pred_topk.items()})
    print(f"saved {args.out}: {sum(t.shape[0] for t in true_topk)} tokens")


if __name__ == "__main__":
    main()
