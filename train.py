"""Training entry point: joint / baseline / posthoc-predictor modes.

Examples:
  python train.py --tier B --steps 50 --data random            # smoke test
  python train.py --tier A --lambda-pred 0.1 --steps 20000     # treatment
  python train.py --tier A --lambda-pred 0.0 --steps 20000     # baseline
  python train.py --mode posthoc --ckpt ckpt_base.pt           # frozen-backbone predictor
"""
import argparse, math, time

import torch

from model import Config, Model, TIER_A, TIER_B


def get_batch_random(cfg, bs, device):
    x = torch.randint(0, cfg.vocab_size, (bs, cfg.ctx + 1), device=device)
    return x[:, :-1], x[:, 1:]


def fineweb_stream(cfg, bs, device, split="train"):
    import tiktoken
    from datasets import load_dataset
    enc = tiktoken.get_encoding("gpt2")
    ds = load_dataset("HuggingFaceFW/fineweb-edu", "sample-10BT",
                      split=split, streaming=True)
    buf, need = [], bs * (cfg.ctx + 1)
    while True:
        for row in ds:
            buf.extend(enc.encode(row["text"]))
            while len(buf) >= need:
                chunk, buf = buf[:need], buf[need:]
                t = torch.tensor(chunk, dtype=torch.long, device=device).view(bs, cfg.ctx + 1)
                yield t[:, :-1], t[:, 1:]


def evaluate(model, batches):
    model.eval()
    tot = {"lm": 0.0, "pred": 0.0}
    hits, diags = {}, []
    with torch.no_grad():
        for x, y in batches:
            out = model(x, y, diag=True)
            tot["lm"] += out["loss_lm"].item()
            tot["pred"] += out["loss_pred"].item()
            diags.append(out["diag"])
            for h, v in out["hits"].items():
                hits.setdefault(h, []).append(sum(v) / len(v))
    model.train()
    d = {k: sum(x[k] for x in diags) / len(diags) for k in diags[0]}
    return (tot["lm"] / len(batches), tot["pred"] / len(batches),
            {h: sum(v) / len(v) for h, v in hits.items()}, d)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--tier", choices=["A", "B"], default="B")
    p.add_argument("--mode", choices=["joint", "posthoc"], default="joint")
    p.add_argument("--lambda-pred", type=float, default=0.0)
    p.add_argument("--lambda-sticky", type=float, default=0.0)
    p.add_argument("--steps", type=int, default=1000)
    p.add_argument("--batch", type=int, default=8)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--data", choices=["fineweb", "random"], default="fineweb")
    p.add_argument("--ckpt", type=str, default="")
    p.add_argument("--save", type=str, default="")
    p.add_argument("--eval-every", type=int, default=200)
    p.add_argument("--device", default="cpu")
    p.add_argument("--horizons", default="1", help="comma-separated layer-ahead horizons")
    args = p.parse_args()

    cfg = TIER_A if args.tier == "A" else TIER_B
    cfg.lambda_pred = args.lambda_pred
    cfg.lambda_sticky = args.lambda_sticky
    cfg.horizons = tuple(int(h) for h in args.horizons.split(","))
    assert all(1 <= h < cfg.n_layers for h in cfg.horizons), "bad horizon"
    torch.manual_seed(0)
    if args.device == "cpu":
        torch.set_num_threads(16)
    model = Model(cfg).to(args.device)
    if args.ckpt:
        model.load_state_dict(torch.load(args.ckpt, weights_only=True))
    if args.mode == "posthoc":
        for n, prm in model.named_parameters():
            prm.requires_grad = n.startswith("predictor.")
        # fresh predictor: never inherit co-trained heads from the ckpt
        for w in model.predictor.heads.values():
            torch.nn.init.normal_(w, std=0.02)
        cfg.lambda_pred = 1.0  # predictor loss is the only thing that matters
    total, active = model.param_counts()
    print(f"params: {total/1e6:.0f}M total / {active/1e6:.0f}M active | "
          f"mode={args.mode} lambda_pred={cfg.lambda_pred}")

    trainable = [p for n, p in model.named_parameters() if p.requires_grad
                 and (cfg.lambda_pred > 0 or not n.startswith("predictor."))]
    opt = torch.optim.AdamW(trainable, lr=args.lr, betas=(0.9, 0.95), weight_decay=0.1)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min((s + 1) / 100,
                           0.5 * (1 + math.cos(math.pi * min(s / args.steps, 1.0)))))

    if args.data == "random":
        data = None
        batch_fn = lambda: get_batch_random(cfg, args.batch, args.device)
    else:
        data = iter(fineweb_stream(cfg, args.batch, args.device))
        batch_fn = lambda: next(data)

    model.train()
    # fixed held-out eval set: drawn once, never trained on, identical across runs
    eval_batches = [batch_fn() for _ in range(16)]
    t0 = time.time()
    for step in range(args.steps):
        x, y = batch_fn()
        with torch.autocast(args.device, dtype=torch.bfloat16):
            out = model(x, y)
        opt.zero_grad()
        out["loss"].backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step(); sched.step()
        if step % 20 == 0:
            dt = time.time() - t0
            toks = (step + 1) * args.batch * cfg.ctx
            print(f"step {step:5d} | lm {out['loss_lm'].item():.3f} | "
                  f"pred {out['loss_pred'].item():.3f} | {toks/dt:.0f} tok/s")
        if (step + 1) % args.eval_every == 0 or step + 1 == args.steps:
            lm, pred, hits, d = evaluate(model, eval_batches)
            hit_str = " | ".join(f"h{h}: {v:.3f}" for h, v in hits.items())
            print(f"  eval step {step+1}: lm {lm:.3f} pred {pred:.3f} hit@k [{hit_str}] "
                  f"| ent {d['entropy']:.3f} persist {d['persist']:.3f} util_max {d['util_max']:.3f}")
    if args.save:
        torch.save(model.state_dict(), args.save)
        print(f"saved {args.save}")


if __name__ == "__main__":
    main()
