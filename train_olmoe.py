"""Tier C: joint predictability fine-tuning of pretrained OLMoE (HuggingFace).

Trains a small expert-predictor jointly with LoRA-adapted OLMoE, folding
prediction accuracy into the LM loss so the backbone itself becomes more
predictable (the backbone-predictability claim, tested at real scale).

Usage (mirrors train.py):
  # treatment: joint predictability fine-tune
  python train_olmoe.py --mode joint --lambda-pred 0.1 --steps 6000 --save ckpt_C_joint.pt
  # baseline: same setup, lambda_pred=0
  python train_olmoe.py --mode joint --lambda-pred 0.0 --steps 6000 --save ckpt_C_base.pt
  # control / isolation test: fresh predictor on a frozen backbone
  python train_olmoe.py --mode posthoc --ckpt ckpt_C_base.pt --steps 2000 --save ckpt_C_posthoc.pt
  # CPU smoke test with random weights (no download needed):
  python train_olmoe.py --tiny --data random --steps 20 --eval-every 20 --device cpu

OLMoE-1B-7B: 16 layers, 64 experts, top-8, MoE in every block (no shared
experts), QK-norm attention. We hook the HF implementation instead of
fighting it: forward_pre_hook on each `model.layers.{i}.mlp`
(OlmoeSparseMoeBlock) captures the pre-MoE hidden state (predictor input);
forward hook on `mlp.gate` (OlmoeTopKRouter) captures router logits and the
realized top-k indices (prediction targets). Gate returns
(router_logits, router_scores, router_indices) in transformers 5.x.
"""
import argparse, math, time

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------- LoRA ----
class LoRALinear(nn.Module):
    """LoRA wrapper around a frozen nn.Linear."""

    def __init__(self, base: nn.Linear, r: int, alpha: float):
        super().__init__()
        self.base = base
        for p in self.base.parameters():
            p.requires_grad = False
        self.lora_A = nn.Parameter(torch.empty(base.in_features, r))
        self.lora_B = nn.Parameter(torch.zeros(r, base.out_features))
        nn.init.normal_(self.lora_A, std=0.02)
        self.scale = alpha / r

    def forward(self, x):
        return self.base(x) + (x @ self.lora_A.to(x.dtype) @ self.lora_B.to(x.dtype)) * self.scale


class LoRAExperts(nn.Module):
    """Low-rank deltas on OlmoeExperts' fused [E, ...] weights.

    Faithfully re-implements OlmoeExperts.forward (transformers 5.14) with
    W_effective = W + B @ A per expert.
    """

    def __init__(self, base, r: int, alpha: float):
        super().__init__()
        self.base = base  # keeps base.gate_up_proj / base.down_proj frozen
        E, twoI, H = base.gate_up_proj.shape
        I = base.down_proj.shape[-1]
        self.gu_A = nn.Parameter(torch.empty(E, r, H))
        self.gu_B = nn.Parameter(torch.zeros(E, twoI, r))
        self.dn_A = nn.Parameter(torch.empty(E, r, I))
        self.dn_B = nn.Parameter(torch.zeros(E, H, r))
        nn.init.normal_(self.gu_A, std=0.02)
        nn.init.normal_(self.dn_A, std=0.02)
        self.scale = alpha / r
        self.act_fn = base.act_fn

    def forward(self, hidden_states, top_k_index, top_k_weights):
        base = self.base
        final = torch.zeros_like(hidden_states)
        with torch.no_grad():
            mask = F.one_hot(top_k_index, num_classes=base.num_experts).permute(2, 1, 0)
            hits = torch.greater(mask.sum(dim=(-1, -2)), 0).nonzero()
        for e in hits:
            e = e[0]
            if e == base.num_experts:
                continue
            pos, tok = torch.where(mask[e])
            dt = base.gate_up_proj.dtype
            gu = base.gate_up_proj[e] + (self.gu_B[e].to(dt) @ self.gu_A[e].to(dt)) * self.scale
            dn = base.down_proj[e] + (self.dn_B[e].to(dt) @ self.dn_A[e].to(dt)) * self.scale
            gate, up = F.linear(hidden_states[tok], gu).chunk(2, dim=-1)
            cur = self.act_fn(gate) * up
            cur = F.linear(cur, dn) * top_k_weights[tok, pos, None]
            final.index_add_(0, tok, cur.to(final.dtype))
        return final


def apply_lora(model, r, alpha):
    """Wrap attn q/k/v/o and the expert weights in every OLMoE block."""
    for layer in model.model.layers:
        for proj in ("q_proj", "k_proj", "v_proj", "o_proj"):
            setattr(layer.self_attn, proj,
                    LoRALinear(getattr(layer.self_attn, proj), r, alpha))
        layer.mlp.experts = LoRAExperts(layer.mlp.experts, r, alpha)


# ----------------------------------------------------------- predictor ----
class Predictor(nn.Module):
    """Per-(layer, horizon) linear heads: pre-MoE hidden at l -> router at l+h."""

    def __init__(self, n_layers, d_model, n_experts, horizons):
        super().__init__()
        self.horizons = horizons
        self.heads = nn.ParameterDict()
        for h in horizons:
            for l in range(n_layers - h):
                w = torch.empty(d_model, n_experts)
                nn.init.normal_(w, std=0.02)
                self.heads[f"h{h}_l{l}"] = nn.Parameter(w)

    def forward(self, layer_hiddens):
        return {h: [layer_hiddens[l] @ self.heads[f"h{h}_l{l}"].to(layer_hiddens[l].dtype)
                    for l in range(len(layer_hiddens) - h)]
                for h in self.horizons}

    def reinit(self):
        for w in self.heads.values():
            nn.init.normal_(w, std=0.02)


# --------------------------------------------------------------- model ----
class OlmoePredictability(nn.Module):
    def __init__(self, lm, horizons, lambda_pred=0.0, lambda_bal=0.0):
        super().__init__()
        self.lm = lm
        cfg = lm.config
        self.n_layers = cfg.num_hidden_layers
        self.n_experts = cfg.num_local_experts
        self.top_k = cfg.num_experts_per_tok
        self.lambda_pred = lambda_pred
        self.lambda_bal = lambda_bal
        self.predictor = Predictor(self.n_layers, cfg.hidden_size,
                                   self.n_experts, horizons)
        self._hid, self._logits, self._idx = [], [], []
        for layer in lm.model.layers:
            layer.mlp.register_forward_pre_hook(self._cap_hidden)
            layer.mlp.gate.register_forward_hook(self._cap_router)

    def _cap_hidden(self, module, args):
        self._hid.append(args[0])  # [B,T,D] pre-MoE hidden (stays in graph)

    def _cap_router(self, module, args, out):
        logits, _, indices = out     # [B*T,E], _, [B*T,k]
        self._logits.append(logits)
        self._idx.append(indices)

    def forward(self, input_ids, labels=None, diag=False):
        self._hid, self._logits, self._idx = [], [], []
        out = self.lm(input_ids=input_ids, labels=labels)
        B, T = input_ids.shape
        E, k = self.n_experts, self.top_k
        topks = [i.view(B, T, k) for i in self._idx]

        loss_pred = input_ids.new_zeros((), dtype=torch.float32)
        hits, n_terms = {}, 0
        pred_logits = self.predictor(self._hid)
        for h, pls in pred_logits.items():
            hit_h = []
            for l, pl in enumerate(pls):
                true = topks[l + h]
                soft = torch.zeros(B, T, E, device=pl.device)
                soft.scatter_(-1, true, 1.0 / k)
                loss_pred = loss_pred + -(soft * F.log_softmax(pl.float(), -1)).sum(-1).mean()
                n_terms += 1
                with torch.no_grad():
                    pt = pl.topk(k, -1).indices
                    hit_h.append((pt.unsqueeze(-1) == true.unsqueeze(-2))
                                 .any(-1).float().mean(-1).mean().item())
            hits[h] = hit_h
        loss_pred = loss_pred / max(n_terms, 1)

        loss_bal = input_ids.new_zeros(())
        if self.lambda_bal > 0:
            # Switch-style: soft importance x top-k dispatch fraction
            for lg, true in zip(self._logits, topks):
                probs = F.softmax(lg.float(), -1)
                f = torch.zeros_like(probs).scatter_add_(
                    1, true.reshape(-1, k), torch.ones_like(probs[:, :1]).expand(-1, k)
                    .reshape(-1, k).float()).mean(0) / k
                loss_bal = loss_bal + E * (probs.mean(0) * f).sum()
            loss_bal = loss_bal / self.n_layers

        loss = out.loss + self.lambda_pred * loss_pred + self.lambda_bal * loss_bal
        res = {"loss": loss, "loss_lm": out.loss, "loss_pred": loss_pred, "hits": hits}
        if diag:
            with torch.no_grad():
                ent = torch.stack([
                    -(F.softmax(lg.float(), -1) *
                      F.log_softmax(lg.float(), -1)).sum(-1).mean()
                    for lg in self._logits]).mean()
                persist = torch.stack([
                    (topks[l].unsqueeze(-1) == topks[l + 1].unsqueeze(-2)).any(-1)
                    .float().mean(-1).mean()
                    for l in range(self.n_layers - 1)]).mean()
                counts = torch.stack([
                    F.one_hot(t.reshape(-1), E).float().mean(0) for t in topks])
                res["diag"] = {"entropy": ent.item(), "persist": persist.item(),
                               "util_max": counts.max(-1).values.mean().item()}
        return res

    def trainable(self, posthoc=False):
        for n, p in self.named_parameters():
            if posthoc:
                p.requires_grad = n.startswith("predictor.")
            else:
                p.requires_grad = ("lora_" in n or n.endswith("mlp.gate.weight")
                                   or n.startswith("predictor."))
        return [p for p in self.parameters() if p.requires_grad]


# ----------------------------------------------------------------- data ----
def fineweb_stream(tokenizer, ctx, bs, device, seed=0):
    """Document-shuffled stream (1000-doc buffer, EOS between docs)."""
    import random
    from datasets import load_dataset
    ds = iter(load_dataset("HuggingFaceFW/fineweb-edu", "sample-10BT",
                           split="train", streaming=True))
    rng = random.Random(seed)
    docs, buf, need = [], [], bs * (ctx + 1)
    while True:
        while len(docs) < 1000:
            docs.append(tokenizer(next(ds)["text"]).input_ids + [tokenizer.eos_token_id])
        buf.extend(docs.pop(rng.randrange(len(docs))))
        while len(buf) >= need:
            chunk, buf = buf[:need], buf[need:]
            t = torch.tensor(chunk, dtype=torch.long, device=device).view(bs, ctx + 1)
            yield t[:, :-1], t[:, 1:]


def evaluate(model, batches):
    model.eval()
    lm = pred = 0.0
    hits, diags = {}, []
    with torch.no_grad():
        for x, y in batches:
            out = model(x, y, diag=True)
            lm += out["loss_lm"].item()
            pred += out["loss_pred"].item()
            diags.append(out["diag"])
            for h, v in out["hits"].items():
                hits.setdefault(h, []).append(sum(v) / len(v))
    model.train()
    d = {k: sum(x[k] for x in diags) / len(diags) for k in diags[0]}
    return (lm / len(batches), pred / len(batches),
            {h: sum(v) / len(v) for h, v in hits.items()}, d)


# ----------------------------------------------------------------- main ----
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=["joint", "posthoc"], default="joint")
    p.add_argument("--model", default="allenai/OLMoE-1B-7B-0924")
    p.add_argument("--tiny", action="store_true", help="random-weight tiny config (smoke test)")
    p.add_argument("--horizons", default="1,2,4")
    p.add_argument("--lambda-pred", type=float, default=0.0)
    p.add_argument("--lambda-bal", type=float, default=0.0)
    p.add_argument("--lora-r", type=int, default=16)
    p.add_argument("--lora-alpha", type=float, default=32.0)
    p.add_argument("--steps", type=int, default=6000)
    p.add_argument("--batch", type=int, default=4)
    p.add_argument("--ctx", type=int, default=512)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--data", choices=["fineweb", "random"], default="fineweb")
    p.add_argument("--eval-every", type=int, default=500)
    p.add_argument("--device", default="cuda")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--ckpt", default="")
    p.add_argument("--save", default="")
    args = p.parse_args()

    from transformers import AutoTokenizer, OlmoeConfig, OlmoeForCausalLM
    torch.manual_seed(args.seed)
    if args.tiny:
        cfg = OlmoeConfig(vocab_size=1000, hidden_size=128, intermediate_size=256,
                          num_hidden_layers=4, num_attention_heads=4,
                          num_key_value_heads=4, num_experts=8,
                          num_experts_per_tok=2, max_position_embeddings=args.ctx,
                          eos_token_id=999, pad_token_id=0)
        lm = OlmoeForCausalLM(cfg)
    else:
        lm = OlmoeForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16)
    apply_lora(lm, args.lora_r, args.lora_alpha)
    horizons = tuple(int(h) for h in args.horizons.split(","))
    n_l = lm.config.num_hidden_layers
    horizons = tuple(h for h in horizons if 1 <= h < n_l)  # clamp for --tiny
    assert horizons, "no valid horizons"
    model = OlmoePredictability(lm, horizons, args.lambda_pred,
                                args.lambda_bal).to(args.device)
    if args.ckpt:
        sd = torch.load(args.ckpt, weights_only=True)
        sd = {k: v for k, v in sd.items() if not k.startswith("predictor.")}
        model.load_state_dict(sd, strict=False)
    params = model.trainable(posthoc=(args.mode == "posthoc"))
    if args.mode == "posthoc":
        model.predictor.reinit()  # never inherit co-trained heads
        args.lambda_pred = 1.0
        model.lambda_pred = 1.0
    n_train = sum(p.numel() for p in params)
    print(f"mode={args.mode} lambda_pred={model.lambda_pred} horizons={horizons} "
          f"| trainable {n_train/1e6:.1f}M / {sum(p.numel() for p in model.parameters())/1e9:.2f}B")

    opt = torch.optim.AdamW(params, lr=args.lr, betas=(0.9, 0.95), weight_decay=0.0)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min((s + 1) / 100,
                           0.5 * (1 + math.cos(math.pi * min(s / args.steps, 1.0)))))

    if args.data == "random":
        V = lm.config.vocab_size
        batch_fn = lambda: (lambda t: (t[:, :-1], t[:, 1:]))(
            torch.randint(0, V, (args.batch, args.ctx + 1), device=args.device))
    else:
        tok = AutoTokenizer.from_pretrained(args.model)
        stream = iter(fineweb_stream(tok, args.ctx, args.batch, args.device, args.seed))
        batch_fn = lambda: next(stream)

    model.train()
    eval_batches = [batch_fn() for _ in range(16)]  # fixed held-out set
    t0 = time.time()
    for step in range(args.steps):
        x, y = batch_fn()
        with torch.autocast(args.device, dtype=torch.bfloat16,
                            enabled=(args.device != "cpu" or True)):
            out = model(x, y)
        opt.zero_grad()
        out["loss"].backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step(); sched.step()
        if step % 20 == 0:
            print(f"step {step:5d} | lm {out['loss_lm'].item():.3f} | "
                  f"pred {out['loss_pred'].item():.3f} | "
                  f"{(step+1)*args.batch*args.ctx/(time.time()-t0):.0f} tok/s")
        if (step + 1) % args.eval_every == 0 or step + 1 == args.steps:
            lm_v, pred_v, hits, d = evaluate(model, eval_batches)
            hit_str = " | ".join(f"h{h}: {v:.3f}" for h, v in hits.items())
            print(f"  eval step {step+1}: lm {lm_v:.3f} pred {pred_v:.3f} "
                  f"hit@k [{hit_str}] | ent {d['entropy']:.3f} "
                  f"persist {d['persist']:.3f} util_max {d['util_max']:.3f}")
    if args.save:
        sd = {k: v for k, v in model.state_dict().items()
              if k.startswith("predictor.") or "lora_" in k or k.endswith("mlp.gate.weight")}
        torch.save(sd, args.save)
        print(f"saved {args.save} ({sum(v.numel() for v in sd.values())/1e6:.0f}M params)")


if __name__ == "__main__":
    main()
