"""GPT-style MoE with a jointly-trained expert predictor.

The predictor forecasts future layers' router decisions from earlier hidden
states. In joint mode its loss backprops into the backbone, making routing
*more predictable*; in posthoc mode the backbone is frozen and only the
predictor trains (SOTA control, cf. arXiv 2511.10676).
"""
from dataclasses import dataclass, field

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class Config:
    vocab_size: int = 50257
    d_model: int = 384
    n_layers: int = 8
    n_heads: int = 6
    d_ff: int = 1024          # per-expert FFN hidden size
    n_experts: int = 8
    top_k: int = 2
    ctx: int = 512
    horizons: tuple = (1,)    # layers-ahead to predict
    lambda_pred: float = 0.0  # predictability loss weight (0 = baseline)
    lambda_bal: float = 0.01  # load-balancing aux loss weight
    lambda_sticky: float = 0.0  # StickyMoE-style temporal consistency loss
    dropout: float = 0.0


TIER_B = Config()  # ~120M total / ~40M active
TIER_A = Config(d_model=512, n_layers=12, n_heads=8, d_ff=1408,
                n_experts=16, top_k=2)  # ~450M total / ~100M active


class Attention(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.n_heads = cfg.n_heads
        self.qkv = nn.Linear(cfg.d_model, 3 * cfg.d_model, bias=False)
        self.proj = nn.Linear(cfg.d_model, cfg.d_model, bias=False)

    def forward(self, x):
        B, T, C = x.shape
        q, k, v = self.qkv(x).chunk(3, dim=-1)
        q = q.view(B, T, self.n_heads, C // self.n_heads).transpose(1, 2)
        k = k.view(B, T, self.n_heads, C // self.n_heads).transpose(1, 2)
        v = v.view(B, T, self.n_heads, C // self.n_heads).transpose(1, 2)
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        return self.proj(y.transpose(1, 2).reshape(B, T, C))


class MoEFFN(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        self.router = nn.Linear(cfg.d_model, cfg.n_experts, bias=False)
        self.w1 = nn.Parameter(torch.empty(cfg.n_experts, cfg.d_model, cfg.d_ff))
        self.w2 = nn.Parameter(torch.empty(cfg.n_experts, cfg.d_ff, cfg.d_model))
        nn.init.normal_(self.w1, std=0.02)
        nn.init.normal_(self.w2, std=0.02)

    def forward(self, x):
        """Returns (output, topk_idx [B*T, k], router_probs [B*T, E], bal_loss)."""
        cfg = self.cfg
        B, T, C = x.shape
        flat = x.reshape(B * T, C)
        logits = self.router(flat)
        probs = F.softmax(logits.float(), dim=-1)
        topk_w, topk_idx = probs.topk(cfg.top_k, dim=-1)
        topk_w = topk_w / topk_w.sum(dim=-1, keepdim=True)

        out = torch.zeros_like(flat)
        for e in range(cfg.n_experts):
            sel = (topk_idx == e).any(dim=-1)
            tok = flat[sel]
            h = F.gelu(tok @ self.w1[e]) @ self.w2[e]
            w = topk_w[sel] * (topk_idx[sel] == e).float()
            out[sel] += h * w.sum(dim=-1, keepdim=True)

        # Load-balancing loss: soft importance x top-k dispatch fraction
        f = torch.zeros_like(probs).scatter_add_(
            1, topk_idx, torch.ones_like(topk_w)).mean(0) / cfg.top_k
        bal = cfg.n_experts * (probs.mean(0) * f).sum()
        # StickyMoE-style routing consistency between consecutive tokens
        p = probs.view(B, T, -1)
        sticky = (p[:, 1:] - p[:, :-1]).pow(2).mean()
        return out.view(B, T, C), topk_idx, probs, bal, sticky


class Block(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.ln1 = nn.LayerNorm(cfg.d_model)
        self.attn = Attention(cfg)
        self.ln2 = nn.LayerNorm(cfg.d_model)
        self.moe = MoEFFN(cfg)

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        h = self.ln2(x)  # pre-MoE hidden state = predictor input
        ffn, topk_idx, probs, bal, sticky = self.moe(h)
        return x + ffn, h, topk_idx, probs, bal, sticky


class Predictor(nn.Module):
    """Per-(layer, horizon) linear heads: hidden at layer l -> router at l+h."""

    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        self.heads = nn.ParameterDict()
        for h in cfg.horizons:
            for l in range(cfg.n_layers - h):
                w = torch.empty(cfg.d_model, cfg.n_experts)
                nn.init.normal_(w, std=0.02)
                self.heads[f"h{h}_l{l}"] = nn.Parameter(w)

    def forward(self, layer_hiddens):
        """layer_hiddens: list of [B,T,C]. Returns dict h -> list of logits."""
        out = {h: [] for h in self.cfg.horizons}
        for h in self.cfg.horizons:
            for l in range(self.cfg.n_layers - h):
                out[h].append(layer_hiddens[l] @ self.heads[f"h{h}_l{l}"])
        return out


class Model(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        self.tok = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.pos = nn.Embedding(cfg.ctx, cfg.d_model)
        self.blocks = nn.ModuleList(Block(cfg) for _ in range(cfg.n_layers))
        self.ln_f = nn.LayerNorm(cfg.d_model)
        self.head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        self.predictor = Predictor(cfg)

    def forward(self, idx, targets=None, diag=False):
        B, T = idx.shape
        x = self.tok(idx) + self.pos(torch.arange(T, device=idx.device))
        hiddens, topks, probs_all, bal_losses, sticky_losses = [], [], [], [], []
        for blk in self.blocks:
            x, h, topk_idx, probs, bal, sticky = blk(x)
            hiddens.append(h)
            topks.append(topk_idx.view(B, T, -1))
            probs_all.append(probs)
            bal_losses.append(bal)
            sticky_losses.append(sticky)
        logits = self.head(self.ln_f(x))

        loss_lm = (F.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))
                   if targets is not None else None)
        loss_bal = torch.stack(bal_losses).mean()
        loss_sticky = torch.stack(sticky_losses).mean()

        # Predictability loss: soft CE of predictor logits against realized top-k
        pred_logits = self.predictor(hiddens)
        loss_pred, hits = x.new_zeros(()), {}
        n_terms = 0
        for h in self.cfg.horizons:
            hit_h = []
            for l, pl in enumerate(pred_logits[h]):
                true = topks[l + h]                       # [B,T,k] realized top-k
                soft = torch.zeros(B, T, self.cfg.n_experts, device=x.device)
                soft.scatter_(-1, true, 1.0 / self.cfg.top_k)
                loss_pred = loss_pred + -(soft * F.log_softmax(pl.float(), -1)).sum(-1).mean()
                n_terms += 1
                with torch.no_grad():
                    pred_topk = pl.topk(self.cfg.top_k, -1).indices
                    hit = (pred_topk.unsqueeze(-1) == true.unsqueeze(-2)).any(-1).float().mean(-1)
                    hit_h.append(hit.mean().item())
            hits[h] = hit_h
        loss_pred = loss_pred / max(n_terms, 1)

        out = {"logits": logits, "loss": None, "loss_lm": loss_lm,
               "loss_bal": loss_bal, "loss_pred": loss_pred, "hits": hits}
        if loss_lm is not None:
            out["loss"] = (loss_lm + self.cfg.lambda_bal * loss_bal
                           + self.cfg.lambda_pred * loss_pred
                           + self.cfg.lambda_sticky * loss_sticky)
        if diag:
            with torch.no_grad():
                ent = torch.stack([
                    -(p * p.clamp_min(1e-9).log()).sum(-1).mean()
                    for p in probs_all]).mean()
                persist = torch.stack([
                    (topks[l].unsqueeze(-1) == topks[l + 1].unsqueeze(-2)).any(-1)
                    .float().mean(-1).mean()
                    for l in range(len(topks) - 1)]).mean()
                counts = torch.stack([
                    F.one_hot(t.reshape(-1), self.cfg.n_experts).float().mean(0)
                    for t in topks])
                out["diag"] = {"entropy": ent.item(), "persist": persist.item(),
                               "util_max": counts.max(-1).values.mean().item()}
        return out

    def param_counts(self):
        total = sum(p.numel() for p in self.parameters())
        active = sum(p.numel() for n, p in self.named_parameters() if "w1" not in n and "w2" not in n)
        expert = sum(p.numel() for n, p in self.named_parameters() if "w1" in n or "w2" in n)
        return total, active + expert * self.cfg.top_k // self.cfg.n_experts
