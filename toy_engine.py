"""Minimal disk-resident MoE inference engine (the PLAN.md end-to-end demo).

Tier A model: dense backbone in RAM, 192 experts (12x16, 2.88MB bf16 each)
on disk, read via O_DIRECT pread (bypasses page cache — real disk behavior).
Cache: LRU with hard budget C. Prefetch: a thread fed by the trained
posthoc predictor (linear heads on pre-MoE hidden states, horizon 1),
one fetch per idle window, demand-priority — the deployable configuration.

Usage:
  python toy_engine.py --model engine_base --policy demand --tokens 300
  python toy_engine.py --model engine_joint --policy prefetch --tokens 300
"""
import argparse, os, threading, time
from collections import OrderedDict

import numpy as np
import torch
import torch.nn.functional as F

D, H, N_LAYERS, N_EXPERTS, TOP_K, DFF = 512, 8, 12, 16, 2, 1408
EXPERT_BYTES = (D * DFF + DFF * D) * 2


class DiskExperts:
    def __init__(self, path):
        self.fd = os.open(path, os.O_RDONLY | os.O_DIRECT)
        raw = np.zeros(EXPERT_BYTES + 512, dtype=np.uint8)
        addr = raw.ctypes.data
        al = (addr + 511) & ~511
        self.buf = raw[al - addr:al - addr + EXPERT_BYTES]

    def read(self, eid):
        os.preadv(self.fd, [self.buf], eid * EXPERT_BYTES)
        u = self.buf.view(np.uint16)
        w1 = torch.from_numpy(u[: D * DFF].copy()).view(torch.bfloat16).view(D, DFF)
        w2 = torch.from_numpy(u[D * DFF:].copy()).view(torch.bfloat16).view(DFF, D)
        return w1, w2


class Cache:
    def __init__(self, disk, budget):
        self.disk, self.budget = disk, budget
        self.cache, self.inflight = OrderedDict(), set()
        self.cond = threading.Condition()
        self.hits = self.misses = 0
        import queue
        self.pq = queue.PriorityQueue()  # (0, eid) demand, (1, eid) prefetch

    def get(self, eid):
        """Demand path: returns (w1, w2), waiting if a fetch is in flight."""
        with self.cond:
            if eid in self.cache:
                self.hits += 1
                self.cache.move_to_end(eid)
                return self.cache[eid]
            self.misses += 1
            if eid not in self.inflight:
                self.inflight.add(eid)
                self.pq.put((0, eid))
            while eid not in self.cache:
                self.cond.wait()
            self.cache.move_to_end(eid)
            return self.cache[eid]

    def request_prefetch(self, eid):
        with self.cond:
            if eid not in self.cache and eid not in self.inflight:
                self.inflight.add(eid)
                self.pq.put((1, eid))

    def _store(self, eid, w):
        with self.cond:
            self.cache[eid] = w
            self.cache.move_to_end(eid)
            while len(self.cache) > self.budget:
                self.cache.popitem(last=False)
            self.inflight.discard(eid)
            self.cond.notify_all()

    def prefetch_worker(self):
        """Single disk; demand fetches always served before prefetches."""
        while True:
            _, eid = self.pq.get()
            if eid is None:
                return
            with self.cond:
                if eid in self.cache:
                    self.inflight.discard(eid)
                    self.cond.notify_all()
                    continue
            self._store(eid, self.disk.read(eid))


class Engine:
    def __init__(self, prefix, budget, policy):
        bb = torch.load(prefix + ".backbone.pt", weights_only=True)
        self.bb = {k: v.to(torch.bfloat16) for k, v in bb.items()}
        pred = torch.load(prefix + ".predictor.pt", weights_only=True)
        # horizon-1 heads only, stacked per layer: [L-1, D, E]
        self.pred_w = torch.stack(
            [pred[f"predictor.heads.h1_l{l}"].to(torch.bfloat16)
             for l in range(N_LAYERS - 1)])
        self.disk = DiskExperts(prefix + ".experts.bin")
        self.cache = Cache(self.disk, budget)
        self.policy = policy
        self.worker = threading.Thread(target=self.cache.prefetch_worker,
                                       daemon=True)
        self.worker.start()
        self.kv = [None] * N_LAYERS

    def request_prefetch(self, eid):
        self.cache.request_prefetch(eid)

    def attention(self, x, l, pos):
        bb, B = self.bb, 1
        qkv = x @ bb[f"blocks.{l}.attn.qkv.weight"].T
        q, k, v = qkv.split(D, dim=-1)
        hd = D // H
        q = q.view(B, 1, H, hd).transpose(1, 2)
        k = k.view(B, 1, H, hd).transpose(1, 2)
        v = v.view(B, 1, H, hd).transpose(1, 2)
        if self.kv[l] is None:
            self.kv[l] = (k, v)
        else:
            pk, pv = self.kv[l]
            self.kv[l] = (torch.cat([pk, k], 2), torch.cat([pv, v], 2))
        k, v = self.kv[l]
        y = F.scaled_dot_product_attention(q, k, v, is_causal=False)
        return y.transpose(1, 2).reshape(B, 1, D) @ bb[f"blocks.{l}.attn.proj.weight"].T

    def ln(self, x, name):
        return F.layer_norm(x, (D,), self.bb[f"{name}.weight"], self.bb[f"{name}.bias"])

    def step(self, tok):
        bb = self.bb
        x = bb["tok.weight"][tok].view(1, 1, D) + bb["pos.weight"][self.pos].view(1, 1, D)
        for l in range(N_LAYERS):
            x = x + self.attention(self.ln(x, f"blocks.{l}.ln1"), l, self.pos)
            h = self.ln(x, f"blocks.{l}.ln2")
            if self.policy == "prefetch" and l + 1 < N_LAYERS:
                logits = h.view(1, D) @ self.pred_w[l]
                for e in logits.topk(TOP_K, -1).indices[0].tolist():
                    self.request_prefetch((l + 1) * N_EXPERTS + e)
            probs = F.softmax((h.view(1, D) @ bb[f"blocks.{l}.moe.router.weight"].T).float(), -1)
            w, idx = probs.topk(TOP_K, -1)
            w = w / w.sum()
            out = torch.zeros(1, D, dtype=torch.bfloat16)
            for j in range(TOP_K):
                e = idx[0, j].item()
                w1, w2 = self.cache.get(l * N_EXPERTS + e)
                out += (F.gelu(h.view(1, D) @ w1) @ w2) * w[0, j].to(torch.bfloat16)
            x = x + out.view(1, 1, D)
        logits = self.ln(x, "ln_f").view(1, D) @ bb["head.weight"].T
        return logits

    def generate(self, prompt, n_tokens, warmup=20):
        self.pos = 0
        times = []
        tok = prompt
        for i in range(n_tokens + warmup):
            t0 = time.perf_counter()
            logits = self.step(torch.tensor(tok))
            dt = time.perf_counter() - t0
            tok = logits.argmax(-1).item()
            self.pos += 1
            if i >= warmup:
                times.append(dt)
        self.cache.pq.put((9, None))
        return times, self.cache.hits, self.cache.misses


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True, help="engine file prefix")
    p.add_argument("--policy", choices=["demand", "prefetch"], default="demand")
    p.add_argument("--budget", type=int, default=64)
    p.add_argument("--tokens", type=int, default=300)
    p.add_argument("--prompt", type=int, default=13)
    args = p.parse_args()
    torch.set_num_threads(8)
    eng = Engine(args.model, args.budget, args.policy)
    times, hits, misses = eng.generate(args.prompt, args.tokens)
    arr = np.array(times)
    print(f"{args.model} pol={args.policy} C={args.budget} prompt={args.prompt}: "
          f"tok/s={1/arr.mean():.2f} p50={np.percentile(arr,50)*1000:.1f}ms "
          f"p99={np.percentile(arr,99)*1000:.1f}ms hit_rate={hits/(hits+misses):.3f}")


if __name__ == "__main__":
    main()
