"""Trace-driven expert-cache simulator (honest disk model).

Single NVMe queue: fetches are served FIFO, each costing latency + size/BW.
Prefetching CANNOT create bandwidth — it only converts serial last-moment
demand stalls into overlapped early fetches when the disk has slack. This is
the real economics of Colibri-style engines: prediction pays in the
latency-bound regime (disk not saturated), not the bandwidth-bound one.

Default timing (this box): compute 0.10 ms/layer/token; NVMe 0.10 ms latency
+ 5 GB/s; Tier-A expert 2.88 MB bf16. Batch-union decode: a decode step of B
sequences reads each UNIQUE expert once per layer. Prefetch is opportunistic:
issued only when the disk is idle.

--synthetic runs a Colibri-geometry scenario (75x256 experts, top-8, 19MB
int4, 4ms/layer dense compute) with synthetic Zipf+locality routing and
predictor accuracies measured at Tier A (base 0.83 / joint 0.89 / oracle 1.0).
"""
import argparse
from collections import OrderedDict

import numpy as np

N_LAYERS, N_EXPERTS = 12, 16
T_LAYER_MS = 0.10
LAT_MS, BW_MB_S = 0.10, 5000.0
EXPERT_MB = 2 * 512 * 1408 * 2 / 1e6  # 2.88 MB bf16


class Sim:
    def __init__(self, true, pred, budget, policy, batch, n_layers=N_LAYERS,
                 n_experts=N_EXPERTS, t_layer=T_LAYER_MS, expert_mb=EXPERT_MB):
        self.true, self.pred = true, pred
        self.C, self.policy, self.B = budget, policy, batch
        self.L, self.E = n_layers, n_experts
        self.T_LAYER, self.EXPERT_MB = t_layer, expert_mb
        self.FETCH_MS = LAT_MS + expert_mb / BW_MB_S * 1000
        self.cache = OrderedDict()
        self.current = None   # (eid, end_time) currently being read from disk
        self.queue = []       # eids waiting for the disk (FIFO)
        self.diskq = {}       # eid -> ready time (current + queued)
        self.t = 0.0
        self.stall = self.pf_mb = self.waste_mb = 0.0
        self.hits = self.reqs = 0
        self.used = set()

    def _recompute(self):
        start = self.current[1]
        for eid in self.queue:
            start += self.FETCH_MS
            self.diskq[eid] = start

    def issue(self, eid, front=False):
        if self.current is None:
            self.current = (eid, self.t + self.FETCH_MS)
            self.diskq[eid] = self.current[1]
        else:
            self.queue.insert(0 if front else len(self.queue), eid)
            self._recompute()

    def _advance(self):
        """Complete disk reads whose end time has passed; insert into cache."""
        while self.current is not None and self.current[1] <= self.t:
            eid = self.current[0]
            del self.diskq[eid]
            self.insert(eid)
            if self.queue:
                nxt = self.queue.pop(0)
                self.current = (nxt, self.diskq[nxt])
            else:
                self.current = None

    def insert(self, eid):
        self.cache[eid] = True
        self.cache.move_to_end(eid)
        while len(self.cache) > self.C:
            old, _ = self.cache.popitem(last=False)
            if old not in self.used:
                self.waste_mb += self.EXPERT_MB

    def prefetch(self, tok0, l):
        tl = l + (self.policy if isinstance(self.policy, int) else 1)
        if tl >= self.L or self.queue or self.current is not None:
            return  # no prediction, or disk not fully idle: never risk
                    # delaying the next layer's demand fetches
        for b in range(self.B):
            src = self.true[tok0 + b, tl] if self.policy == "oracle" else self.pred[tok0 + b, l]
            for pe in src:
                if pe < 0:
                    continue
                eid = tl * self.E + int(pe)
                if eid not in self.cache and eid not in self.diskq:
                    self.issue(eid)
                    self.pf_mb += self.EXPERT_MB
                    return  # one prefetch per idle window

    def run(self):
        N = self.true.shape[0]
        for tok0 in range(0, N - self.B + 1, self.B):
            for l in range(self.L):
                need, seen = [], set()
                for b in range(self.B):
                    for e in self.true[tok0 + b, l]:
                        self.reqs += 1
                        eid = l * self.E + int(e)
                        if eid in self.cache:
                            self.hits += 1
                            self.used.add(eid)
                            self.cache.move_to_end(eid)
                        elif eid not in seen:
                            seen.add(eid)
                            need.append(eid)
                for eid in need:  # unique demand fetches (batch-union)
                    if eid in self.diskq:
                        if eid in self.queue:  # prioritize the demand fetch
                            self.queue.remove(eid)
                            self.queue.insert(0, eid)
                            self._recompute()
                        ready = self.diskq[eid]
                        self.hits += 1  # served by in-flight prefetch
                    else:
                        self.issue(eid, front=True)
                        ready = self.diskq[eid]
                    if ready > self.t:
                        self.stall += ready - self.t
                        self.t = ready
                    self.used.add(eid)
                    self._advance()
                    self.insert(eid)
                if self.policy != "demand":
                    self.prefetch(tok0, l)  # into the compute window
                self.t += self.T_LAYER * self.B
                self._advance()
        for eid in self.diskq:
            if eid not in self.used:
                self.waste_mb += self.EXPERT_MB
        ntok = (N // self.B) * self.B
        return {"tok_s": ntok / (self.t / 1000.0),
                "stall_ms_tok": self.stall / ntok,
                "hit_rate": self.hits / self.reqs,
                "pf_mb_tok": self.pf_mb / ntok,
                "waste_mb_tok": self.waste_mb / ntok}


def synth_traces(ntok, L, E, k, acc, rho=0.3, zipf_a=1.0, seed=0):
    """Synthetic routing: Zipf popularity + rho temporal reuse; predictor
    emulated at per-expert hit probability acc (from measured hit@k)."""
    rng = np.random.default_rng(seed)
    ranks = np.arange(1, E + 1)
    p = ranks ** -zipf_a
    p /= p.sum()
    true = np.empty((ntok, L, k), dtype=np.int64)
    pred = np.empty((ntok, L, k), dtype=np.int64)
    prev = rng.choice(E, size=(L, k), p=p)
    for t in range(ntok):
        for l in range(L):
            reuse = rng.random(k) < rho
            fresh = rng.choice(E, size=k, p=p)
            cur = np.where(reuse, prev[l], fresh)
            prev[l] = cur
            true[t, l] = cur
            keep = rng.random(k) < acc
            pred[t, l] = np.where(keep, cur, rng.integers(0, E, size=k))
    return true, pred


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--traces", nargs="+")
    p.add_argument("--budgets", default="16,32,64,128,192")
    p.add_argument("--batches", default="1,8")
    p.add_argument("--synthetic", action="store_true",
                   help="Colibri-geometry scenario with measured accuracies")
    args = p.parse_args()

    if args.synthetic:
        L, E, K = 75, 256, 8
        print(f"scenario: {L} layers x {E} experts, top-{K}, 19MB int4 experts, "
              "4ms/layer dense compute, C in {500, 2000}")
        for label, acc in {"base": 0.83, "joint": 0.89, "oracle": 1.0}.items():
            true, pred = synth_traces(2000, L, E, K, acc)
            for c in (500, 2000):
                for pol in ("demand", 1, 4):
                    r = Sim(true, pred, c, pol, 1, L, E, 4.0, 19.0).run()
                    print(f"acc={label:6s} C={c:5d} pol={str(pol):6s} "
                          f"tok/s={r['tok_s']:6.3f} stall={r['stall_ms_tok']:7.1f}ms "
                          f"hit={r['hit_rate']:.3f} pf={r['pf_mb_tok']:8.1f}MB "
                          f"waste={r['waste_mb_tok']:6.1f}MB")
        return

    budgets = [int(b) for b in args.budgets.split(",")]
    batches = [int(b) for b in args.batches.split(",")]
    policies = ["demand", 1, 2, 4, "oracle"]
    import csv
    rows = []
    for label, path in (x.split("=") for x in args.traces):
        d = np.load(path)
        for bsz in batches:
            for c in budgets:
                for pol in policies:
                    pred = d["true_topk"] if pol == "oracle" else (
                        d[f"pred_h{pol}"] if isinstance(pol, int) else d["true_topk"])
                    r = Sim(d["true_topk"], pred, c, pol, bsz).run()
                    rows.append((label, bsz, c, str(pol), r))
                    print(f"{label:6s} B={bsz} C={c:3d} pol={str(pol):6s} "
                          f"tok/s={r['tok_s']:6.1f} stall={r['stall_ms_tok']:5.2f}ms "
                          f"hit={r['hit_rate']:.3f} pf={r['pf_mb_tok']:5.1f}MB "
                          f"waste={r['waste_mb_tok']:4.1f}MB")
    with open("results_cache_sim.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["trace", "batch", "budget", "policy", "tok_s", "stall_ms_tok",
                    "hit_rate", "pf_mb_tok", "waste_mb_tok"])
        for label, bsz, c, pol, r in rows:
            w.writerow([label, bsz, c, pol, f"{r['tok_s']:.2f}", f"{r['stall_ms_tok']:.4f}",
                        f"{r['hit_rate']:.4f}", f"{r['pf_mb_tok']:.3f}", f"{r['waste_mb_tok']:.3f}"])
    print("wrote results_cache_sim.csv")


if __name__ == "__main__":
    main()
