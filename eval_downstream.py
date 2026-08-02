"""Zero-shot downstream evaluation for Tier A checkpoints.

Scores multiple-choice benchmarks by length-normalized continuation logprob
(mean logprob per continuation token), the standard protocol for small LMs:
the model never sees task training data; quality = picking the right
continuation. Complements val-LM loss as the quality axis for the paper
(closes the "val-LM only" limitation in PAPER.md section 7).

Benchmarks: HellaSwag (subsampled), ARC-Easy, ARC-Challenge, PIQA.
Usage:
  python eval_downstream.py --ckpt ckpt_A_base.pt --device cuda
  python eval_downstream.py --ckpt ckpt_A_lam0.3.pt --device cuda --n 2000
"""
import argparse, json, time

import torch
import torch.nn.functional as F

from model import Model, TIER_A

CTX = 512


def load_model(ckpt, device):
    cfg = TIER_A
    cfg.horizons = (1, 2, 4)  # match training so ckpt keys load; predictor unused
    model = Model(cfg).to(device)
    model.load_state_dict(torch.load(ckpt, weights_only=True, map_location=device))
    model.eval()
    return model


def score_continuations(model, enc, prompt, choices, device):
    """Mean per-token logprob of each choice given prompt (one batched fwd).
    Right-padded so pads never influence real tokens under causal attention."""
    p_ids = enc.encode(prompt)
    eot = enc.eot_token
    seqs, cont_lens = [], []
    for ch in choices:
        c_ids = enc.encode(" " + ch)
        ids = (p_ids + c_ids)[-CTX:]
        seqs.append(ids)
        cont_lens.append(min(len(c_ids), len(ids)))
    L = max(len(s) for s in seqs)
    x = torch.tensor([s[:-1] + [eot] * (L - len(s)) for s in seqs], device=device)
    y = torch.tensor([s[1:] + [eot] * (L - len(s)) for s in seqs], device=device)
    with torch.no_grad(), torch.autocast(device, dtype=torch.bfloat16,
                                         enabled=(device != "cpu")):
        out = model(x)
    lp = F.log_softmax(out["logits"].float(), -1)
    tok_lp = lp.gather(-1, y.unsqueeze(-1)).squeeze(-1)  # [n_choice, L-1]
    scores = []
    for i, (ids, n) in enumerate(zip(seqs, cont_lens)):
        # continuation = tokens ids[-n:], scored at tok_lp[len(ids)-n-1 : len(ids)-1]
        scores.append(tok_lp[i, len(ids) - n - 1: len(ids) - 1].mean().item())
    return scores


def iter_hellaswag(n, seed):
    from datasets import load_dataset
    ds = load_dataset("Rowan/hellaswag", split="validation").shuffle(seed=seed)
    for i, ex in enumerate(ds):
        if i >= n:
            break
        prompt = ex["activity_label"] + ": " + ex["ctx_a"] + " " + ex["ctx_b"]
        yield prompt, ex["endings"], int(ex["label"])


def iter_arc(name, n, seed):
    from datasets import load_dataset
    ds = load_dataset("allenai/ai2_arc", name, split="test").shuffle(seed=seed)
    for i, ex in enumerate(ds):
        if i >= n:
            break
        choices = ex["choices"]["text"]
        labels = ex["choices"]["label"]
        yield "Question: " + ex["question"] + "\nAnswer:", choices, labels.index(ex["answerKey"])


def iter_piqa(n, seed):
    from datasets import load_dataset
    # script datasets are unsupported in datasets>=4; use the parquet mirror
    url = ("https://huggingface.co/datasets/ybisk/piqa/resolve/"
           "refs%2Fconvert%2Fparquet/plain_text/validation/0000.parquet")
    ds = load_dataset("parquet", data_files=url, split="train").shuffle(seed=seed)
    for i, ex in enumerate(ds):
        if i >= n:
            break
        yield ex["goal"], [ex["sol1"], ex["sol2"]], int(ex["label"])


BENCHES = {"hellaswag": iter_hellaswag, "arc_easy": lambda n, s: iter_arc("ARC-Easy", n, s),
           "arc_challenge": lambda n, s: iter_arc("ARC-Challenge", n, s),
           "piqa": iter_piqa}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--device", default="cuda")
    p.add_argument("--n", type=int, default=1000, help="examples per benchmark")
    p.add_argument("--seed", type=int, default=123, help="subsample seed (fixed)")
    p.add_argument("--bench", default=",".join(BENCHES))
    p.add_argument("--out", default="")
    args = p.parse_args()

    import tiktoken
    enc = tiktoken.get_encoding("gpt2")
    model = load_model(args.ckpt, args.device)
    results = {}
    for name in args.bench.split(","):
        t0, correct, total = time.time(), 0, 0
        for prompt, choices, gold in BENCHES[name](args.n, args.seed):
            s = score_continuations(model, enc, prompt, choices, args.device)
            correct += int(max(range(len(s)), key=lambda i: s[i]) == gold)
            total += 1
            if total % 200 == 0:
                print(f"  {name} {total}: acc {correct/total:.3f} "
                      f"({total/(time.time()-t0):.1f} ex/s)", flush=True)
        acc = correct / total
        results[name] = {"acc": acc, "n": total}
        print(f"{name}: acc {acc:.4f} (n={total}, {time.time()-t0:.0f}s)", flush=True)
    results["ckpt"] = args.ckpt
    out = args.out or f"downstream_{args.ckpt.replace('.pt','')}.json"
    with open(out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
