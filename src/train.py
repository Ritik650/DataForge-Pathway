"""Train a small BDH-GPU on MQAR until in-context associative recall works.

This is the Block-2 gate. Nothing downstream (ablation, interference sweeps,
the explainer) means anything unless recall is well above chance AND the
behavioural controls pass: deleting the queried binding must collapse recall to
chance, and rebinding it must move the answer.

Run: python src/train.py --iters 8000
"""

import argparse
import json
import pathlib
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

from bdh import BDH, BDHConfig  # noqa: E402
import mqar  # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parents[1]


@torch.no_grad()
def evaluate(model, device, block, n_seq=512, seed=1234,
             pairs_range=(2, 8), queries_range=(1, 4), filler_range=(0, 4)):
    """Recall accuracy over every query position, plus chance levels."""
    model.eval()
    rng = np.random.default_rng(seed)
    hits = n = 0
    ctx_city_hits = 0
    chance = []
    for i in range(0, n_seq, 128):
        bs = min(128, n_seq - i)
        x, _, meta = mqar.make_batch(rng, bs, block, pairs_range=pairs_range,
                                     queries_range=queries_range,
                                     filler_range=filler_range)
        logits, _ = model(torch.from_numpy(x).to(device))
        for b, ex in enumerate(meta):
            ctx = {mqar.CITY_OFF + c for _, c in ex["kv"]}
            for pos, ans in zip(ex["ans_positions"], ex["answers"]):
                pred = int(logits[b, pos].argmax())
                hits += int(pred == ans)
                ctx_city_hits += int(pred in ctx)
                chance.append(1.0 / max(1, len(ctx)))
                n += 1
    model.train()
    return {
        "recall_acc": hits / n,
        "picks_a_context_city": ctx_city_hits / n,
        "chance_if_copying_some_context_city": float(np.mean(chance)),
        "chance_uniform_over_cities": 1.0 / len(mqar.CITIES),
        "n": n,
    }


@torch.no_grad()
def eval_controls(model, device, block, n_seq=512, seed=4321, n_pairs=6):
    """The two interventions that decide whether recall is really in state.

    Paired design: each trial uses one base sequence (fixed per-trial seed) and
    is re-generated under all three conditions, so the conditions differ only
    by the intervention.

        normal -- the binding is present; recall should be high
        drop   -- the binding is deleted; the answer is now unavailable from
                  context, so recall must fall to chance. If it does not, the
                  model is reading the answer from somewhere it should not.
        swap   -- the binding is rewritten to a different city; if the answer
                  follows the swap, recall is tracking context, not weights.
    """
    model.eval()
    hits = {"normal": 0, "drop": 0, "swap": 0}
    swap_follows = 0
    n = 0

    def run(ex):
        logits, _ = model(torch.from_numpy(ex["ids"][:-1]).unsqueeze(0).to(device))
        return int(logits[0, ex["ans_pos"]].argmax())

    for i in range(n_seq):
        qi = int(np.random.default_rng(seed + i).integers(0, n_pairs))
        base = mqar.make_example(np.random.default_rng(seed + i), n_pairs=n_pairs,
                                 n_queries=1, query_idx=qi, block=block + 1)
        hits["normal"] += int(run(base) == base["answer"])

        drop = mqar.make_example(np.random.default_rng(seed + i), n_pairs=n_pairs,
                                 n_queries=1, query_idx=qi, drop_queried=True,
                                 block=block + 1)
        hits["drop"] += int(run(drop) == drop["answer"])

        taken = {c for j, (_, c) in enumerate(base["pairs"]) if j != qi}
        free = [c for c in range(len(mqar.CITIES)) if c not in taken]
        target = int(free[(seed + i) % len(free)])
        swap = mqar.make_example(np.random.default_rng(seed + i), n_pairs=n_pairs,
                                 n_queries=1, query_idx=qi,
                                 swap_queried_to=target, block=block + 1)
        pred = run(swap)
        hits["swap"] += int(pred == swap["answer"])
        swap_follows += int(pred == swap["answer"])
        n += 1

    model.train()
    return {
        "normal": hits["normal"] / n,
        "drop": hits["drop"] / n,
        "swap_follows_context": hits["swap"] / n,
        "chance": 1.0 / n_pairs,
        "n": n,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--iters", type=int, default=8000)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--block", type=int, default=64)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--n-layer", type=int, default=2)
    ap.add_argument("--n-embd", type=int, default=64)
    ap.add_argument("--mult", type=int, default=8)
    ap.add_argument("--n-head", type=int, default=1)
    ap.add_argument("--dropout", type=float, default=0.0)
    ap.add_argument("--answer-weight", type=float, default=1.0)
    ap.add_argument("--pairs-max", type=int, default=8)
    ap.add_argument("--filler-max", type=int, default=4)
    ap.add_argument("--u-decay", type=float, default=1.0,
                    help="damping case of the paper's U; 1.0 == reference (RoPE only)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=str, default="data/bdh_mqar.pt")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    cfg = BDHConfig(
        n_layer=args.n_layer, n_embd=args.n_embd, n_head=args.n_head,
        mlp_internal_dim_multiplier=args.mult, dropout=args.dropout,
        vocab_size=mqar.VOCAB_SIZE, u_decay=args.u_decay,
    )
    model = BDH(cfg).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"device={device}  n={cfg.N * cfg.n_head} neurons  d={cfg.n_embd}  "
          f"layers={cfg.n_layer}  u_decay={cfg.u_decay}  params={n_params:,}")

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.1)
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=args.lr, total_steps=args.iters, pct_start=0.1
    )

    rng = np.random.default_rng(args.seed + 7)
    t0 = time.time()
    hist = []
    best = -1.0

    for it in range(1, args.iters + 1):
        x, y, meta = mqar.make_batch(
            rng, args.batch, args.block,
            pairs_range=(2, args.pairs_max), filler_range=(0, args.filler_max))
        xt = torch.from_numpy(x).to(device)
        yt = torch.from_numpy(y).to(device)

        logits, _ = model(xt)
        loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), yt.reshape(-1),
                               ignore_index=mqar.PAD)
        if args.answer_weight != 1.0:
            bi, pi = [], []
            for b, m in enumerate(meta):
                for p in m["ans_positions"]:
                    bi.append(b)
                    pi.append(p)
            bi = torch.tensor(bi, device=device)
            pi = torch.tensor(pi, device=device)
            loss = loss + (args.answer_weight - 1.0) * F.cross_entropy(
                logits[bi, pi], yt[bi, pi]
            )

        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()

        if it % 500 == 0 or it == args.iters:
            ev = evaluate(model, device, args.block, n_seq=384,
                          pairs_range=(2, args.pairs_max),
                          filler_range=(0, args.filler_max))
            hist.append({"iter": it, "loss": float(loss), **ev})
            print(f"  it {it:5d}  loss {float(loss):.4f}  "
                  f"recall {ev['recall_acc']*100:5.1f}%  "
                  f"(context-city {ev['picks_a_context_city']*100:5.1f}%, "
                  f"copy-chance {ev['chance_if_copying_some_context_city']*100:4.1f}%)  "
                  f"{time.time()-t0:5.0f}s")
            if ev["recall_acc"] > best:
                best = ev["recall_acc"]
                out = ROOT / args.out
                out.parent.mkdir(parents=True, exist_ok=True)
                torch.save({"cfg": cfg.__dict__, "model": model.state_dict(),
                            "iter": it, "eval": ev, "args": vars(args)}, out)

    ck = torch.load(ROOT / args.out, weights_only=False)
    model.load_state_dict(ck["model"])
    final = evaluate(model, device, args.block, n_seq=2048, seed=99,
                     pairs_range=(2, args.pairs_max),
                     filler_range=(0, args.filler_max))
    ctrl = eval_controls(model, device, args.block, n_seq=400)
    print("\nfinal (2048 sequences, best-recall checkpoint):")
    print(json.dumps(final, indent=2))
    print("\nbehavioural controls (6 pairs, 1 query):")
    print(json.dumps(ctrl, indent=2))
    (ROOT / "data" / "train_history.json").write_text(
        json.dumps({"history": hist, "final": final, "controls": ctrl,
                    "config": cfg.__dict__, "n_params": n_params,
                    "args": vars(args)}, indent=2)
    )
    print(f"\nbest checkpoint: {ROOT / args.out}  (recall {best*100:.1f}%)")


if __name__ == "__main__":
    main()
