"""Is the recall dip a positional blind spot, and does it track distance?

Observed: with 8 bindings and one query, recall by binding index reads
(oldest -> newest)

    u=1.00   98.2 100.0 100.0 100.0 100.0 100.0  58.2 100.0
    u=0.98   99.5 100.0 100.0 100.0 100.0 100.0  99.5 100.0
    u=0.95  100.0 100.0 100.0 100.0 100.0 100.0  39.0  93.8

A sharp isolated failure at index 6 that recurs across independent training
runs, so it is unlikely to be seed noise. Index 6 of 8 puts the queried
binding's city 3 tokens before the query.

Two candidate explanations, which this script separates:

  DISTANCE -- the failure sits at a fixed query-to-binding offset. BDH applies
              RoPE to the sparse activations, and the attention score is
              <rope_t(x_t), rope_tau(x_tau)>, so the retrieval kernel is a
              function of (t - tau). A specific offset can fall in a
              phase-cancellation zone, giving a genuine positional blind spot.

  POSITION -- the failure sits at a fixed absolute index, which would point at
              something about sequence construction instead.

Varying the number of bindings moves absolute position and relative distance
apart. If the dip tracks the offset, it is the RoPE kernel.
"""

import json
import pathlib
import sys

import numpy as np
import torch

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from bdh import BDH, BDHConfig  # noqa: E402
from ablate import wilson  # noqa: E402
import mqar  # noqa: E402


@torch.no_grad()
def scan(model, device, n_pairs, trials, seed):
    rows = []
    for qi in range(n_pairs):
        hits = n = 0
        for i in range(trials):
            ex = mqar.make_example(np.random.default_rng(seed + i),
                                   n_pairs=n_pairs, n_queries=1, query_idx=qi)
            ids = torch.from_numpy(ex["ids"][:-1]).unsqueeze(0).to(device)
            logits, _ = model(ids)
            hits += int(int(logits[0, ex["ans_pos"]].argmax()) == ex["answer"])
            n += 1
        # the queried binding's city token, and the query position
        city_pos = ex["write_pos"]
        q_pos = ex["ans_pos"]
        rows.append({
            "query_idx": qi, "n_pairs": n_pairs,
            "city_pos": city_pos, "query_pos": q_pos,
            "offset": q_pos - city_pos,
            "recall": hits / n, "ci": wilson(hits, n), "n": n,
        })
    return rows


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    ck_path = ROOT / "data" / "decay_u095.pt"  # the run with the largest dip
    if not ck_path.exists():
        print(f"missing {ck_path}")
        return
    ck = torch.load(ck_path, weights_only=False)
    cfg = BDHConfig(**ck["cfg"])
    model = BDH(cfg).to(device)
    model.load_state_dict(ck["model"])
    model.eval()
    print(f"checkpoint {ck_path.name} (u_decay={cfg.u_decay}, iter {ck['iter']})\n")

    out = []
    for n_pairs in (6, 7, 8, 9, 10):
        rows = scan(model, device, n_pairs, trials=300, seed=77)
        out.extend(rows)
        worst = min(rows, key=lambda r: r["recall"])
        line = "  ".join(f"{r['recall']*100:5.1f}" for r in rows)
        print(f"n_pairs={n_pairs:2d}  {line}")
        print(f"{'':13s}worst: idx {worst['query_idx']} "
              f"(city_pos {worst['city_pos']}, query_pos {worst['query_pos']}, "
              f"offset {worst['offset']}) = {worst['recall']*100:.1f}%\n")

    (ROOT / "artifact" / "data").mkdir(parents=True, exist_ok=True)
    (ROOT / "artifact" / "data" / "position_dip.json").write_text(
        json.dumps({"checkpoint": ck_path.name, "config": cfg.__dict__,
                    "rows": out}, indent=2))

    # does the minimum sit at a constant offset or a constant index?
    by_np = {}
    for r in out:
        by_np.setdefault(r["n_pairs"], []).append(r)
    print("summary")
    for np_, rows in sorted(by_np.items()):
        w = min(rows, key=lambda r: r["recall"])
        print(f"  n_pairs={np_:2d}: worst at index {w['query_idx']}, "
              f"offset {w['offset']}, recall {w['recall']*100:.1f}%")


if __name__ == "__main__":
    main()
