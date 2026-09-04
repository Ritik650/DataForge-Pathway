"""Characterise the positional failure. It has been "observed, not characterised"
since DESIGN_NOTES 3c; this maps it well enough to state a rule.

What we already know, and what we got wrong twice:

  - A severe isolated recall failure lands near the end of the binding block.
  - "Offset 3 from the query" was refuted: adding filler did not move the
    failure to whichever binding then sat at offset 3.
  - "Two filler tokens abolish it" was refuted: filler MOVED the failure for
    u=1.00 rather than removing it, and u=0.98 never showed it at all.
  - "The interference sweep is immune because it queries index 0" was refuted:
    at 2 bindings, index 0 IS the second-to-last binding.

So this scan varies all three axes at once on the SHIPPED substrate and asks a
single question: as a function of (n_pairs, query_idx, n_filler), where does
recall fail? Specifically, is the failing position better described by

    ABSOLUTE     query_idx == some fixed index
    RECENCY      n_pairs - 1 - query_idx == some fixed distance from the end
    OFFSET       token distance from the query back to the binding

Recency and offset are separable here because filler changes the token distance
without changing how many bindings follow the queried one.

Output feeds Act 2 of the artifact, so it must be a measured map, not a story.
"""

import argparse
import json
import pathlib
import sys

import numpy as np
import torch

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ablate import load, wilson  # noqa: E402
import mqar  # noqa: E402


@torch.no_grad()
def cell(model, device, n_pairs, query_idx, n_filler, trials, seed):
    hits = n = 0
    ex = None
    for i in range(trials):
        ex = mqar.make_example(np.random.default_rng(seed + i), n_pairs=n_pairs,
                               n_queries=1, n_filler=n_filler,
                               query_idx=query_idx)
        ids = torch.from_numpy(ex["ids"][:-1]).unsqueeze(0).to(device)
        logits, _ = model(ids)
        hits += int(int(logits[0, ex["ans_pos"]].argmax()) == ex["answer"])
        n += 1
    return hits, n, ex["ans_pos"] - ex["write_pos"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="data/artifact_d32m8.pt")
    ap.add_argument("--trials", type=int, default=250)
    ap.add_argument("--seed", type=int, default=202)
    ap.add_argument("--fail-below", type=float, default=0.90)
    ap.add_argument("--out", default="artifact/data/positional_map.json")
    args = ap.parse_args()

    model, cfg, device, ck = load(args.ckpt)
    tr = ck.get("args", {})
    print(f"substrate {args.ckpt}: d={cfg.n_embd} N={cfg.N*cfg.n_head} "
          f"trained 2-{tr.get('pairs_max')} bindings, 0-{tr.get('filler_max')} filler")
    print(f"scanning n_pairs x query_idx x n_filler, n={args.trials} per cell\n")

    rows = []
    for nf in (0, 1, 2, 3):
        print(f"n_filler={nf}")
        for npairs in range(2, 9):
            accs = []
            for qi in range(npairs):
                h, n, off = cell(model, device, npairs, qi, nf,
                                 args.trials, args.seed)
                acc = h / n
                accs.append(acc)
                rows.append({
                    "n_pairs": npairs, "query_idx": qi, "n_filler": nf,
                    "from_end": npairs - 1 - qi, "token_offset": int(off),
                    "recall": acc, "ci": wilson(h, n), "n": n,
                    "fail": acc < args.fail_below,
                })
            line = "  ".join(f"{a*100:5.1f}" for a in accs)
            worst = int(np.argmin(accs))
            flag = "  <-- fail" if accs[worst] < args.fail_below else ""
            print(f"  pairs={npairs}  {line}   worst idx {worst} "
                  f"(from end {npairs-1-worst}){flag}")
        print()

    fails = [r for r in rows if r["fail"]]
    print(f"cells below {args.fail_below*100:.0f}%: {len(fails)} of {len(rows)}")

    if fails:
        for key, label in (("query_idx", "ABSOLUTE index"),
                           ("from_end", "RECENCY (bindings after it)"),
                           ("token_offset", "OFFSET (tokens back to binding)")):
            vals = [r[key] for r in fails]
            uniq = sorted(set(vals))
            # how concentrated is the failure on one value of this descriptor?
            best = max(uniq, key=lambda v: vals.count(v))
            frac = vals.count(best) / len(vals)
            print(f"  by {label:32s} -> {frac*100:5.1f}% of failures at "
                  f"{key}={best}  (values seen: {uniq})")

        print("\n  failing cells:")
        for r in sorted(fails, key=lambda r: r["recall"]):
            print(f"    pairs={r['n_pairs']} idx={r['query_idx']} "
                  f"filler={r['n_filler']}  from_end={r['from_end']} "
                  f"offset={r['token_offset']}  recall {r['recall']*100:5.1f}% "
                  f"[{r['ci'][0]*100:.1f},{r['ci'][1]*100:.1f}]")

    out = ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "checkpoint": args.ckpt, "config": cfg.__dict__,
        "trained_pairs_max": tr.get("pairs_max"),
        "trained_filler_max": tr.get("filler_max"),
        "trials_per_cell": args.trials, "fail_below": args.fail_below,
        "rows": rows,
    }, indent=2))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
