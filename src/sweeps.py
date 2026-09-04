"""The three precomputed datasets the explainer ships.

1. dose      -- recall vs number of ablated synapses m, for targeted /
                magnitude-matched / uniform-random selection. The dose-response
                curve is what turns "ablation hurts" into "ablation hurts in
                proportion to how much of THIS binding you remove".

2. interference -- recall vs number of competing bindings, AND recall vs an
                equal number of neutral filler tokens. Two curves on one axis.
                If only the first bends, forgetting is interference between
                stored associations rather than sequence-length decay. Running
                one without the other proves nothing, which is why they are
                produced together and plotted together.

3. capacity  -- recall vs interference load at several state widths, to show
                the fixed-size state has a capacity rather than a horizon.

Every proportion is reported with a Wilson 95% interval and its n.
"""

import argparse
import json
import pathlib
import sys

import numpy as np
import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

from bdh import BDH, BDHConfig  # noqa: E402
from ablate import (load, run_trial, wilson)  # noqa: E402
import mqar  # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parents[1]


# Dose ladder as FRACTIONS of the state, not absolute counts. An absolute
# ladder calibrated on one width is meaningless on another: m=2048 is 6.25% of
# a 32,768-entry state but 100% of a 2,048-entry one, which collapses the curve.
# Fractions keep the span identical (0.05% -> 12.5%) and make dose curves
# directly comparable across state widths.
DOSE_FRACTIONS = [0.0005, 0.001, 0.002, 0.004, 0.008, 0.016, 0.031, 0.0625, 0.125]


def dose_response(model, cfg, device, fractions, trials, seed, pairs=6):
    layer = cfg.n_layer - 1
    n_state = cfg.n_embd * cfg.N * cfg.n_head
    rng = np.random.default_rng(seed)
    rows = []
    # de-duplicate: on a narrow state, small fractions can round to the same m
    m_list = sorted({max(1, int(round(f * n_state))) for f in fractions})
    for m in m_list:
        agg = {k: 0 for k in ("targeted", "matched", "random")}
        mags = {k: [] for k in agg}
        n_used = 0
        for i in range(trials):
            ex = mqar.make_example(np.random.default_rng(seed + i), n_pairs=pairs,
                                   n_queries=1, block=65)
            r = run_trial(model, cfg, device, ex, layer, m, rng)
            if not r["base_correct"]:
                continue
            n_used += 1
            for k in agg:
                agg[k] += r["correct"][k]
                mags[k].append(r["removed_magnitude"][k])
        row = {"m": m, "n": n_used,
               "frac_of_state": m / (cfg.n_embd * cfg.N * cfg.n_head)}
        for k in agg:
            row[k] = agg[k] / max(1, n_used)
            row[k + "_ci"] = wilson(agg[k], n_used)
            row[k + "_mass"] = float(np.mean(mags[k])) if mags[k] else 0.0
        rows.append(row)
        print(f"  m={m:4d} ({row['frac_of_state']*100:6.3f}% of state)  "
              f"targeted {row['targeted']*100:5.1f}%  "
              f"matched {row['matched']*100:5.1f}%  "
              f"random {row['random']*100:5.1f}%   n={n_used}")
    return rows


@torch.no_grad()
def recall_at(model, device, n_pairs, n_filler, trials, seed, query_first=True):
    """Recall of the FIRST-written binding under a given load."""
    hits = n = 0
    for i in range(trials):
        rng = np.random.default_rng(seed + i)
        ex = mqar.make_example(rng, n_pairs=n_pairs, n_queries=1,
                               n_filler=n_filler, query_idx=0, block=None)
        ids = torch.from_numpy(ex["ids"][:-1]).unsqueeze(0).to(device)
        logits, _ = model(ids)
        hits += int(int(logits[0, ex["ans_pos"]].argmax()) == ex["answer"])
        n += 1
    return hits, n


def interference(model, device, loads, trials, seed, pairs_max=8, filler_max=4):
    """Two matched curves: competing bindings vs neutral filler.

    Both cost 2 tokens per unit, so at load k the sequences are the same length.
    The only difference is whether those tokens carry an association.

    pairs_max/filler_max come from the checkpoint's own training arguments, not
    from a constant. Different checkpoints were trained on different ranges, and
    mislabelling which points are in-distribution is exactly the confound that
    invalidated the first version of this experiment (DESIGN_NOTES 3b).
    """
    rows = []
    for k in loads:
        # k competing bindings after the queried one, no filler
        hb, nb = recall_at(model, device, n_pairs=1 + k, n_filler=0,
                           trials=trials, seed=seed)
        # 1 binding, k filler units -> same token count, no competition
        hf, nf = recall_at(model, device, n_pairs=1, n_filler=k,
                           trials=trials, seed=seed + 777)
        rows.append({
            "load": k,
            "bindings_recall": hb / nb, "bindings_ci": wilson(hb, nb), "n": nb,
            "filler_recall": hf / nf, "filler_ci": wilson(hf, nf),
            "seq_len_tokens": 1 + 2 * (1 + k) + 2,
            # beyond the checkpoint's own training range the model is
            # extrapolating and the artifact must say so
            "bindings_in_training_range": (1 + k) <= pairs_max,
            "filler_in_training_range": k <= filler_max,
        })
        print(f"  load {k:2d}  competing bindings {hb/nb*100:5.1f}% "
              f"[{wilson(hb,nb)[0]*100:.1f},{wilson(hb,nb)[1]*100:.1f}]   "
              f"equal-length filler {hf/nf*100:5.1f}% "
              f"[{wilson(hf,nf)[0]*100:.1f},{wilson(hf,nf)[1]*100:.1f}]   n={nb}")
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="data/bdh_mqar.pt")
    ap.add_argument("--trials", type=int, default=200)
    ap.add_argument("--seed", type=int, default=21)
    ap.add_argument("--out", default="artifact/data")
    ap.add_argument("--name", default=None,
                    help="output basename; defaults to 'sweeps'")
    args = ap.parse_args()

    model, cfg, device, ck = load(args.ckpt)
    n_state = cfg.n_embd * cfg.N * cfg.n_head
    print(f"checkpoint {args.ckpt} (iter {ck['iter']})  "
          f"n={cfg.N*cfg.n_head} neurons, d={cfg.n_embd}, "
          f"state = {n_state:,} entries per layer\n")

    print("dose-response (ablated synapses vs recall):")
    dose = dose_response(model, cfg, device, DOSE_FRACTIONS,
                         args.trials, args.seed)

    tr = ck.get("args", {})
    pairs_max = tr.get("pairs_max", 8)
    filler_max = tr.get("filler_max", 4)
    print(f"\ninterference: competing bindings vs equal-length neutral filler"
          f"  (this checkpoint trained on 2-{pairs_max} bindings, "
          f"0-{filler_max} filler)")
    inter = interference(model, device, list(range(0, 15)), args.trials,
                         args.seed, pairs_max=pairs_max, filler_max=filler_max)

    out = pathlib.Path(ROOT / args.out)
    out.mkdir(parents=True, exist_ok=True)
    payload = {
        "checkpoint": args.ckpt, "iter": ck["iter"], "config": cfg.__dict__,
        "n_state_entries_per_layer": n_state,
        "trials_per_point": args.trials, "seed": args.seed,
        "trained_pairs_max": pairs_max, "trained_filler_max": filler_max,
        "n_params": sum(p.numel() for p in model.parameters()),
        "dose_fractions": DOSE_FRACTIONS,
        "dose_response": dose,
        "interference": inter,
    }
    name = (args.name or "sweeps") + ".json"
    (out / name).write_text(json.dumps(payload, indent=2))
    print(f"\nwrote {out / name}")


if __name__ == "__main__":
    main()
