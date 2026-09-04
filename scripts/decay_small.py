"""The stability-plasticity trade-off, measured where capacity actually binds.

The first decay experiment was inconclusive, and for the same reason the first
interference experiment was invalid: at d=64/mult=8 the state holds 32,768
entries per layer, which 8 bindings never stress. Every model sat at ~100% at
almost every position, so there was no forgetting for damping to trade against.
A trade-off cannot be measured in a regime where nothing is lost.

Here we use the narrow state (d=32, mult=2 -> 2,048 entries per layer) from the
capacity sweep, where recall of the oldest binding falls from 99.6% to 7.2% as
competing bindings accumulate. In that regime damping has something to do.

Prediction under test
---------------------
    u = 1     nothing decays; old bindings persist but every binding keeps
              competing for the same fixed state
    u < 1     old writes shrink geometrically, so recent bindings face less
              competition, at the cost of losing old ones

If that is right the curves should CROSS: damping worse for old bindings, no
worse -- possibly better -- for recent ones. If they do not cross, the
stability-plasticity story does not hold here and we report that instead.

Measurement hygiene
-------------------
Filler units separate the query from the binding block. A sharp isolated
failure can land on one of the two most recent bindings (DESIGN_NOTES 3c); its
index is model-dependent, so the two newest positions are reported but flagged,
and no decay conclusion is drawn from them.
"""

import json
import pathlib
import subprocess
import sys

import numpy as np
import torch

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

U_VALUES = [1.0, 0.95, 0.90, 0.80]
ITERS = 8000
D, MULT = 32, 2
PAIRS_MAX, FILLER_MAX, BLOCK = 14, 14, 96
N_PAIRS = 10
N_FILLER = 2
TRIALS = 400


def train_one(u):
    tag = f"u{str(u).replace('.', '')}"
    out = f"data/decaysm_{tag}.pt"
    if (ROOT / out).exists():
        print(f"  [{u}] checkpoint exists, skipping", flush=True)
        return out
    cmd = [
        sys.executable, str(ROOT / "src" / "train.py"),
        "--iters", str(ITERS), "--batch", "64", "--n-layer", "2",
        "--n-embd", str(D), "--mult", str(MULT), "--answer-weight", "8",
        "--pairs-max", str(PAIRS_MAX), "--filler-max", str(FILLER_MAX),
        "--block", str(BLOCK), "--u-decay", str(u), "--out", out,
    ]
    p = subprocess.run(cmd, capture_output=True, text=True, cwd=ROOT)
    if p.returncode != 0:
        print(f"  [{u}] FAILED\n{p.stderr[-1500:]}", flush=True)
        return None
    line = [l for l in p.stdout.splitlines() if "best checkpoint" in l]
    print(f"  [{u}] {line[-1].strip() if line else 'done'}", flush=True)
    return out


@torch.no_grad()
def recall_by_age(model, device, trials, seed):
    import mqar
    from ablate import wilson
    rows = []
    for qi in range(N_PAIRS):
        hits = n = 0
        for i in range(trials):
            ex = mqar.make_example(np.random.default_rng(seed + i),
                                   n_pairs=N_PAIRS, n_queries=1,
                                   n_filler=N_FILLER, query_idx=qi)
            ids = torch.from_numpy(ex["ids"][:-1]).unsqueeze(0).to(device)
            logits, _ = model(ids)
            hits += int(int(logits[0, ex["ans_pos"]].argmax()) == ex["answer"])
            n += 1
        rows.append({
            "query_idx": qi,
            "n_written_after": N_PAIRS - 1 - qi,
            "recall": hits / n, "ci": wilson(hits, n), "n": n,
            # the two newest positions can carry the DESIGN_NOTES 3c artifact
            "flagged_position": qi >= N_PAIRS - 2,
        })
    return rows


def main():
    from bdh import BDH, BDHConfig
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"decay at narrow state: d={D} mult={MULT} "
          f"(state {D*D*MULT:,}/layer), {N_PAIRS} bindings, {ITERS} iters\n",
          flush=True)

    results = []
    for u in U_VALUES:
        ckpt = train_one(u)
        if ckpt is None:
            continue
        ck = torch.load(ROOT / ckpt, weights_only=False)
        cfg = BDHConfig(**ck["cfg"])
        model = BDH(cfg).to(device)
        model.load_state_dict(ck["model"])
        model.eval()
        rows = recall_by_age(model, device, TRIALS, seed=53)
        results.append({"u_decay": u, "overall_recall": ck["eval"]["recall_acc"],
                        "by_age": rows})
        curve = "  ".join(
            (f"{r['recall']*100:5.1f}" + ("*" if r["flagged_position"] else " "))
            for r in rows)
        print(f"  u={u:<5} oldest->newest  {curve}   (overall "
              f"{ck['eval']['recall_acc']*100:.1f}%)", flush=True)

    print("\n  * = one of the two newest positions; may carry the "
          "DESIGN_NOTES 3c artifact, no decay conclusion drawn from these")

    (ROOT / "artifact" / "data").mkdir(parents=True, exist_ok=True)
    (ROOT / "artifact" / "data" / "decay_small.json").write_text(json.dumps({
        "note": ("Damping case of U from Definition 4 of arXiv:2509.26507, at a "
                 "state width where interference is measurable. u=1.0 is the "
                 "public reference behaviour (rotation only). Each u is a "
                 "separate training run. The two newest positions are flagged "
                 "because a model-dependent positional artifact can land there."),
        "d": D, "mult": MULT, "state_entries_per_layer": D * D * MULT,
        "n_pairs": N_PAIRS, "n_filler": N_FILLER, "iters": ITERS,
        "trials_per_point": TRIALS, "results": results,
    }, indent=2))
    print(f"\nwrote {ROOT / 'artifact' / 'data' / 'decay_small.json'}")


if __name__ == "__main__":
    main()
