"""Interference, measured without the extrapolation confound.

The first attempt at the interference curve was invalid. Training covered 2-8
bindings but only 0-4 filler units, so each condition began degrading shortly
after leaving ITS OWN training range -- filler at load 7, bindings at load 9.
The curves were measuring distance-outside-distribution, not the mechanism.
Inside the shared training range both sat at 100%, because a state of 32,768
entries is nowhere near stressed by 16 bindings.

Two changes make the comparison mean something:

1. Matched coverage. Training now spans 2-14 bindings AND 0-14 filler units, so
   every point on both curves is in-distribution. Neither condition gets to
   look worse merely by being further from what the model saw.

2. Capacity that actually binds. The state must be small enough that bindings
   compete for it. We sweep the state width and look for a model where the
   binding curve bends while the filler curve stays flat -- both in-distribution.

The prediction being tested: recall degrades with the number of competing
BINDINGS but not with an equal number of tokens carrying no association. If
both curves bend together, the loss is length-driven and the interference claim
is wrong. If neither bends, the state is too big to be stressed and the
experiment says nothing.
"""

import json
import pathlib
import subprocess
import sys

import numpy as np
import torch

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

# (n_embd, mult, iters). Iteration counts differ ON PURPOSE. Comparing state
# widths at a fixed budget confounds capacity with training quality: at 8000
# iters the two widest models failed at load 0 -- a single binding, with nothing
# to interfere with -- and their filler controls degraded too. A model that
# cannot hold ONE binding says nothing about how many it can hold. Wider models
# get the budget they need to clear the same load-0 gate.
CONFIGS = [
    (32, 2, 8000),    # N=64,   state 2,048
    (32, 4, 8000),    # N=128,  state 4,096
    (32, 8, 24000),   # N=256,  state 8,192
    (64, 8, 24000),   # N=512,  state 32,768
]
# A width only enters the capacity comparison if it recalls a lone binding this
# reliably. Below it, the curve measures undertraining, not capacity.
LOAD0_GATE = 0.95
PAIRS_MAX = 14
FILLER_MAX = 14
BLOCK = 96
LOADS = list(range(0, 14))   # every point in-distribution for both conditions
TRIALS = 250


def train_one(d, mult, iters):
    tag = f"d{d}m{mult}i{iters}"
    out = f"data/cap_{tag}.pt"
    if (ROOT / out).exists():
        print(f"  [{tag}] checkpoint exists, skipping training", flush=True)
        return out
    cmd = [
        sys.executable, str(ROOT / "src" / "train.py"),
        "--iters", str(iters), "--batch", "64", "--n-layer", "2",
        "--n-embd", str(d), "--mult", str(mult), "--answer-weight", "8",
        "--pairs-max", str(PAIRS_MAX), "--filler-max", str(FILLER_MAX),
        "--block", str(BLOCK), "--out", out,
    ]
    p = subprocess.run(cmd, capture_output=True, text=True, cwd=ROOT)
    if p.returncode != 0:
        print(f"  [{tag}] FAILED\n{p.stderr[-1500:]}", flush=True)
        return None
    line = [l for l in p.stdout.splitlines() if "best checkpoint" in l]
    print(f"  [{tag}] {line[-1].strip() if line else 'done'}", flush=True)
    return out


@torch.no_grad()
def curves(model, device, loads, trials, seed):
    import mqar
    from ablate import wilson
    rows = []
    for k in loads:
        res = {}
        for cond in ("bindings", "filler"):
            hits = n = 0
            for i in range(trials):
                rng = np.random.default_rng(seed + i + (0 if cond == "bindings" else 90000))
                if cond == "bindings":
                    ex = mqar.make_example(rng, n_pairs=1 + k, n_queries=1,
                                           n_filler=0, query_idx=0)
                else:
                    ex = mqar.make_example(rng, n_pairs=1, n_queries=1,
                                           n_filler=k, query_idx=0)
                ids = torch.from_numpy(ex["ids"][:-1]).unsqueeze(0).to(device)
                logits, _ = model(ids)
                hits += int(int(logits[0, ex["ans_pos"]].argmax()) == ex["answer"])
                n += 1
            res[cond] = {"recall": hits / n, "ci": wilson(hits, n), "n": n}
        rows.append({"load": k, **res,
                     "both_in_training_range": (1 + k) <= PAIRS_MAX and k <= FILLER_MAX})
        print(f"    load {k:2d}  bindings {res['bindings']['recall']*100:5.1f}%"
              f" [{res['bindings']['ci'][0]*100:.1f},{res['bindings']['ci'][1]*100:.1f}]"
              f"   filler {res['filler']['recall']*100:5.1f}%"
              f" [{res['filler']['ci'][0]*100:.1f},{res['filler']['ci'][1]*100:.1f}]",
              flush=True)
    return rows


def main():
    from bdh import BDH, BDHConfig
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"training {len(CONFIGS)} widths to a load-0 gate of "
          f"{LOAD0_GATE*100:.0f}%, bindings 2-{PAIRS_MAX} and filler "
          f"0-{FILLER_MAX} both in training\n", flush=True)

    results = []
    for d, mult, iters in CONFIGS:
        tag = f"d{d}m{mult}i{iters}"
        ckpt = train_one(d, mult, iters)
        if ckpt is None:
            continue
        ck = torch.load(ROOT / ckpt, weights_only=False)
        cfg = BDHConfig(**ck["cfg"])
        model = BDH(cfg).to(device)
        model.load_state_dict(ck["model"])
        model.eval()
        state = cfg.n_embd * cfg.N * cfg.n_head
        print(f"  [{tag}] n={cfg.N*cfg.n_head} d={cfg.n_embd} "
              f"state={state:,}/layer  overall recall "
              f"{ck['eval']['recall_acc']*100:.1f}%", flush=True)
        rows = curves(model, device, LOADS, TRIALS, seed=41)
        load0 = rows[0]["bindings"]["recall"]
        passed = load0 >= LOAD0_GATE
        print(f"    load-0 gate: {load0*100:.1f}% "
              f"({'PASS' if passed else 'FAIL - excluded from capacity comparison'})",
              flush=True)
        results.append({"tag": tag, "iters": iters, "load0_recall": load0,
                        "passes_load0_gate": passed, "n_embd": cfg.n_embd,
                        "n_neurons": cfg.N * cfg.n_head,
                        "state_entries_per_layer": state,
                        "n_params": sum(p.numel() for p in model.parameters()),
                        "overall_recall": ck["eval"]["recall_acc"],
                        "curves": rows})

    (ROOT / "artifact" / "data").mkdir(parents=True, exist_ok=True)
    (ROOT / "artifact" / "data" / "interference.json").write_text(json.dumps({
        "note": ("Both conditions are in-distribution at every point: training "
                 f"covered 2-{PAIRS_MAX} bindings and 0-{FILLER_MAX} filler units. "
                 "Each filler unit costs the same 2 tokens as one binding but "
                 "carries no association, so the two curves hold sequence length "
                 "matched and vary only whether the tokens bind anything."),
        "trials_per_point": TRIALS, "loads": LOADS,
        "load0_gate": LOAD0_GATE,
        "gate_note": ("Widths are trained to a quality gate, not a fixed step "
                      "budget. A width failing load-0 recall cannot support any "
                      "capacity claim, since it cannot hold a single binding."),
        "results": results,
    }, indent=2))
    print(f"\nwrote {ROOT / 'artifact' / 'data' / 'interference.json'}")


if __name__ == "__main__":
    main()
