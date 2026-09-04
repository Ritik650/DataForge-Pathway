"""The stability-plasticity trade-off, driven by the paper's own U matrix.

Definition 4 of arXiv:2509.26507 writes the state update with a right
multiplication by U, "a diagonal or block-diagonal matrix representing local
rotation or damping of state (such as ALiBi or RoPE)". The public reference
implementation instantiates U as RoPE -- rotation only, u = 1, no damping.
Here we train models across the damping case of that same U.

Why train separate models rather than turning u down at inference
-----------------------------------------------------------------
Changing u only at inference would confound the effect of damping with a
train/test mismatch: the model would be run under dynamics it never learned,
and any degradation would be uninterpretable. Each u gets its own training run.

What the experiment measures
----------------------------
Recall as a function of the queried binding's AGE -- how many bindings were
written after it -- at fixed total load. That is the axis on which damping
should trade:

    u = 1     nothing decays, so old bindings persist, but every binding keeps
              competing for the same fixed-size state -> interference
    u < 1     old writes shrink geometrically, so recent bindings face less
              competition, at the cost of losing old ones

If that is what the architecture does, the curves should cross: damping should
be worse for old bindings and no worse -- possibly better -- for recent ones.
If they do not cross, the trade-off story is wrong and we report that instead.
"""

import json
import pathlib
import subprocess
import sys

import numpy as np
import torch

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

U_VALUES = [1.0, 0.98, 0.95, 0.90]
ITERS = 6000
N_PAIRS = 8
TRIALS = 400


def train_one(u):
    tag = f"u{str(u).replace('.', '')}"
    out = f"data/decay_{tag}.pt"
    if (ROOT / out).exists():
        print(f"  [{u}] checkpoint exists, skipping training")
        return out
    cmd = [
        sys.executable, str(ROOT / "src" / "train.py"),
        "--iters", str(ITERS), "--batch", "64", "--n-layer", "2",
        "--mult", "8", "--answer-weight", "8", "--u-decay", str(u),
        "--out", out,
    ]
    p = subprocess.run(cmd, capture_output=True, text=True, cwd=ROOT)
    if p.returncode != 0:
        print(f"  [{u}] FAILED\n{p.stderr[-1500:]}")
        return None
    line = [l for l in p.stdout.splitlines() if "best checkpoint" in l]
    print(f"  [{u}] {line[-1].strip() if line else 'done'}")
    return out


@torch.no_grad()
def recall_by_age(model, device, n_pairs, trials, seed, n_filler=1):
    """Recall vs number of bindings written AFTER the queried one.

    n_filler=1 is deliberate, not decoration. With the query placed directly
    against the binding block, the second-to-last binding suffers a severe
    isolated failure (see docs/DESIGN_NOTES.md 3c) that has nothing to do with
    damping. Two filler tokens abolish it, so every point here is measuring the
    effect of U rather than that artifact.
    """
    import mqar
    rows = []
    for qi in range(n_pairs):
        hits = n = 0
        for i in range(trials):
            ex = mqar.make_example(np.random.default_rng(seed + i),
                                   n_pairs=n_pairs, n_queries=1,
                                   n_filler=n_filler, query_idx=qi)
            ids = torch.from_numpy(ex["ids"][:-1]).unsqueeze(0).to(device)
            logits, _ = model(ids)
            hits += int(int(logits[0, ex["ans_pos"]].argmax()) == ex["answer"])
            n += 1
        rows.append({"query_idx": qi, "n_written_after": n_pairs - 1 - qi,
                     "recall": hits / n, "n": n})
    return rows


def main():
    from bdh import BDH, BDHConfig
    from ablate import wilson

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"training {len(U_VALUES)} models across the damping case of U "
          f"({ITERS} iters each)")
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
        rows = recall_by_age(model, device, N_PAIRS, TRIALS, seed=31)
        for r in rows:
            k = int(round(r["recall"] * r["n"]))
            r["ci"] = wilson(k, r["n"])
        results.append({"u_decay": u, "overall_recall": ck["eval"]["recall_acc"],
                        "by_age": rows})
        curve = "  ".join(f"{r['recall']*100:5.1f}" for r in rows)
        print(f"  u={u}: recall by age (oldest->newest) {curve}")

    (ROOT / "artifact" / "data").mkdir(parents=True, exist_ok=True)
    (ROOT / "artifact" / "data" / "decay.json").write_text(json.dumps({
        "note": ("Each u is a separate training run. u=1.0 reproduces the public "
                 "BDH reference (U = RoPE rotation only). u<1 is the damping case "
                 "of the same U from Definition 4 of arXiv:2509.26507. One filler "
                 "unit separates the query from the binding block to avoid the "
                 "second-to-last-binding artifact documented in DESIGN_NOTES 3c."),
        "iters": ITERS, "n_pairs": N_PAIRS, "trials_per_point": TRIALS,
        "results": results,
    }, indent=2))
    print(f"\nwrote {ROOT / 'artifact' / 'data' / 'decay.json'}")


if __name__ == "__main__":
    main()
