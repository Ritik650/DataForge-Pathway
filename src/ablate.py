"""Localisation: is a specific binding held in a specific, findable set of
synapses, or is it smeared across the state?

The experiment
--------------
For one sequence with a known queried binding:

  1. Read the rank-1 Hebbian write `delta = LN(E y) x^T` deposited at the token
     that carries the binding. Its largest-magnitude entries are the candidate
     synapses.
  2. Ablate them: force those entries of `rho` to zero from the write onward,
     so the state evolves as if that mass had never been deposited.
  3. Re-run and read the answer.

Step 3 alone proves nothing. Removing any state mass degrades a model. The
result only means something against a control that removes *the same amount of
state mass from somewhere else*:

  MATCHED  -- m entries drawn from outside the target set whose |rho| values are
              nearest-neighbour matched to the target's. This is the control
              that can falsify the claim.
  RANDOM   -- m entries drawn uniformly. Reported to show that the naive control
              is too weak: it removes far less magnitude and would flatter us.

If targeted and MATCHED ablation degrade recall equally, localisation is false
and we say so.

Specificity
-----------
A localised binding should be separable from its neighbours. With several
bindings queried in one sequence, ablating the synapses of binding A must break
A's recall while leaving B and C intact. A model whose state is smeared will
lose all three together.
"""

import argparse
import json
import pathlib
import sys

import numpy as np
import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

from bdh import BDH, BDHConfig  # noqa: E402
import mqar  # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parents[1]


def load(ckpt="data/bdh_mqar.pt", device=None):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    ck = torch.load(ROOT / ckpt, weights_only=False)
    cfg = BDHConfig(**ck["cfg"])
    model = BDH(cfg).to(device)
    model.load_state_dict(ck["model"])
    model.eval()
    return model, cfg, device, ck


def make_mask_ablator(layer, from_t, idx):
    """Zero state entries `idx` in `layer` from timestep `from_t` onward.

    idx is a tuple of index tensors addressing rho[b, h, d, n].
    """
    def ablate(l, t, rho):
        if l == layer and t >= from_t:
            rho = rho.clone()
            rho[idx] = 0.0
        return rho
    return ablate


def select_targeted(delta_write, m):
    """Top-m entries of the Hebbian write by magnitude. delta: (B,nh,D,N)."""
    flat = delta_write.abs().flatten()
    top = torch.topk(flat, m).indices
    return top


def select_matched(rho_final, targeted_flat, m, rng):
    """Nearest-neighbour match on |rho| from outside the target set.

    For each targeted entry we find an untargeted entry with the closest |rho|,
    sampling among the closest candidates so the control is not deterministic.
    """
    flat = rho_final.abs().flatten()
    n = flat.numel()
    mask = torch.ones(n, dtype=torch.bool, device=flat.device)
    mask[targeted_flat] = False
    cand_idx = torch.nonzero(mask, as_tuple=True)[0]
    cand_val = flat[cand_idx]

    order = torch.argsort(cand_val)
    cand_idx_sorted = cand_idx[order]
    cand_val_sorted = cand_val[order]

    targets = flat[targeted_flat]
    picked, used = [], set()
    for v in targets.tolist():
        j = int(torch.searchsorted(cand_val_sorted, torch.tensor(v, device=flat.device)))
        # walk outward from the insertion point to the nearest unused candidate
        lo, hi = j - 1, j
        while True:
            cands = []
            if hi < len(cand_idx_sorted):
                cands.append((abs(float(cand_val_sorted[hi]) - v), hi))
            if lo >= 0:
                cands.append((abs(float(cand_val_sorted[lo]) - v), lo))
            if not cands:
                break
            cands.sort()
            _, k = cands[0]
            if k not in used:
                used.add(k)
                picked.append(int(cand_idx_sorted[k]))
                break
            if k == hi:
                hi += 1
            else:
                lo -= 1
    return torch.tensor(picked, device=flat.device, dtype=torch.long)


def select_random(rho_final, targeted_flat, m, rng):
    n = rho_final.numel()
    mask = np.ones(n, dtype=bool)
    mask[targeted_flat.cpu().numpy()] = False
    pool = np.nonzero(mask)[0]
    pick = rng.choice(pool, size=m, replace=False)
    return torch.tensor(pick, device=rho_final.device, dtype=torch.long)


def unflatten(flat_idx, shape):
    return torch.unravel_index(flat_idx, shape)


@torch.no_grad()
def run_trial(model, cfg, device, ex, layer, m, rng):
    """One sequence: baseline, targeted, matched-random, uniform-random."""
    ids = torch.from_numpy(ex["ids"][:-1]).unsqueeze(0).to(device)
    ans_pos, answer = ex["ans_pos"], ex["answer"]
    write_pos = ex["write_pos"]

    base_logits, info = model.forward_recurrent(ids, record=True)
    base_pred = int(base_logits[0, ans_pos].argmax())

    delta_write = info["delta"][layer][write_pos]  # (B,nh,D,N)
    rho_final = info["rho"][layer]
    shape = rho_final.shape

    tgt = select_targeted(delta_write, m)
    mat = select_matched(rho_final, tgt, m, rng)
    rnd = select_random(rho_final, tgt, m, rng)

    out = {
        "base_correct": int(base_pred == answer),
        "removed_magnitude": {},
        "correct": {},
        "answer_prob": {},
    }
    probs_base = torch.softmax(base_logits[0, ans_pos], -1)
    out["answer_prob"]["base"] = float(probs_base[answer])

    for name, sel in (("targeted", tgt), ("matched", mat), ("random", rnd)):
        idx = unflatten(sel, shape)
        out["removed_magnitude"][name] = float(rho_final.abs().flatten()[sel].sum())
        logits, _ = model.forward_recurrent(
            ids, ablate=make_mask_ablator(layer, write_pos, idx)
        )
        p = torch.softmax(logits[0, ans_pos], -1)
        out["correct"][name] = int(int(logits[0, ans_pos].argmax()) == answer)
        out["answer_prob"][name] = float(p[answer])

    return out


@torch.no_grad()
def run_specificity(model, cfg, device, ex, layer, m, rng):
    """Ablate binding 0's synapses; check binding 0 breaks and others survive."""
    ids = torch.from_numpy(ex["ids"][:-1]).unsqueeze(0).to(device)
    _, info = model.forward_recurrent(ids, record=True)
    delta_write = info["delta"][layer][ex["write_pos"]]
    rho_final = info["rho"][layer]
    tgt = select_targeted(delta_write, m)
    idx = unflatten(tgt, rho_final.shape)

    base_logits, _ = model.forward_recurrent(ids)
    abl_logits, _ = model.forward_recurrent(
        ids, ablate=make_mask_ablator(layer, ex["write_pos"], idx)
    )
    res = []
    for k, (pos, ans) in enumerate(zip(ex["ans_positions"], ex["answers"])):
        res.append({
            "is_target": k == 0,
            "base_correct": int(int(base_logits[0, pos].argmax()) == ans),
            "abl_correct": int(int(abl_logits[0, pos].argmax()) == ans),
        })
    return res


def wilson(k, n, z=1.96):
    """Wilson score interval. Reported on every proportion in this project."""
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, c - h), min(1.0, c + h))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="data/bdh_mqar.pt")
    ap.add_argument("--trials", type=int, default=200)
    ap.add_argument("--m", type=int, default=64, help="synapses ablated")
    ap.add_argument("--layer", type=int, default=-1, help="-1 = last layer")
    ap.add_argument("--pairs", type=int, default=6)
    ap.add_argument("--seed", type=int, default=11)
    ap.add_argument("--out", default="artifact/data/ablation.json")
    args = ap.parse_args()

    model, cfg, device, ck = load(args.ckpt)
    layer = args.layer if args.layer >= 0 else cfg.n_layer - 1
    print(f"checkpoint {args.ckpt} (iter {ck['iter']}), layer {layer}, "
          f"m={args.m} of {cfg.n_embd * cfg.N * cfg.n_head} state entries "
          f"({100*args.m/(cfg.n_embd*cfg.N*cfg.n_head):.3f}%)")

    rng = np.random.default_rng(args.seed)
    agg = {k: 0 for k in ("base", "targeted", "matched", "random")}
    mags = {k: [] for k in ("targeted", "matched", "random")}
    probs = {k: [] for k in ("base", "targeted", "matched", "random")}
    n_used = 0

    for i in range(args.trials):
        ex = mqar.make_example(np.random.default_rng(args.seed + i),
                               n_pairs=args.pairs, n_queries=1, block=65)
        r = run_trial(model, cfg, device, ex, layer, args.m, rng)
        if not r["base_correct"]:
            continue  # only meaningful where the model got it right unablated
        n_used += 1
        agg["base"] += 1
        for k in ("targeted", "matched", "random"):
            agg[k] += r["correct"][k]
            mags[k].append(r["removed_magnitude"][k])
        for k in ("base", "targeted", "matched", "random"):
            probs[k].append(r["answer_prob"][k])

    print(f"\ntrials where baseline was correct: {n_used} / {args.trials}")
    print(f"{'condition':<12}{'recall':>10}{'95% CI':>18}{'mean p(answer)':>17}"
          f"{'state mass removed':>21}")
    for k in ("base", "targeted", "matched", "random"):
        acc = agg[k] / max(1, n_used)
        lo, hi = wilson(agg[k], n_used)
        mag = f"{np.mean(mags[k]):.3f}" if k in mags else "-"
        print(f"{k:<12}{acc*100:9.1f}%  [{lo*100:5.1f}, {hi*100:5.1f}]"
              f"{np.mean(probs[k]):16.3f}{mag:>21}")

    # specificity
    spec_t = {"base": 0, "abl": 0, "n": 0}
    spec_o = {"base": 0, "abl": 0, "n": 0}
    for i in range(args.trials):
        ex = mqar.make_example(np.random.default_rng(args.seed + 5000 + i),
                               n_pairs=args.pairs, n_queries=3, block=65)
        for row in run_specificity(model, cfg, device, ex, layer, args.m, rng):
            d = spec_t if row["is_target"] else spec_o
            d["base"] += row["base_correct"]
            d["abl"] += row["abl_correct"]
            d["n"] += 1
    print(f"\nspecificity (ablating the target binding's synapses only):")
    for nm, d in (("targeted binding", spec_t), ("other bindings", spec_o)):
        print(f"  {nm:<18} baseline {d['base']/d['n']*100:5.1f}%  "
              f"after ablation {d['abl']/d['n']*100:5.1f}%   (n={d['n']})")

    result = {
        "checkpoint": args.ckpt, "iter": ck["iter"], "layer": layer,
        "m": args.m, "n_state_entries": cfg.n_embd * cfg.N * cfg.n_head,
        "trials_requested": args.trials, "trials_used": n_used,
        "recall": {k: agg[k] / max(1, n_used) for k in agg},
        "recall_ci": {k: wilson(agg[k], n_used) for k in agg},
        "mean_answer_prob": {k: float(np.mean(probs[k])) for k in probs},
        "mean_state_mass_removed": {k: float(np.mean(v)) for k, v in mags.items()},
        "specificity": {"targeted": spec_t, "others": spec_o},
        "config": cfg.__dict__,
    }
    out = ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
