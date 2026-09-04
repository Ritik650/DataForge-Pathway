"""Choose the demo preset on the JOINT criterion, not on effect size.

The claim is "ablate them and THAT recall breaks while the rest survives". Both
halves matter, and picking on the localisation gap alone cost us the second
half: moving to 7 bindings / 3 filler took the gap from 12 to 50 points but
dropped untouched bystanders from 96.0% to 74.5%.

That trade is mechanically sensible -- more bindings packed into the same 8,192
state entries means the top-m write entries of one binding overlap its
neighbours more -- which is why it needs measuring rather than assuming.

Selection rule, in order:
  1. baseline recall            >= 95%   (a preset that fails unablated is dead)
  2. bystanders after ablation  >= 90%   (the "rest survives" half)
  3. maximise the localisation gap (targeted vs top_other)

Candidates avoid the blind offset bands. With query_idx=0 the queried binding
sits at token offset 2P + 2f - 1, and the positional scan shows nulls at offset
3 and 13-17, so candidates are chosen to land at 9, 11, or 19.
"""

import argparse
import json
import pathlib
import sys

import numpy as np
import torch

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ablate import (load, run_trial, wilson, select_targeted,  # noqa: E402
                    unflatten)
import mqar  # noqa: E402

# (n_pairs, n_filler) -> offset of the queried binding when query_idx = 0
# (n_pairs, n_filler, m). The dose m is part of the preset, not a constant:
# bystander bleed scales with how much of the state we remove, so fixing m at 32
# and then discovering no preset clears the bystander bar was searching one axis
# of a two-axis problem.
CANDIDATES = [(P, f, m) for (P, f) in [(3, 2), (4, 2), (5, 1), (8, 2)]
              for m in (4, 8, 16, 32)]


@torch.no_grad()
def evaluate_preset(model, cfg, device, P, f, M, trials, seed):
    layer = cfg.n_layer - 1
    rng = np.random.default_rng(seed)
    offset = 2 * P + 2 * f - 1

    agg = {k: 0 for k in ("targeted", "top_other", "matched", "random")}
    mass = {k: [] for k in agg}
    used = 0
    for i in range(trials):
        ex = mqar.make_example(np.random.default_rng(seed + i), n_pairs=P,
                               n_queries=1, n_filler=f, query_idx=0)
        r = run_trial(model, cfg, device, ex, layer, M, rng)
        if not r["base_correct"]:
            continue
        used += 1
        for k in agg:
            agg[k] += r["correct"][k]
            mass[k].append(r["removed_magnitude"][k])

    # specificity: ablate binding 0, watch the other queried bindings
    tgt = {"base": 0, "abl": 0, "n": 0}
    oth = {"base": 0, "abl": 0, "n": 0}
    nq = min(3, P)
    for i in range(trials):
        ex = mqar.make_example(np.random.default_rng(seed + 5000 + i),
                               n_pairs=P, n_queries=nq, n_filler=f, query_idx=0)
        ids = torch.from_numpy(ex["ids"][:-1]).unsqueeze(0).to(device)
        base_logits, info = model.forward_recurrent(ids, record=True)
        delta = info["delta"][layer][ex["write_pos"]]
        sel = select_targeted(delta, M)
        idx = unflatten(sel, info["rho"][layer].shape)

        def ab(l, t, rho, _l=layer, _f=ex["write_pos"], _i=idx):
            if l == _l and t >= _f:
                rho = rho.clone(); rho[_i] = 0.0
            return rho
        abl_logits, _ = model.forward_recurrent(ids, ablate=ab)
        for j, (pos, ans) in enumerate(zip(ex["ans_positions"], ex["answers"])):
            d = tgt if j == 0 else oth
            d["base"] += int(int(base_logits[0, pos].argmax()) == ans)
            d["abl"] += int(int(abl_logits[0, pos].argmax()) == ans)
            d["n"] += 1

    baseline = used / trials
    gap = (agg["top_other"] - agg["targeted"]) / max(1, used)
    return {
        "n_pairs": P, "n_filler": f, "m": M, "query_offset": offset,
        "baseline_recall": baseline, "trials_used": used,
        "targeted": agg["targeted"] / max(1, used),
        "targeted_ci": wilson(agg["targeted"], used),
        "top_other": agg["top_other"] / max(1, used),
        "top_other_ci": wilson(agg["top_other"], used),
        "matched": agg["matched"] / max(1, used),
        "random": agg["random"] / max(1, used),
        "gap": gap,
        "mass_ratio": float(np.mean(mass["top_other"]) / max(1e-9, np.mean(mass["targeted"]))),
        "mass_targeted": float(np.mean(mass["targeted"])),
        "mass_top_other": float(np.mean(mass["top_other"])),
        "bystander_base": oth["base"] / max(1, oth["n"]),
        "bystander_abl": oth["abl"] / max(1, oth["n"]),
        "bystander_ci": wilson(oth["abl"], oth["n"]),
        "bystander_n": oth["n"],
        "target_base": tgt["base"] / max(1, tgt["n"]),
        "target_abl": tgt["abl"] / max(1, tgt["n"]),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="data/artifact_d32m8.pt")
    ap.add_argument("--trials", type=int, default=250)
    ap.add_argument("--seed", type=int, default=303)
    args = ap.parse_args()

    model, cfg, device, ck = load(args.ckpt)
    n_state = cfg.n_embd * cfg.N * cfg.n_head
    print(f"substrate {args.ckpt}, state {n_state:,}/layer\n")
    print(f"{'preset':>12}{'off':>5}{'base':>7}{'targ':>7}{'top_o':>7}"
          f"{'gap':>7}{'mass':>7}{'bys.base':>10}{'bys.abl':>9}")

    rows = []
    for P, f, M in CANDIDATES:
        r = evaluate_preset(model, cfg, device, P, f, M, args.trials, args.seed)
        rows.append(r)
        print(f"{f'p{P}/f{f}/m{M}':>12}{r['query_offset']:>5}"
              f"{r['baseline_recall']*100:>6.1f}%{r['targeted']*100:>6.1f}%"
              f"{r['top_other']*100:>6.1f}%{r['gap']*100:>6.1f}"
              f"{r['mass_ratio']:>6.2f}x{r['bystander_base']*100:>9.1f}%"
              f"{r['bystander_abl']*100:>8.1f}%")

    ok = [r for r in rows
          if r["baseline_recall"] >= 0.95 and r["bystander_abl"] >= 0.90]
    print(f"\npassing joint criterion (baseline >=95%, bystanders after "
          f"ablation >=90%): {len(ok)} of {len(rows)}")
    for r in sorted(ok, key=lambda r: -r["gap"]):
        print(f"  p{r['n_pairs']}/f{r['n_filler']}/m{r['m']}  gap {r['gap']*100:5.1f} pts  "
              f"bystanders {r['bystander_abl']*100:5.1f}%  "
              f"mass {r['mass_ratio']:.2f}x")
    best = max(ok, key=lambda r: r["gap"]) if ok else None
    if best:
        print(f"\nSELECTED: {best['n_pairs']} bindings / {best['n_filler']} filler "
              f"/ m={best['m']} ({best['m']/n_state*100:.3f}% of state)")
    else:
        print("\nNO CANDIDATE PASSES - report and widen the search")

    out = ROOT / "artifact" / "data" / "preset_selection.json"
    out.write_text(json.dumps({
        "checkpoint": args.ckpt, "trials": args.trials,
        "criterion": {"baseline_recall_min": 0.95, "bystander_abl_min": 0.90,
                      "then": "maximise targeted-vs-top_other gap"},
        "candidates": rows, "selected": best,
    }, indent=2))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
