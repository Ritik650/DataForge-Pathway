"""Find a config that clears the Block-2 recall gate.

Hypothesis being tested: BDH's attention is symmetric (Q is K), so the usual
two-layer induction circuit -- match the query name against the earlier name,
read the token after it -- cannot be built directly, because the query matches
the earlier NAME more strongly than the CITY beside it. An extra layer gives
the model room to move name identity into the subspace the match needs.
Separately, answer_weight concentrates gradient on the recall positions, which
are otherwise ~1 in 30 tokens and nearly invisible in the total loss.
"""

import itertools
import subprocess
import sys
import json
import pathlib

ROOT = pathlib.Path(__file__).resolve().parents[1]

CONFIGS = [
    # (layers, heads, mult, answer_weight, tag)
    (2, 1, 8, 8.0, "L2_aw8"),
    (3, 1, 8, 8.0, "L3_aw8"),
    (4, 1, 8, 8.0, "L4_aw8"),
    (4, 2, 8, 8.0, "L4_H2_aw8"),
    (3, 1, 8, 1.0, "L3_aw1"),
]

ITERS = 4000

if __name__ == "__main__":
    results = []
    for nl, nh, mult, aw, tag in CONFIGS:
        print(f"\n{'='*70}\n{tag}: layers={nl} heads={nh} mult={mult} answer_weight={aw}\n{'='*70}")
        cmd = [
            sys.executable, str(ROOT / "src" / "train.py"),
            "--iters", str(ITERS), "--batch", "64",
            "--n-layer", str(nl), "--n-head", str(nh), "--mult", str(mult),
            "--answer-weight", str(aw),
            "--out", f"data/sweep_{tag}.pt",
        ]
        p = subprocess.run(cmd, capture_output=True, text=True, cwd=ROOT)
        tail = p.stdout.strip().splitlines()[-30:]
        print("\n".join(tail))
        if p.returncode != 0:
            print("FAILED:", p.stderr[-2000:])
            continue
        hist = json.loads((ROOT / "data" / "train_history.json").read_text())
        results.append({"tag": tag, "layers": nl, "heads": nh, "aw": aw,
                        "recall": hist["final"]["recall_acc"],
                        "copy_chance": hist["final"]["chance_if_copying_some_context_city"],
                        "controls": hist["controls"]})
        (ROOT / "data" / f"history_{tag}.json").write_text(
            (ROOT / "data" / "train_history.json").read_text())

    print(f"\n\n{'='*70}\nSUMMARY\n{'='*70}")
    for r in sorted(results, key=lambda r: -r["recall"]):
        c = r["controls"]
        print(f"  {r['tag']:12s} recall {r['recall']*100:5.1f}%  "
              f"(copy-chance {r['copy_chance']*100:4.1f}%)  "
              f"controls: normal {c['normal']*100:5.1f}% / drop {c['drop']*100:5.1f}% "
              f"/ swap {c['swap_follows_context']*100:5.1f}% (chance {c['chance']*100:.1f}%)")
    (ROOT / "data" / "sweep_results.json").write_text(json.dumps(results, indent=2))
