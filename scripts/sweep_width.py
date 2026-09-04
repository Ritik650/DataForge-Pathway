"""Second sweep: neuron-space width.

The reference BDH config is n_embd=256, n_head=4, mlp_internal_dim_multiplier=128,
i.e. N = 8192 neurons per head and n/d = 128. Our first sweep used n/d = 8.
BDH's design premise is a large sparse neuron space (n >> d) in which activations
are sparse and positive -- the paper reports ~5% of the y vector active. At
n/d = 8 there may simply not be enough neurons for distinct associations to
occupy distinct subspaces, which is exactly what associative recall needs.

This sweep varies the expansion ratio, holding d fixed.
"""

import subprocess
import sys
import json
import pathlib

ROOT = pathlib.Path(__file__).resolve().parents[1]

CONFIGS = [
    # (layers, heads, mult, d, answer_weight, tag)  -> n/d = mult
    (3, 1, 16, 64, 8.0, "L3_m16"),
    (3, 1, 32, 64, 8.0, "L3_m32"),
    (3, 1, 64, 64, 8.0, "L3_m64"),
    (3, 4, 32, 64, 8.0, "L3_H4_m32"),
    (2, 4, 32, 64, 8.0, "L2_H4_m32"),
]
ITERS = 4000

if __name__ == "__main__":
    results = []
    for nl, nh, mult, d, aw, tag in CONFIGS:
        n = mult * d  # total neurons = N*nh = (mult*d/nh)*nh
        print(f"\n{'='*70}\n{tag}: layers={nl} heads={nh} d={d} n={n} (n/d={mult}) aw={aw}\n{'='*70}",
              flush=True)
        cmd = [
            sys.executable, str(ROOT / "src" / "train.py"),
            "--iters", str(ITERS), "--batch", "64",
            "--n-layer", str(nl), "--n-head", str(nh), "--mult", str(mult),
            "--n-embd", str(d), "--answer-weight", str(aw),
            "--out", f"data/sweep_{tag}.pt",
        ]
        p = subprocess.run(cmd, capture_output=True, text=True, cwd=ROOT)
        print("\n".join(p.stdout.strip().splitlines()[-28:]), flush=True)
        if p.returncode != 0:
            print("FAILED:", p.stderr[-2000:], flush=True)
            continue
        hist = json.loads((ROOT / "data" / "train_history.json").read_text())
        results.append({"tag": tag, "layers": nl, "heads": nh, "mult": mult,
                        "n_neurons": n, "n_params": hist["n_params"],
                        "recall": hist["final"]["recall_acc"],
                        "copy_chance": hist["final"]["chance_if_copying_some_context_city"],
                        "controls": hist["controls"]})
        (ROOT / "data" / f"history_{tag}.json").write_text(
            (ROOT / "data" / "train_history.json").read_text())

    print(f"\n\n{'='*70}\nSUMMARY (width sweep)\n{'='*70}", flush=True)
    for r in sorted(results, key=lambda r: -r["recall"]):
        c = r["controls"]
        print(f"  {r['tag']:12s} n={r['n_neurons']:5d} params={r['n_params']:>9,}  "
              f"recall {r['recall']*100:5.1f}% (copy-chance {r['copy_chance']*100:4.1f}%)  "
              f"normal {c['normal']*100:5.1f}% / drop {c['drop']*100:5.1f}% "
              f"/ swap {c['swap_follows_context']*100:5.1f}%", flush=True)
    (ROOT / "data" / "sweep_width_results.json").write_text(json.dumps(results, indent=2))
