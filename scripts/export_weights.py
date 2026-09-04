"""Export the substrate for the browser, and prove the export is lossless.

Two artifacts:

  artifact/data/weights.bin   flat Float32Array, little-endian
  artifact/data/fixture.json  one sequence with its reference logits and final
                              rho, so the JS port can assert equivalence on load

Why a round-trip assert
-----------------------
A silent export bug -- a transposed matrix, a wrong offset, a float64 -> float32
surprise -- would invalidate every number the page displays while leaving it
looking perfectly functional. So this script reloads the binary it just wrote
into a fresh model and asserts forward_recurrent reproduces the original logits.
The binary is only written if that passes.

The JS side asserts the other direction against fixture.json. Between the two,
a discrepancy anywhere in the chain shows up as a visible failure rather than as
a plausible-looking wrong answer.
"""

import argparse
import json
import pathlib
import struct
import sys

import numpy as np
import torch

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from bdh import BDH, BDHConfig  # noqa: E402
import mqar  # noqa: E402

# Order is the contract with js/bdh.js. Do not reorder without changing both.
TENSOR_ORDER = [
    ("embed", "embed.weight"),
    ("encoder", "encoder"),
    ("encoder_v", "encoder_v"),
    ("decoder", "decoder"),
    ("lm_head", "lm_head"),
    ("freqs", "freqs"),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="data/artifact_d32m4.pt")
    ap.add_argument("--out", default="artifact/data")
    args = ap.parse_args()

    device = "cpu"  # export and verify on cpu so the fixture is reproducible
    ck = torch.load(ROOT / args.ckpt, weights_only=False)
    cfg = BDHConfig(**ck["cfg"])
    model = BDH(cfg).to(device)
    model.load_state_dict(ck["model"])
    model.eval()

    sd = dict(model.state_dict())
    sd["freqs"] = model.freqs

    blob = bytearray()
    manifest = []
    offset = 0
    for name, key in TENSOR_ORDER:
        t = sd[key].detach().to(torch.float32).cpu().contiguous()
        arr = t.numpy().ravel()
        blob += arr.tobytes()
        manifest.append({"name": name, "shape": list(t.shape),
                         "offset": offset, "count": int(arr.size)})
        offset += int(arr.size)
        print(f"  {name:<10} shape={list(t.shape)} count={arr.size}")

    total = offset
    n_params = sum(p.numel() for p in model.parameters())
    print(f"\n  {total} floats = {total*4/1024:.1f} KB "
          f"({n_params:,} params + {total-n_params} buffer)")

    # ---- round trip -------------------------------------------------------
    fresh = BDH(cfg).to(device)
    buf = np.frombuffer(bytes(blob), dtype="<f4")
    new_sd = {}
    for entry in manifest:
        seg = buf[entry["offset"]: entry["offset"] + entry["count"]]
        tens = torch.from_numpy(seg.reshape(entry["shape"]).copy())
        key = dict(TENSOR_ORDER)[entry["name"]]
        new_sd[key] = tens
    fresh.load_state_dict({k: v for k, v in new_sd.items() if k != "freqs"},
                          strict=False)
    fresh.freqs.copy_(new_sd["freqs"])
    fresh.eval()

    rng = np.random.default_rng(7)
    ex = mqar.make_example(rng, n_pairs=4, n_queries=1, n_filler=1, query_idx=0)
    ids = torch.from_numpy(ex["ids"][:-1]).unsqueeze(0)

    with torch.no_grad():
        ref_logits, ref_info = model.forward_recurrent(ids)
        rt_logits, _ = fresh.forward_recurrent(ids)
    err = (ref_logits - rt_logits).abs().max().item()
    print(f"  round-trip max|logit diff| = {err:.3e}")
    assert err < 1e-6, f"EXPORT IS LOSSY ({err}) - binary not written"

    outdir = ROOT / args.out
    outdir.mkdir(parents=True, exist_ok=True)
    (outdir / "weights.bin").write_bytes(bytes(blob))

    fixture = {
        "note": ("Reference values from src/bdh.py forward_recurrent on cpu. "
                 "The browser port asserts against these on load and shows a "
                 "pass/fail badge. Tolerance 1e-5."),
        "config": cfg.__dict__,
        "manifest": manifest,
        "total_floats": total,
        "n_params": n_params,
        "tokens": [int(t) for t in ids[0].tolist()],
        "token_strings": [mqar.ITOS[int(t)] for t in ids[0].tolist()],
        "ans_pos": ex["ans_pos"],
        "answer": ex["answer"],
        "answer_string": mqar.ITOS[ex["answer"]],
        "write_pos": ex["write_pos"],
        "logits": [[round(float(v), 6) for v in row]
                   for row in ref_logits[0].tolist()],
        "rho_final_checksum": [
            {"layer": l,
             "abs_sum": round(float(r.abs().sum()), 5),
             "max": round(float(r.max()), 6),
             "min": round(float(r.min()), 6)}
            for l, r in enumerate(ref_info["rho"])
        ],
        "tolerance": 1e-5,
    }
    (outdir / "fixture.json").write_text(json.dumps(fixture, indent=2))

    vocab = {"itos": mqar.ITOS, "names": mqar.NAMES, "cities": mqar.CITIES,
             "filler": mqar.FILLER, "NAME_OFF": mqar.NAME_OFF,
             "CITY_OFF": mqar.CITY_OFF, "FILL_OFF": mqar.FILL_OFF,
             "BOS": mqar.BOS, "PAD": mqar.PAD,
             "vocab_size": mqar.VOCAB_SIZE}
    (outdir / "vocab.json").write_text(json.dumps(vocab, indent=2))

    print(f"\n  wrote {outdir/'weights.bin'} ({len(blob)/1024:.1f} KB)")
    print(f"  wrote {outdir/'fixture.json'}")
    print(f"  wrote {outdir/'vocab.json'}")
    print("\nexport verified lossless")


if __name__ == "__main__":
    main()
