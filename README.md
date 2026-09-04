# Where a fact lives: synaptic memory in Dragon Hatchling

**DataForge 2026 — Pathway Track ("Explain the Frontier")**
Topic: *Synaptic Plasticity as Short-Term Memory*

> **Status: in progress.** The substrate, the science, and the source
> verification are done. The interactive explainer is not built yet. Every
> number below is measured by the committed scripts; nothing here is
> illustrative or placeholder. Results that were produced and then invalidated
> are documented as such in [`docs/DESIGN_NOTES.md`](docs/DESIGN_NOTES.md)
> rather than deleted.

---

## The claim

> **In BDH, a fact the model just read is held in a small, locatable set of
> synapses rather than in its weights — ablate those synapses and that specific
> recall breaks while the rest of the model's behaviour survives; leave them
> alone and enough competing facts erase it anyway, with no parameter ever
> changing.**

It is falsifiable in two independent directions, and the artifact can show
either failing:

- **Localisation is false if** ablating the identified synapses degrades recall
  no more than removing an equal amount of state mass from elsewhere. That
  magnitude-matched control is the single most important element of the project.
- **Volatility is false if** the fact survives arbitrarily many competing facts,
  or if it degrades just as fast against an equal number of tokens that bind
  nothing.

Both controls have been run. Results below.

## Audience, prerequisites, objectives

**Audience.** An ML practitioner or final-year student who understands a
Transformer forward pass and softmax attention, and has never read the BDH paper.

**Prerequisites.** Matrix multiplication; what a KV cache is; what "in-context
learning" means. No neuroscience.

**After using the artifact a learner can:**

1. State where within-session memory physically lives in BDH, versus in a Transformer.
2. Predict what happens to a specific recall when specific state entries are removed, and be right.
3. Explain why a fixed-size synaptic state forgets through *interference* rather than eviction or truncation.
4. Say what BDH's published evidence does and does not establish, and where BDH-CQ extends the idea.
5. Name a limitation of the artifact itself.

---

## Results so far

All figures from the committed scripts. Proportions carry Wilson 95% intervals
and an explicit `n`.

### The model clears its gate

A small BDH-GPU trained by us on synthetic MQAR — 2 layers, 512 neurons, d=64,
**104,704 parameters**, minutes on one consumer GPU.

| metric | value |
|---|---|
| Recall on held-out sequences | **99.66%** (n=4653) |
| Chance if copying a random context city | 22.8% |
| Delete the queried binding from context | **7.5%** (uniform chance 6.25%) |
| Rebind the name to a different city | **99.75%** follow the context |

The binding cannot have come from the weights: pairings are resampled uniformly
per sequence, so the weight-optimal prior for "which city follows Mira" is
uniform. Verified empirically — pair frequencies over 20k sequences are uniform
to Poisson noise.

### Localisation: a binding lives in specific state entries

Ablating the largest-magnitude entries of the Hebbian write that laid down one
binding, against a control that removes **the same state mass from elsewhere**:

| synapses ablated | % of state | targeted | magnitude-matched control |
|---|---|---|---|
| 8 | 0.024% | 97.3% | 100.0% |
| 64 | 0.195% | 85.2% | 100.0% |
| 256 | 0.781% | **47.7%** | 99.3% |
| 1024 | 3.13% | 24.2% | 98.0% |
| 2048 | 6.25% | **12.1%** | 97.3% |

Targeted ablation falls monotonically with dose; the matched control barely
moves. At m=64 the two conditions removed 1235.8 vs 1230.6 units of state mass
— statistically identical — for a 13-point vs 0.3-point cost.

**Specificity:** ablating one binding's synapses takes that binding from 100% to
90.0% while other bindings in the same sequence go 100% → 99.8%.

### Interference: forgetting tracks competing facts, not length

Each filler unit costs exactly the same 2 tokens as a binding but stores no
association, so the two curves hold sequence length matched and vary only
whether the tokens bind anything. Narrow state (2,048 entries/layer); every
point in-distribution for both conditions.

| competing bindings | recall | equal-length filler | recall |
|---|---|---|---|
| 0 | 99.6% | 0 | 99.6% |
| 1 | 62.8% | 1 | 100.0% |
| 3 | 35.2% | 3 | 100.0% |
| 7 | 16.4% | 7 | 100.0% |
| 13 | 7.2% | 13 | 100.0% |

Recall collapses against competing bindings and is flat against neutral tokens.
That is interference, not truncation or length decay.

**The capacity reading is not established.** The comparison was redone with each
width trained to a **load-0 quality gate** — a width enters only if it recalls a
lone binding at ≥95% — rather than a fixed step budget. Tripling the budget was
not enough for the two wide models:

| width | state/layer | steps | load-0 recall | gate |
|---|---|---|---|---|
| d32m2 | 2,048 | 8,000 | 99.6% | PASS |
| d32m4 | 4,096 | 8,000 | 100.0% | PASS |
| d32m8 | 8,192 | 24,000 | 83.6% (from 62.4%) | FAIL |
| d64m8 | 32,768 | 24,000 | 92.4% (from 83.2%) | FAIL |

Both wide models improved substantially and neither converged, which points at
the training recipe rather than the architecture — lr, schedule and
`answer_weight` were tuned on the narrow models and carried over unchanged. Not
tested, so it stays a hypothesis.

Among the two widths that clear the gate the wider state is better at every load
(load-3 recall 35.2% → 52.8%). That is a two-point comparison, consistent with
the capacity reading and nowhere near enough to assert it. See
`DESIGN_NOTES.md` §3d.

> **An earlier version of this experiment was invalid and was discarded.**
> Training covered 2–8 bindings but only 0–4 filler units, so each curve fell
> shortly after leaving *its own* training range — it measured extrapolation
> distance, not mechanism. See `DESIGN_NOTES.md` §3b.

---

## What is live, precomputed, synthetic, and unverified

| Component | Status |
|---|---|
| MQAR dataset | **Synthetic**, generated from committed seeds. No external data. |
| Model weights | **Trained by us.** Not an official Pathway model, not a released checkpoint, far below any reported BDH scale. |
| Forward pass and ablation in the artifact | **Live** (planned) — the learner's choice is computed for real |
| Dose-response and interference sweeps | **Precomputed**, shipped as JSON in `artifact/data/` |
| Decay / stability–plasticity | **Not demonstrated.** Control cut from the artifact — see below. |
| Across-width capacity ordering | **Not established.** Two of four widths fail the load-0 quality gate. |
| Positional dip (`DESIGN_NOTES` §3c) | **Observed but not characterised.** Excluded from the artifact as a teachable claim. |

### Results we tried for and did not get

Reported because a submission that only lists its successes cannot be checked.

**The stability–plasticity trade-off did not replicate.** Damping is the `u < 1`
case of the paper's own `U` matrix, and the prediction was that curves for
different `u` should cross — damping worse for old bindings, better for recent
ones. Across three regimes (wide state where nothing is forgotten, narrow state
where everything is floored, and narrow state at loads with headroom), **no
crossing appears anywhere.** `u=1.0` matches or beats every damped model at
every binding age; `u<0.95` is catastrophic (overall recall 21.3% → 7.5%).

We cannot distinguish "the trade-off is absent in this toy regime" from "damped
models need their own hyperparameters and our fixed recipe disadvantages them",
so this is *not demonstrated*, not *refuted*. **The decay slider is cut from the
artifact** — a control whose effect we cannot demonstrate would be a decorative
slider dressed as a concept variable. The artifact ships two controls, both
backed by results that survived their own falsification tests.

## Reproducing

```bash
pip install -r requirements.txt

python tests/test_equivalence.py                  # correctness gate, run this first
python src/mqar.py                                # task generator + invariant checks
python src/train.py --iters 12000 --n-layer 2 --answer-weight 8 \
       --out data/bdh_mqar_final.pt               # ~8 min on one GPU
python src/ablate.py --ckpt data/bdh_mqar_final.pt --trials 300 --m 64
python src/sweeps.py --ckpt data/bdh_mqar_final.pt --trials 150
python scripts/capacity_experiment.py             # interference, matched coverage
python scripts/decay_small.py                     # damping case of U
```

`tests/test_equivalence.py` is not optional. It asserts that the token-parallel
training path and the step-by-step recurrent path compute the same function
(agreement ~2e-7). If they diverged, every ablation result would describe a
state the trained model never uses.

## Repository map

| Path | Role |
|---|---|
| `src/bdh.py` | BDH-GPU, derived from Pathway's MIT reference. Adds a recurrent forward that materialises the synaptic state `rho`, ablation hooks, and the damping case of `U`. |
| `src/mqar.py` | Deterministic MQAR generator with the drop/swap interventions. |
| `src/train.py` | Training loop, recall evaluation, behavioural controls. |
| `src/ablate.py` | Localisation experiment and its magnitude-matched control. |
| `src/sweeps.py` | Dose-response and interference datasets. |
| `scripts/` | Architecture, width, capacity, decay, and positional-artifact experiments. |
| `docs/CLAIMS.md` | Every externally sourced number, with source wording and verification status. |
| `docs/DESIGN_NOTES.md` | Decisions, dead ends, and invalidated experiments. |
| `NOTICE.md` | Source, licence, and provenance record. |

## A note on the state we ablate

Eq. (6) of the BDH paper defines the conceptual state `sigma` on neuron-neuron
edges. Eq. (8) defines BDH-GPU, whose state is
`rho_{t,l} := (rho_{t-1,l} + LN(E y_{t,l-1}) x_{t,l}^T) U` — an outer product of
a `d`-vector and an `n`-vector, so `rho` is `d x n`, not `n x n`. They relate by
`sigma ~ D_y rho`, rank at most `d`.

We ablate entries of **`rho`**, because that is the state the trained model
actually carries. Describing this as "editing neuron-neuron synapses" without
that caveat would be a misstatement.

## Credits, licences, AI disclosure

- Code licence: MIT (see `LICENSE`). `src/bdh.py` derives from
  [pathwaycom/bdh](https://github.com/pathwaycom/bdh), MIT, © 2025 Pathway
  Technology, Inc. Full provenance in `NOTICE.md`.
- Primary sources and verification status: `docs/CLAIMS.md`.
- No third-party fonts, icons, datasets, or model weights are used.

**AI assistance.** This project was built with AI assistance (Claude) for code
authoring, experiment scaffolding, source retrieval and drafting, under the
author's direction and review. The track requires this disclosure; it will be
expanded to component-level detail before submission. All experimental design
decisions, invalidated results, and corrections are recorded in
`docs/DESIGN_NOTES.md` so that every claim can be traced and defended.
