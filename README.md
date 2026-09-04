# Where a fact lives: synaptic memory in Dragon Hatchling

**DataForge 2026 — Pathway Track ("Explain the Frontier")**
Topic: *Synaptic Plasticity as Short-Term Memory*

### ▶ **[Open the explainer](https://ritik650.github.io/DataForge-Pathway/)** · [one-page summary (PDF)](docs/concept-summary.pdf)

The page runs a 27,776-parameter BDH-GPU **live in your browser**. Click a
condition, drag the dose, press *new sequence* — every number is recomputed by
the real forward pass at about 2 ms a run. The badge in the header asserts that
this browser port still matches `src/bdh.py` to 1e-5 before any panel renders;
if it ever fails, the page says the figures are not trustworthy.

> **Status.** Substrate, science, source verification, the interactive artifact
> and the one-page summary are done. Remaining: blog PDF, mobile and
> accessibility passes, component-level AI disclosure. Every number below is
> measured by the committed scripts; nothing is illustrative or placeholder.
> Results that were produced and then invalidated are documented in
> [`docs/DESIGN_NOTES.md`](docs/DESIGN_NOTES.md) rather than deleted.

---

## The claim

> **In BDH, a fact the model just read is held in a small, locatable set of
> synapses rather than in its weights — ablate those synapses and that recall
> breaks, while removing even more state mass from elsewhere leaves it intact,
> and no parameter ever changes.**

**Scope.** The claim is localisation, and only localisation. An earlier version
also asserted volatility — that enough competing facts erase the binding anyway.
That half is **withdrawn, not pending**: the interference curve it rested on
failed its own control (Test C, `DESIGN_NOTES` §3f — recall moves 50%→100% when
only neutral filler changes and the binding count is held fixed, so the load
axis is confounded with sequence geometry). It does not appear in the artifact
and does not support the claim. One claim, fully backed, beats two with one
wobbling.

Localisation is falsifiable in one sharp direction, and the artifact can show it
failing:

- **Localisation is false if** ablating the identified synapses degrades recall
  no more than removing an equal — or greater — amount of state mass from
  elsewhere. That control is the single most important element of the project.

The decisive control is `top_other`: the m largest-magnitude state entries
*outside* the binding's own write. It necessarily removes at least as much state
mass as the targeted set, so surviving it cannot be explained by how much state
was removed.

## Audience, prerequisites, objectives

**Audience.** An ML practitioner or final-year student who understands a
Transformer forward pass and softmax attention, and has never read the BDH paper.

**Prerequisites.** Matrix multiplication; what a KV cache is; what "in-context
learning" means. No neuroscience.

**After using the artifact a learner can:**

1. State where within-session memory physically lives in BDH, versus in a Transformer.
2. Predict what happens to a specific recall when specific state entries are removed, and be right.
3. Predict, from a rule they can state, exactly which binding this model cannot retrieve — and check it.
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

### The shipped substrate

d=32, **N=256**, 2 layers, state **8,192 entries/layer**, **27,776 params**,
trained on 2–8 bindings and 0–8 filler — the range the artifact displays.
Weights ship as a 109.5 KB `Float32Array`, export verified lossless by
round-trip (0.000e+00) before the binary is written.

### The demo preset

**8 bindings · 2 filler units · query the oldest binding · dose m=8.**

The queried binding sits at token offset 19, inside a clean band (§ *periodic
bands* below). Baseline recall at this preset is **99.8%** over n=400.

This is the only preset the artifact ships. An earlier draft used 7 bindings /
3 filler at m=32; it was replaced when dose was added to the selection search,
and the superseded figures do not appear anywhere in this repo. Selection
procedure and the full candidate table: `scripts/select_preset.py` and
`artifact/data/preset_selection.json`.

**How it was chosen, and what it misses.** The first selection rule was
*baseline ≥95%, bystanders after ablation ≥90%, then maximise the localisation
gap*. Applied literally it picks a 9.3-point gap — recall stays at 89.3% and
nothing visibly breaks. That is a flaw in the rule: dose and effect size are the
same knob pulled in opposite directions, so "largest gap above a bystander
floor" lands on the smallest dose that still registers, and **no candidate with
a teachable gap clears 90% bystanders on this substrate at all.**

The applied rule maximises the **selectivity ratio** — the targeted binding's
drop divided by untouched bindings' drop — which is what "that recall breaks
while the rest survives" actually asserts. The shipped preset is the maximum on
that measure at every sample size tested.

It lands at **88.0% bystanders, 2.4 points below the original 90% bar.** That is
stated here, in the selection script, and on the page rather than rounded past.

All quoted figures for this preset come from the n=400 confirmation in
`dose_panel.json`. The n=250 search in `preset_selection.json` identifies the
preset but is not the source of any number quoted elsewhere — its selectivity
reads 6.1× against the canonical 5.9×, which is two sample sizes, not two
presets.

### Dose–response, and why m=8

n=400 per row, `artifact/data/dose_panel.json`.

| m | % of state | targeted | top_other | matched | random | p(answer) | bystanders | selectivity |
|---|---|---|---|---|---|---|---|---|
| 2 | 0.024% | 100.0% | 100.0% | 99.2% | 99.5% | 0.799 | 92.4% | 0.0× |
| 4 | 0.049% | 98.7% | 100.0% | 97.7% | 99.7% | 0.782 | 91.0% | 0.5× |
| **8** | **0.098%** | **77.2%** | **100.0%** | 95.7% | 100.0% | **0.614** | 88.0% | **5.9×** |
| 16 | 0.195% | 53.1% | 98.7% | 87.0% | 98.5% | 0.438 | 82.4% | 4.8× |
| 32 | 0.391% | 29.8% | 88.0% | 81.0% | 97.7% | 0.287 | 77.0% | 4.3× |
| 64 | 0.781% | 14.3% | 78.7% | 77.7% | 95.5% | 0.138 | 63.2% | 2.9× |
| 128 | 1.562% | 10.8% | 79.2% | 76.2% | 90.5% | 0.098 | 48.6% | 2.1× |

Targeted ablation collapses from 100% to 10.8% across the ladder while uniform
random barely moves (99.5% → 90.5%), and `top_other` — which removes **more**
state mass than targeted at every dose — stays above 78%.

**Selectivity peaks at m=8** (5.9×, the ratio of the targeted binding's drop to
untouched bystanders' drop), so the preset is the measured optimum of the ladder
rather than a taste call.

**Dose is the artifact's second control, and the trade-off is the lesson.**
Small doses are surgical but mild; large doses are dramatic but bleed —
bystanders fall monotonically from 92.4% to 48.6%. A learner slides it and sees
specificity being spent.

Some of that bleed is substrate, not dose: this model holds 8,192 state entries
where the earlier one held 32,768, so writes overlap about 4× more.

### The sixty-second moment, at the shipped preset

At **m=8** — 8 entries out of 8,192 — with n=400 and baseline recall 99.8%:

| condition | recall | 95% CI | state mass removed |
|---|---|---|---|
| baseline | 99.8% | [98.6, 100.0] | — |
| **targeted** | **77.2%** | [72.8, 81.0] | 1.00× |
| **top_other** | **100.0%** | [99.0, 100.0] | **2.06×** |
| magnitude-matched | 95.7% | — | 0.94× |
| uniform random | 100.0% | — | 0.13× |

The control that removes **twice** the state mass costs nothing; the eight
entries this binding actually wrote cost 22.6 points of recall and drop
p(answer) from 0.96 to 0.614. Intervals disjoint.

Specificity at the same dose: the targeted binding falls 100% → 74.2% while
untouched bindings in the same sequence go 92.4% → 88.0% — a 5.9× selectivity
ratio, the maximum over the dose ladder.

### Localisation across state widths

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

**It replicates across state widths.** The same experiment on the two narrow
checkpoints, at a matched dose of 0.391% of state and 3 bindings:

| model | params | state/layer | targeted | magnitude-matched | gap | baseline correct |
|---|---|---|---|---|---|---|
| d64m8 | 104,704 | 32,768 | 86.6% | 99.7% | 13.1 pts | 298/300 |
| d32m2 | 9,344 | 2,048 | 80.7% | 94.8% | 14.1 pts | 135/400 |
| d32m4 | 15,488 | 4,096 | 75.5% | 96.5% | **21.0 pts** | 143/400 |

Confidence intervals are disjoint in every row, and in every row the matched
control removes state mass identical to the targeted condition (for d32m4,
221.043 vs 221.074). The mechanism is not an artifact of one model size.

The last column is the catch, and it drives the substrate choice: the narrow
models are correct on only ~35% of trials *before* any ablation, because they
were trained across 2–14 bindings — a range the artifact will never show.

### Interference: withdrawn

An interference result — recall collapsing against competing bindings while an
equal number of neutral filler tokens leaves it untouched — was measured and is
**not shipped**. It is withdrawn from the claim, the artifact, and the results
above. This is a decision, not a queue: no further measurement is planned.

Three reasons, all found by controls rather than by inspection:

1. **It fails Test C.** Hold the binding count fixed at 7 and vary only neutral
   filler, and recall reads 50.4% → 40.0% → 73.6% → 100.0% across filler 0–3
   (`positional_map.json`). Recall moves while the competing-binding count does
   not, so the load axis is confounded with sequence geometry and any curve
   built on it is uninterpretable. Full definition in `DESIGN_NOTES` §3f.
2. **The binding curve is not monotone.** On the shipped substrate it reads
   39.7% [36.5, 42.9] at 6 competing bindings and rises to 51.1% [47.8, 54.4]
   at 7 (n=900, disjoint intervals). Seven competing bindings are reproducibly
   harder than eight, and we cannot explain it.
3. **The measurement was contaminated** by the positional blind spot below. With
   no filler separating query from binding block, one load in the sweep read
   exactly 0.0% — the offset-3 failure, not interference.

The dissociation may well hold. It has not been shown to hold cleanly here, and
a claim that needs an asterisk is not ready to teach.

### Retrieval fails in periodic bands of relative offset

Scanning `n_pairs` × `query_idx` × `n_filler` (140 cells, n=250 each,
`scripts/positional_map.py`) and grouping by the token distance from the query
back to the queried binding:

| token offset | 1 | 3 | 5 | 7 | 9 | 11 | 13 | 15 | 17 | 19 | 21 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| cells | 7 | 14 | 20 | 25 | 22 | 18 | 14 | 10 | 6 | 3 | 1 |
| failures | 0 | **9** | 0 | 2 | 0 | 0 | 2 | **8** | **6** | 0 | 0 |
| mean recall | 100.0 | **42.8** | 99.7 | 97.3 | 98.4 | 99.1 | 93.7 | **68.7** | **71.3** | 100.0 | 100.0 |

Two null bands — at offset 3, and again across offsets 13–17 — separated by a
clean stretch at 5–11 where mean recall is 97–100%. Offsets 19 and 21 recover,
but on 3 and 1 cells respectively, so **the shape beyond 17 is unresolved.**

The sharpest single result sits inside the first band: with no filler, the
binding at offset 3 (the second-to-last) scores **exactly 0.0%** in all seven
cells from 2 to 8 bindings, against 98–100% at neighbouring positions.

**Offset alone does not determine failure, and an earlier version of this
section wrongly said it did.** Of 27 failing cells only 9 are at offset 3, and
of the 14 cells at offset 3, five pass — all at one filler unit with fewer than
seven bindings. Load and offset interact; neither predicts failure by itself.
That claim was checked against our own committed `positional_map.json` and did
not survive it.

**Mechanism: hypothesis, not result.** BDH scores attention as
`⟨rope_t(x_t), rope_τ(x_τ)⟩`, a function of `t − τ`. Periodic nulls in relative
offset are what phase cancellation predicts, and a periodic band structure is a
better fit to that hypothesis than a single blind spot would have been. We have
not demonstrated it, and the artifact says so.

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

# correctness gates — run these first
python tests/test_equivalence.py          # parallel vs recurrent, ~2e-7
node   tests/test_js_equivalence.mjs      # browser port vs Python, 2.6e-6
node   tests/test_page_logic.mjs          # the page's computation path

python tests/test_equivalence.py                  # correctness gate, run this first
python src/mqar.py                                # task generator + invariant checks
python src/train.py --iters 12000 --n-layer 2 --answer-weight 8 \
       --out data/bdh_mqar_final.pt               # ~8 min on one GPU
python src/ablate.py --ckpt data/bdh_mqar_final.pt --trials 300 --m 64
python src/sweeps.py --ckpt data/bdh_mqar_final.pt --trials 150
python scripts/capacity_experiment.py             # interference, matched coverage
python scripts/decay_small.py                     # damping case of U

# artifact
python scripts/export_weights.py --ckpt data/artifact_d32m8.pt
python scripts/build_summary_pdf.py && python scripts/check_pdf.py
python -m http.server -d artifact 8000            # then open localhost:8000
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

**AI assistance, by component.** This project was built with AI assistance
(Claude, via Claude Code) throughout, under the author's direction and review.
Component-level detail:

| Component | AI involvement |
|---|---|
| `src/bdh.py` | Port and adaptation of Pathway's MIT-licensed reference. AI-written; the recurrent formulation, ablation hook and `u_decay` were specified by the author and verified against Python by `tests/test_equivalence.py`. |
| `src/mqar.py`, `src/train.py` | AI-written to the author's task specification. The held-out-pair design was AI-proposed, **found to be wrong by measurement**, and replaced (`DESIGN_NOTES` §1). |
| `src/ablate.py`, `src/sweeps.py`, `scripts/*` | AI-written. The `top_other` control was added after the magnitude-matched control was found insufficient. |
| `artifact/js/*` | AI-written port of the Python forward pass and selectors, gated by `tests/test_js_equivalence.mjs`. |
| `artifact/index.html`, `css/` | AI-written from the design in `design/` and `docs/UI_SPEC.md`. |
| Design canvas (`design/`) | AI-generated via the Claude design tooling. Stitch was attempted first per the author's standing preference and produced nothing (two timeouts, no screens); `docs/UI_SPEC.md` exists so the design can be regenerated in Stitch by hand. |
| `docs/CONCEPT_SUMMARY.md`, README, `DESIGN_NOTES` | AI-drafted, author-reviewed. |
| Experimental design and interpretation | Iterative between author and AI. Several AI-proposed hypotheses were **refuted by the project's own data** and are recorded as such: the offset-3 rule, the "d is the bottleneck" prediction, and the first interference curve. |
| Source verification | AI-retrieved from arXiv and the Pathway repository; every figure logged in `docs/CLAIMS.md` with the source's verbatim wording. |

No AI-generated figure appears anywhere without a committed script that
produces it. All experimental design decisions, invalidated results and
corrections are recorded in `docs/DESIGN_NOTES.md` so every claim can be traced
and defended.

**Mentorship.** None. No mentor advised, reviewed or challenged this work.
