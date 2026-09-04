# Design notes — decisions, dead ends, and why

A running record of choices that are not obvious from the code, kept so every
team member can defend them. Judges award 15 points for whether the team can
trace the system and predict the result of changes.

---

## 1. Held-out (name, city) pairs were removed. They measured the wrong thing.

**What we tried first.** Fix a bijection `HELD_OUT[name] -> city`, forbid those
pairs during training, and evaluate recall only on them. The reasoning was the
one in the project plan: if the pair never appeared in training, the answer
cannot have been stored in the weights, so success must come from state.

**What actually happened.** Recall on held-out pairs fell to **1.8%** — well
*below* the 20.6% a model would score by copying a random city from context —
and it got worse the longer the model trained. Measured on the same checkpoint,
held-out pairings scored 15.5% against 22.5% for ordinary pairings.

**Why.** Withholding a fixed pair set does not just prevent memorisation; it
teaches the complementary rule. The training distribution says *"Mira is never
paired with Oslo"*, which is learnable signal. Evaluating on exactly those pairs
then measures **learned avoidance**, not failure to recall. The metric was
anti-correlated with the thing it was supposed to measure, and it degraded with
training — the signature of a model getting better at a rule we accidentally
taught it.

**What we do instead.** Pairings are resampled uniformly at random for every
sequence, so each name is paired with each city equally often and the
weight-optimal prior for "which city follows Mira" is uniform. Any name-specific
preference in the weights would *raise* training loss. Verified empirically:
pair frequencies over 20k sequences are uniform to Poisson noise
(mean 469, sd 21, expected sd `sqrt(469) = 21.7`).

The claim is then checked **behaviourally, by intervention**, which is stronger
than a held-out split because it acts on the individual sequence:

| Control | What it does | What must happen |
|---|---|---|
| `drop_queried` | deletes the queried binding from the KV block | recall collapses to chance — nothing else can supply the answer |
| `swap_queried_to` | rebinds the name to a different city | the answer follows the context, not any weight-level prior |

Both live in `eval_controls()` in `src/train.py`.

**Lesson worth keeping in the write-up.** "Held out from training" and "cannot be
answered from weights" are not the same property, and the gap between them is
measurable. This is a good candidate for the artifact's misconception section.

---

## 2. The task format is true MQAR, not prose sentences.

The original plan used `Mira lives in Oslo .` (5 tokens per fact) with one query
per sequence. Two problems:

1. **One query per sequence is one recall gradient per ~30 tokens.** Most of the
   loss sits on first occurrences of names and cities, which are uniform by
   construction and therefore irreducible (`log 16 = 2.77` nats). Recall
   improvements barely move total loss, so they barely get gradient. The
   published MQAR task (Arora et al., *Zoology*, 2023) uses **multiple queries
   per sequence** for exactly this reason.
2. **The queried fact was always first**, so "copy the first city" is a valid
   strategy that never reads the name. Fixed by randomising which pair is
   queried.

Current format: `<bos> k1 v1 k2 v2 ... [filler] ... q1 a1 q2 a2 ...`, 2 tokens
per binding, several supervised recall positions per sequence.

Filler tokens are inserted in units of 2 tokens — the same length as one
binding, but carrying no association. That separation is the point: if recall
degrades with the number of competing **bindings** but not with an equal number
of **filler** tokens, forgetting is interference between stored associations,
not truncation or length-driven decay. One knob cannot distinguish those.

---

## 3. BDH attention is symmetric, which makes induction harder than usual.

The reference implementation asserts `K is Q`: the score matrix is
`QR @ QR.mT`, a similarity of each token's sparse activation with its own
history. There are no separate query and key projections.

This blocks the textbook two-layer induction circuit. To answer `... Mira Oslo
... Mira ?`, the query token `Mira` should attend to the position holding
`Oslo`. But with symmetric attention the query `Mira` matches the *earlier
`Mira` token* far more strongly than the `Oslo` beside it — identical
embeddings give a near-maximal score. The model must therefore build the match
in a subspace where the earlier name's *successor* is what looks similar, which
needs an extra layer of routing.

This is a real architectural property of BDH, not a bug in our port, and it is
worth teaching: it is a concrete trade the architecture makes.

**But it was not what blocked us, and we should not claim it was.** We predicted
that a third layer would be needed to route around the symmetry, and swept
layers on that basis. The sweep refuted it:

All runs: 4000 iterations, batch 64, identical seeds. Copy-a-random-context-city
chance is 22.8%; the drop/swap controls use 6 bindings (chance 16.7%).

| config | recall | drop | swap |
|---|---|---|---|
| **2 layers, answer_weight 8** | **87.4%** | 0.0% | 82.2% |
| 3 layers, answer_weight 8 | 31.3% | 0.0% | 12.5% |
| 4 layers, answer_weight 8 | 31.1% | 0.0% | 15.2% |
| 4 layers, 2 heads, answer_weight 8 | 31.1% | 3.2% | 19.2% |
| 3 layers, **answer_weight 1** | 5.6% | 7.2% | 7.5% |

Two layers were sufficient; three and four were *worse* at an equal training
budget. The last row is the controlled comparison that identifies the real
cause: holding depth fixed at 3 and changing only the answer weighting moves
recall from 5.6% to 31.3%.
The binding constraint was **gradient dilution**, fixed by `answer_weight`
(see §2): recall positions are ~1 in 30 tokens and the rest of the loss is
largely irreducible, so the recall signal was being drowned rather than being
architecturally impossible. The symmetric-attention observation stands as a
description of the architecture; it is not the explanation for our training
failure, and the write-up must not imply it was.

---

## 3b. The first interference curve was invalid. Its own control caught it.

**What we ran.** Recall of the first binding as a function of (a) the number of
competing bindings written after it, and (b) an equal number of neutral filler
tokens. Filler units cost the same 2 tokens as a binding but carry no
association, so the pair of curves was supposed to separate *interference
between stored associations* from *sequence-length decay*.

**What came out.** Filler degraded **earlier and harder** than competing
bindings across the middle of the range — the opposite of the prediction.

| load | competing bindings | equal-length filler |
|---|---|---|
| 0–6 | 100% | 100% |
| 7 | 100% | 87.3% |
| 10 | 93.3% | 62.7% |
| 14 | 30.0% | 55.3% |

**Why it was meaningless.** Training covered 2–8 bindings but only 0–4 filler
units. Each curve began falling shortly after leaving *its own* training range
— filler at load 7, bindings at load 9. Both were measuring distance outside
the training distribution, not the mechanism. And inside the shared range both
sat at 100%, because a state of 32,768 entries per layer is nowhere near
stressed by at most 16 bindings. The experiment could not have supported the
claim even if the numbers had come out the "right" way.

**Fix** (`scripts/capacity_experiment.py`), two parts, both necessary:

1. **Matched coverage** — training spans 2–14 bindings *and* 0–14 filler units,
   so every plotted point is in-distribution for both conditions.
2. **Capacity that binds** — sweep the state width downward until bindings
   actually compete for it. A curve that only bends outside the training
   distribution is an extrapolation result, and must not be labelled
   interference.

**Rule this establishes for the project.** An asymmetry between two conditions
is only evidence about mechanism if the conditions are matched on everything
else — including how much of each the model was trained on. We would have
shipped a wrong central claim here if the control had not been run alongside.

---

## 3d. Comparing state widths at a fixed step budget measures training, not capacity.

Having fixed the interference experiment (§3b), the obvious next claim was the
capacity one: a wider fixed-size state should hold more bindings before
interference bites. Sweeping four widths at a fixed 8,000 steps appeared to
start well and then fell apart.

Recall of the oldest binding, both conditions in-distribution, n=250 per point:

| load | d32m2 (2,048) | d32m4 (4,096) | d32m8 (8,192) | d64m8 (32,768) |
|---|---|---|---|---|
| 0 | 99.6% | 100.0% | **62.4%** | **83.2%** |
| 3 | 35.2% | 52.8% | 38.8% | 61.2% |
| 7 | 16.4% | 23.2% | 13.6% | 18.8% |

The first two widths behave: 4,096 sits above 2,048 everywhere. Then 8,192
falls *below* 2,048 despite four times the state.

**Why the ordering is meaningless.** The two widest models fail at **load 0** —
one binding, nothing competing with it. Their filler controls also degrade
(minima 62.8% and 80.8%) instead of staying flat at 100%. Both are simply
undertrained: the wider models need more steps to fit the harder 2–14 binding
distribution, and 8,000 was tuned for the narrow ones. A model that cannot hold
a single binding tells us nothing about how many bindings it can hold, so its
degradation curve is not a capacity curve.

**Fix.** Widths are trained to a **quality gate**, not a step budget: a width
enters the capacity comparison only if it recalls a lone binding at ≥95%
(`LOAD0_GATE`). Wider models get the steps they need to clear it. Any width
that fails the gate is reported but excluded, with the failure stated.

**Outcome after the fix: still not established, and now we know why.** Tripling
the budget for the two wide models was not enough to clear the gate.

| width | state/layer | steps | load-0 recall | gate |
|---|---|---|---|---|
| d32m2 | 2,048 | 8,000 | 99.6% | PASS |
| d32m4 | 4,096 | 8,000 | 100.0% | PASS |
| d32m8 | 8,192 | 24,000 | 83.6% (was 62.4% at 8k) | FAIL |
| d64m8 | 32,768 | 24,000 | 92.4% (was 83.2% at 8k) | FAIL |

Both wide models improved substantially with more steps and neither converged.
That points at the training recipe, not the architecture: learning rate,
schedule, and `answer_weight` were tuned on the narrow models and carried over
unchanged, and the wider models plausibly need their own. We have not tested
that, so it stays a hypothesis.

**What can honestly be said.** Among the two widths that clear the gate, the
wider state is better at every load (load-3 recall 35.2% → 52.8%). That is a
two-point comparison, consistent with the capacity reading and nowhere near
sufficient to assert it as a law. The artifact will show the binding-vs-filler
dissociation, which is solid, and will not claim a capacity scaling result.

**What survives regardless.** The binding-vs-filler dissociation is unaffected —
it holds cleanly on both models that clear the gate, and it is a within-model
comparison, so training quality cancels. Only the *across-width ordering*
needed the gate.

**The general rule, now twice-learned.** §3b failed because two conditions had
unmatched training coverage; §3d failed because four models had unmatched
training quality. Both are the same mistake: a comparison is evidence about
mechanism only when everything except the variable of interest is matched —
and "same number of training steps" is not the same as "equally well trained".

---

## 3c. A reproducible failure case: the query must not abut the binding block.

Found while sanity-checking the decay runs, not looked for. With `n_pairs`
bindings and the query placed immediately after them, recall of the
**second-to-last** binding collapses while every other position stays at ~100%.
It recurs across independently trained models (`u=1.00`, `u=0.95`), so it is not
seed noise, and it worsens sharply with load:

| bindings | worst index | query-to-binding offset | recall |
|---|---|---|---|
| 6 | 4 | 3 | 90.0% |
| 7 | 5 | 3 | 75.0% |
| 8 | 6 | 3 | 38.7% |
| 9 | 7 | 3 | 5.7% |
| 10 | 8 | 3 | **1.0%** |

**A hypothesis we tested and rejected.** The constant offset of 3 looked like a
RoPE phase-cancellation zone: BDH scores attention as
`<rope_t(x_t), rope_tau(x_tau)>`, a function of `t - tau`, so a fixed bad offset
is a plausible mechanism. Inserting one filler unit (2 tokens) between the
binding block and the query removed the dip entirely on the `u=0.95` model — if
a fixed offset were the cause, the failure should have moved to whichever
binding then sat at offset 3, and it did not.

**And then a correction, because that test was run on one model.** Repeating it
across all four independently trained models shows the effect is real but
neither universal nor cleanly removable. Recall by binding index, 8 bindings,
n=300 per point:

| model | filler=0 | filler=1 | filler=2 |
|---|---|---|---|
| u=1.00 | **57.3% @ idx 6** | **54.7% @ idx 7** | clean |
| u=0.98 | clean (99.3%) | clean | 97.0% @ idx 1 |
| u=0.95 | **39.0% @ idx 6** | clean | clean |
| u=0.90 | **53.7% @ idx 6** | 95.7% @ idx 2 | 91.3% @ idx 4 |

So the earlier note overstated it. What survives across models is only this:
**a sharp isolated recall failure can land on one of the two most recent
bindings, and it is severe when it appears (39–57%) against ~100% everywhere
else.** Three of four models show it at index 6 with the query abutting the
block; `u=0.98` never shows it; and for `u=1.00` a filler unit *moves* the
failure to index 7 rather than removing it. The offset-3 rule, the
"two tokens abolish it" rule, and the adjacency-masking story are all
unsupported as stated.

**Status: an observed instability, not a characterised phenomenon.** It should
not be presented in the artifact as a clean teachable law until we can predict
in advance which index will fail for a given model. Doing the n_pairs scan on a
single checkpoint made it look far more lawful than it is — a reminder that
"reproducible across sequence lengths" and "reproducible across training runs"
are different claims.

**Consequences we must act on:**

1. The decay "recall by age" curves are contaminated wherever this lands, and
   the contaminated index is *not* fixed. Re-running with one filler unit did
   not clean it — it moved the `u=1.00` failure to index 7 (54.7%). Any decay
   claim must be read off positions away from the last two bindings, or from a
   configuration verified clean for that specific model.
2. The interference sweeps are unaffected: they query index 0, the oldest
   binding, which is never the failing position in any model or filler setting.

---

## 3e. The shipped substrate: what it is, and three things Block 0 corrected.

**Disclosure.** The model in the artifact (`data/artifact_d32m8.pt`, d=32,
N=256, 2 layers, state 8,192/layer, **27,776 params**) is trained on **2–8
bindings and 0–8 filler units** — the range the artifact actually displays. The
width-comparison family in §3d was trained on 2–14. **They are different
models**, and the §3d width table stays in these notes as the not-established
result it already is, out of the artifact. Choosing training coverage to match
display coverage is a normal design decision; it is only a problem if it is
silent, so it is stated here, in the README, and on the page.

The filler range matches the binding range (0–8 against 2–8) on purpose. If
filler left the training distribution before bindings did, the matched-length
control would degrade for the wrong reason and §3b would return in mirror image.

### Correction 1: the weak baseline was capacity, not training coverage.

We predicted the ~35% baseline came from training on 2–14 bindings. Retraining
`d32m4` on 2–8 made it **worse** where it mattered: 21.3% at 6 bindings, with
`swap_follows_context` at 14.5% — below the 16.7% chance line. That model was
not doing associative recall at all. The hypothesis was wrong.

### Correction 2: the constraint is neuron count, not embedding width or state size.

Two candidates were trained at an **identical 8,192-entry state**:

| candidate | N | d | state | recall | normal | swap |
|---|---|---|---|---|---|---|
| d=32, mult=8 | **256** | 32 | 8,192 | **91.4%** | 79.8% | 77.3% |
| d=64, mult=2 | 128 | 64 | 8,192 | 31.7% | 16.0% | 16.3% |

Same state size, opposite outcomes. Across every model trained in this project,
N=128 fails and N≥256 works, at both d=32 and d=64. So the binding constraint is
the **number of sparse neurons**, not `d` and not the size of the state — which
is what BDH's own premise of a large sparse neuron space (n >> d) predicts. The
prediction on record before this run was that `d` was the bottleneck; it was
wrong, and the plan's `mult=8` fallback was right.

### Correction 3: the interference sweep is NOT immune to the §3c artifact.

§3c closed by asserting the interference sweeps were unaffected "because they
query index 0, the oldest binding, which is never the failing position." **That
is false at small loads.** With 2 bindings, index 0 *is* the second-to-last one:
oldest and second-to-last are the same position until there are at least three
bindings. It showed up as load 1 reading **0.0%** between load 0 at 100% and
load 2 at 100% — a hole in the middle of a curve we were about to publish.

Fixed by separating the query from the binding block with one filler unit in
**both** conditions, which leaves their token counts identical (7 + 2k either
way) and so leaves the matched-length comparison intact. Load 1 returns to 100%.

## Open questions (recorded, not investigated — science is frozen)

1. **The binding curve is not monotone at the tail.** Recall reads 39.7%
   [36.5, 42.9] at load 6 and rises to 51.1% [47.8, 54.4] at load 7, n=900,
   disjoint intervals — real, not sampling noise. Seven competing bindings are
   harder than eight, reproducibly. No explanation. The dissociation the claim
   rests on is unaffected: filler is flat at 100.0% [99.6, 100.0] at every load.
   The artifact shows the measured curve including this wobble, and says it is
   unexplained.
2. **Why N≥256 and not N=128?** The threshold is sharp and sits between two
   powers of two we happened to test. Where it actually lies, and whether it
   tracks the number of bindings to be stored, is untested.
3. **§3c remains uncharacterised** — see that section.

## 4. `u_decay` is grounded in the paper, but the public code does not use it.

Definition 4 of the BDH paper defines the state update with a right-multiplication
by `U`, described as "a diagonal or block-diagonal matrix representing local
rotation or damping of state (such as ALiBi or RoPE)". The public reference
instantiates `U` as **RoPE only** — pure rotation, no damping.

So a decay control is *not* an invention of ours, but it is also *not* what the
released implementation runs. Our default is `u_decay = 1.0`, which reproduces
the reference exactly; setting it below 1 explores the damping case of the same
published `U`. The artifact must label it that way rather than implying the
released model has a forgetting knob.

### 4b. The stability-plasticity trade-off did not replicate. The decay control is cut.

The plan's §5.3 wanted a decay slider showing the classic trade: slower decay
retains longer but interferes more, faster decay is cleaner but shorter-lived.
The prediction was that curves for different `u` should **cross** — damping
worse for old bindings, no worse or better for recent ones.

Three attempts, each fixing the previous regime error:

1. **Wide state, 8 bindings.** Everything at ~100% almost everywhere. Nothing is
   forgotten, so damping has nothing to trade against. Inconclusive.
2. **Narrow state, 10 bindings.** Everything floored at ~10%. The state is
   already exhausted at that load, so again no dynamic range. Inconclusive.
3. **Narrow state, 3 and 5 bindings** — loads chosen to have headroom. This one
   could have shown the effect, and did not.

Recall by binding age, narrow state (2,048 entries), n=400 per point:

| bindings | u | oldest → newest |
|---|---|---|
| 3 | **1.00** | 42.5  31.2  23.5 |
| 3 | 0.95 | 39.0  32.5  19.0 |
| 3 | 0.90 | 7.8  7.2  5.8 |
| 3 | 0.80 | 7.8  7.8  5.8 |
| 5 | **1.00** | 24.5  23.2  24.5  13.0  12.5 |
| 5 | 0.95 | 24.8  23.8  19.2  14.5  10.2 |
| 5 | 0.90 | 5.5  5.2  9.8  7.0  5.0 |
| 5 | 0.80 | 5.0  5.0  9.5  7.2  4.0 |

**No crossing anywhere.** `u=1.0` is at least as good as every damped model at
every age and every load. Damping below 0.95 is catastrophic — overall recall
falls from 21.3% to 7.5% — and it never buys anything for recent bindings,
which is the entire content of the trade-off claim.

**What we cannot distinguish.** Whether the trade-off is absent in this toy
regime, or whether damped models need their own learning-rate and schedule and
our fixed recipe disadvantages them. Both are live; we have not tested the
second. So this is reported as *not demonstrated*, not as *refuted*.

**Decision: the decay slider is cut from the artifact.** A control whose
underlying effect we cannot demonstrate would be a decorative slider dressed as
a concept variable, which is exactly what the brief says to cut, and defending
it in a live review would be impossible. The artifact ships two controls —
which synapses are ablated, and interference load — both backed by results that
survived their own falsification tests.

`scripts/decay_small.py` and the data stay in the repo as a recorded negative
result. The `u_decay` parameter stays in `src/bdh.py` because it is a faithful
implementation of the published `U`, and is documented as unused by default.

---

## 5. The state we ablate is `rho` (d x n), not the neuron-neuron `sigma`.

Eq. (6) of the paper defines the conceptual BDH state as `sigma` on neuron-neuron
edges. Eq. (8) defines BDH-GPU, whose state is
`rho_{t,l} := (rho_{t-1,l} + LN(E y_{t,l-1}) x_{t,l}^T) U`, an outer product of a
`d`-vector and an `n`-vector — so `rho` is `d x n`, not `n x n`.

The two relate by the low-rank factorisation `sigma ~ D_y rho`: the neuron-neuron
synapse matrix is the trained `D_y` applied to the fast state `rho`, hence rank at
most `d`. `BDH.sigma()` materialises it for small `N`.

We ablate entries of **`rho`**, because that is the state the trained model
actually carries and mutates. Presenting `sigma` as the thing being edited would
be a misstatement. The explainer shows `sigma` as the neuron-neuron *view* and
says plainly that it is a rank-limited projection of `rho`.

---

## 6. Verified numerically before any science was run

`tests/test_equivalence.py` asserts the token-parallel training path and the
step-by-step recurrent path agree to `< 1e-4` (observed: ~2e-7) across layer
counts, head counts, and `u_decay` values. If they disagreed, every ablation
result computed on `rho` would describe a state the trained model never uses.
The same file checks causality and checks that wiping the state actually changes
the output.
