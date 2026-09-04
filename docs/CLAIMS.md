# Claims ledger

Every externally sourced number used anywhere in this project, with the primary
source and its verbatim wording. **Nothing enters the artifact, README, blog, or
one-page summary unless it appears here with status VERIFIED.**

Verification pass: 2026-09-04, against `arxiv.org/html/2509.26507v1`,
`arxiv.org/html/2608.09888v1`, and `github.com/pathwaycom/bdh`. Local copies of
the fetched sources are in `refs/`.

---

## BDH — arXiv:2509.26507 (Kosowski, Uznański, Chorowski, Stamirowska, Bartoszkiewicz; 30 Sep 2025)

| # | Claim | Status | Source wording |
|---|---|---|---|
| B1 | Title and authors as cited | VERIFIED | "The Dragon Hatchling: The Missing Link between the Transformer and Models of the Brain"; 5 authors as listed; submitted 30 September 2025 |
| B2 | Rivals GPT-2 at matched parameters, 10M–1B | VERIFIED | "empirically BDH rivals GPT2 performance on language and translation tasks, at the same number of parameters (10M to 1B), for the same training data" |
| B3 | Activations sparse and positive | VERIFIED | "Activation vectors of BDH are sparse and positive." |
| B4 | ~5% sparsity | VERIFIED — **wording matters** | §6.4: "The positive activations of BDH-GPU exhibit sparsity (at about 5% level) in the **y vectors of its state space dynamics**." Cite as a property of the `y` vectors, not loosely as "5% of neurons". |
| B5 | Interpretability of state is architectural | VERIFIED | "Interpretability of state, which goes beyond interpretability of neurons and model parameters, is an inherent feature of the BDH architecture." |
| B6 | Working memory is synaptic plasticity with Hebbian learning | VERIFIED | "The working memory of BDH during inference entirely relies on synaptic plasticity with Hebbian learning using spiking neurons." |
| B7 | Heavy-tailed / high-modularity connectivity | VERIFIED | "The neuron interaction network of BDH is a graph of high modularity with heavy-tailed degree distribution." |
| B8 | Monosemanticity demonstrated | VERIFIED | "We demonstrate monosemanticity in BDH on language tasks." |
| B9 | **Synapse localisation across prompts** — the direct precedent for our claim | VERIFIED | §6.3: "In-context state of BDH-GPU attention is shown to localize on the same synapses (neuron-neuron links) consistently across multiple prompts, allowing for some basic features, the interpretation of the current in-context state based on the reading of state of an individual synapse." |
| B10 | BDH-GPU state update, Eq. (8) | VERIFIED | `rho_{t,l} := (rho_{t-1,l} + LN(E y_{t,l-1}) x_{t,l}^T) U` |
| B11 | Conceptual BDH state update, Eq. (6) | VERIFIED | `sigma_{t,l} := (sigma_{t-1,l} + ((y_{t,l-1} x_{t,l}^T) ⊙ G_s)) U` |
| B12 | `U` is rotation **or damping** | VERIFIED — load-bearing for our decay control | Definition 4: "Here, `U ∈ R^{n×n}` is a diagonal or block-diagonal matrix representing local rotation or damping of state (such as ALiBi or RoPE)" |
| B13 | Parameter count `3nd + 2|Ω|d` | VERIFIED | "The model has `3nd+2Ωd=(3+o(1))nd` parameters" |
| B14 | Sudoku Extreme 97.4% is **not** reproducible from the public repo | VERIFIED | Repo README: "The Sudoku Extreme result refers to Pathway's internal BDH implementation, not to the current open-source repository. This repository contains the implementation of the baseline variant as described in our public paper and does not reproduce the 97.4% benchmark result out of the box." |
| B15 | Reference repo licence | VERIFIED | MIT, "Copyright 2025 Pathway Technology, Inc." |

**Do not say:** that BDH is a state-space model in the Mamba sense. The paper
does call BDH "a practical, performant state-of-the-art attention-based state
space sequence learning architecture", so the phrase "state space" *does* appear
in the abstract — but BDH-GPU is built from ReLU-low-rank transformations with
linear attention, which is a different construction from selective SSMs. The
track brief flags this specific confusion. Quote the abstract if the distinction
comes up rather than paraphrasing either way.

---

## BDH-CQ — arXiv:2608.09888 (Engdahl, Kosowski, Chorowski, Stamirowska, Uznański, Jiang, Phadke, Kinas, Zhong; 10 Aug 2026)

| # | Claim | Status | Source wording |
|---|---|---|---|
| C1 | 150M config, 29.5% pass@2, $0.00070/task | VERIFIED | "A 150M-parameter configuration reaches 29.5% pass@2 on ARC-AGI-1 at a computed $0.00070 per task" |
| C2 | pass@1 = 24.25% (97 tasks); pass@2 = 118 tasks, CI [25.24, 34.15] | VERIFIED | Table 1 |
| C3 | GPT-5.6 Luna (Low) scores **34.2%** at $0.040 | VERIFIED | "57x cheaper than GPT 5.6 Luna (Low) which scores 34.2% at $0.040 (ARC Prize Foundation, 2026)" |
| C4 | ~57× cheaper, or ~11× after the 30 Jul 2026 price cut | VERIFIED — **correct the date** | "OpenAI's 80% public API price reduction of GPT 5.6 Luna on July 30, 2026, which is not reflected in ARC Prize's data **as of August 6, 2026**". The project plan said "as of July 2026"; the paper says August 6, 2026. |
| C5 | This is a **cost-efficiency** result, not an accuracy win | VERIFIED | Luna (Low) scores higher (34.2% vs 29.5%). The paper claims "a new state of the art in benchmark **cost efficiency**". |
| C6 | Independent audit was by **co-authors** | VERIFIED — evidence labelling | "An independent black-box audit conducted by **co-authors from Bielik and New York University** reproduced the deployed system's 29.5% pass@2 score... without access to model weights." Report it as a co-author audit, not an external reproduction. |
| C7 | ConceptARC semantic IDs 59.38% pass@2, CI [51.63, 66.68]; test-pair 77.92% | VERIFIED | Table 1; "The 18.5-point gap between semantic ConceptARC pair accuracy (77.92%) and strict task accuracy (59.38%)" |
| C8 | Coverage: depth-five nesting 19/24 → 24/24 with matched support | VERIFIED | "matched support raises depth-five nesting from 19/24 to 24/24 exact outputs at pass@2" |
| C9 | Coverage: length-eight ordering 0/24 → 13/24 | VERIFIED | "For ordering, it recovers 13/24 outputs from a 0/24 baseline"; inputs are "byte-identical length-eight ordering and depth-five nesting" |
| C10 | Composition: rotation+relocation 72/72; reflection+relocation 47/72; colour swap+relocation **0/72** | VERIFIED | "Rotation composed with relocation is solved on 72/72 tasks, while reflection composed with relocation is solved on 47/72 tasks... the model never learns to compose it with relocation (0/72)" |
| C11 | Colour swap acquired atomically only in the original family (26/72 pooled) | VERIFIED | as quoted above |
| C12 | Effort: HIGH 29.5% / 0% cost reduction; MEDIUM 27% / 11%; LOW 21% / 22% | VERIFIED | Table 5 |
| C13 | Internals are proprietary | VERIFIED | "Dimensions, exact update rules, and implementation details remain proprietary"; "the complete internal training recipe remains proprietary" |
| C14 | Two distinct state objects: contextual memory `S_t`, reasoning workspace `H_r` | VERIFIED | "`S_t` changes as evidence is encountered and supports in-context learning. `H_r` carries the ongoing computation used to answer the current query." |
| C15 | Scaling 1B–600B | VERIFIED — **but note what it is** | "Early experiments confirm Transformer-like scaling laws apply during pretraining at scales from 1B to 600B parameters, while preserving the latent reasoning capabilities specific to BDH-CQ." This is an asserted result; the report presents no multi-scale table for it. Label it as a developer-reported claim without published supporting data. |
| C16 | BDH provenance description | VERIFIED | "a post-Transformer sequence-model architecture built around high-dimensional positive activations, low-rank communication, and a recurrent associative state... Its GPU-oriented formulation uses BDH layers combining ReLU-low-rank transformations with linear attention in a large neuron or feature space." |

---

## Corrections to the project plan's Appendix A

1. **C4** — the ARC Prize cost snapshot is "as of August 6, 2026", not July 2026.
2. **B4** — the 5% sparsity figure is stated for the `y` vectors of the state
   space dynamics. "Roughly 5% of neurons are active" is a loose paraphrase;
   use the paper's scoping.
3. **A.3 addition** — the MQAR task we build on has a primary source of its own
   (Arora et al., *Zoology*, arXiv:2312.04927, 2023). It should be cited beside
   the task description, and it counts toward the ≥3 recent primary papers.
4. **§5.3 (decay)** — grounded, but not in the way the plan implied. The decay
   control is the damping case of the published `U` (B12); the public reference
   implementation instantiates `U` as RoPE rotation only. Say both.

---

## Our own measured numbers

These are results we produce, not sourced claims. They are listed so the two
categories never get mixed in the write-up. Each must carry `n`, seed, and an
interval.

Values are listed, not just quantities, because `scripts/verify_claims.py`
asserts that every figure appearing on the artifact is present in this ledger.
A number on the page that is not here fails the build.

| # | Quantity | Value | Where produced |
|---|---|---|---|
| M1 | Parallel vs recurrent agreement | ~2e-7 | `tests/test_equivalence.py` |
| M2 | Browser port vs Python | 2.6e-6, tol 1e-5 | `tests/test_js_equivalence.mjs` |
| M3 | Weight export round-trip | 0.000e+00 | `scripts/export_weights.py` |
| M4 | Recall, gate model | 99.66%, n=4653 | `src/train.py::evaluate` |
| M5 | Copy-a-context-city chance | 22.8% | same |
| M6 | Drop the binding from context | **7.5%** (uniform chance 6.25%) | `src/train.py::eval_controls` |
| M7 | Rebind — answer follows context | 99.75% | same |
| M8 | Shipped substrate | 27,776 params, 8,192 state entries/layer | `scripts/export_weights.py` |
| M9 | Preset baseline recall | 99.8%, n=400 | `artifact/data/dose_panel.json` |
| M10 | Targeted ablation at m=8 | 77.2% [72.8, 81.0] | same |
| M11 | `top_other` control at m=8 | 100.0% [99.0, 100.0] at 2.06× mass | same |
| M12 | Bystanders at m=8 | 92.4% → 88.0% | same |
| M13 | Selectivity ratio at m=8 | 5.9× (n=400); 6.1× in the n=250 search | `dose_panel.json`, `preset_selection.json` |
| M14 | Targeted at the largest dose | 10.8% at m=128 | `dose_panel.json` |
| M15 | Offset-3 band mean recall | 42.8% over 14 cells | `artifact/data/positional_map.json` |
| M16 | Offset-3 at zero filler | 0.0%, all 7 cells, 2–8 bindings | same |
| M17 | Ablation flips the answer at m=8 | ~23% of sequences | `tests/test_page_logic.mjs` |
| M18 | **Test C** — binding count fixed, filler 0→3 | **50.4%** → 40.0% → 73.6% → 100.0% | `positional_map.json` |
| M19 | Interference non-monotonicity | 39.7% [36.5,42.9] at load 6 → 51.1% [47.8,54.4] at load 7, n=900 | `DESIGN_NOTES` §3f |
| M20 | Decay, undamped overall recall | **21.3%** | `artifact/data/decay_small.json` |
| M21 | Decay at u<0.95 | **7.5%** overall — collapse | same |
| M22 | Capacity gate failures | 83.6% and 92.4% load-0 recall at 24k steps | `artifact/data/interference.json` |

**Note on M6 and M21.** Both read 7.5% and they are unrelated: M6 is recall
after deleting the binding from context on the gate model; M21 is overall recall
of a heavily damped model. The coincidence is why values live in this table
rather than being matched by string alone.
