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

| Quantity | Where produced | Status |
|---|---|---|
| Parallel/recurrent agreement (~2e-7) | `tests/test_equivalence.py` | measured |
| Recall accuracy vs copy-chance | `src/train.py::evaluate` | in progress |
| drop / swap behavioural controls | `src/train.py::eval_controls` | in progress |
| Targeted vs magnitude-matched random ablation | `src/ablate.py` | not yet run |
| Interference curve (bindings vs filler) | `src/sweeps.py` | not yet run |
| Decay trade-off | `src/sweeps.py` | not yet run |
