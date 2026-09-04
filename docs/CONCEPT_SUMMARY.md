# Where a fact lives: synaptic memory in Dragon Hatchling

**Concept:** Synaptic plasticity as short-term memory · **Artifact:** an interactive explainer with a live 27,776-parameter BDH-GPU running in the browser

---

**The design pressure.** A Transformer remembers what it has read by keeping it: the
KV cache grows linearly with context, so memory cost tracks how much has been
said rather than how much matters. Eviction, compression, retrieval, linear
attention and state-space models are all attempts to hold within-session memory
in something of fixed size. Fixed size buys bounded cost and forces a question
the Transformer never has to answer: when two facts compete for the same finite
state, what happens to them?

**What BDH changes.** Dragon Hatchling (BDH) reformulates attention as a
*synaptic write*. Its GPU formulation carries a state `ρ` updated by

> `ρ_{t,l} := (ρ_{t-1,l} + LN(E y_{t,l-1}) x_{t,l}ᵀ) U`  — Eq. (8), arXiv:2509.26507

Each token deposits a rank-1 outer product: reading strengthens a specific set
of connections, Hebbian-style, and the model answers by reading that state back.
Two details matter and are routinely garbled. First, `ρ` is `d × n` — in our
model 32 × 256 — **not** neuron-by-neuron; the conceptual `σ` of Eq. (6) relates
to it by `σ ≈ D_y ρ`, rank at most `d`. Second, `U` is defined as "local
rotation *or damping*"; the public reference implements rotation only (RoPE),
so BDH as released has no forgetting term. Activations are sparse and
non-negative — the paper reports about 5% of the `y` vectors active. The trade
is that nothing is addressable by position: there is no cache entry to evict,
only superposition in one fixed tensor.

| | Softmax + KV cache | Linear attention | BDH synaptic state |
|---|---|---|---|
| Memory growth | linear in tokens | fixed-size state | fixed-size state |
| Where session memory lives | cached keys/values | accumulated state matrix | `ρ`, synaptic writes over (channel, neuron) |
| Failure mode | eviction, context limit | interference | interference; positional nulls (measured here) |
| Interpretability handle | attention maps over tokens | state matrix | individual synapses; reported monosemantic at concept level |

**Our claim, and the control that carries it.** A fact the model just read is
held in a small, *locatable* set of synapses rather than in its weights. We
train a 27,776-parameter BDH-GPU on multi-query associative recall (Arora et al.,
*Zoology*, arXiv:2312.04927), where name–city pairings are resampled uniformly
per sequence — so the weight-optimal prior is uniform and the answer cannot come
from the parameters. Deleting the binding from context drops recall to 7.5%
against a 6.25% chance line; rebinding it moves the answer 99.75% of the time.

Ablating the **8 largest-magnitude entries of the Hebbian write** that laid one
binding down — 0.098% of an 8,192-entry state — takes recall from 99.8% to
**77.2%** [72.8, 81.0]. The result means nothing without a control, so we remove
the largest-magnitude entries *outside* that write: this necessarily removes at
least as much state mass, here **2.06×** as much, and recall stays at **100.0%**
[99.0, 100.0]. Across a dose ladder, targeted ablation falls to 10.8% while that
control never drops below 78%. Untouched bindings in the same sequence fall only
92.4% → 88.0%. The effect is therefore about *which* state was removed, not how
much. This is the mechanism the BDH paper reports at §6.3, where in-context
state "localize[s] on the same synapses consistently across multiple prompts".

**BDH-CQ's role, labelled.** BDH-CQ (arXiv:2608.09888) carries the same idea one
level up: a contextual memory `S_t` that demonstrations update at inference, and
a latent workspace `H_r` iterated to answer — no parameter updates, no verbalised
chain of thought. Its controlled interventions make our point at task scale:
with byte-identical held-out inputs, adding one demonstration at the test
complexity lifts depth-five nesting from 19/24 to 24/24, and recovers 13/24 from
a 0/24 baseline at ordering length eight. What is in state decides. Label the
evidence precisely: its 150M configuration reaches 29.5% pass@2 on public
ARC-AGI-1 at a computed $0.00070/task — a **cost-efficiency** result, not an
accuracy win, since GPT-5.6 Luna (Low) scores 34.2% on the same set; the audit
reproducing 29.5% was run by **co-authors**, so it is not an external
reproduction; and capability is structured rather than scalar — rotation
composes with relocation on 72/72, colour swap on 0/72. BDH-CQ's dimensions and
update rules are proprietary, and nothing we built reproduces them.

**The most important limitation.** We set out to show both halves of a claim —
that the binding is localised *and* that competing facts erase it — and only the
first survived. Our interference curve failed its own control: holding the
number of competing bindings fixed and varying only neutral filler still moved
recall from 50.4% to 100%, so the load axis was confounded with sequence
geometry. It is withdrawn. Two further results did not stand: damping never
produced the stability–plasticity crossover it predicts, and a capacity ordering
across state widths was confounded by training quality. Separately, our model
has an exact failure mode — retrieval fails in periodic bands of query-to-binding
offset, at 3 and 13–17 — whose mechanism (RoPE phase cancellation) remains a
hypothesis. Beyond this project, the open problem is consolidation: BDH's fast
state is erased between sessions, and turning useful fast state into durable
slow weights is unsolved. Our model is our own small training run of the
published architecture on a synthetic task — not an official Pathway model, and
far below any reported BDH scale. It is evidence about the mechanism, not a
reproduction of Pathway's results.

**Continue with** arXiv:2509.26507 §3.2 and §6.3, arXiv:2608.09888 §6, the
reference implementation at `github.com/pathwaycom/bdh`, and Arora et al.
(arXiv:2312.04927) for the recall task.

---

*Artifact, source and every figure's provenance: `github.com/Ritik650/DataForge-Pathway`. All measured values carry Wilson 95% intervals and an explicit n; results we withdrew are documented rather than deleted.*
