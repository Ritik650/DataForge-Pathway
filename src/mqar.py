"""Deterministic multi-query associative recall (MQAR) generator.

Sequence layout (the standard MQAR format, cf. Arora et al., "Zoology", 2023):

    <bos>  Mira Oslo  Devan Cairo  Nia Lima  ... [filler] ...  Nia ?  Mira ?

The key-value block binds each name to a city. The query block then re-presents
some of the names; at each query position the model must emit that name's city.
Every query position is a supervised recall event, which is why the format has
many queries per sequence -- one query per sequence would give one gradient
signal per ~30 tokens and the association never gets learned.

Why the answer cannot live in the weights
-----------------------------------------
Pairings are resampled uniformly at random for every sequence, so across
training each name is paired with every city equally often. The weight-optimal
prior for "which city follows Mira" is therefore uniform, and any name-specific
city preference in the weights would raise training loss. The binding used to
answer a query exists only in the recurrent state built while reading THIS
sequence.

We do not hold out specific (name, city) pairs. Withholding a fixed pair set
teaches the model the complementary rule -- "Mira is never paired with Oslo" --
and then evaluating on exactly those pairs measures learned avoidance rather
than failure to recall. We measured this: held-out recall fell BELOW
random-copy chance and degraded further with training. See docs/DESIGN_NOTES.md.

Instead the claim is checked behaviourally, by intervention:
    drop_fact  -- delete the queried pair from the KV block; recall must
                  collapse to chance, since nothing else could supply it.
    swap_fact  -- rebind the queried name to a different city; the answer must
                  follow the context, not any weight-level prior.
Both are implemented in eval_controls().
"""

import numpy as np

NAMES = [
    "Mira", "Devan", "Anya", "Kofi", "Rosa", "Ilya", "Nia", "Tomas",
    "Suri", "Bram", "Lena", "Omar", "Freya", "Jian", "Ada", "Pavel",
]
CITIES = [
    "Oslo", "Cairo", "Lima", "Perth", "Kyoto", "Quito", "Sofia", "Dakar",
    "Riga", "Tunis", "Bogota", "Hanoi", "Malmo", "Accra", "Cusco", "Vienna",
]
FILLER = [
    "the", "day", "was", "quiet", "and", "cold", "rain", "fell",
    "a", "bell", "rang", "twice", "wind", "moved", "slowly", "here",
]

PAD, BOS = 0, 1
_SPECIAL = 2

NAME_OFF = _SPECIAL
CITY_OFF = NAME_OFF + len(NAMES)
FILL_OFF = CITY_OFF + len(CITIES)
VOCAB_SIZE = FILL_OFF + len(FILLER)

ITOS = ["<pad>", "<bos>"] + NAMES + CITIES + FILLER


def decode(ids):
    return " ".join(ITOS[int(i)] for i in ids)


def is_name(tok):
    return NAME_OFF <= tok < CITY_OFF


def is_city(tok):
    return CITY_OFF <= tok < FILL_OFF


def make_example(rng, n_pairs=6, n_queries=3, n_filler=0, query_idx=None,
                 drop_queried=False, swap_queried_to=None, block=None):
    """Build one MQAR sequence.

    n_pairs      : bindings in the key-value block
    n_queries    : how many of them are queried afterwards
    n_filler     : neutral filler tokens inserted between the blocks, in units
                   of 2 tokens so one filler unit costs exactly one pair's
                   worth of sequence length but stores no association
    query_idx    : force the FIRST query to target this pair index. The number
                   of pairs written after it (n_pairs - 1 - query_idx) is the
                   interference load on that binding.
    drop_queried : remove the queried pair from the KV block (control: the
                   answer becomes unavailable, recall must fall to chance)
    swap_queried_to : rebind the queried name to this city index in the KV
                   block, leaving the query unchanged (control: the answer
                   must follow context)
    """
    names = rng.choice(len(NAMES), size=n_pairs, replace=False)
    cities = rng.choice(len(CITIES), size=n_pairs, replace=False)
    pairs = [(int(a), int(b)) for a, b in zip(names, cities)]

    if query_idx is None:
        query_idx = int(rng.integers(0, n_pairs))
    q_name, q_city = pairs[query_idx]

    if swap_queried_to is not None:
        # rebind in context; the truth for this sequence becomes the new city
        taken = {c for i, (_, c) in enumerate(pairs) if i != query_idx}
        assert swap_queried_to not in taken, "swap target already bound"
        pairs[query_idx] = (q_name, int(swap_queried_to))
        q_city = int(swap_queried_to)

    kv = list(pairs)
    if drop_queried:
        kv = [p for i, p in enumerate(kv) if i != query_idx]

    ids = [BOS]
    for nm, ct in kv:
        ids += [NAME_OFF + nm, CITY_OFF + ct]
    for _ in range(n_filler):
        ids += [FILL_OFF + int(w) for w in rng.choice(len(FILLER), size=2)]

    # query block: the targeted pair first, then other distinct names
    others = [i for i in range(n_pairs) if i != query_idx]
    rng.shuffle(others)
    q_order = [query_idx] + others[: max(0, n_queries - 1)]

    ans_positions, answers = [], []
    for qi in q_order:
        nm, ct = pairs[qi]
        ans_positions.append(len(ids))  # logits here predict the city
        answers.append(CITY_OFF + ct)
        ids += [NAME_OFF + nm, CITY_OFF + ct]

    if block is not None:
        if len(ids) > block:
            raise ValueError(f"sequence {len(ids)} exceeds block {block}")
        ids = ids + [PAD] * (block - len(ids))

    return {
        "ids": np.array(ids, dtype=np.int64),
        "ans_positions": ans_positions,
        "answers": answers,
        # the primary (first) query, used by every sweep
        "ans_pos": ans_positions[0],
        "answer": answers[0],
        "n_pairs": n_pairs,
        "n_queries": len(q_order),
        "n_filler": n_filler,
        "query_idx": query_idx,
        "n_after": n_pairs - 1 - query_idx,  # interference load
        "pairs": pairs,
        "kv": kv,
        "dropped": drop_queried,
        # token positions of the queried binding inside the KV block
        "name_pos": None if drop_queried else 1 + 2 * kv.index(pairs[query_idx]),
        "write_pos": None if drop_queried else 1 + 2 * kv.index(pairs[query_idx]) + 1,
    }


def make_batch(rng, batch_size, block, pairs_range=(2, 8), queries_range=(1, 4),
               filler_range=(0, 4)):
    """Training batch. Loss is masked to non-pad positions."""
    xs, ys, meta = [], [], []
    for _ in range(batch_size):
        npair = int(rng.integers(pairs_range[0], pairs_range[1] + 1))
        nq = int(rng.integers(queries_range[0], min(queries_range[1], npair) + 1))
        nf = int(rng.integers(filler_range[0], filler_range[1] + 1))
        ex = make_example(rng, npair, nq, nf, block=block + 1)
        xs.append(ex["ids"][:-1])
        ys.append(ex["ids"][1:])
        meta.append(ex)
    return np.stack(xs), np.stack(ys), meta


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    print(f"vocab_size = {VOCAB_SIZE}")

    ex = make_example(rng, n_pairs=4, n_queries=2, n_filler=2, query_idx=0)
    print("\nexample:")
    print(" ", decode(ex["ids"]))
    print(f"  queries at {ex['ans_positions']} -> "
          f"{[ITOS[a] for a in ex['answers']]}")
    print(f"  primary query targets pair {ex['query_idx']} "
          f"({ITOS[NAME_OFF + ex['pairs'][ex['query_idx']][0]]}), "
          f"interference load n_after={ex['n_after']}")
    print(f"  binding written at pos {ex['write_pos']} -> {ITOS[ex['ids'][ex['write_pos']]]}")

    rng2 = np.random.default_rng(5)
    e = make_example(rng2, n_pairs=4, n_queries=1, query_idx=1)
    print("\ncontrol: drop the queried binding")
    print("  full: ", decode(e["ids"]))
    rng2 = np.random.default_rng(5)
    d = make_example(rng2, n_pairs=4, n_queries=1, query_idx=1, drop_queried=True)
    print("  drop: ", decode(d["ids"]))

    rng2 = np.random.default_rng(5)
    s = make_example(rng2, n_pairs=4, n_queries=1, query_idx=1, swap_queried_to=11)
    print("  swap: ", decode(s["ids"]), "-> truth now", ITOS[s["answer"]])

    # uniformity check: no name should prefer any city in the training marginal
    from collections import Counter
    cnt = Counter()
    r = np.random.default_rng(3)
    for _ in range(20000):
        e = make_example(r, n_pairs=6, n_queries=2)
        for nm, ct in e["pairs"]:
            cnt[(nm, ct)] += 1
    v = np.array(list(cnt.values()))
    print(f"\npair-frequency spread over 20k sequences: "
          f"min={v.min()} max={v.max()} mean={v.mean():.1f} std={v.std():.1f} "
          f"(uniform => no name->city signal in the weights)")
