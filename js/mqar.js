/**
 * MQAR sequence construction, mirroring src/mqar.py's format.
 *
 *   <bos> k1 v1 k2 v2 ... [filler] ... q  ->  the city bound to q
 *
 * The page builds its own sequences rather than shipping one frozen example,
 * so a learner can press "new sequence" and watch the same mechanism operate on
 * bindings it has never seen. Pairings are drawn uniformly, exactly as in
 * training, which is what makes the answer unavailable from the weights.
 *
 * This does NOT reproduce numpy's RNG stream, and does not need to: the offline
 * figures are aggregates over Python-sampled sequences, and the page says so.
 * What it reproduces exactly is the token LAYOUT -- 2 tokens per binding, 2 per
 * filler unit, query last -- because the offsets that layout produces are what
 * the positional bands are measured against.
 */

export class Mqar {
  constructor(vocab) {
    this.v = vocab;
  }

  /** mulberry32 -- small, seeded, reproducible across reloads. */
  static rng(seed) {
    let s = seed >>> 0;
    return () => {
      s = (s + 0x6d2b79f5) >>> 0;
      let x = Math.imul(s ^ (s >>> 15), 1 | s);
      x = (x + Math.imul(x ^ (x >>> 7), 61 | x)) ^ x;
      return ((x ^ (x >>> 14)) >>> 0) / 4294967296;
    };
  }

  static sample(rnd, n, k) {
    const pool = Array.from({ length: n }, (_, i) => i);
    for (let i = pool.length - 1; i > 0; i--) {
      const j = Math.floor(rnd() * (i + 1));
      [pool[i], pool[j]] = [pool[j], pool[i]];
    }
    return pool.slice(0, k);
  }

  /**
   * @param {number} nPairs   bindings in the key-value block
   * @param {number} nFiller  filler units, 2 tokens each, binding nothing
   * @param {number} queryIdx which binding is queried (0 = oldest)
   * @param {number} seed
   */
  build(nPairs, nFiller, queryIdx, seed) {
    const v = this.v;
    const rnd = Mqar.rng(seed);
    const names = Mqar.sample(rnd, v.names.length, nPairs);
    const cities = Mqar.sample(rnd, v.cities.length, nPairs);

    const ids = [v.BOS];
    const chips = [{ text: "⟨bos⟩", role: "bos" }];
    const pairs = [];

    for (let i = 0; i < nPairs; i++) {
      const nameTok = v.NAME_OFF + names[i];
      const cityTok = v.CITY_OFF + cities[i];
      pairs.push({ name: names[i], city: cities[i], nameTok, cityTok });
      // the city token's position is where this binding's write lands
      if (i === queryIdx) {
        pairs[i].namePos = ids.length;
        pairs[i].writePos = ids.length + 1;
      }
      ids.push(nameTok, cityTok);
      const role = i === queryIdx ? "target" : "competing";
      chips.push({ text: v.itos[nameTok], role, kind: "name" });
      chips.push({ text: v.itos[cityTok], role, kind: "city" });
    }

    for (let f = 0; f < nFiller; f++) {
      const a = v.FILL_OFF + Math.floor(rnd() * v.filler.length);
      const b = v.FILL_OFF + Math.floor(rnd() * v.filler.length);
      ids.push(a, b);
      chips.push({ text: v.itos[a], role: "filler" });
      chips.push({ text: v.itos[b], role: "filler" });
    }

    const q = pairs[queryIdx];
    const ansPos = ids.length; // logits here predict the answer
    ids.push(q.nameTok);
    chips.push({ text: v.itos[q.nameTok], role: "query" });

    return {
      ids,
      chips,
      pairs,
      ansPos,
      answer: q.cityTok,
      answerText: v.itos[q.cityTok],
      queryText: v.itos[q.nameTok],
      writePos: q.writePos,
      namePos: q.namePos,
      // distance from the query back to the queried binding's city token,
      // the axis the positional bands are measured on
      offset: ansPos - q.writePos,
      nPairs,
      nFiller,
      queryIdx,
    };
  }
}
