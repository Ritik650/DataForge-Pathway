/**
 * Reading and ablating the synaptic state, in the browser.
 *
 * Every selector here is a port of the corresponding function in src/ablate.py.
 * They must stay in step: the page shows a learner one of these conditions and
 * tells them what the offline measurement found for it, so if the browser's
 * "top_other" is not Python's "top_other", the page is lying about its own
 * numbers.
 *
 *   targeted   top-m entries of the Hebbian write that laid this binding down
 *   matched    nearest-neighbour matched on |rho| from outside the target set
 *   top_other  the m largest |rho| entries outside the target set. The strict
 *              control: it removes AT LEAST as much state mass as targeted, so
 *              surviving it cannot be explained by how much was removed.
 *   random     uniform. Weak on purpose -- it removes far less mass, and shows
 *              why the naive control would have flattered us.
 */

/** Indices of the m largest-magnitude entries of a Float32Array. */
function topIndices(values, m, excludeSet = null) {
  const idx = [];
  for (let i = 0; i < values.length; i++) {
    if (excludeSet && excludeSet.has(i)) continue;
    idx.push(i);
  }
  idx.sort((a, b) => Math.abs(values[b]) - Math.abs(values[a]));
  return idx.slice(0, m);
}

/**
 * The rank-1 Hebbian write deposited at one token, for one layer.
 * @returns {Float32Array} same shape as rho (D*N), flattened
 */
export function writeAt(model, tokens, layer, t) {
  const { writes } = model.forward(tokens, null, true);
  return writes[layer][t];
}

export function selectTargeted(writeDelta, m) {
  return topIndices(writeDelta, m);
}

export function selectTopOther(rho, targeted, m) {
  return topIndices(rho, m, new Set(targeted));
}

export function selectMatched(rho, targeted, m, seed = 1) {
  // Nearest-neighbour on |rho| from outside the target set, without reuse.
  const excl = new Set(targeted);
  const cand = [];
  for (let i = 0; i < rho.length; i++) {
    if (!excl.has(i)) cand.push(i);
  }
  cand.sort((a, b) => Math.abs(rho[a]) - Math.abs(rho[b]));
  const vals = cand.map((i) => Math.abs(rho[i]));

  const used = new Set();
  const picked = [];
  for (const ti of targeted.slice(0, m)) {
    const target = Math.abs(rho[ti]);
    // binary search for the insertion point
    let lo = 0, hi = vals.length;
    while (lo < hi) {
      const mid = (lo + hi) >> 1;
      if (vals[mid] < target) lo = mid + 1;
      else hi = mid;
    }
    let a = lo - 1, b = lo;
    while (a >= 0 || b < cand.length) {
      const da = a >= 0 ? Math.abs(vals[a] - target) : Infinity;
      const db = b < cand.length ? Math.abs(vals[b] - target) : Infinity;
      const k = da <= db ? a : b;
      if (k < 0 || k >= cand.length) break;
      if (!used.has(k)) { used.add(k); picked.push(cand[k]); break; }
      if (k === a) a--; else b++;
    }
  }
  return picked;
}

export function selectRandom(rho, targeted, m, seed = 42) {
  const excl = new Set(targeted);
  // mulberry32, so the "random" control is reproducible across reloads
  let s = seed >>> 0;
  const rnd = () => {
    s = (s + 0x6d2b79f5) >>> 0;
    let x = Math.imul(s ^ (s >>> 15), 1 | s);
    x = (x + Math.imul(x ^ (x >>> 7), 61 | x)) ^ x;
    return ((x ^ (x >>> 14)) >>> 0) / 4294967296;
  };
  const pool = [];
  for (let i = 0; i < rho.length; i++) if (!excl.has(i)) pool.push(i);
  for (let i = pool.length - 1; i > 0; i--) {
    const j = Math.floor(rnd() * (i + 1));
    [pool[i], pool[j]] = [pool[j], pool[i]];
  }
  return pool.slice(0, m);
}

/** Total |rho| removed by a selection -- the quantity the controls equalise. */
export function removedMass(rho, indices) {
  let s = 0;
  for (const i of indices) s += Math.abs(rho[i]);
  return s;
}

/**
 * Build an ablation hook for model.forward.
 * Zeroes `indices` in `layer` from timestep `fromT` onward, matching
 * make_mask_ablator in src/ablate.py.
 */
export function ablator(layer, fromT, indices) {
  const idx = Int32Array.from(indices);
  return (l, t, rho) => {
    if (l === layer && t >= fromT) {
      for (let k = 0; k < idx.length; k++) rho[idx[k]] = 0;
    }
  };
}

/**
 * Run one condition end to end and report what the learner needs to see:
 * the answer probability, the argmax, and how much state mass was removed.
 */
export function runCondition(model, tokens, layer, writePos, ansPos, answer,
                             condition, m) {
  const baseRun = model.forward(tokens, null, true);
  const rho = baseRun.rho[layer];
  const delta = baseRun.writes[layer][writePos];

  let indices = [];
  if (condition !== "none") {
    const targeted = selectTargeted(delta, m);
    if (condition === "targeted") indices = targeted;
    else if (condition === "top_other") indices = selectTopOther(rho, targeted, m);
    else if (condition === "matched") indices = selectMatched(rho, targeted, m);
    else if (condition === "random") indices = selectRandom(rho, targeted, m);
  }

  const run = indices.length
    ? model.forward(tokens, ablator(layer, writePos, indices))
    : baseRun;

  const probs = model.probsAt(run.logits, ansPos);
  const top = [...probs.keys()]
    .sort((a, b) => probs[b] - probs[a])
    .slice(0, 5)
    .map((tok) => ({ token: tok, p: probs[tok] }));

  return {
    indices,
    removedMass: removedMass(rho, indices),
    pAnswer: probs[answer],
    predicted: model.argmaxAt(run.logits, ansPos),
    correct: model.argmaxAt(run.logits, ansPos) === answer,
    top5: top,
    rho,
    delta,
  };
}
