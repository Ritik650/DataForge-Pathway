/**
 * Exercises the page's computation path outside the browser: sequence
 * construction, all four selectors, the ablation hook, and the numbers the
 * panels bind to. Catches API drift between app.js and the modules it calls
 * without needing a DOM.
 *
 * Run: node tests/test_page_logic.mjs
 */

import { readFileSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";

import { BDH } from "../artifact/js/bdh.js";
import { Mqar } from "../artifact/js/mqar.js";
import { runCondition, removedMass } from "../artifact/js/state.js";
import { Claims } from "../artifact/js/claims.js";

const here = dirname(fileURLToPath(import.meta.url));
const D = join(here, "..", "artifact", "data");
const read = (f) => JSON.parse(readFileSync(join(D, f), "utf8"));

const fixture = read("fixture.json");
const vocab = read("vocab.json");
const raw = readFileSync(join(D, "weights.bin"));
const model = new BDH(raw.buffer.slice(raw.byteOffset, raw.byteOffset + raw.byteLength), fixture);

const claims = new Claims({
  fixture, vocab, gates: read("block0_gates.json"),
  preset: read("preset_selection.json"), presetFinal: read("preset_final.json"),
  dose: read("dose_panel.json"), positional: read("positional_map.json"),
});

let fails = 0;
const ok = (c, label, detail = "") => {
  console.log(`  ${c ? "PASS" : "FAIL"}  ${label}${detail ? "  " + detail : ""}`);
  if (!c) fails++;
};

console.log("page logic\n");

// ── claims layer ──────────────────────────────────────────
const p = claims.preset;
ok(p.n_pairs === 8 && p.n_filler === 2 && p.m === 8, "preset from claims",
  `${p.n_pairs} bindings / ${p.n_filler} filler / m=${p.m}`);
ok(claims.stateEntries === 8192, "state entries", String(claims.stateEntries));
ok(claims.nParams === 27776, "param count", String(claims.nParams));

const bands = claims.offsetBands;
ok(bands.length === 11, "offset bands", `${bands.length} offsets`);
const b3 = bands.find((b) => b.offset === 3);
ok(b3 && b3.meanRecall < 0.5, "offset-3 band is a null", `mean ${(b3.meanRecall * 100).toFixed(1)}%`);
ok(bands.filter((b) => b.undersampled).every((b) => b.cells <= 3),
  "undersampled flag matches cell count");

// ── sequence construction ─────────────────────────────────
const mq = new Mqar(vocab);
const ex = mq.build(p.n_pairs, p.n_filler, 0, 12345);
ok(ex.ids.length === 1 + 2 * p.n_pairs + 2 * p.n_filler + 1, "token count",
  `${ex.ids.length} tokens`);
ok(ex.offset === 2 * p.n_pairs + 2 * p.n_filler - 1, "query offset matches layout formula",
  `offset ${ex.offset}, expected ${2 * p.n_pairs + 2 * p.n_filler - 1}`);
ok(ex.ids[ex.writePos] === ex.answer, "write position holds the answer token",
  `${vocab.itos[ex.ids[ex.writePos]]}`);
ok(ex.chips.length === ex.ids.length, "one chip per token");

// distinct names and cities, as the generator promises
const names = ex.pairs.map((q) => q.name), cities = ex.pairs.map((q) => q.city);
ok(new Set(names).size === names.length, "names distinct");
ok(new Set(cities).size === cities.length, "cities distinct");

// ── the live computation ──────────────────────────────────
const layer = fixture.config.n_layer - 1;
const args = [model, ex.ids, layer, ex.writePos, ex.ansPos, ex.answer];
const none = runCondition(...args, "none", 8);
ok(none.correct, "baseline recalls the binding",
  `p(${ex.answerText}) = ${none.pAnswer.toFixed(3)}`);
ok(none.indices.length === 0 && none.removedMass === 0, "none removes nothing");

const tgt = runCondition(...args, "targeted", 8);
const top = runCondition(...args, "top_other", 8);
const mat = runCondition(...args, "matched", 8);
const rnd = runCondition(...args, "random", 8);

ok(tgt.indices.length === 8, "targeted selects m entries");
ok(new Set(tgt.indices).size === 8, "targeted indices unique");
ok(top.indices.every((i) => !tgt.indices.includes(i)), "top_other disjoint from targeted");
ok(mat.indices.every((i) => !tgt.indices.includes(i)), "matched disjoint from targeted");
ok(rnd.indices.every((i) => !tgt.indices.includes(i)), "random disjoint from targeted");

// the load-bearing property: the strict control removes AT LEAST as much mass
ok(top.removedMass >= tgt.removedMass, "top_other removes >= targeted mass",
  `${top.removedMass.toFixed(1)} vs ${tgt.removedMass.toFixed(1)} (${(top.removedMass / tgt.removedMass).toFixed(2)}x)`);
ok(rnd.removedMass < tgt.removedMass, "random removes less (the weak control)",
  `${rnd.removedMass.toFixed(1)}`);

// NOT asserted per-sequence: that targeted ablation lowers p(answer) on THIS
// sequence. It is a stochastic effect -- at m=8 the answer flips on about 23%
// of sequences -- and on seed 12345 p(answer) actually rises slightly. An
// earlier version of this test asserted the per-sequence direction and failed,
// which is the same overclaim we corrected in the offset rule: an aggregate
// effect does not license a per-case prediction. The aggregate is checked below.

// top5 shape, which the bars bind to
ok(tgt.top5.length === 5 && tgt.top5[0].p >= tgt.top5[4].p, "top5 sorted descending");
ok(tgt.top5.every((t) => vocab.itos[t.token] !== undefined), "top5 tokens resolve to strings");

// ── aggregate over sequences, the "run trials" path ───────
let base = 0, tHit = 0, oHit = 0, pDropped = 0;
for (let i = 0; i < 60; i++) {
  const e = mq.build(p.n_pairs, p.n_filler, 0, 500000 + i);
  const a = [model, e.ids, layer, e.writePos, e.ansPos, e.answer];
  const b = runCondition(...a, "none", 8);
  if (!b.correct) continue;
  base++;
  const t = runCondition(...a, "targeted", 8);
  if (t.correct) tHit++;
  if (t.pAnswer < b.pAnswer) pDropped++;
  if (runCondition(...a, "top_other", 8).correct) oHit++;
}
ok(pDropped / base > 0.6, "targeted lowers p(answer) on most sequences (aggregate)",
  `${pDropped}/${base} = ${(pDropped / base * 100).toFixed(0)}%`);
const tPct = (tHit / base) * 100, oPct = (oHit / base) * 100;
console.log(`\n  live over ${base} baseline-correct sequences: ` +
  `targeted ${tPct.toFixed(1)}%, top_other ${oPct.toFixed(1)}%`);
const off = claims.doseRow(8);
console.log(`  offline (n=${claims.raw.dose.trials}):            ` +
  `targeted ${(off.targeted * 100).toFixed(1)}%, top_other ${(off.top_other * 100).toFixed(1)}%`);
ok(oPct > tPct, "live run reproduces the direction of the offline result",
  `gap ${(oPct - tPct).toFixed(1)} pts live vs ${((off.top_other - off.targeted) * 100).toFixed(1)} offline`);

console.log(`\n${fails === 0 ? "ALL PAGE LOGIC CHECKS PASS" : `${fails} CHECK(S) FAILED`}`);
process.exit(fails === 0 ? 0 : 1);
