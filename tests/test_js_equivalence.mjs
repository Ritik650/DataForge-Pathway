/**
 * HARD GATE. The browser port must reproduce src/bdh.py forward_recurrent.
 *
 * Nothing in the artifact means anything if this fails: the page claims to be
 * really computing, and every ablation a learner performs is computed by this
 * code. A drift here produces plausible-looking wrong answers rather than a
 * visible error, which is the worst failure mode available to us.
 *
 * Run: node tests/test_js_equivalence.mjs
 *
 * Checks, in order:
 *   1. weights.bin matches the manifest's declared float count
 *   2. logits match the Python reference to fixture.tolerance (1e-5)
 *   3. the argmax at the answer position matches
 *   4. the final rho checksums match, so the STATE agrees and not merely the
 *      output -- ablation acts on rho, so a correct logit with a wrong rho
 *      would still break every interactive result
 *   5. ablation actually changes the output (the hook is wired to the seam)
 *   6. forward pass latency, since the page must respond in under a second
 */

import { readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { dirname, join } from "node:path";

import { BDH, verifyFixture } from "../artifact/js/bdh.js";

const here = dirname(fileURLToPath(import.meta.url));
const dataDir = join(here, "..", "artifact", "data");

const fixture = JSON.parse(readFileSync(join(dataDir, "fixture.json"), "utf8"));
const raw = readFileSync(join(dataDir, "weights.bin"));
const buffer = raw.buffer.slice(raw.byteOffset, raw.byteOffset + raw.byteLength);

let failures = 0;
const ok = (cond, label, detail = "") => {
  console.log(`  ${cond ? "PASS" : "FAIL"}  ${label}${detail ? "  " + detail : ""}`);
  if (!cond) failures++;
};

console.log("JS <-> Python equivalence\n");

// 1. binary integrity
const floats = buffer.byteLength / 4;
ok(floats === fixture.total_floats, "weights.bin float count",
  `${floats} vs manifest ${fixture.total_floats}`);

const model = new BDH(buffer, fixture);
console.log(`  model d=${model.D} N=${model.N} layers=${model.L} vocab=${model.V}\n`);

// 2 + 3. logits and prediction
const res = verifyFixture(model, fixture);
ok(res.maxErr < fixture.tolerance, "logits match Python",
  `max|diff| = ${res.maxErr.toExponential(3)} (tol ${fixture.tolerance})`);
ok(res.predicted === res.expected, "argmax at answer position",
  `got ${res.predicted} (${fixture.token_strings ? "" : ""}) expected ${res.expected} = "${fixture.answer_string}"`);

// 4. state agreement, not just output agreement
const { rho } = model.forward(fixture.tokens);
for (const ref of fixture.rho_final_checksum) {
  const r = rho[ref.layer];
  let absSum = 0, mx = -Infinity, mn = Infinity;
  for (let i = 0; i < r.length; i++) {
    absSum += Math.abs(r[i]);
    if (r[i] > mx) mx = r[i];
    if (r[i] < mn) mn = r[i];
  }
  const relErr = Math.abs(absSum - ref.abs_sum) / Math.max(1e-9, Math.abs(ref.abs_sum));
  ok(relErr < 1e-4, `rho layer ${ref.layer} abs_sum`,
    `${absSum.toFixed(5)} vs ${ref.abs_sum} (rel ${relErr.toExponential(2)})`);
  ok(Math.abs(mx - ref.max) < 1e-4 && Math.abs(mn - ref.min) < 1e-4,
    `rho layer ${ref.layer} range`,
    `[${mn.toFixed(6)}, ${mx.toFixed(6)}] vs [${ref.min}, ${ref.max}]`);
}

// 5. the ablation hook is wired to the write/read seam
const wiped = model.forward(fixture.tokens, (l, t, r) => r.fill(0));
const base = model.forward(fixture.tokens);
let diff = 0;
for (let i = 0; i < base.logits.length; i++) {
  diff = Math.max(diff, Math.abs(base.logits[i] - wiped.logits[i]));
}
ok(diff > 1e-3, "ablation hook changes the output",
  `max|base - wiped| = ${diff.toExponential(3)}`);

// 6. latency budget
const t0 = performance.now();
const REPS = 20;
for (let i = 0; i < REPS; i++) model.forward(fixture.tokens);
const ms = (performance.now() - t0) / REPS;
ok(ms < 50, "forward latency", `${ms.toFixed(2)} ms/forward over ${REPS} reps`);

console.log(
  `\n${failures === 0 ? "ALL CHECKS PASS - JS port is equivalent" : `${failures} CHECK(S) FAILED`}`
);
process.exit(failures === 0 ? 0 : 1);
