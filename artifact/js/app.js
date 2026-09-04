/**
 * Wiring. Everything a learner changes is recomputed by the real forward pass;
 * nothing on this page is a stored animation of a result.
 *
 * Load order matters for honesty: the fixture check runs FIRST and its outcome
 * is rendered before any panel. If this port has drifted from src/bdh.py, the
 * badge says so and the numbers below it are not to be trusted.
 */

import { BDH, verifyFixture } from "./bdh.js";
import { Claims, fmt } from "./claims.js";
import { Mqar } from "./mqar.js";
import { runCondition } from "./state.js";

const $ = (id) => document.getElementById(id);

const CONDITIONS = [
  ["none", "None"],
  ["targeted", "Targeted"],
  ["matched", "Matched"],
  ["top_other", "Top-other"],
  ["random", "Random"],
];
const DOSES = [2, 4, 8, 16, 32, 64, 128];

const S = { model: null, claims: null, mqar: null, ex: null,
            condition: "targeted", doseIdx: 2, seed: 20260905, layer: 1 };

// ── boot ──────────────────────────────────────────────────
(async function boot() {
  let fixture, vocab;
  try {
    [fixture, vocab] = await Promise.all([
      fetch("data/fixture.json").then((r) => r.json()),
      fetch("data/vocab.json").then((r) => r.json()),
    ]);
    const buf = await fetch("data/weights.bin").then((r) => r.arrayBuffer());
    S.model = new BDH(buf, fixture);
    S.claims = await Claims.load();
  } catch (err) {
    badge("fail", `could not load the model — ${err.message}`);
    return;
  }

  // the gate: does this port still compute what Python computes?
  const v = verifyFixture(S.model, fixture);
  if (v.pass) {
    badge("pass", `forward pass verified · max|diff| ${v.maxErr.toExponential(1)}`);
  } else {
    badge("fail", `forward pass MISMATCH ${v.maxErr.toExponential(1)} — figures below are not trustworthy`);
  }

  const c = S.claims;
  S.layer = c.config.n_layer - 1;
  const sh = c.stateShape;
  $("substrate-line").textContent =
    `${fmt.int(c.nParams)} params · ${fmt.int(c.stateEntries)} state entries · ${v.forwardMs.toFixed(1)} ms/forward`;
  $("state-caption").textContent =
    `layer ${c.config.n_layer} · ρ · ${sh.d} rows × ${sh.n} columns = ${fmt.int(c.stateEntries)} entries`;

  S.mqar = new Mqar(vocab);
  buildControls();
  drawDoseChart();
  drawBandsChart();
  newSequence(true);
})();

function badge(kind, text) {
  const el = $("badge");
  el.className = `badge ${kind === "pass" ? "" : "fail"}`;
  const icon = kind === "pass"
    ? '<svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round"><polyline points="20 6 9 17 4 12"/></svg>'
    : '<svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5" stroke-linecap="round"><line x1="18" y1="6" x2="6" y2="18"/><line x1="6" y1="6" x2="18" y2="18"/></svg>';
  el.innerHTML = `${icon}<span class="num">${text}</span>`;
}

// ── controls ──────────────────────────────────────────────
function buildControls() {
  const seg = $("seg");
  seg.innerHTML = "";
  for (const [id, label] of CONDITIONS) {
    const b = document.createElement("button");
    b.textContent = label;
    b.setAttribute("aria-pressed", String(id === S.condition));
    b.onclick = () => { S.condition = id; syncSeg(); recompute(); };
    seg.appendChild(b);
  }
  $("dose-ticks").innerHTML = DOSES.map((m) => `<span>${m}</span>`).join("");
  const slider = $("dose");
  slider.value = String(S.doseIdx);
  slider.oninput = () => { S.doseIdx = +slider.value; recompute(); };
  $("new-seq").onclick = () => { S.seed = (S.seed * 1103515245 + 12345) >>> 0; newSequence(false); };
  $("run-trials").onclick = runTrials;
}

function syncSeg() {
  [...$("seg").children].forEach((b, i) =>
    b.setAttribute("aria-pressed", String(CONDITIONS[i][0] === S.condition)));
}

/**
 * Build the next sequence.
 *
 * `selected` is used once, for the opening view. At the default dose the
 * targeted ablation flips the answer on roughly a quarter of sequences -- it is
 * a probabilistic effect, and a page that opened on an unselected draw would
 * show nothing happening three times in four. So the FIRST sequence is chosen
 * by a stated rule: the lowest seed at which targeted ablation flips the answer
 * at the default dose. That is disclosed on the page, and "new sequence" always
 * draws unselected, so the learner meets the variability immediately.
 */
function newSequence(selected = false) {
  const p = S.claims.preset;
  const qi = p.query_idx ?? 0;
  const m = DOSES[S.doseIdx];

  if (selected) {
    for (let s = 1; s <= 400; s++) {
      const cand = S.mqar.build(p.n_pairs, p.n_filler, qi, s);
      const a = [S.model, cand.ids, S.layer, cand.writePos, cand.ansPos, cand.answer];
      if (!runCondition(...a, "none", m).correct) continue;
      if (!runCondition(...a, "targeted", m).correct) { S.seed = s; S.ex = cand; break; }
    }
    S.selectedOpening = true;
  }
  if (!S.ex || !selected) {
    S.selectedOpening = false;
    S.ex = S.mqar.build(p.n_pairs, p.n_filler, qi, S.seed);
  }
  renderSequence();
  recompute();
}

function renderSequence() {
  const seq = $("seq");
  seq.innerHTML = "";
  S.ex.chips.forEach((chip, i) => {
    const col = document.createElement("div");
    col.className = "chip-col";
    const d = document.createElement("div");
    d.className = `chip ${chip.role}${chip.kind === "city" ? " city" : ""}`;
    d.textContent = chip.text;
    const pos = document.createElement("div");
    pos.className = "chip-pos";
    pos.textContent = i === 0 ? "0" : String(i);
    col.append(d, pos);
    seq.appendChild(col);
  });
  const ex = S.ex;
  $("seq-caption").innerHTML =
    `${ex.nPairs} bindings · ${ex.nFiller} filler · querying the oldest · offset ${ex.offset}` +
    (S.selectedOpening
      ? ` · <span style="color:var(--accent)">opening sequence selected: lowest seed where the ablation flips the answer</span>`
      : ` · <span style="color:var(--pass)">unselected draw, seed ${S.seed}</span>`);
  $("truth-word").textContent = ex.answerText;
}

// ── the live computation ──────────────────────────────────
function recompute() {
  const m = DOSES[S.doseIdx];
  const ex = S.ex;
  const t0 = performance.now();
  const r = runCondition(S.model, ex.ids, S.layer, ex.writePos, ex.ansPos,
                         ex.answer, S.condition, m);
  const ms = performance.now() - t0;

  const nState = S.claims.stateEntries;
  $("dose-label").textContent =
    `${m} ${m === 1 ? "entry" : "entries"} · ${(m / nState * 100).toFixed(3)}% of state`;
  $("latency").textContent = `${ms.toFixed(1)} ms`;

  // state mass, expressed against the targeted selection so the control's
  // "removes more and survives" is legible without arithmetic
  const ref = runCondition(S.model, ex.ids, S.layer, ex.writePos, ex.ansPos,
                           ex.answer, "targeted", m);
  const ratio = ref.removedMass > 0 ? r.removedMass / ref.removedMass : 0;
  $("mass-targeted").textContent =
    S.condition === "none" ? "0.00×" : `${ratio.toFixed(2)}×`;

  const broken = !r.correct;
  drawBars(r, broken);
  $("p-answer").textContent = r.pAnswer.toFixed(3);
  $("pbig").className = `pbig${broken ? " broken" : ""}`;
  $("p-caption").textContent =
    S.condition === "none"
      ? `p(${ex.answerText}) · nothing removed`
      : `p(${ex.answerText}) · ${r.indices.length} entries removed · ${broken ? "recall broken" : "recall survives"}`;

  $("callout-text").textContent =
    `The ${ref.indices.length} entries the ${ex.queryText}→${ex.answerText} write deposited. ` +
    `${(ref.indices.length / nState * 100).toFixed(3)}% of the state.`;

  drawState(r, ref);
  renderReadout();
}

function drawBars(r, broken) {
  const itos = S.claims.raw.vocab.itos;
  $("bars").innerHTML = r.top5.map((t, i) => {
    const pct = (t.p * 100).toFixed(1);
    return `<div class="bar-row${i === 0 ? " top" : ""}${i === 0 && broken ? " broken" : ""}">
      <div class="bar-tok">${itos[t.token]}</div>
      <div class="bar-track"><div class="bar-fill" style="width:${pct}%"></div></div>
      <div class="bar-val">${t.p.toFixed(3)}</div>
    </div>`;
  }).join("");
}

function renderReadout() {
  const d = S.claims.doseRow(DOSES[S.doseIdx]);
  if (!d) { $("readout").innerHTML = ""; return; }
  $("readout").innerHTML = `
    <span>Offline at this dose · targeted <span class="num" style="color:var(--accent)">${fmt.pct(d.targeted)}</span></span>
    <span>top-other <span class="num" style="color:var(--pass)">${fmt.pct(d.top_other)}</span> at <span class="num">${fmt.ratio(d.mass_ratio)}</span> the mass</span>
    <span>untouched bindings <span class="num">${fmt.pct(d.bystander_base)} → ${fmt.pct(d.bystander_abl)}</span></span>
    <span>selectivity <span class="num">${d.selectivity_ratio.toFixed(1)}×</span></span>
    <span style="color:var(--fainter)">n=${S.claims.raw.dose.trials}, precomputed</span>`;
}

// ── the state heatmap ─────────────────────────────────────
function drawState(r, ref) {
  const cv = $("state-canvas");
  const ctx = cv.getContext("2d");
  const sh = S.claims.stateShape;
  const CW = cv.width / sh.n, CH = cv.height / sh.d;

  const rho = r.rho;
  let max = 0;
  for (let i = 0; i < rho.length; i++) { const a = Math.abs(rho[i]); if (a > max) max = a; }
  const scale = max > 0 ? 1 / max : 0;

  ctx.clearRect(0, 0, cv.width, cv.height);
  for (let d = 0; d < sh.d; d++) {
    for (let n = 0; n < sh.n; n++) {
      const v = Math.min(1, Math.pow(Math.abs(rho[d * sh.n + n]) * scale, 0.45));
      ctx.fillStyle = `rgb(${Math.round(251 - v * 213)},${Math.round(250 - v * 218)},${Math.round(249 - v * 227)})`;
      ctx.fillRect(n * CW, d * CH, CW, CH);
    }
  }

  const removed = new Set(r.indices);
  // always outline where this binding wrote, so the learner can see the target
  // whether or not it is the thing currently being removed
  for (const idx of ref.indices) {
    const d = Math.floor(idx / sh.n), n = idx % sh.n;
    const gone = removed.has(idx);
    if (gone) { ctx.fillStyle = "#ffffff"; ctx.fillRect(n * CW, d * CH, CW, CH); }
    ctx.strokeStyle = "#b3401b";
    ctx.lineWidth = gone ? 1.5 : 1;
    ctx.globalAlpha = gone ? 1 : 0.45;
    ctx.strokeRect(n * CW - 1.5, d * CH - 1.5, CW + 3, CH + 3);
    ctx.globalAlpha = 1;
  }
  // entries removed by a control condition, marked differently so the two are
  // never confused by colour alone
  if (S.condition !== "targeted" && S.condition !== "none") {
    ctx.strokeStyle = "#2f6f5e";
    ctx.lineWidth = 1;
    for (const idx of r.indices) {
      const d = Math.floor(idx / sh.n), n = idx % sh.n;
      ctx.fillStyle = "#ffffff";
      ctx.fillRect(n * CW, d * CH, CW, CH);
      ctx.strokeRect(n * CW - 1, d * CH - 1, CW + 2, CH + 2);
    }
  }
}

// ── learner-run trials ────────────────────────────────────
function runTrials() {
  const btn = $("run-trials");
  btn.disabled = true;
  $("trials-out").textContent = "running…";
  setTimeout(() => {
    const p = S.claims.preset;
    const m = DOSES[S.doseIdx];
    const N = 100;
    let base = 0, tgt = 0, top = 0;
    const t0 = performance.now();
    for (let i = 0; i < N; i++) {
      const ex = S.mqar.build(p.n_pairs, p.n_filler, p.query_idx ?? 0, 900000 + i);
      const a = runCondition(S.model, ex.ids, S.layer, ex.writePos, ex.ansPos, ex.answer, "none", m);
      if (!a.correct) continue;
      base++;
      if (runCondition(S.model, ex.ids, S.layer, ex.writePos, ex.ansPos, ex.answer, "targeted", m).correct) tgt++;
      if (runCondition(S.model, ex.ids, S.layer, ex.writePos, ex.ansPos, ex.answer, "top_other", m).correct) top++;
    }
    const ms = performance.now() - t0;
    $("trials-out").innerHTML =
      `your run, n=${base} of ${N} baseline-correct, m=${m} · ` +
      `targeted <span style="color:var(--accent)">${(tgt / base * 100).toFixed(1)}%</span> · ` +
      `top-other <span style="color:var(--pass)">${(top / base * 100).toFixed(1)}%</span> · ` +
      `${(ms / 1000).toFixed(1)}s`;
    btn.disabled = false;
  }, 20);
}

// ── charts ────────────────────────────────────────────────
const SVGNS = "http://www.w3.org/2000/svg";
function el(name, attrs, text) {
  const n = document.createElementNS(SVGNS, name);
  for (const [k, v] of Object.entries(attrs)) n.setAttribute(k, v);
  if (text != null) n.textContent = text;
  return n;
}

function axes(svg, W, H, yLabels) {
  svg.appendChild(el("line", { x1: 46, y1: 10, x2: 46, y2: H - 44, stroke: "#d5d2c8" }));
  svg.appendChild(el("line", { x1: 46, y1: H - 44, x2: W - 14, y2: H - 44, stroke: "#d5d2c8" }));
  for (const p of yLabels) {
    const y = (H - 44) - (p / 100) * (H - 54);
    svg.appendChild(el("line", { x1: 46, y1: y, x2: W - 14, y2: y, stroke: "#f0eee8" }));
    svg.appendChild(el("text", { x: 38, y: y + 4, "text-anchor": "end",
      "font-family": "IBM Plex Mono, monospace", "font-size": 10.5, fill: "#a8a79f" }, `${p}%`));
  }
}

function drawDoseChart() {
  const rows = S.claims.doseLadder;
  const W = 580, H = 230, plotW = W - 76;
  const svg = el("svg", { width: W, height: H, viewBox: `0 0 ${W} ${H}`, role: "img",
    "aria-label": "Dose-response: targeted ablation falls steeply while the top-other control stays high." });
  axes(svg, W, H, [0, 50, 100]);
  const px = (i) => 56 + (i / (rows.length - 1)) * (plotW - 20);
  const py = (v) => (H - 44) - v * (H - 54);
  const series = [
    ["random", "#c9c6bb", 2],
    ["top_other", "#2f6f5e", 2.5],
    ["targeted", "#b3401b", 2.5],
  ];
  for (const [key, colour, w] of series) {
    svg.appendChild(el("polyline", {
      points: rows.map((r, i) => `${px(i).toFixed(1)},${py(r[key]).toFixed(1)}`).join(" "),
      fill: "none", stroke: colour, "stroke-width": w }));
    rows.forEach((r, i) => svg.appendChild(
      el("circle", { cx: px(i).toFixed(1), cy: py(r[key]).toFixed(1), r: 3.2, fill: colour })));
  }
  rows.forEach((r, i) => svg.appendChild(el("text", { x: px(i).toFixed(1), y: H - 27,
    "text-anchor": "middle", "font-family": "IBM Plex Mono, monospace",
    "font-size": 10.5, fill: "#a8a79f" }, String(r.m))));
  svg.appendChild(el("text", { x: W / 2, y: H - 8, "text-anchor": "middle",
    "font-family": "IBM Plex Mono, monospace", "font-size": 10.5, fill: "#85847c" }, "entries removed"));
  svg.appendChild(el("text", { x: W - 20, y: 34, "text-anchor": "end",
    "font-family": "IBM Plex Mono, monospace", "font-size": 11, fill: "#2f6f5e" }, "top-other control"));
  svg.appendChild(el("text", { x: W - 20, y: H - 60, "text-anchor": "end",
    "font-family": "IBM Plex Mono, monospace", "font-size": 11, fill: "#b3401b" }, "targeted"));
  $("chart-dose").appendChild(svg);
  $("dose-note").textContent = `n=${S.claims.raw.dose.trials} per point · precomputed · the control removes more state mass at every dose`;
}

function drawBandsChart() {
  const bands = S.claims.offsetBands;
  const W = 580, H = 230;
  const svg = el("svg", { width: W, height: H, viewBox: `0 0 ${W} ${H}`, role: "img",
    "aria-label": "Mean recall by token offset. Null bands at offset 3 and offsets 13 to 17." });
  axes(svg, W, H, [0, 50, 100]);
  const bw = (W - 76) / bands.length;
  bands.forEach((b, i) => {
    const x = 52 + i * bw;
    const h = b.meanRecall * (H - 54);
    const weak = b.meanRecall < 0.9;
    svg.appendChild(el("rect", {
      x: x.toFixed(1), y: ((H - 44) - h).toFixed(1), width: (bw - 8).toFixed(1), height: h.toFixed(1),
      fill: weak ? "#f6e6de" : "#eeece5",
      stroke: weak ? "#b3401b" : (b.undersampled ? "#c4bfb2" : "#cfccc2"),
      "stroke-width": weak ? 1.5 : 1,
      "stroke-dasharray": b.undersampled ? "3 2" : "0" }));
    svg.appendChild(el("text", { x: (x + (bw - 8) / 2).toFixed(1), y: H - 27, "text-anchor": "middle",
      "font-family": "IBM Plex Mono, monospace", "font-size": 10.5, fill: "#a8a79f" }, String(b.offset)));
  });
  svg.appendChild(el("text", { x: W / 2, y: H - 8, "text-anchor": "middle",
    "font-family": "IBM Plex Mono, monospace", "font-size": 10.5, fill: "#85847c" }, "token offset from query"));
  $("chart-bands").appendChild(svg);
}
