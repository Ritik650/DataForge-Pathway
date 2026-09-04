/**
 * The single source of every number the page displays.
 *
 * Rule for this artifact: no figure is written as a literal in markup or panel
 * code. Everything a reader sees is read from the JSON that a committed script
 * produced. We have already been bitten once by a hand-typed number -- a
 * parameter count that stayed at 106,240 after the real value became 104,704 --
 * and scripts/verify_claims.py exists to make that class of error uncommittable.
 *
 * If you need a number on the page and it is not in here, the answer is to
 * measure it and add it to a JSON file, not to type it into the HTML.
 */

const FILES = {
  fixture: "data/fixture.json",
  vocab: "data/vocab.json",
  gates: "data/block0_gates.json",
  preset: "data/preset_selection.json",
  presetFinal: "data/preset_final.json",
  dose: "data/dose_panel.json",
  positional: "data/positional_map.json",
};

export class Claims {
  constructor(data) {
    this.raw = data;
  }

  static async load(base = "") {
    const entries = await Promise.all(
      Object.entries(FILES).map(async ([key, path]) => {
        const res = await fetch(base + path);
        if (!res.ok) throw new Error(`claims: cannot load ${path} (${res.status})`);
        return [key, await res.json()];
      })
    );
    return new Claims(Object.fromEntries(entries));
  }

  // ---- substrate ---------------------------------------------------------
  get config() { return this.raw.fixture.config; }
  get nParams() { return this.raw.fixture.n_params; }
  get stateEntries() {
    const c = this.config;
    return c.n_embd * ((c.mlp_internal_dim_multiplier * c.n_embd) / c.n_head) * c.n_head;
  }
  get stateShape() {
    const c = this.config;
    return { d: c.n_embd, n: (c.mlp_internal_dim_multiplier * c.n_embd) / c.n_head };
  }
  get trainedRange() {
    return {
      pairsMax: this.raw.gates.trained_pairs_max,
      fillerMax: this.raw.gates.trained_filler_max,
    };
  }

  // ---- the demo preset ---------------------------------------------------
  get preset() {
    const p = this.raw.dose.preset;
    return { ...p, m: this.raw.dose.default_m };
  }

  /** The dose ladder row for a given m, or the default. */
  doseRow(m = null) {
    const target = m ?? this.raw.dose.default_m;
    return this.raw.dose.rows.find((r) => r.m === target) ?? null;
  }

  get doseLadder() { return this.raw.dose.rows; }

  // ---- gates -------------------------------------------------------------
  get gates() { return this.raw.gates.gates; }
  get gateCaveats() { return this.raw.gates.caveats ?? []; }

  // ---- positional bands --------------------------------------------------
  /** Mean recall and failure count grouped by token offset. */
  get offsetBands() {
    const by = new Map();
    for (const r of this.raw.positional.rows) {
      if (!by.has(r.token_offset)) by.set(r.token_offset, []);
      by.get(r.token_offset).push(r);
    }
    return [...by.entries()]
      .sort((a, b) => a[0] - b[0])
      .map(([offset, rows]) => ({
        offset,
        cells: rows.length,
        failures: rows.filter((r) => r.fail).length,
        meanRecall: rows.reduce((s, r) => s + r.recall, 0) / rows.length,
        minRecall: Math.min(...rows.map((r) => r.recall)),
        // 19 and 21 rest on 3 and 1 cells; the page must not draw them as solid
        undersampled: rows.length <= 3,
      }));
  }

  // ---- verification ------------------------------------------------------
  get tolerance() { return this.raw.fixture.tolerance; }

  /** Numbers that must agree wherever they appear. verify_claims.py cross-checks these. */
  get canonical() {
    const d = this.doseRow();
    return {
      n_params: this.nParams,
      state_entries: this.stateEntries,
      preset_pairs: this.preset.n_pairs,
      preset_filler: this.preset.n_filler,
      preset_m: this.preset.m,
      targeted_recall: d?.targeted,
      top_other_recall: d?.top_other,
      mass_ratio: d?.mass_ratio,
      bystander_base: d?.bystander_base,
      bystander_abl: d?.bystander_abl,
      selectivity_ratio: d?.selectivity_ratio,
    };
  }
}

/** Format helpers so percentages and intervals are rendered one way only. */
export const fmt = {
  pct: (x, dp = 1) => `${(x * 100).toFixed(dp)}%`,
  ci: ([lo, hi], dp = 1) =>
    `[${(lo * 100).toFixed(dp)}, ${(hi * 100).toFixed(dp)}]`,
  int: (x) => x.toLocaleString("en-US"),
  ratio: (x, dp = 2) => `${x.toFixed(dp)}×`,
  sci: (x, dp = 3) => x.toExponential(dp),
};
