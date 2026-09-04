/**
 * BDH-GPU forward pass, ported from src/bdh.py `forward_recurrent`.
 *
 * Why the recurrent form and not the token-parallel one: the entire artifact
 * depends on the seam between the write at step t and the read at step t+1.
 * That seam is where a learner's ablation lands. The parallel form computes the
 * same function without ever building rho, so it has no seam to hook, and an
 * ONNX graph would have none either.
 *
 * Correctness is not assumed. fixture.json carries reference logits from the
 * Python implementation and the page asserts against them on load. If this file
 * drifts, the badge in the footer goes red.
 *
 * Shapes for the shipped substrate (nh=1, D=32, N=128, L=2):
 *   embed     (50, 32)      encoder   (1, 32, 128)
 *   encoder_v (1, 32, 128)  decoder   (128, 32)
 *   lm_head   (32, 50)      freqs     (128,)
 *   rho[l]    (32, 128)  -- d x n, NOT n x n. See README.
 */

export class BDH {
  /**
   * @param {ArrayBuffer} buffer  contents of weights.bin
   * @param {object} fixture      fixture.json (manifest + config)
   */
  constructor(buffer, fixture) {
    const all = new Float32Array(buffer);
    this.cfg = fixture.config;
    this.w = {};
    for (const e of fixture.manifest) {
      this.w[e.name] = all.subarray(e.offset, e.offset + e.count);
      this.w[e.name + "_shape"] = e.shape;
    }
    this.D = this.cfg.n_embd;
    this.nh = this.cfg.n_head;
    this.N = (this.cfg.mlp_internal_dim_multiplier * this.D) / this.nh;
    this.L = this.cfg.n_layer;
    this.V = this.cfg.vocab_size;
    this.u = this.cfg.u_decay;
  }

  /** LayerNorm with elementwise_affine=False: normalise only, no parameters. */
  static layerNorm(v, off, d, eps = 1e-5) {
    let mean = 0;
    for (let i = 0; i < d; i++) mean += v[off + i];
    mean /= d;
    let varr = 0;
    for (let i = 0; i < d; i++) {
      const x = v[off + i] - mean;
      varr += x * x;
    }
    varr /= d;
    const inv = 1 / Math.sqrt(varr + eps);
    for (let i = 0; i < d; i++) v[off + i] = (v[off + i] - mean) * inv;
  }

  /**
   * RoPE, matching src/bdh.py `rope`: pairs are (0,1), (2,3), ...
   * phase = (t * freq) mod 1, scaled to 2*pi.
   */
  ropeInPlace(vec, t) {
    const f = this.w.freqs;
    for (let i = 0; i + 1 < this.N; i += 2) {
      let ph = (t * f[i]) % 1;
      if (ph < 0) ph += 1;
      const a = ph * 2 * Math.PI;
      const c = Math.cos(a);
      const s = Math.sin(a);
      const v0 = vec[i];
      const v1 = vec[i + 1];
      // v * cos + rot(v) * sin, where rot = (-v1, v0)
      vec[i] = v0 * c - v1 * s;
      vec[i + 1] = v1 * c + v0 * s;
    }
  }

  /**
   * Run the sequence.
   * @param {number[]} ids
   * @param {?function(number, number, Float32Array): void} ablate
   *        called as (layer, t, rho) after the write at step t and before the
   *        read at t+1 -- the same contract as the Python `ablate` hook.
   * @returns {{logits: Float32Array, rho: Float32Array[], writes: Float32Array[][]}}
   */
  forward(ids, ablate = null, record = false) {
    const T = ids.length;
    const { D, N, L, V } = this;

    // x: (T, D) residual stream, layer-normalised embeddings
    let x = new Float32Array(T * D);
    for (let t = 0; t < T; t++) {
      const src = ids[t] * D;
      for (let i = 0; i < D; i++) x[t * D + i] = this.w.embed[src + i];
      BDH.layerNorm(x, t * D, D);
    }

    const rhos = [];
    const writes = [];

    for (let l = 0; l < L; l++) {
      // x_sparse = relu(x @ encoder)   (T, N)
      const xs = new Float32Array(T * N);
      for (let t = 0; t < T; t++) {
        for (let j = 0; j < N; j++) {
          let acc = 0;
          for (let i = 0; i < D; i++) acc += x[t * D + i] * this.w.encoder[i * N + j];
          xs[t * N + j] = acc > 0 ? acc : 0;
        }
      }

      const rho = new Float32Array(D * N); // rho is d x n
      const a = new Float32Array(T * D);
      const layerWrites = [];
      const qr = new Float32Array(N);

      for (let t = 0; t < T; t++) {
        for (let j = 0; j < N; j++) qr[j] = xs[t * N + j];
        this.ropeInPlace(qr, t);

        // READ from state written by tau < t
        for (let i = 0; i < D; i++) {
          let acc = 0;
          const base = i * N;
          for (let j = 0; j < N; j++) acc += rho[base + j] * qr[j];
          a[t * D + i] = acc;
        }

        // WRITE: rank-1 Hebbian outer product of the residual and the rotated
        // sparse activation, then damping (u = 1 for the shipped model)
        if (record) layerWrites.push(new Float32Array(D * N));
        for (let i = 0; i < D; i++) {
          const xi = x[t * D + i];
          const base = i * N;
          for (let j = 0; j < N; j++) {
            const d = xi * qr[j];
            rho[base + j] = (rho[base + j] + d) * this.u;
            if (record) layerWrites[layerWrites.length - 1][base + j] = d;
          }
        }

        if (ablate) ablate(l, t, rho);
      }

      // yKV = LN(a); y_sparse = relu(yKV @ encoder_v); gate by x_sparse
      for (let t = 0; t < T; t++) BDH.layerNorm(a, t * D, D);

      const xy = new Float32Array(T * N);
      for (let t = 0; t < T; t++) {
        for (let j = 0; j < N; j++) {
          let acc = 0;
          for (let i = 0; i < D; i++) acc += a[t * D + i] * this.w.encoder_v[i * N + j];
          const ys = acc > 0 ? acc : 0;
          xy[t * N + j] = xs[t * N + j] * ys;
        }
      }

      // yMLP = xy @ decoder; x = LN(x + LN(yMLP))
      const y = new Float32Array(T * D);
      for (let t = 0; t < T; t++) {
        for (let i = 0; i < D; i++) {
          let acc = 0;
          for (let j = 0; j < N; j++) acc += xy[t * N + j] * this.w.decoder[j * D + i];
          y[t * D + i] = acc;
        }
        BDH.layerNorm(y, t * D, D);
        for (let i = 0; i < D; i++) x[t * D + i] += y[t * D + i];
        BDH.layerNorm(x, t * D, D);
      }

      rhos.push(rho);
      if (record) writes.push(layerWrites);
    }

    // logits = x @ lm_head
    const logits = new Float32Array(T * V);
    for (let t = 0; t < T; t++) {
      for (let v = 0; v < V; v++) {
        let acc = 0;
        for (let i = 0; i < D; i++) acc += x[t * D + i] * this.w.lm_head[i * V + v];
        logits[t * V + v] = acc;
      }
    }

    return { logits, rho: rhos, writes };
  }

  /** Softmax over one row of the logits. */
  probsAt(logits, t) {
    const V = this.V;
    const row = logits.subarray(t * V, (t + 1) * V);
    let mx = -Infinity;
    for (let i = 0; i < V; i++) if (row[i] > mx) mx = row[i];
    const out = new Float32Array(V);
    let s = 0;
    for (let i = 0; i < V; i++) {
      out[i] = Math.exp(row[i] - mx);
      s += out[i];
    }
    for (let i = 0; i < V; i++) out[i] /= s;
    return out;
  }

  argmaxAt(logits, t) {
    const V = this.V;
    let best = 0;
    let bv = -Infinity;
    for (let i = 0; i < V; i++) {
      const v = logits[t * V + i];
      if (v > bv) { bv = v; best = i; }
    }
    return best;
  }
}

/**
 * Assert this port reproduces the Python reference. Returns a result object
 * rather than throwing, so the page can render a visible badge either way.
 */
export function verifyFixture(model, fixture) {
  const t0 = performance.now();
  const { logits } = model.forward(fixture.tokens);
  const ms = performance.now() - t0;

  const V = model.V;
  let maxErr = 0;
  for (let t = 0; t < fixture.logits.length; t++) {
    for (let v = 0; v < V; v++) {
      const err = Math.abs(logits[t * V + v] - fixture.logits[t][v]);
      if (err > maxErr) maxErr = err;
    }
  }
  const pred = model.argmaxAt(logits, fixture.ans_pos);
  return {
    pass: maxErr < fixture.tolerance && pred === fixture.answer,
    maxErr,
    tolerance: fixture.tolerance,
    predicted: pred,
    expected: fixture.answer,
    forwardMs: ms,
  };
}
