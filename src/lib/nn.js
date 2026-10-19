// A very small neural network with hand-written backprop and Adam, enough for the policies,
// critics and Q-networks on these pages. Hidden layers use tanh; the output is linear.
import { rng } from './rng.js';

export function mlp(sizes, seed = 1, scaleOut = 0.01) {
  const rnd = rng(seed);
  const layers = [];
  for (let l = 0; l < sizes.length - 1; l++) {
    const nIn = sizes[l], nOut = sizes[l + 1];
    const last = l === sizes.length - 2;
    const sd = last ? scaleOut : Math.sqrt(1 / nIn);
    layers.push({
      nIn, nOut,
      W: Float64Array.from({ length: nIn * nOut }, () => rnd.normal() * sd),
      b: new Float64Array(nOut),
    });
  }
  const net = { sizes, layers, opt: null };
  return net;
}

// Forward one input; keeps the activations for backward.
export function forward(net, x) {
  const acts = [Float64Array.from(x)];
  let h = acts[0];
  net.layers.forEach((L, l) => {
    const out = new Float64Array(L.nOut);
    for (let o = 0; o < L.nOut; o++) {
      let s = L.b[o];
      const w = o * L.nIn;
      for (let i = 0; i < L.nIn; i++) s += L.W[w + i] * h[i];
      out[o] = l === net.layers.length - 1 ? s : Math.tanh(s);
    }
    acts.push(out);
    h = out;
  });
  return { out: h, acts };
}

export function zeroGrads(net) {
  return net.layers.map((L) => ({ W: new Float64Array(L.W.length), b: new Float64Array(L.b.length) }));
}

// Accumulate gradients of a scalar loss given dL/d(output) for one forward pass.
export function backward(net, fwd, gOut, grads) {
  let g = Float64Array.from(gOut);
  for (let l = net.layers.length - 1; l >= 0; l--) {
    const L = net.layers[l], hIn = fwd.acts[l], G = grads[l];
    const gIn = new Float64Array(L.nIn);
    for (let o = 0; o < L.nOut; o++) {
      const go = g[o];
      if (!go) continue;
      G.b[o] += go;
      const w = o * L.nIn;
      for (let i = 0; i < L.nIn; i++) { G.W[w + i] += go * hIn[i]; gIn[i] += go * L.W[w + i]; }
    }
    if (l > 0) for (let i = 0; i < L.nIn; i++) gIn[i] *= 1 - hIn[i] * hIn[i]; // tanh'
    g = gIn;
  }
}

// Adam step (Kingma & Ba 2015) that descends the accumulated gradient, scaled by 1/n.
export function adam(net, grads, { lr = 1e-2, n = 1, clip = 5 } = {}) {
  if (!net.opt) net.opt = { t: 0, m: grads.map((g) => ({ W: new Float64Array(g.W.length), b: new Float64Array(g.b.length) })), v: grads.map((g) => ({ W: new Float64Array(g.W.length), b: new Float64Array(g.b.length) })) };
  const o = net.opt;
  o.t++;
  let norm = 0;
  for (const g of grads) { for (const v of g.W) norm += (v / n) ** 2; for (const v of g.b) norm += (v / n) ** 2; }
  norm = Math.sqrt(norm);
  const k = norm > clip ? clip / norm : 1;
  const c1 = 1 - 0.9 ** o.t, c2 = 1 - 0.999 ** o.t;
  net.layers.forEach((L, l) => {
    for (const key of ['W', 'b']) {
      const p = L[key], g = grads[l][key], m = o.m[l][key], v = o.v[l][key];
      for (let i = 0; i < p.length; i++) {
        const gi = (g[i] / n) * k;
        m[i] = 0.9 * m[i] + 0.1 * gi;
        v[i] = 0.999 * v[i] + 0.001 * gi * gi;
        p[i] -= (lr * m[i]) / c1 / (Math.sqrt(v[i] / c2) + 1e-8);
      }
    }
  });
  return norm;
}

export function copyNet(net) {
  return { sizes: net.sizes, layers: net.layers.map((L) => ({ ...L, W: Float64Array.from(L.W), b: Float64Array.from(L.b) })), opt: null };
}
