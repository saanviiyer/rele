// Seeded randomness, so every curve on the page can be reproduced and the node tests see the
// same numbers as the browser.

export function rng(seed = 1) {
  // mulberry32
  let a = seed >>> 0;
  const f = () => {
    a = (a + 0x6d2b79f5) >>> 0;
    let t = a;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
  f.normal = () => {
    const u = f() || 1e-12, v = f();
    return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
  };
  // Beta(a, b) for integer-ish a, b via two Gammas (Marsaglia & Tsang)
  f.gamma = (k) => {
    if (k < 1) return f.gamma(k + 1) * Math.pow(f() || 1e-12, 1 / k);
    const d = k - 1 / 3, c = 1 / Math.sqrt(9 * d);
    for (;;) {
      let x, v;
      do { x = f.normal(); v = 1 + c * x; } while (v <= 0);
      v = v * v * v;
      const u = f();
      if (u < 1 - 0.0331 * x ** 4 || Math.log(u) < 0.5 * x * x + d * (1 - v + Math.log(v))) return d * v;
    }
  };
  f.beta = (x, y) => { const g = f.gamma(x); return g / (g + f.gamma(y)); };
  f.int = (n) => Math.floor(f() * n);
  return f;
}

// argmax with random tie-breaking: an agent that always picks the first of equal values
// explores in a biased way, which shows up in the curves
export function argmax(arr, rnd) {
  let best = -Infinity, idx = [];
  for (let i = 0; i < arr.length; i++) {
    if (arr[i] > best + 1e-12) { best = arr[i]; idx = [i]; }
    else if (Math.abs(arr[i] - best) <= 1e-12) idx.push(i);
  }
  return idx.length === 1 || !rnd ? idx[0] : idx[Math.floor(rnd() * idx.length)];
}

export const mean = (a) => a.reduce((s, x) => s + x, 0) / (a.length || 1);
export function softmax(z, temp = 1) {
  const m = Math.max(...z);
  const e = z.map((v) => Math.exp((v - m) / temp));
  const s = e.reduce((a, b) => a + b, 0);
  return e.map((v) => v / s);
}
export function sample(p, rnd) {
  let u = rnd(), i = 0;
  for (; i < p.length - 1; i++) { u -= p[i]; if (u < 0) return i; }
  return i;
}
