// Small worked examples, one per idea, each matching a slide or a section of the notes.
import { rng, mean } from './rng.js';

// ---------- L01: p_theta(tau) is a product of policy and dynamics terms ----------
// A 2-state, 2-action MDP with horizon T. Enumerate every trajectory tau = (s1, a1, ..., sT, aT)
// and its probability p(s1) prod_t pi(a_t|s_t) prod_{t<T} p(s_{t+1}|s_t, a_t).
export function enumerateTrajectories({ p1 = [0.5, 0.5], pi = [[0.7, 0.3], [0.2, 0.8]], P = [[[0.9, 0.1], [0.3, 0.7]], [[0.5, 0.5], [0.1, 0.9]]], r = [[0, 1], [2, 0]], T = 2 } = {}) {
  const out = [];
  const rec = (t, s, prob, factors, steps) => {
    for (let a = 0; a < 2; a++) {
      const f2 = [...factors, { kind: 'pi', s, a, p: pi[s][a] }];
      const st = [...steps, { s, a, r: r[s][a] }];
      const pa = prob * pi[s][a];
      if (t === T) { out.push({ steps: st, factors: f2, prob: pa, ret: st.reduce((x, y) => x + y.r, 0) }); continue; }
      for (let s2 = 0; s2 < 2; s2++) rec(t + 1, s2, pa * P[s][a][s2], [...f2, { kind: 'P', s, a, s2, p: P[s][a][s2] }], st);
    }
  };
  for (let s = 0; s < 2; s++) rec(1, s, p1[s], [{ kind: 'p1', s, p: p1[s] }], []);
  return { trajectories: out, total: out.reduce((x, o) => x + o.prob, 0), J: out.reduce((x, o) => x + o.prob * o.ret, 0) };
}

// ---------- L03: which way does the policy gradient push each trajectory? ----------
// Per-step forward velocities for the lecture's humanoid trajectories. The numbers are
// illustrative (the slides show videos, not values); their signs follow the descriptions.
export const HUMANOID = {
  set1: [
    { label: 'falls backwards', r: [-0.2, -0.6, -1.0, -1.2] },
    { label: 'small step forward, then falls backwards', r: [0.4, -0.4, -0.9, -1.0] },
    { label: 'falls forwards', r: [0.3, 0.8, 1.2, 0.6] },
    { label: 'stands still', r: [0, 0, 0, 0] },
    { label: 'one large step backwards, then small steps forward', r: [-1.5, 0.3, 0.3, 0.3] },
  ],
  set2: [
    { label: 'falls forwards', r: [0.3, 0.8, 1.2, 0.6] },
    { label: 'stumbles forward slowly', r: [0.4, 0.4, 0.3, 0.4] },
    { label: 'walks steadily', r: [0.8, 0.8, 0.8, 0.8] },
    { label: 'runs', r: [1.5, 1.6, 1.6, 1.7] },
  ],
};

// The weight each step's grad log pi gets: R(tau) (vanilla), reward-to-go (causality), and either
// minus the batch-average of the same quantity (baseline).
export function pgWeights(trajs, { causality = false, baseline = false } = {}) {
  const raw = trajs.map((tr) => tr.r.map((_, t) => (causality ? tr.r.slice(t) : tr.r).reduce((a, b) => a + b, 0)));
  if (!baseline) return raw;
  // a constant baseline per time step: the average over trajectories of that step's weight
  const T = Math.max(...raw.map((w) => w.length));
  const b = Array.from({ length: T }, (_, t) => mean(raw.map((w) => w[t]).filter((v) => v != null)));
  return raw.map((w) => w.map((v, t) => v - b[t]));
}

// ---------- L03: importance weights over a horizon ----------
// Per-step ratio pi_new/pi_old for a two-action policy, multiplied over T steps; returns quantiles
// of the product to show how its spread grows with T.
export function importanceSpread({ pOld = 0.5, pNew = 0.6, T = 10, samples = 4000, seed = 4 } = {}) {
  const rnd = rng(seed);
  const w = [];
  for (let i = 0; i < samples; i++) {
    let prod = 1;
    for (let t = 0; t < T; t++) {
      const a = rnd() < pOld ? 0 : 1;
      prod *= a === 0 ? pNew / pOld : (1 - pNew) / (1 - pOld);
    }
    w.push(prod);
  }
  w.sort((a, b) => a - b);
  const q = (p) => w[Math.min(w.length - 1, Math.floor(p * w.length))];
  const m = mean(w);
  return { mean: m, median: q(0.5), p05: q(0.05), p95: q(0.95), max: w[w.length - 1], sd: Math.sqrt(mean(w.map((x) => (x - m) ** 2))), ess: (w.reduce((a, b) => a + b, 0) ** 2) / w.reduce((a, b) => a + b * b, 0) / samples };
}

// Importance sampling itself: E_p[f] estimated from samples of q.
export function isEstimate({ muP = 1, muQ = 0, n = 50000, seed = 2, f = (x) => x * x } = {}) {
  const rnd = rng(seed);
  const pdf = (x, m) => Math.exp(-0.5 * (x - m) ** 2);
  let s = 0;
  for (let i = 0; i < n; i++) { const x = muQ + rnd.normal(); s += (pdf(x, muP) / pdf(x, muQ)) * f(x); }
  return { estimate: s / n, truth: muP * muP + 1 };
}

// ---------- L04: GAE on a fixed trajectory ----------
export function gae(r, V, { gamma = 0.99, lambda = 0.95, vLast = 0 } = {}) {
  const T = r.length;
  const delta = r.map((rt, t) => rt + gamma * (t + 1 < T ? V[t + 1] : vLast) - V[t]);
  const A = new Array(T);
  let a = 0;
  for (let t = T - 1; t >= 0; t--) { a = delta[t] + gamma * lambda * a; A[t] = a; }
  // forward-sum form, to check the recursion
  const Afwd = delta.map((_, t) => delta.slice(t).reduce((s, d, l) => s + (gamma * lambda) ** l * d, 0));
  const G = r.map((_, t) => r.slice(t).reduce((s, x, l) => s + gamma ** l * x, 0) + gamma ** (T - t) * vLast);
  return { delta, A, Afwd, G, mcAdv: G.map((g, t) => g - V[t]) };
}

// n-step targets on one trajectory: n = 1 is TD, n >= T is Monte Carlo.
export function nStepTargets(r, V, n, gamma = 1) {
  const T = r.length;
  return r.map((_, t) => {
    let y = 0;
    for (let k = 0; k < n && t + k < T; k++) y += gamma ** k * r[t + k];
    if (t + n < T) y += gamma ** n * V[t + n];
    return y;
  });
}

// ---------- L04: the MC vs TD think-pair-share ----------
// Two logged trajectories through a shared state X. Path 1: A -> X -> end with -1. Path 2:
// B -> X ... Only one trajectory ever passes from C into X and later ends +1; MC values C by the
// one return it saw, TD by X's value, which pools every trajectory through X.
export function mcVsTdShared({ episodes, gamma = 1 }) {
  // episodes: [{ states: ['A','X'], rewards: [0,-1] }]
  const mc = {}, mcN = {}, td = {};
  for (const ep of episodes) {
    let G = 0;
    for (let t = ep.states.length - 1; t >= 0; t--) {
      G = ep.rewards[t] + gamma * G;
      const s = ep.states[t];
      mc[s] = (mc[s] ?? 0) + G; mcN[s] = (mcN[s] ?? 0) + 1;
    }
  }
  for (const s of Object.keys(mc)) mc[s] /= mcN[s];
  // batch TD(0): iterate to the fixed point of the certainty-equivalence model
  const states = Object.keys(mc);
  states.forEach((s) => (td[s] = 0));
  for (let k = 0; k < 500; k++) {
    const acc = {}, n = {};
    for (const ep of episodes) for (let t = 0; t < ep.states.length; t++) {
      const s = ep.states[t], next = ep.states[t + 1];
      const y = ep.rewards[t] + (next ? gamma * td[next] : 0);
      acc[s] = (acc[s] ?? 0) + y; n[s] = (n[s] ?? 0) + 1;
    }
    for (const s of states) td[s] = acc[s] / n[s];
  }
  return { mc, td };
}
