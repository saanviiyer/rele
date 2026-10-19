// Offline RL: learn only from a fixed dataset, with no new interaction. Fitted Q-iteration
// bootstraps from max_a' Q(s', a'), and for actions the data never tried, Q is whatever the
// function approximator happens to say. The max picks those guesses, so values for unseen
// actions are overestimated and the greedy policy steers toward them (Levine et al. 2020).
// Two fixes are shown: constrain the max to actions seen in the data (in the spirit of BCQ,
// Fujimoto et al. 2019), or push Q down on all actions and up on data actions (CQL, Kumar et al.
// 2020).
import { rng, argmax } from './rng.js';
import { valueIteration, evaluate } from './dp.js';

// A dataset from a behaviour policy: epsilon-noisy around a given policy (e.g. a mediocre one).
export function collect(env, behaviour, { episodes = 30, eps = 0.3, seed = 3, maxSteps = 60 } = {}) {
  const rnd = rng(seed);
  const D = [];
  for (let e = 0; e < episodes; e++) {
    let s = env.start;
    for (let t = 0; t < maxSteps; t++) {
      const a = rnd() < eps ? rnd.int(4) : behaviour[s];
      const o = env.stepEnv(s, a, rnd);
      D.push({ s, a, r: o.r, s2: o.s2, done: o.done });
      s = o.s2;
      if (o.done) break;
    }
  }
  return D;
}

// method: 'naive' | 'constrained' | 'cql'. Unseen (s, a) start from noise of size `noise`,
// standing in for the arbitrary outputs of a network on inputs it was never trained on.
export function fittedQ(env, D, { method = 'naive', gamma = env.gamma, iters = 200, noise = 1, alpha = 1, seed = 8, lr = 0.5 } = {}) {
  const rnd = rng(seed);
  const Q = Array.from({ length: env.N }, () => [0, 1, 2, 3].map(() => noise * rnd.normal()));
  const seen = Array.from({ length: env.N }, () => [false, false, false, false]);
  for (const d of D) seen[d.s][d.a] = true;
  const nextVal = (s2) => {
    if (method === 'constrained') {
      const opts = [0, 1, 2, 3].filter((a) => seen[s2][a]);
      return opts.length ? Math.max(...opts.map((a) => Q[s2][a])) : 0;
    }
    return Math.max(...Q[s2]);
  };
  for (let k = 0; k < iters; k++) {
    // regress Q(s, a) onto r + gamma max Q(s', .) for every transition in the data
    const sum = Array.from({ length: env.N }, () => [0, 0, 0, 0]), cnt = Array.from({ length: env.N }, () => [0, 0, 0, 0]);
    for (const d of D) { sum[d.s][d.a] += d.r + (d.done ? 0 : gamma * nextVal(d.s2)); cnt[d.s][d.a]++; }
    for (let s = 0; s < env.N; s++) for (let a = 0; a < 4; a++) if (cnt[s][a]) Q[s][a] += lr * (sum[s][a] / cnt[s][a] - Q[s][a]);
    if (method === 'cql') {
      // the CQL regulariser's gradient: lower every action in proportion to softmax(Q), raise data actions
      const dataS = new Set(D.map((d) => d.s));
      for (const s of dataS) {
        const m = Math.max(...Q[s]);
        const e = Q[s].map((q) => Math.exp(q - m)), z = e.reduce((x, y) => x + y, 0);
        const nData = cnt[s].reduce((x, y) => x + y, 0);
        for (let a = 0; a < 4; a++) Q[s][a] -= lr * alpha * (e[a] / z - cnt[s][a] / nData);
      }
    }
  }
  // the constrained learner also acts only with actions the data contains, where it has any
  const choose = (s) => {
    if (method !== 'constrained' || !seen[s].some(Boolean)) return argmax(Q[s]);
    return argmax(Q[s].map((q, a) => (seen[s][a] ? q : -Infinity)));
  };
  const pi = Int8Array.from({ length: env.N }, (_, s) => (env.isTerminal(s) || env.isWall(s) ? -1 : choose(s)));
  // what the learner thinks its policy is worth at the start, and what it is really worth
  const predicted = Q[env.start][pi[env.start]];
  const actual = evaluate(env, pi, { gamma }).V[env.start];
  return { Q, seen, pi, predicted, actual };
}

export function offlineSummary(env, D, opts = {}) {
  const best = valueIteration(env).V[env.start];
  return { best, naive: fittedQ(env, D, { ...opts, method: 'naive' }), constrained: fittedQ(env, D, { ...opts, method: 'constrained' }), cql: fittedQ(env, D, { ...opts, method: 'cql' }) };
}

// ---------- L07: stitching ----------
// Nine states in a 3x3 grid, s1 top-left to s9 bottom-right; moves right/down/left/up; r = +1 on
// entering s9, -1 on entering s6 (both end the episode). The log holds three trajectories:
//   tau1: s1 down s4 right s5 right s6   (return -1: a good first move, then a bad finish)
//   tau2: s5 left s4 down s7 right s8 right s9  (return +1: a good finish from s4)
//   tau3: s1 right s2 right s3             (return 0: safe and goes nowhere)
// From s1 the best route is s1 s4 s7 s8 s9, which no single trajectory contains. Methods that
// value actions by the returns they were logged with (BC, filtered BC, AWR with Monte Carlo
// values) prefer tau3's "right" at s1; a TD method like IQL carries s4's good value back to s1.
export const STITCH = {
  W: 3,
  trajs: [
    { name: 'tau1', s: [1, 4, 5, 6], a: ['down', 'right', 'right'] },
    { name: 'tau2', s: [5, 4, 7, 8, 9], a: ['left', 'down', 'right', 'right'] },
    { name: 'tau3', s: [1, 2, 3], a: ['right', 'right'] },
  ],
};
const MOVES = { up: [-1, 0], right: [0, 1], down: [1, 0], left: [0, -1] };
export const stitchNext = (s, a) => {
  const r = Math.floor((s - 1) / 3) + MOVES[a][0], c = ((s - 1) % 3) + MOVES[a][1];
  return r < 0 || r > 2 || c < 0 || c > 2 ? s : r * 3 + c + 1;
};
export const stitchReward = (s2) => (s2 === 9 ? 1 : s2 === 6 ? -1 : 0);
const terminal = (s) => s === 9 || s === 6;

function transitions() {
  const T = [];
  for (const tr of STITCH.trajs) {
    const rewards = tr.a.map((_, t) => stitchReward(tr.s[t + 1]));
    const ret = rewards.reduce((a, b) => a + b, 0);
    tr.a.forEach((a, t) => T.push({ s: tr.s[t], a, r: rewards[t], s2: tr.s[t + 1], done: terminal(tr.s[t + 1]), G: rewards.slice(t).reduce((x, y) => x + y, 0), traj: tr.name, ret }));
  }
  return T;
}

// Each method returns a policy: state -> action (or null where it has no data).
export function stitchPolicies({ alpha = 0.3, tau = 0.9, gamma = 0.95, keep = 0.34 } = {}) {
  const D = transitions();
  const byState = (data, w = () => 1) => {
    const out = {};
    for (const d of data) { out[d.s] ??= {}; out[d.s][d.a] = (out[d.s][d.a] ?? 0) + w(d); }
    return Object.fromEntries(Object.entries(out).map(([s, m]) => [s, Object.entries(m).sort((x, y) => y[1] - x[1])[0][0]]));
  };
  const bc = byState(D);
  // filtered BC: keep the top fraction of trajectories by return
  const rets = [...new Set(D.map((d) => d.traj))].map((n) => ({ n, ret: D.find((d) => d.traj === n).ret })).sort((a, b) => b.ret - a.ret);
  const kept = new Set(rets.slice(0, Math.max(1, Math.round(keep * rets.length))).map((x) => x.n));
  const fbc = byState(D.filter((d) => kept.has(d.traj)));
  // AWR: Monte Carlo V(s) = mean logged return-to-go from s; weight exp(A / alpha)
  const vmc = {};
  for (const d of D) { vmc[d.s] ??= []; vmc[d.s].push(d.G); }
  for (const s in vmc) vmc[s] = vmc[s].reduce((a, b) => a + b, 0) / vmc[s].length;
  const awr = byState(D, (d) => Math.exp((d.G - vmc[d.s]) / alpha));
  // IQL: expectile V over data actions, Q by one-step TD on data, policy weighted by exp((Q - V)/alpha)
  const Q = {}, V = {};
  for (const d of D) { Q[`${d.s}|${d.a}`] = 0; V[d.s] = 0; V[d.s2] ??= 0; }
  for (let it = 0; it < 300; it++) {
    for (const s in V) {
      const qs = D.filter((d) => String(d.s) === s).map((d) => Q[`${d.s}|${d.a}`]);
      if (qs.length) V[s] = expectile(qs, tau);
    }
    for (const d of D) Q[`${d.s}|${d.a}`] = d.r + (d.done ? 0 : gamma * V[d.s2]);
  }
  const iql = byState(D, (d) => Math.exp((Q[`${d.s}|${d.a}`] - V[d.s]) / alpha));
  return { D, bc, fbc, awr, iql, vmc, V, Q, kept: [...kept] };
}

// The tau-expectile of samples: the value v minimising sum |tau - 1(x < v)| (x - v)^2.
// tau = 0.5 is the mean; tau -> 1 approaches the max. (CS224R writes the same thing as a loss on
// V - Q with a small lambda; this is the IQL paper's form, tau = 1 - lambda.)
export function expectile(xs, tau) {
  let v = xs.reduce((a, b) => a + b, 0) / xs.length;
  for (let k = 0; k < 100; k++) {
    let num = 0, den = 0;
    for (const x of xs) { const w = x >= v ? tau : 1 - tau; num += w * x; den += w; }
    v = num / den;
  }
  return v;
}

export function stitchRollout(pi, start = 1, maxSteps = 8) {
  let s = start, G = 0;
  const path = [s];
  for (let t = 0; t < maxSteps && !terminal(s); t++) {
    const a = pi[s];
    if (!a) break;
    const s2 = stitchNext(s, a);
    G += stitchReward(s2);
    s = s2; path.push(s);
  }
  return { path, G, reached: s };
}

// ---------- L07: out-of-distribution actions in a 1D continuous action space ----------
// True reward r(a) = 1 - (a - 0.5)^2; the data only has actions in [-0.2, 0.2]. A flexible fit
// (polynomial of degree `deg`) matches the data and says anything outside it; its argmax over
// [-1, 1] lands where there is no data and the prediction there is far above the truth.
export function oodBandit({ deg = 5, n = 30, lo = -0.2, hi = 0.2, noise = 0.05, seed = 5, ridge = 1e-6 } = {}) {
  const rnd = rng(seed);
  const truth = (a) => 1 - (a - 0.5) ** 2;
  const X = [], y = [];
  for (let i = 0; i < n; i++) { const a = lo + (hi - lo) * rnd(); X.push(a); y.push(truth(a) + noise * rnd.normal()); }
  // least squares on a scaled basis via normal equations
  const sc = (a) => a / Math.max(Math.abs(lo), Math.abs(hi));
  const phi = (a) => Array.from({ length: deg + 1 }, (_, k) => sc(a) ** k);
  const P = deg + 1;
  const A = Array.from({ length: P }, () => new Array(P).fill(0)), b = new Array(P).fill(0);
  X.forEach((a, i) => { const f = phi(a); for (let j = 0; j < P; j++) { b[j] += f[j] * y[i]; for (let k = 0; k < P; k++) A[j][k] += f[j] * f[k]; } });
  for (let j = 0; j < P; j++) A[j][j] += ridge;
  const w = solve(A, b);
  const pred = (a) => phi(a).reduce((s, f, k) => s + f * w[k], 0);
  const grid = Array.from({ length: 201 }, (_, i) => -1 + i / 100);
  const best = grid.reduce((m, a) => (pred(a) > pred(m) ? a : m), grid[0]);
  const inData = grid.filter((a) => a >= lo && a <= hi);
  const bestIn = inData.reduce((m, a) => (pred(a) > pred(m) ? a : m), inData[0]);
  return { X, y, grid, pred: grid.map(pred), truth: grid.map(truth), best, predBest: pred(best), trueBest: truth(best), bestIn, predIn: pred(bestIn), trueIn: truth(bestIn) };
}

function solve(A, b) {
  const n = b.length, M = A.map((r, i) => [...r, b[i]]);
  for (let i = 0; i < n; i++) {
    let p = i;
    for (let k = i + 1; k < n; k++) if (Math.abs(M[k][i]) > Math.abs(M[p][i])) p = k;
    [M[i], M[p]] = [M[p], M[i]];
    for (let k = i + 1; k < n; k++) { const f = M[k][i] / M[i][i]; for (let j = i; j <= n; j++) M[k][j] -= f * M[i][j]; }
  }
  const x = new Array(n).fill(0);
  for (let i = n - 1; i >= 0; i--) { let s = M[i][n]; for (let j = i + 1; j < n; j++) s -= M[i][j] * x[j]; x[i] = s / M[i][i]; }
  return x;
}
