// Learning values from experience alone. Monte Carlo waits for the whole return G_t; temporal
// difference (TD) learning updates from one step and its own next estimate. The random walk is
// Sutton & Barto's Example 6.2: five states A..E, start in C, +1 for leaving on the right.
import { rng } from './rng.js';

export const WALK_TRUE = [1, 2, 3, 4, 5].map((i) => i / 6);

// One episode of the walk: a list of states 1..5 visited, and the final reward.
export function walkEpisode(rnd) {
  let s = 3;
  const states = [s];
  for (;;) {
    s += rnd() < 0.5 ? -1 : 1;
    if (s === 0) return { states, reward: 0 };
    if (s === 6) return { states, reward: 1 };
    states.push(s);
  }
}

// TD(0): V(s) <- V(s) + alpha [r + gamma V(s') - V(s)]
export function tdUpdate(V, ep, alpha) {
  const { states, reward } = ep;
  for (let t = 0; t < states.length; t++) {
    const s = states[t], last = t === states.length - 1;
    const target = last ? reward : V[states[t + 1]];
    V[s] += alpha * (target - V[s]);
  }
}

// Constant-alpha every-visit Monte Carlo: V(s) <- V(s) + alpha [G_t - V(s)]
export function mcUpdate(V, ep, alpha) {
  for (const s of ep.states) V[s] += alpha * (ep.reward - V[s]);
}

export const rms = (V) => Math.sqrt(WALK_TRUE.reduce((a, v, i) => a + (V[i + 1] - v) ** 2, 0) / 5);

// RMS error after each episode, averaged over runs (Sutton & Barto Figure 6.2, right).
export function walkCurves({ method = 'td', alpha = 0.1, runs = 100, episodes = 100, seed = 3 }) {
  const err = new Float64Array(episodes + 1);
  for (let run = 0; run < runs; run++) {
    const rnd = rng(seed * 1009 + run);
    const V = [0, 0.5, 0.5, 0.5, 0.5, 0.5, 0];
    err[0] += rms(V) / runs;
    for (let e = 1; e <= episodes; e++) {
      const ep = walkEpisode(rnd);
      (method === 'td' ? tdUpdate : mcUpdate)(V, ep, alpha);
      err[e] += rms(V) / runs;
    }
  }
  return Array.from(err);
}

// The spread of the two targets for one state: MC's target is the return (0 or 1, high
// variance, unbiased); TD's is the current estimate of the next state (low variance, biased
// while that estimate is wrong).
export function targetSpread(V, s, runs = 2000, seed = 5) {
  const rnd = rng(seed);
  const mc = [], td = [];
  for (let i = 0; i < runs; i++) {
    // start the walk at s
    let x = s, first = true, next = null, reward = 0;
    for (;;) {
      x += rnd() < 0.5 ? -1 : 1;
      if (first) { next = x; first = false; }
      if (x === 0) { reward = 0; break; }
      if (x === 6) { reward = 1; break; }
    }
    mc.push(reward);
    td.push(next === 6 ? 1 : next === 0 ? 0 : V[next]);
  }
  const stats = (a) => { const m = a.reduce((p, q) => p + q, 0) / a.length; return { mean: m, sd: Math.sqrt(a.reduce((p, q) => p + (q - m) ** 2, 0) / a.length) }; };
  return { mc: stats(mc), td: stats(td), truth: WALK_TRUE[s - 1] };
}
