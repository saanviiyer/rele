// Dynamic programming: when the rules of the world are known, the best behaviour can be computed
// by repeatedly applying the Bellman equation (Bellman 1957; Sutton & Barto 2018, ch. 4).
import { argmax } from './rng.js';

// Expected value of action a in state s, given state values V: the Bellman backup.
export function qFromV(env, V, s, a, gamma = env.gamma) {
  let q = 0;
  for (const { p, s2, r, done } of env.trans(s, a)) q += p * (r + (done ? 0 : gamma * V[s2]));
  return q;
}

export function backupTerms(env, V, s, a, gamma = env.gamma) {
  return env.trans(s, a).map(({ p, s2, r, done }) => ({ p, s2, r, v: done ? 0 : V[s2], done, total: p * (r + (done ? 0 : gamma * V[s2])) }));
}

const live = (env) => env.states.filter((s) => !env.isTerminal(s));

// One synchronous sweep of value iteration: every state takes its best backup from the old values.
export function viSweep(env, V, gamma = env.gamma) {
  const V2 = Float64Array.from(V);
  let delta = 0;
  for (const s of live(env)) {
    let best = -Infinity;
    for (let a = 0; a < 4; a++) best = Math.max(best, qFromV(env, V, s, a, gamma));
    V2[s] = best;
    delta = Math.max(delta, Math.abs(best - V[s]));
  }
  return { V: V2, delta };
}

// Run value iteration to convergence, keeping every sweep so the page can replay it.
export function valueIteration(env, { gamma = env.gamma, theta = 1e-6, maxSweeps = 1000 } = {}) {
  let V = new Float64Array(env.N);
  const history = [{ V, delta: Infinity }];
  for (let k = 0; k < maxSweeps; k++) {
    const r = viSweep(env, V, gamma);
    V = r.V;
    history.push(r);
    if (r.delta < theta) break;
  }
  return { V, history, policy: greedy(env, V, gamma) };
}

export function greedy(env, V, gamma = env.gamma) {
  const pi = new Int8Array(env.N).fill(-1);
  for (const s of live(env)) pi[s] = argmax([0, 1, 2, 3].map((a) => qFromV(env, V, s, a, gamma)));
  return pi;
}

// Iterative policy evaluation of a deterministic policy (pi[s] = action).
export function evaluate(env, pi, { gamma = env.gamma, theta = 1e-8, maxSweeps = 10000 } = {}) {
  let V = new Float64Array(env.N), sweeps = 0;
  for (; sweeps < maxSweeps; sweeps++) {
    const V2 = Float64Array.from(V);
    let delta = 0;
    for (const s of live(env)) { V2[s] = qFromV(env, V, s, pi[s], gamma); delta = Math.max(delta, Math.abs(V2[s] - V[s])); }
    V = V2;
    if (delta < theta) break;
  }
  return { V, sweeps };
}

// Policy iteration (Howard 1960): evaluate, improve, repeat until the policy stops changing.
export function policyIteration(env, { gamma = env.gamma, maxIters = 100 } = {}) {
  let pi = new Int8Array(env.N).fill(-1);
  for (const s of live(env)) pi[s] = 0;
  const steps = [];
  for (let it = 0; it < maxIters; it++) {
    const { V, sweeps } = evaluate(env, pi, { gamma, maxSweeps: gamma >= 1 ? 2000 : 10000 });
    const pi2 = greedy(env, V, gamma);
    // keep the old action on ties so the loop ends
    for (const s of live(env)) if (Math.abs(qFromV(env, V, s, pi[s], gamma) - qFromV(env, V, s, pi2[s], gamma)) < 1e-9) pi2[s] = pi[s];
    const changed = live(env).filter((s) => pi2[s] !== pi[s]).length;
    steps.push({ V, pi: Int8Array.from(pi), sweeps, changed });
    pi = pi2;
    if (!changed) break;
  }
  return { pi, steps };
}

// Follow a policy from the start, without slipping, to draw its path.
export function rollout(env, pi, maxSteps = 200) {
  const path = [env.start];
  let s = env.start;
  const seen = new Map();
  for (let t = 0; t < maxSteps && !env.isTerminal(s); t++) {
    const tr = env.trans(s, pi[s]);
    if (!tr.length) break;
    const o = tr.reduce((a, b) => (b.p > a.p ? b : a));
    s = o.s2;
    path.push(s);
    seen.set(s, (seen.get(s) ?? 0) + 1);
    if (seen.get(s) > 3) return { path, loops: true };
  }
  return { path, loops: false };
}
