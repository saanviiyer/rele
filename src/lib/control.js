// Tabular control from experience: Q-learning (off-policy, Watkins 1989) and SARSA (on-policy,
// Rummery & Niranjan 1994) on any gridworld, with an epsilon-greedy behaviour policy.
// The learner is a stepper so the page can animate one move at a time.
import { rng, argmax } from './rng.js';

export function makeQ(env, q0 = 0) {
  return Array.from({ length: env.N }, () => [q0, q0, q0, q0]);
}

export function epsGreedyAct(Q, s, eps, rnd) {
  return rnd() < eps ? rnd.int(4) : argmax(Q[s], rnd);
}

// method: 'q' (Q-learning), 'sarsa', 'expected' (expected SARSA)
export function learner(env, { method = 'q', alpha = 0.5, eps = 0.1, gamma = env.gamma, seed = 1, maxSteps = 500, q0 = 0 } = {}) {
  const rnd = rng(seed);
  const Q = makeQ(env, q0);
  let s = env.start, a = epsGreedyAct(Q, s, eps, rnd), t = 0, G = 0, episode = 0;
  const returns = [], lengths = [];
  let path = [s];
  const lastPaths = [];
  const target = (s2, a2) => {
    if (method === 'q') return Math.max(...Q[s2]);
    if (method === 'sarsa') return Q[s2][a2];
    const best = argmax(Q[s2]);
    return Q[s2].reduce((acc, q, b) => acc + q * ((b === best ? 1 - eps : 0) + eps / 4), 0);
  };
  const api = {
    Q, returns, lengths, lastPaths,
    get state() { return s; }, get episode() { return episode; }, get path() { return path; },
    set eps(v) { eps = v; }, set alpha(v) { alpha = v; },
    // one environment step; returns what happened, including the TD error
    step() {
      const o = env.stepEnv(s, a, rnd);
      const a2 = epsGreedyAct(Q, o.s2, eps, rnd);
      const y = o.r + (o.done ? 0 : gamma * target(o.s2, a2));
      const delta = y - Q[s][a];
      Q[s][a] += alpha * delta;
      const ev = { s, a, r: o.r, s2: o.s2, done: o.done, delta, slipped: o.dir !== a };
      G += o.r; t++;
      if (o.done || t >= maxSteps) {
        returns.push(G); lengths.push(t); episode++;
        lastPaths.push(path); if (lastPaths.length > 5) lastPaths.shift();
        s = env.start; a = epsGreedyAct(Q, s, eps, rnd); t = 0; G = 0; path = [s];
        ev.end = true;
      } else {
        s = o.s2; a = a2; path.push(s);
      }
      return ev;
    },
    runEpisodes(n) { const stop = episode + n; let guard = 0; while (episode < stop && guard++ < n * maxSteps + 10) api.step(); },
    greedy: () => Q.map((q, st) => (env.isTerminal(st) || env.isWall(st) ? -1 : argmax(q))),
  };
  return api;
}

// Sum of rewards per episode, averaged over runs: Sutton & Barto Figure 6.4 on the cliff.
export function controlCurves(env, { method = 'q', alpha = 0.5, eps = 0.1, runs = 50, episodes = 500, seed = 11 } = {}) {
  const out = new Float64Array(episodes);
  for (let run = 0; run < runs; run++) {
    const L = learner(env, { method, alpha, eps, seed: seed * 101 + run });
    L.runEpisodes(episodes);
    for (let e = 0; e < episodes; e++) out[e] += L.returns[e] / runs;
  }
  return Array.from(out);
}

// Follow the greedy policy from the start with no exploration and no slip.
export function greedyPath(env, Q, maxSteps = 100) {
  let s = env.start;
  const path = [s];
  for (let t = 0; t < maxSteps && !env.isTerminal(s); t++) {
    const a = argmax(Q[s]);
    const tr = env.trans(s, a);
    if (!tr.length) break;
    s = tr.reduce((x, y) => (y.p > x.p ? y : x)).s2;
    path.push(s);
    if (path.length > 3 && path.slice(0, -1).filter((v) => v === s).length > 2) break;
  }
  return path;
}

// Maximisation bias (van Hasselt 2010): with noisy estimates of equal true values, the max of the
// estimates is biased upward; double Q-learning picks with one estimate and scores with another.
export function maxBias({ actions = 10, samples = 5, trials = 4000, seed = 2 } = {}) {
  const rnd = rng(seed);
  let single = 0, double = 0;
  for (let i = 0; i < trials; i++) {
    const A = [], B = [];
    for (let k = 0; k < actions; k++) {
      let a = 0, b = 0;
      for (let j = 0; j < samples; j++) { a += rnd.normal(); b += rnd.normal(); }
      A.push(a / samples); B.push(b / samples);
    }
    single += Math.max(...A) / trials;
    double += B[argmax(A)] / trials;
  }
  return { single, double, truth: 0 };
}

// Sutton & Barto Example 6.7. From A, "right" ends with 0; "left" goes to B with 0, and from B
// each of `nB` actions ends with a reward drawn from N(-0.1, 1). Left is worse in expectation,
// but Q-learning's max over noisy B-estimates makes it look better at first. Double Q-learning
// keeps two tables, picks with one and scores with the other (van Hasselt 2010).
export function maxBiasMDP({ method = 'q', episodes = 300, runs = 1000, alpha = 0.1, eps = 0.1, nB = 10, seed = 6 } = {}) {
  const left = new Float64Array(episodes);
  for (let run = 0; run < runs; run++) {
    const rnd = rng(seed * 7 + run);
    const QA = [[0, 0], [0, 0]], QB = [new Array(nB).fill(0), new Array(nB).fill(0)];
    for (let e = 0; e < episodes; e++) {
      const qa = method === 'double' ? QA[0].map((v, i) => v + QA[1][i]) : QA[0];
      const a = rnd() < eps ? rnd.int(2) : argmax(qa, rnd); // 0 left, 1 right
      if (a === 0) left[e] += 1 / runs;
      const k = method === 'double' && rnd() < 0.5 ? 1 : 0; // which table to update
      if (a === 1) { QA[k][1] += alpha * (0 - QA[k][1]); continue; }
      // in B
      const qb = method === 'double' ? QB[0].map((v, i) => v + QB[1][i]) : QB[0];
      const b = rnd() < eps ? rnd.int(nB) : argmax(qb, rnd);
      const target = method === 'double' ? QB[1 - k][argmax(QB[k], rnd)] : Math.max(...QB[0]);
      QA[k][0] += alpha * (0 + target - QA[k][0]);
      const r = -0.1 + rnd.normal();
      const kb = method === 'double' && rnd() < 0.5 ? 1 : 0;
      QB[kb][b] += alpha * (r - QB[kb][b]);
    }
  }
  return Array.from(left);
}
