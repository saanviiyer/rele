// Imitation learning on a gridworld. The expert is the optimal policy from value iteration.
// Behaviour cloning (BC) copies the expert's action in every state it saw in the demos; in a state
// the demos never reached it has no data and acts at random. A slip pushes it off the expert's
// path, where errors compound (Ross et al. 2011). DAgger rolls out the learner, asks the expert
// to label the states the learner itself reached, and retrains on everything so far.
import { rng, argmax } from './rng.js';
import { valueIteration } from './dp.js';

export function expertOf(env) {
  return valueIteration(env).policy;
}

// One episode under a policy given as act(s, rnd).
export function episode(env, act, rnd, maxSteps = 100) {
  let s = env.start, G = 0, t = 0;
  const states = [s], actions = [];
  for (; t < maxSteps; t++) {
    const a = act(s, rnd);
    const o = env.stepEnv(s, a, rnd);
    actions.push(a);
    G += o.r;
    s = o.s2;
    states.push(s);
    if (o.done) break;
  }
  const cell = env.cells[s];
  return { states, actions, G, success: cell === 'G', failed: cell === 'L', steps: t + 1 };
}

// counts[s][a]: how often the labelled data says a in s
export function fitBC(env, data) {
  const counts = Array.from({ length: env.N }, () => [0, 0, 0, 0]);
  for (const { s, a } of data) counts[s][a]++;
  return counts;
}
export function bcAct(counts) {
  return (s, rnd) => {
    const c = counts[s];
    return c[0] + c[1] + c[2] + c[3] === 0 ? rnd.int(4) : argmax(c, rnd);
  };
}

export function demos(env, expert, n, rnd) {
  const data = [], eps = [];
  for (let i = 0; i < n; i++) {
    const ep = episode(env, (s) => expert[s], rnd);
    eps.push(ep);
    ep.actions.forEach((a, t) => data.push({ s: ep.states[t], a }));
  }
  return { data, eps };
}

export function evaluatePolicy(env, act, { episodes = 200, seed = 99 } = {}) {
  const rnd = rng(seed);
  let succ = 0, fail = 0, G = 0;
  const visits = new Float64Array(env.N);
  for (let i = 0; i < episodes; i++) {
    const ep = episode(env, act, rnd);
    if (ep.success) succ++;
    if (ep.failed) fail++;
    G += ep.G;
    for (const s of ep.states) visits[s]++;
  }
  return { success: succ / episodes, fail: fail / episodes, ret: G / episodes, visits };
}

// DAgger: iteration 0 is plain BC on the expert demos; each later iteration adds the expert's
// labels on states the current learner visits.
export function dagger(env, { demosN = 5, iters = 6, rolloutsPerIter = 5, seed = 4 } = {}) {
  const rnd = rng(seed);
  const expert = expertOf(env);
  let { data } = demos(env, expert, demosN, rnd);
  const out = [];
  for (let k = 0; k <= iters; k++) {
    const counts = fitBC(env, data);
    const act = bcAct(counts);
    const ev = evaluatePolicy(env, act, { seed: 1000 + k });
    const covered = counts.filter((c) => c[0] + c[1] + c[2] + c[3] > 0).length;
    out.push({ iter: k, ...ev, labels: data.length, covered, counts });
    if (k === iters) break;
    for (let r = 0; r < rolloutsPerIter; r++) {
      const ep = episode(env, act, rnd);
      ep.states.slice(0, -1).forEach((s) => { if (!env.isTerminal(s)) data = data.concat([{ s, a: expert[s] }]); });
    }
  }
  return { expert, iters: out };
}

// BC with the same number of expert labels as DAgger used, for a fair comparison.
export function bcMatched(env, labels, { seed = 4 } = {}) {
  const rnd = rng(seed);
  const expert = expertOf(env);
  let data = [];
  while (data.length < labels) data = data.concat(demos(env, expert, 1, rnd).data);
  const counts = fitBC(env, data.slice(0, labels));
  return { ...evaluatePolicy(env, bcAct(counts), { seed: 77 }), counts };
}

// ---------- L02: what an l2-trained policy does on multimodal data ----------
// A car drives right along x from 0 to 10 at unit speed; its action is lateral speed in [-1, 1].
// An obstacle sits at x in [4.5, 5.5], |y| < 0.6. Half the demonstrators pass above, half below.
// The squared-error policy outputs the mean action, which near the start is 0: straight into the
// obstacle. A categorical policy over {-1, 0, +1} samples one mode and commits to it.
export const CAR = { dx: 0.25, xEnd: 10, obs: { x0: 4.5, x1: 5.5, half: 0.6 } };
const carExpert = (x, y, side) => {
  if (x < 1.5) return 0;
  if (x < 4.5) return Math.abs(y) < 1.1 ? side : 0;
  if (x < 5.5) return 0;
  return Math.abs(y) > 0.05 ? -Math.sign(y) : 0;
};
const bin = (x, y) => `${Math.round(x / CAR.dx)}:${Math.round(y / 0.25)}`;
export function carDemos(n = 20, seed = 1) {
  const rnd = rng(seed);
  const trajs = [];
  for (let i = 0; i < n; i++) {
    const side = i % 2 ? 1 : -1;
    let y = (rnd() - 0.5) * 0.1;
    const pts = [];
    for (let x = 0; x <= CAR.xEnd; x += CAR.dx) {
      const a = carExpert(x, y, side);
      pts.push({ x, y, a });
      y += a * CAR.dx * 0.6;
    }
    trajs.push({ side, pts });
  }
  return trajs;
}
// Fit per state bin: mean action (l2) and action histogram (categorical).
export function carFit(trajs) {
  const table = new Map();
  for (const t of trajs) for (const p of t.pts) {
    const k = bin(p.x, p.y);
    if (!table.has(k)) table.set(k, []);
    table.get(k).push(p.a);
  }
  return table;
}
// the nearest occupied bins in y; equally near bins on both sides are pooled, as a smooth
// function approximator would pool them
const nearestBin = (table, x, y) => {
  const xi = Math.round(x / CAR.dx), yi = Math.round(y / 0.25);
  for (let d = 0; d < 12; d++) {
    const acts = (d ? [d, -d] : [0]).flatMap((dy) => table.get(`${xi}:${yi + dy}`) ?? []);
    if (acts.length) return acts;
  }
  return [0];
};
export function carRollout(table, kind, rnd) {
  let y = (rnd() - 0.5) * 0.1;
  const pts = [];
  let crashed = false;
  for (let x = 0; x <= CAR.xEnd; x += CAR.dx) {
    const acts = nearestBin(table, x, y);
    const a = kind === 'l2' ? acts.reduce((s, v) => s + v, 0) / acts.length : acts[Math.floor(rnd() * acts.length)];
    pts.push({ x, y });
    if (x >= CAR.obs.x0 && x <= CAR.obs.x1 && Math.abs(y) < CAR.obs.half) { crashed = true; break; }
    y += a * CAR.dx * 0.6;
  }
  return { pts, crashed };
}
export function carCompare({ demos = 20, rollouts = 40, seed = 3 } = {}) {
  const table = carFit(carDemos(demos, seed));
  const rnd = rng(seed + 50);
  const run = (kind) => Array.from({ length: rollouts }, () => carRollout(table, kind, rnd));
  const l2 = run('l2'), cat = run('cat');
  const rate = (rs) => rs.filter((r) => r.crashed).length / rs.length;
  return { table, l2, cat, crashL2: rate(l2), crashCat: rate(cat) };
}
