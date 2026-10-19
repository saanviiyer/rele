// Multi-armed bandits: the one-state version of reinforcement learning, where the only problem is
// whether to try something new or use what already seems best (Sutton & Barto 2018, ch. 2).
import { rng, argmax, softmax, sample } from './rng.js';

// The 10-armed testbed: each arm's true mean is drawn from N(0, 1), each pull pays N(mean, 1).
export function gaussianBandit(k = 10, seed = 1) {
  const r = rng(seed);
  const q = Array.from({ length: k }, () => r.normal());
  return { k, q, best: argmax(q), pull: (a, rnd) => q[a] + rnd.normal(), kind: 'gaussian' };
}

// Slot machines that pay 1 with a hidden probability.
export function bernoulliBandit(p) {
  return { k: p.length, q: p, best: argmax(p), pull: (a, rnd) => (rnd() < p[a] ? 1 : 0), kind: 'bernoulli' };
}

// Agents share one interface: act(t) picks an arm, learn(a, r) updates.
export const AGENTS = {
  greedy: { label: 'greedy', make: (k, o, rnd) => epsGreedy(k, { ...o, eps: 0 }, rnd) },
  eps: { label: 'epsilon-greedy', make: (k, o, rnd) => epsGreedy(k, o, rnd) },
  optimistic: { label: 'optimistic greedy', make: (k, o, rnd) => epsGreedy(k, { ...o, eps: 0, q0: o.q0 ?? 5, alpha: 0.1 }, rnd) },
  ucb: { label: 'UCB', make: (k, o, rnd) => ucb(k, o, rnd) },
  thompson: { label: 'Thompson sampling', make: (k, o, rnd) => thompson(k, o, rnd) },
  gradient: { label: 'gradient bandit', make: (k, o, rnd) => gradient(k, o, rnd) },
};

export function epsGreedy(k, { eps = 0.1, q0 = 0, alpha = null } = {}, rnd) {
  const Q = new Array(k).fill(q0), N = new Array(k).fill(0);
  return {
    Q, N,
    act: () => (rnd() < eps ? rnd.int(k) : argmax(Q, rnd)),
    learn(a, r) { N[a]++; Q[a] += (alpha ?? 1 / N[a]) * (r - Q[a]); },
    // what to draw as an uncertainty bar: nothing for this agent
    spread: () => null,
  };
}

// Upper confidence bound (Auer et al. 2002): pick the arm whose optimistic estimate is highest.
export function ucb(k, { c = 2 } = {}, rnd) {
  const Q = new Array(k).fill(0), N = new Array(k).fill(0);
  let t = 0;
  const bonus = (a) => (N[a] ? c * Math.sqrt(Math.log(t + 1) / N[a]) : Infinity);
  return {
    Q, N,
    act: () => argmax(Q.map((q, a) => q + bonus(a)), rnd),
    learn(a, r) { t++; N[a]++; Q[a] += (r - Q[a]) / N[a]; },
    spread: () => Q.map((q, a) => (N[a] ? [q - bonus(a), q + bonus(a)] : null)),
  };
}

// Thompson sampling (Thompson 1933): keep a belief about each arm, draw one guess per arm from
// it, and play the arm whose guess is highest. Beta beliefs for 0/1 payouts, Gaussian otherwise.
export function thompson(k, { kind = 'gaussian' } = {}, rnd) {
  const N = new Array(k).fill(0), S = new Array(k).fill(0);
  const Q = new Array(k).fill(0);
  return {
    Q, N,
    act() {
      const draws = Q.map((_, a) => (kind === 'bernoulli'
        ? rnd.beta(1 + S[a], 1 + N[a] - S[a])
        : S[a] / (N[a] + 1) + rnd.normal() / Math.sqrt(N[a] + 1)));
      return argmax(draws, rnd);
    },
    learn(a, r) {
      N[a]++; S[a] += r;
      Q[a] = kind === 'bernoulli' ? (1 + S[a]) / (2 + N[a]) : S[a] / (N[a] + 1);
    },
    spread: () => Q.map((q, a) => {
      if (kind === 'bernoulli') { const al = 1 + S[a], be = 1 + N[a] - S[a]; const sd = Math.sqrt((al * be) / ((al + be) ** 2 * (al + be + 1))); return [q - 2 * sd, q + 2 * sd]; }
      const sd = 1 / Math.sqrt(N[a] + 1);
      return [q - 2 * sd, q + 2 * sd];
    }),
  };
}

// Gradient bandit: learn a preference per arm and act by softmax. It is the simplest policy
// gradient method (Sutton & Barto 2018, section 2.8).
export function gradient(k, { alpha = 0.1, baseline = true } = {}, rnd) {
  const H = new Array(k).fill(0), N = new Array(k).fill(0);
  let rbar = 0, t = 0;
  const pi = () => softmax(H);
  return {
    Q: H, N, H,
    act: () => sample(pi(), rnd),
    learn(a, r) {
      t++; N[a]++;
      const p = pi(), b = baseline ? rbar : 0;
      for (let i = 0; i < k; i++) H[i] += alpha * (r - b) * ((i === a ? 1 : 0) - p[i]);
      rbar += (r - rbar) / t;
    },
    policy: pi,
    spread: () => null,
  };
}

// Average many independent runs: mean reward per step, share of steps on the best arm, and
// cumulative regret (what the best arm would have paid in expectation, minus what was expected).
export function runBandits({ agent = 'eps', opts = {}, runs = 200, steps = 1000, seed = 7, env = 'gaussian', k = 10 }) {
  const reward = new Float64Array(steps), optimal = new Float64Array(steps), regret = new Float64Array(steps);
  for (let run = 0; run < runs; run++) {
    const b = env === 'gaussian' ? gaussianBandit(k, seed * 1000 + run) : bernoulliBandit(Array.from({ length: k }, (_, i) => rng(seed * 1000 + run + i * 17)()));
    const rnd = rng(seed * 7919 + run);
    const ag = AGENTS[agent].make(b.k, { ...opts, kind: b.kind }, rnd);
    let reg = 0;
    for (let t = 0; t < steps; t++) {
      const a = ag.act(t);
      const r = b.pull(a, rnd);
      ag.learn(a, r);
      reward[t] += r / runs;
      optimal[t] += (a === b.best ? 1 : 0) / runs;
      reg += b.q[b.best] - b.q[a];
      regret[t] += reg / runs;
    }
  }
  return { reward: Array.from(reward), optimal: Array.from(optimal), regret: Array.from(regret) };
}
