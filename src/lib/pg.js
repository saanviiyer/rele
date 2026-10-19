// Policy gradients on CartPole. The policy pi_theta(a|s) is a small network with a softmax over
// {left, right}. Every variant uses the same estimator,
//   grad J(theta) ~ (1/N) sum_i sum_t grad log pi_theta(a_t|s_t) * Psi_t,
// and differs only in Psi_t (Williams 1992; Schulman et al. 2016):
//   'vanilla'  the whole trajectory's reward R(tau) at every step
//   'rtg'      reward-to-go: rewards from t onward only (causality)
//   'baseline' reward-to-go minus a learned V_phi(s_t)
//   'gae'      generalized advantage estimation with lambda (actor-critic)
// PPO (Schulman et al. 2017) reuses each batch for several epochs with a clipped ratio.
import { rng, softmax, sample } from './rng.js';
import { mlp, forward, backward, zeroGrads, adam } from './nn.js';
import { cpReset, cpStep, cpFeatures, CP } from './cartpole.js';

export const VARIANTS = {
  vanilla: 'R(tau) at every step',
  rtg: 'reward-to-go',
  baseline: 'reward-to-go minus V(s)',
  gae: 'GAE advantage (actor-critic)',
  ppo: 'PPO, clipped',
};

export function rolloutCP(policy, rnd, { maxSteps = CP.maxSteps, greedy = false } = {}) {
  let s = cpReset(rnd);
  const S = [], A = [], R = [], P = [], states = [s];
  for (let t = 0; t < maxSteps; t++) {
    const phi = cpFeatures(s);
    const p = softmax(Array.from(forward(policy, phi).out));
    const a = greedy ? (p[1] > p[0] ? 1 : 0) : sample(p, rnd);
    const { s2, r, done } = cpStep(s, a);
    S.push(phi); A.push(a); R.push(r); P.push(p[a]);
    s = s2; states.push(s);
    if (done) return { S, A, R, P, states, done: true, last: cpFeatures(s) };
  }
  return { S, A, R, P, states, done: false, last: cpFeatures(s) };
}

function discounted(R, gamma) {
  const G = new Array(R.length);
  let g = 0;
  for (let t = R.length - 1; t >= 0; t--) { g = R[t] + gamma * g; G[t] = g; }
  return G;
}

export function makeTrainer({ variant = 'rtg', gamma = 0.99, lambda = 0.95, lr = 0.01, vlr = 0.01, batch = 8, hidden = 16, seed = 1, normalize = false, clip = 0.2, epochs = 4, entropy = 0 } = {}) {
  const rnd = rng(seed);
  const policy = mlp([4, hidden, 2], seed, 0.01);
  const critic = mlp([4, hidden, 1], seed + 99, 0.1);
  if (variant === 'ppo' && lr === 0.01) lr = 0.003;
  const history = [];
  let iter = 0;
  const VS = 100; // the critic predicts returns divided by 100, so its output stays near unit size
  const V = (phi) => VS * forward(critic, phi).out[0];

  function fitCritic(S, targets, passes = 1) {
    for (let k = 0; k < passes; k++) {
      const g = zeroGrads(critic);
      S.forEach((phi, i) => { const f = forward(critic, phi); backward(critic, f, [2 * (f.out[0] - targets[i] / VS)], g); });
      adam(critic, g, { lr: vlr, n: S.length });
    }
  }

  function iterate() {
    const trajs = Array.from({ length: batch }, () => rolloutCP(policy, rnd));
    const S = [], A = [], Psi = [], Pold = [], Ret = [];
    for (const tr of trajs) {
      const G = discounted(tr.R, gamma);
      const total = tr.R.reduce((a, b) => a + b, 0);
      let adv;
      if (variant === 'vanilla') adv = tr.R.map(() => total);
      else if (variant === 'rtg') adv = G;
      else if (variant === 'baseline') adv = G.map((g, t) => g - V(tr.S[t]));
      else {
        // GAE: A_t = sum_l (gamma lambda)^l delta_{t+l}, delta_t = r_t + gamma V(s_{t+1}) - V(s_t)
        const v = tr.S.map(V);
        const vLast = tr.done ? 0 : V(tr.last);
        adv = new Array(tr.R.length);
        let a = 0;
        for (let t = tr.R.length - 1; t >= 0; t--) {
          const next = t === tr.R.length - 1 ? vLast : v[t + 1];
          a = tr.R[t] + gamma * next - v[t] + gamma * lambda * a;
          adv[t] = a;
        }
      }
      tr.S.forEach((phi, t) => { S.push(phi); A.push(tr.A[t]); Psi.push(adv[t]); Pold.push(tr.P[t]); Ret.push(G[t]); });
    }
    let psi = Psi;
    if (normalize || variant === 'ppo' || variant === 'gae') {
      const m = psi.reduce((a, b) => a + b, 0) / psi.length;
      const sd = Math.sqrt(psi.reduce((a, b) => a + (b - m) ** 2, 0) / psi.length) || 1;
      psi = psi.map((x) => (x - m) / sd);
    } else {
      psi = psi.map((x) => x / 100); // same learning rate across variants without centring
    }
    let gradNorm = 0, clipFrac = 0;
    if (variant === 'ppo') {
      // several epochs over the same batch; the ratio pi/pi_old is clipped to [1-eps, 1+eps]
      for (let ep = 0; ep < epochs; ep++) {
        const g = zeroGrads(policy);
        let clipped = 0;
        S.forEach((phi, i) => {
          const f = forward(policy, phi);
          const p = softmax(Array.from(f.out));
          const ratio = p[A[i]] / Pold[i];
          const A_ = psi[i];
          const active = A_ >= 0 ? ratio < 1 + clip : ratio > 1 - clip;
          if (!active) { clipped++; }
          const H = -p.reduce((s, q) => s + q * Math.log(q + 1e-12), 0);
          const gOut = p.map((pk, k) => (active ? -A_ * ratio * ((k === A[i] ? 1 : 0) - pk) : 0) + entropy * pk * (Math.log(pk + 1e-12) + H));
          backward(policy, f, gOut, g);
        });
        gradNorm = adam(policy, g, { lr, n: S.length });
        clipFrac = clipped / S.length;
      }
    } else {
      const g = zeroGrads(policy);
      S.forEach((phi, i) => {
        const f = forward(policy, phi);
        const p = softmax(Array.from(f.out));
        backward(policy, f, p.map((pk, k) => -psi[i] * ((k === A[i] ? 1 : 0) - pk)), g);
      });
      gradNorm = adam(policy, g, { lr, n: S.length });
    }
    if (variant === 'baseline' || variant === 'gae' || variant === 'ppo') fitCritic(S, Ret, variant === 'ppo' ? 2 * epochs : 4);
    iter++;
    const lens = trajs.map((t) => t.R.length);
    const rec = { iter, mean: lens.reduce((a, b) => a + b, 0) / lens.length, min: Math.min(...lens), max: Math.max(...lens), gradNorm, clipFrac, steps: S.length };
    history.push(rec);
    return rec;
  }
  return { policy, critic, history, iterate, rnd, V };
}

// Mean episode length per iteration for one variant over several seeds.
export function pgCurves(variant, { seeds = 3, iters = 60, ...o } = {}) {
  return Array.from({ length: seeds }, (_, k) => {
    const T = makeTrainer({ variant, seed: 1 + k * 13, ...o });
    for (let i = 0; i < iters; i++) T.iterate();
    return T.history.map((h) => h.mean);
  });
}

// The PPO surrogate for one sample as a function of the ratio, for the clipping picture.
export function clippedObjective(ratio, adv, eps) {
  return Math.min(ratio * adv, Math.max(1 - eps, Math.min(1 + eps, ratio)) * adv);
}
