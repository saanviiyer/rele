// Learning what to optimise: goal classifiers (L08), Bradley-Terry reward models (L08, L09),
// KL-regularised RLHF and DPO (L09; the notes, sections 5-6), and GRPO, Dr GRPO and DAPO
// (the notes, sections 7-9). Everything is small enough to solve exactly or train in a second.
import { rng, softmax, sample } from './rng.js';

export const sigmoid = (x) => 1 / (1 + Math.exp(-x));

// ---------- L08: a goal classifier as the reward, and how RL exploits it ----------
// States are points in [-1, 1]^2. The goal is a disk at (0.55, 0.55), radius 0.25. A logistic
// classifier on quadratic features scores "is this a goal state". The agent hill-climbs the
// score from (-0.7, -0.7). With the classifier frozen, the agent finds high-score states the
// classifier never saw. VICE-style training adds the agent's visited states as negatives
// (balanced batches), so the score peak moves onto the real goal (Fu et al. 2018).
// The agent's episode ends when the classifier says it has succeeded (score above 0.5), as when a
// learned success detector is the reward and the termination signal.
export const GOAL = { cx: 0.55, cy: 0.55, r: 0.25, start: [-0.7, -0.7] };
const NF = 6;
const feats = ([x, y]) => [1, x, y, x * x, y * y, x * y];
export const inGoal = ([x, y]) => (x - GOAL.cx) ** 2 + (y - GOAL.cy) ** 2 < GOAL.r ** 2;
export const score = (w, s) => sigmoid(feats(s).reduce((a, f, i) => a + f * w[i], 0));

function fitLogistic(pos, neg, w0, { iters = 1500, lr = 2, l2 = 1e-5 } = {}) {
  const w = w0 ? [...w0] : new Array(NF).fill(0);
  for (let it = 0; it < iters; it++) {
    const g = new Array(NF).fill(0);
    // balanced: positives and negatives get equal total weight
    for (const [set, y] of [[pos, 1], [neg, 0]]) for (const s of set) {
      const f = feats(s), p = sigmoid(f.reduce((a, v, i) => a + v * w[i], 0));
      for (let i = 0; i < NF; i++) g[i] += ((p - y) * f[i]) / (2 * set.length);
    }
    for (let i = 0; i < NF; i++) w[i] -= lr * (g[i] + l2 * w[i]);
  }
  return w;
}

// The agent: RL finds whatever the reward favours, so here it heads in a straight line for the
// state the classifier scores highest, and the episode ends at the first state scoring above 0.5.
function climb(w) {
  let best = GOAL.start, bs = -1;
  for (let i = 0; i <= 40; i++) for (let j = 0; j <= 40; j++) {
    const c = [-1 + i / 20, -1 + j / 20], sc = score(w, c);
    if (sc > bs) { bs = sc; best = c; }
  }
  const path = [GOAL.start];
  const d = Math.hypot(best[0] - GOAL.start[0], best[1] - GOAL.start[1]);
  const n = Math.max(1, Math.ceil(d / 0.04));
  for (let k = 1; k <= n; k++) {
    const s = [GOAL.start[0] + ((best[0] - GOAL.start[0]) * k) / n, GOAL.start[1] + ((best[1] - GOAL.start[1]) * k) / n];
    path.push(s);
    if (score(w, s) > 0.5) break;
  }
  return path;
}

export function goalClassifier({ rounds = 25, nPos = 80, nNeg = 30, seed = 4 } = {}) {
  const rnd = rng(seed);
  const pos = Array.from({ length: nPos }, () => { const a = rnd() * 2 * Math.PI, rr = GOAL.r * Math.sqrt(rnd()); return [GOAL.cx + rr * Math.cos(a), GOAL.cy + rr * Math.sin(a)]; });
  const neg0 = Array.from({ length: nNeg }, () => [GOAL.start[0] + (rnd() - 0.5) * 0.6, GOAL.start[1] + (rnd() - 0.5) * 0.6]);
  const w0 = fitLogistic(pos, neg0);
  const frozen = climb(w0);
  let neg = [...neg0], w = w0;
  const vice = [];
  for (let k = 0; k < rounds; k++) {
    const path = climb(w);
    vice.push({ w: [...w], path, end: path.at(-1), success: inGoal(path.at(-1)), score: score(w, path.at(-1)) });
    neg = neg.concat(path);
    w = fitLogistic(pos, neg, w);
  }
  return { pos, neg0, w0, frozen, frozenEnd: frozen.at(-1), frozenScore: score(w0, frozen.at(-1)), frozenSuccess: inGoal(frozen.at(-1)), vice };
}

// ---------- Bradley-Terry: recover a hidden reward from pairwise preferences ----------
// Each item has features x; the hidden reward is w* . x. A person prefers a over b with
// probability sigmoid(r(a) - r(b)). Fit w by maximising sum log sigmoid(r(winner) - r(loser)).
export function bradleyTerry({ dim = 3, items = 40, pairs = 300, wTrue = [1.5, -1, 0.5], seed = 9, iters = 500, lr = 0.5 } = {}) {
  const rnd = rng(seed);
  const X = Array.from({ length: items }, () => Array.from({ length: dim }, () => rnd.normal()));
  const r = (w, x) => x.reduce((a, v, i) => a + v * w[i], 0);
  const P = [];
  for (let k = 0; k < pairs; k++) {
    const i = rnd.int(items), j = rnd.int(items);
    if (i === j) continue;
    const pij = sigmoid(r(wTrue, X[i]) - r(wTrue, X[j]));
    P.push(rnd() < pij ? [i, j] : [j, i]);
  }
  const w = new Array(dim).fill(0);
  const losses = [];
  for (let it = 0; it < iters; it++) {
    const g = new Array(dim).fill(0);
    let L = 0;
    for (const [a, b] of P) {
      const d = r(w, X[a]) - r(w, X[b]);
      L -= Math.log(sigmoid(d) + 1e-12);
      const c = 1 - sigmoid(d);
      for (let i = 0; i < dim; i++) g[i] -= c * (X[a][i] - X[b][i]);
    }
    for (let i = 0; i < dim; i++) w[i] -= (lr * g[i]) / P.length;
    losses.push(L / P.length);
  }
  const dot = w.reduce((a, v, i) => a + v * wTrue[i], 0);
  const cos = dot / Math.hypot(...w) / Math.hypot(...wTrue);
  return { w, wTrue, cos, losses, pairs: P.length };
}

// ---------- L09 and the notes: a tiny "language model" ----------
// One prompt, N candidate responses. Each has a true quality q (what we want) and a length.
// Annotators judge by quality but also like length: they prefer a over b with probability
// sigmoid(k (q_a - q_b) + bias (len_a - len_b)). pi_ref is the pretrained/SFT model.
export function makeLM({ N = 30, seed = 4, lenBias = 1.2 } = {}) {
  const rnd = rng(seed);
  const resp = Array.from({ length: N }, (_, i) => {
    const len = 0.2 + 0.8 * rnd();
    // longer answers are not better on average; a few long, poor ones exist
    const q = Math.max(0, Math.min(1, 0.55 + 0.3 * rnd.normal() - 0.25 * Math.max(0, len - 0.7) * 3 * rnd()));
    return { id: i, q, len };
  });
  const logits = resp.map((r) => 1.2 * r.q - 1.5 * (r.len - 0.45) ** 2 + 0.5 * rnd.normal());
  const ref = softmax(logits);
  return { resp, ref, lenBias, k: 6 };
}

export function preferencePairs(lm, { n = 400, seed = 7 } = {}) {
  const rnd = rng(seed);
  const pairs = [];
  for (let t = 0; t < n; t++) {
    // pairs are sampled from pi_ref, as in practice
    const a = sample(lm.ref, rnd), b = sample(lm.ref, rnd);
    if (a === b) continue;
    const p = sigmoid(lm.k * (lm.resp[a].q - lm.resp[b].q) + lm.lenBias * 6 * (lm.resp[a].len - lm.resp[b].len));
    pairs.push(rnd() < p ? { w: a, l: b } : { w: b, l: a });
  }
  return pairs;
}

// A reward model r_phi on features [q, len] fitted to the pairs (Bradley-Terry).
export function fitRewardModel(lm, pairs, { iters = 600, lr = 1 } = {}) {
  const f = (i) => [lm.resp[i].q, lm.resp[i].len];
  const w = [0, 0];
  for (let it = 0; it < iters; it++) {
    const g = [0, 0];
    for (const { w: a, l: b } of pairs) {
      const fa = f(a), fb = f(b);
      const c = 1 - sigmoid(w[0] * (fa[0] - fb[0]) + w[1] * (fa[1] - fb[1]));
      g[0] -= c * (fa[0] - fb[0]); g[1] -= c * (fa[1] - fb[1]);
    }
    w[0] -= (lr * g[0]) / pairs.length; w[1] -= (lr * g[1]) / pairs.length;
  }
  return { w, r: lm.resp.map((_, i) => w[0] * f(i)[0] + w[1] * f(i)[1]) };
}

// The KL-regularised optimum: pi*(y) = pi_ref(y) exp(r(y)/beta) / Z.
export function klOptimal(ref, r, beta) {
  const z = ref.map((p, i) => Math.log(p) + r[i] / beta);
  return softmax(z);
}
export const kl = (p, q) => p.reduce((a, pi, i) => a + (pi > 0 ? pi * Math.log(pi / q[i]) : 0), 0);
export const expect = (p, v) => p.reduce((a, pi, i) => a + pi * v[i], 0);

// Sweep beta: proxy (reward model) and true (quality) reward of pi*, against KL to pi_ref.
export function overoptimisation(lm, rm, betas = Array.from({ length: 40 }, (_, i) => 10 ** (1 - i / 12))) {
  const q = lm.resp.map((r) => r.q), len = lm.resp.map((r) => r.len);
  return betas.map((beta) => {
    const p = klOptimal(lm.ref, rm.r, beta);
    return { beta, kl: kl(p, lm.ref), proxy: expect(p, rm.r), truth: expect(p, q), len: expect(p, len) };
  });
}

// DPO (Rafailov et al. 2023): fit the policy's logits directly to the pairs with
// -log sigmoid(beta log(pi(yw)/pi_ref(yw)) - beta log(pi(yl)/pi_ref(yl))).
export function dpo(lm, pairs, { beta = 0.5, iters = 400, lr = 0.5 } = {}) {
  const N = lm.ref.length;
  const logits = lm.ref.map((p) => Math.log(p)); // start at pi_ref
  const lref = [...logits];
  const hist = [];
  for (let it = 0; it <= iters; it++) {
    const lp = softmax(logits).map((p) => Math.log(p));
    const g = new Array(N).fill(0);
    let L = 0;
    for (const { w, l } of pairs) {
      const h = beta * (lp[w] - lref[w]) - beta * (lp[l] - lref[l]);
      L -= Math.log(sigmoid(h) + 1e-12);
      // d/dlogits of log pi(w) - log pi(l) is onehot(w) - onehot(l) (the softmax terms cancel)
      const c = -(1 - sigmoid(h)) * beta;
      g[w] += c; g[l] -= c;
    }
    if (it % 20 === 0) hist.push({ it, loss: L / pairs.length });
    if (it === iters) break;
    for (let i = 0; i < N; i++) logits[i] -= (lr * g[i]) / pairs.length;
  }
  const pi = softmax(logits);
  return { pi, hist, implicit: pi.map((p, i) => beta * Math.log(p / lm.ref[i])) };
}

// ---------- GRPO, Dr GRPO, DAPO ----------
// Group-relative advantages: A_i = (r_i - mean) / std over G samples of one prompt
// (population std, as in the notes' code). Dr GRPO drops the division by std.
export function groupAdvantages(rewards, { std = true } = {}) {
  const m = rewards.reduce((a, b) => a + b, 0) / rewards.length;
  const sd = Math.sqrt(rewards.reduce((a, b) => a + (b - m) ** 2, 0) / rewards.length);
  return rewards.map((r) => (std ? (sd > 0 ? (r - m) / sd : 0) : r - m));
}
// The k3 estimator of KL(pi || pi_ref) from one sample: ratio - log ratio - 1, with ratio = pi_ref/pi.
export const k3 = (ratio) => ratio - Math.log(ratio) - 1;

// Expected total |advantage| a prompt contributes per group as a function of its accuracy p,
// with and without the std division: the question-difficulty bias of Dr GRPO.
export function difficultyWeight(p, G = 8, std = true) {
  // exact expectation over the number of correct answers k ~ Binomial(G, p)
  let tot = 0;
  const C = (n, k) => { let c = 1; for (let i = 1; i <= k; i++) c = (c * (n - k + i)) / i; return c; };
  for (let k = 0; k <= G; k++) {
    const pk = C(G, k) * p ** k * (1 - p) ** (G - k);
    const r = Array.from({ length: G }, (_, i) => (i < k ? 1 : 0));
    tot += pk * groupAdvantages(r, { std }).reduce((a, b) => a + Math.abs(b), 0);
  }
  return tot;
}
// Share of groups whose rewards are all equal (all right or all wrong): zero advantage, no
// gradient. DAPO's dynamic sampling drops and refills them.
export const zeroGroupShare = (p, G = 8) => p ** G + (1 - p) ** G;

// The length bias. Each prompt's policy chooses among four kinds of answer: right or wrong,
// short (10 tokens) or long (40 tokens). GRPO weights a sample's gradient by 1/|o_i|; Dr GRPO by
// a constant. Under 1/|o_i|, long wrong answers are penalised less than short wrong ones, so
// wrong answers drift longer (Liu et al. 2025).
export const KINDS = [
  { label: 'right, short', ok: 1, len: 10 },
  { label: 'right, long', ok: 1, len: 40 },
  { label: 'wrong, short', ok: 0, len: 10 },
  { label: 'wrong, long', ok: 0, len: 40 },
];
export function lengthBiasRun({ norm = 'grpo', prompts = 24, G = 8, iters = 150, lr = 0.5, seed = 12 } = {}) {
  const rnd = rng(seed);
  // prompts differ in difficulty: the initial chance of a right answer
  const P = Array.from({ length: prompts }, (_, j) => {
    const pr = 0.15 + 0.7 * (j / (prompts - 1));
    return [Math.log(pr / 2), Math.log(pr / 2), Math.log((1 - pr) / 2), Math.log((1 - pr) / 2)];
  });
  const hist = [];
  const measure = (it) => {
    let acc = 0, lenWrong = 0, wWrong = 0, lenAll = 0;
    for (const L of P) {
      const p = softmax(L);
      acc += (p[0] + p[1]) / prompts;
      lenAll += KINDS.reduce((a, k, i) => a + p[i] * k.len, 0) / prompts;
      lenWrong += (p[2] * 10 + p[3] * 40); wWrong += p[2] + p[3];
    }
    hist.push({ it, acc, lenAll, lenWrong: wWrong ? lenWrong / wWrong : 0 });
  };
  measure(0);
  for (let it = 1; it <= iters; it++) {
    for (const L of P) {
      const p = softmax(L);
      const ks = Array.from({ length: G }, () => sample(p, rnd));
      const A = groupAdvantages(ks.map((k) => KINDS[k].ok), { std: norm === 'grpo' });
      const g = [0, 0, 0, 0];
      ks.forEach((k, i) => {
        const w = A[i] * (norm === 'grpo' ? 1 / KINDS[k].len : 1 / 40);
        for (let m = 0; m < 4; m++) g[m] += w * ((m === k ? 1 : 0) - p[m]);
      });
      for (let m = 0; m < 4; m++) L[m] += (lr * 10 * g[m]) / G;
    }
    measure(it);
  }
  return hist;
}
