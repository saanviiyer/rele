// Every number ReLe states is checked here against a textbook value or an exact identity.
// Run: npm test
import { gridOf } from '../src/lib/grid.js';
import { valueIteration, policyIteration } from '../src/lib/dp.js';
import { controlCurves, learner, greedyPath, maxBias, maxBiasMDP } from '../src/lib/control.js';
import { runBandits } from '../src/lib/bandit.js';
import { walkCurves, WALK_TRUE } from '../src/lib/td.js';
import { pgCurves, clippedObjective } from '../src/lib/pg.js';
import { makeDQN } from '../src/lib/dqn.js';
import { dagger, carCompare, evaluatePolicy } from '../src/lib/imitation.js';
import { stitchPolicies, stitchRollout, oodBandit, expectile, collect, fittedQ } from '../src/lib/offline.js';
import * as P from '../src/lib/prefs.js';
import { enumerateTrajectories, pgWeights, HUMANOID, gae, nStepTargets, importanceSpread, isEstimate } from '../src/lib/toys.js';
import { makeGrid } from '../src/lib/grid.js';

let fail = 0;
const ok = (c, m) => { console.log(`${c ? 'ok  ' : 'FAIL'} ${m}`); if (!c) fail++; };
const near = (a, b, tol) => Math.abs(a - b) <= tol;
const avg = (a) => a.reduce((x, y) => x + y, 0) / a.length;

// L01: trajectory probabilities
{
  const e = enumerateTrajectories();
  ok(near(e.total, 1, 1e-12), `p_theta(tau) sums to 1 over all ${e.trajectories.length} trajectories`);
}
// DP: Russell & Norvig 4x3 utilities (U = V + R(s) with R(s) = -0.04)
{
  const env = gridOf('classic');
  const vi = valueIteration(env);
  const U = (r, c) => vi.V[r * 4 + c] - 0.04;
  const ref = [[0, 0, 0.812], [0, 1, 0.868], [0, 2, 0.918], [1, 0, 0.762], [1, 2, 0.66], [2, 0, 0.705], [2, 1, 0.655], [2, 2, 0.611], [2, 3, 0.388]];
  ok(ref.every(([r, c, u]) => near(U(r, c), u, 0.0015)), 'value iteration reproduces the Russell & Norvig 4x3 utilities to 3 decimals');
  const pi = policyIteration(env);
  ok(pi.steps.length < 10, `policy iteration converges in ${pi.steps.length} improvement steps`);
  const e8 = gridOf('eight');
  const a = valueIteration(e8).V, b = policyIteration(e8).steps.at(-1).V;
  ok(e8.states.every((s) => near(a[s], b[s], 1e-5)), 'value iteration and policy iteration agree on the 8x8 world');
  // deterministic corridor: V*(d steps from the goal) = gamma^(d-1) with the reward on entry
  const c = makeGrid({ rows: ['S....G'], step: 0, slip: 0, gamma: 0.9, rewards: { G: 1 } });
  const v = valueIteration(c).V;
  ok([1, 2, 3, 4, 5].every((d) => near(v[5 - d], 0.9 ** (d - 1), 1e-9)), 'corridor values are gamma^(d-1)');
}
// FrozenLake
{
  const det = gridOf('frozen', { slip: 0 });
  const L = learner(det, { method: 'q', alpha: 0.1, eps: 0.2, seed: 2, maxSteps: 100 });
  L.runEpisodes(2000);
  const g = L.greedy();
  ok(evaluatePolicy(det, (s) => g[s], { episodes: 500 }).success === 1, 'deterministic FrozenLake: tabular Q-learning reaches 100% greedy success');
  const sl = gridOf('frozen', { slip: 2 / 3 });
  const vi = valueIteration(sl, { gamma: 0.99 });
  const opt = evaluatePolicy(sl, (s) => vi.policy[s], { episodes: 4000 }).success;
  ok(near(opt, 0.74, 0.03), `slippery FrozenLake: the optimal policy succeeds ${(opt * 100).toFixed(1)}% of the time within 100 steps`);
}
// Bandits (supplementary)
{
  const e = runBandits({ agent: 'eps', opts: { eps: 0.1 } }), g = runBandits({ agent: 'greedy' });
  ok(avg(e.reward.slice(-200)) > avg(g.reward.slice(-200)), 'epsilon = 0.1 earns more than greedy on the 10-armed testbed');
}
// MC vs TD on the random walk
{
  const td = walkCurves({ method: 'td', alpha: 0.1 }).at(-1), mc = walkCurves({ method: 'mc', alpha: 0.02 }).at(-1);
  ok(td < mc, `random walk: TD(0) RMS ${td.toFixed(3)} below MC ${mc.toFixed(3)} after 100 episodes (Sutton & Barto Fig. 6.2)`);
  ok(near(WALK_TRUE[2], 0.5, 1e-12), 'true value of the centre state is 1/2');
}
// Cliff: Q-learning vs SARSA (supplementary)
{
  const cliff = gridOf('cliff');
  const q = avg(controlCurves(cliff, { method: 'q', runs: 20 }).slice(400)), s = avg(controlCurves(cliff, { method: 'sarsa', runs: 20 }).slice(400));
  ok(s > q, `cliff walk: SARSA earns ${s.toFixed(1)} per episode online, Q-learning ${q.toFixed(1)}`);
  const Lq = learner(cliff, { method: 'q', seed: 3 }); Lq.runEpisodes(500);
  ok(greedyPath(cliff, Lq.Q).length - 1 === 13, 'Q-learning greedy path hugs the cliff (13 steps)');
}
// Overestimation
{
  const m = maxBias();
  ok(m.single > 0.3 && Math.abs(m.double) < 0.05, `max of noisy zero-mean estimates is ${m.single.toFixed(2)}; double estimate ${m.double.toFixed(3)}`);
  const q = maxBiasMDP({ method: 'q', runs: 500 }), d = maxBiasMDP({ method: 'double', runs: 500 });
  ok(avg(q.slice(0, 20)) > avg(d.slice(0, 20)) + 0.1, 'Q-learning chooses the bad left action more often than double Q early on');
}
// Imitation
{
  const car = carCompare();
  ok(car.crashL2 > 0.9 && car.crashCat < 0.1, `multimodal demos: l2 policy crashes ${(car.crashL2 * 100).toFixed(0)}%, categorical ${(car.crashCat * 100).toFixed(0)}%`);
  const d = dagger(gridOf('corridor'), { demosN: 3, iters: 6, rolloutsPerIter: 3 });
  ok(d.iters.at(-1).success > d.iters[0].success + 0.15, `DAgger lifts success from ${d.iters[0].success.toFixed(2)} to ${d.iters.at(-1).success.toFixed(2)}`);
}
// Policy gradient bookkeeping
{
  const w = pgWeights(HUMANOID.set1, { causality: true, baseline: true });
  const T = 4;
  ok([...Array(T).keys()].every((t) => near(avg(w.map((x) => x[t])), 0, 1e-12)), 'with the mean baseline each step\'s weights average to 0');
  ok(pgWeights(HUMANOID.set2).every((x) => x.every((v) => v > 0)), 'set 2 without a baseline: every weight is positive');
  const a = importanceSpread({ T: 2 }), b = importanceSpread({ T: 50 });
  ok(b.ess < a.ess, `importance weights degrade with horizon: effective sample share ${a.ess.toFixed(2)} at T=2, ${b.ess.toFixed(2)} at T=50`);
  const is = isEstimate();
  ok(near(is.estimate, is.truth, 0.25), `importance sampling estimates E_p[x^2] = ${is.estimate.toFixed(2)} (truth ${is.truth})`);
}
// GAE and n-step
{
  const r = [0, 0, 1, 0, 2], V = [0.5, 0.4, 0.9, 0.3, 1.1];
  const g0 = gae(r, V, { lambda: 0 }), g1 = gae(r, V, { lambda: 1 });
  ok(g0.A.every((x, t) => near(x, g0.delta[t], 1e-12)), 'GAE with lambda = 0 is the TD error delta_t');
  ok(g1.A.every((x, t) => near(x, g1.mcAdv[t], 1e-9)), 'GAE with lambda = 1 is G_t - V(s_t)');
  ok(gae(r, V).A.every((x, t, arr) => near(x, gae(r, V).Afwd[t], 1e-9)), 'backward recursion equals the forward sum');
  ok(nStepTargets(r, V, 1).every((y, t) => near(y, r[t] + (t + 1 < 5 ? V[t + 1] : 0), 1e-12)), 'n = 1 targets are TD targets');
  ok(nStepTargets(r, V, 5).every((y, t) => near(y, r.slice(t).reduce((a, b) => a + b, 0), 1e-12)), 'n = T targets are Monte Carlo returns');
}
// PPO clip
{
  const e = 0.2;
  ok(near(clippedObjective(1.5, 1, e), 1.2, 1e-12) && near(clippedObjective(0.5, 1, e), 0.5, 1e-12) && near(clippedObjective(0.5, -1, e), -0.8, 1e-12) && near(clippedObjective(1.5, -1, e), -1.5, 1e-12), 'PPO surrogate at ratios 1.5 / 0.5 with A = +1 / -1');
  ok(Math.round((2000 * 10) / 64) === 313, 'slide example: 2000 steps x 10 epochs / batch 64 is about 300 gradient steps');
}
// CartPole learners (quick)
{
  const pg = pgCurves('rtg', { seeds: 2, iters: 60 });
  ok(pg.every((c) => Math.max(...c.slice(-10)) > 300), 'REINFORCE with reward-to-go balances CartPole past 300 steps within 60 iterations');
  const D = makeDQN({ seed: 1 }); D.run(25000);
  ok(avg(D.episodes.slice(-20)) > 300, `DQN with replay and target network: ${avg(D.episodes.slice(-20)).toFixed(0)} steps over the last 20 episodes`);
}
// Offline
{
  const r = stitchPolicies();
  ok(stitchRollout(r.iql).reached === 9 && stitchRollout(r.awr).reached !== 9 && stitchRollout(r.bc).reached !== 9, 'only IQL stitches s1 to s9');
  ok(near(expectile([0, 1, 2, 10], 0.5), 3.25, 1e-9) && expectile([0, 1, 2, 10], 0.99) > 9, 'expectile 0.5 is the mean; near 1 it approaches the max');
  const o = oodBandit({ deg: 5 });
  ok((o.best < -0.2 || o.best > 0.2) && o.predBest - o.trueBest > 1, `OOD: the fitted argmax a = ${o.best.toFixed(2)} lies outside the data and is overestimated by ${(o.predBest - o.trueBest).toFixed(1)}`);
  const env = makeGrid({ rows: ['S......', '.#####.', '.#...#.', '.#.L.#.', '...#..G'], step: -1, slip: 0, gamma: 0.95, rewards: { G: 10, L: -10 } });
  const D = collect(env, valueIteration(env).policy, { episodes: 20, eps: 0.1 });
  const n = fittedQ(env, D, { method: 'naive', noise: 2 }), c = fittedQ(env, D, { method: 'constrained', noise: 2 });
  ok(n.predicted - n.actual > 5 && near(c.predicted, c.actual, 1e-6), `fitted Q on logged data: naive predicts ${n.predicted.toFixed(2)} but earns ${n.actual.toFixed(2)}; constrained ${c.predicted.toFixed(2)} and ${c.actual.toFixed(2)}`);
}
// Reward learning and LLM post-training
{
  const g = P.goalClassifier();
  ok(!g.frozenSuccess && g.vice.at(-1).success, 'a frozen goal classifier is fooled; adding visited states as negatives fixes it');
  ok(near(P.sigmoid(0), 0.5, 1e-12), 'sigma(0) = 1/2');
  const bt = P.bradleyTerry();
  ok(bt.cos > 0.95, `Bradley-Terry recovers the hidden reward direction (cosine ${bt.cos.toFixed(3)})`);
  const lm = P.makeLM(), pairs = P.preferencePairs(lm), rm = P.fitRewardModel(lm, pairs);
  const big = P.klOptimal(lm.ref, rm.r, 1e6), small = P.klOptimal(lm.ref, rm.r, 1e-4);
  ok(big.every((p, i) => near(p, lm.ref[i], 1e-4)), 'beta -> infinity: the KL-regularised optimum is pi_ref');
  ok(Math.max(...small) > 0.999, 'beta -> 0: it puts all mass on the reward model\'s favourite');
  const oo = P.overoptimisation(lm, rm);
  const peak = Math.max(...oo.map((o) => o.truth));
  ok(peak > oo.at(-1).truth + 0.02, `over-optimisation: true quality peaks at ${peak.toFixed(3)} then falls to ${oo.at(-1).truth.toFixed(3)} as KL grows`);
  const d = P.dpo(lm, pairs);
  ok(near(d.hist[0].loss, Math.log(2), 1e-9), 'DPO loss at pi = pi_ref is log 2');
  const A = P.groupAdvantages([1, 0, 0, 1]);
  ok(A.every((x, i) => near(x, [1, -1, -1, 1][i], 1e-12)), 'GRPO advantages for rewards [1, 0, 0, 1] are [1, -1, -1, 1]');
  ok(P.groupAdvantages([1, 1, 1, 1]).every((x) => x === 0), 'an all-equal group has zero advantage');
  ok(P.k3(1) === 0 && P.k3(0.5) > 0 && P.k3(2) > 0, 'the k3 KL estimate is 0 at ratio 1 and positive elsewhere');
  const g1 = P.lengthBiasRun({ norm: 'grpo' }), g2 = P.lengthBiasRun({ norm: 'dr' });
  ok(g1.at(-1).lenWrong > g1[0].lenWrong + 3 && Math.abs(g2.at(-1).lenWrong - g2[0].lenWrong) < 2, `1/|o| normalisation: wrong answers grow from ${g1[0].lenWrong.toFixed(1)} to ${g1.at(-1).lenWrong.toFixed(1)} tokens; Dr GRPO ${g2.at(-1).lenWrong.toFixed(1)}`);
  ok(near(0.01 * 1.2, 0.012, 1e-12) && near(0.01 * 1.28, 0.0128, 1e-12), 'clip-higher: a p = 0.01 token can reach 0.012 at eps 0.2 and 0.0128 at eps_high 0.28');
}
console.log(fail ? `\n${fail} FAILED` : '\nall passed');
process.exit(fail ? 1 : 0);
