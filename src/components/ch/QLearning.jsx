// Chapter 8 (CS224R lecture 6; the notes, section 2): Q-learning on FrozenLake, Q-learning vs
// SARSA on the cliff, maximisation bias and double Q, and DQN with its stabilisers as switches.
import React, { useEffect, useMemo, useState } from 'react';
import { Card, Btn, Slider, Term, Cite, Eq, NotShown, Stat } from '../ui.jsx';
import GridView from '../GridView.jsx';
import Chart, { bandOf } from '../Chart.jsx';
import CartPoleView from '../CartPoleView.jsx';
import { gridOf } from '../../lib/grid.js';
import { valueIteration } from '../../lib/dp.js';
import { learner, controlCurves, greedyPath, maxBias, maxBiasMDP } from '../../lib/control.js';
import { evaluatePolicy } from '../../lib/imitation.js';
import { makeDQN } from '../../lib/dqn.js';
import { useLoop, later } from '../../lib/useLoop.js';

const movingAvg = (a, k = 50) => a.map((_, i) => { const s = Math.max(0, i - k + 1); let t = 0; for (let j = s; j <= i; j++) t += a[j]; return t / (i - s + 1); });

function Frozen() {
  const [slippery, setSlippery] = useState(false);
  const [alpha, setAlpha] = useState(0.1);
  const [eps, setEps] = useState(0.2);
  const [speed, setSpeed] = useState(1);
  const [seed, setSeed] = useState(2);
  const env = useMemo(() => gridOf('frozen', { slip: slippery ? 2 / 3 : 0 }), [slippery]);
  const L = useMemo(() => learner(env, { method: 'q', alpha, eps, seed, maxSteps: 100 }), [env, seed]); // eslint-disable-line
  useEffect(() => { L.alpha = alpha; L.eps = eps; }, [L, alpha, eps]);
  const [, tick] = useState(0);
  const [on, setOn] = useState(false);
  const [succ, setSucc] = useState([]);
  useEffect(() => { setSucc([]); setOn(false); }, [L]);
  const optimal = useMemo(() => { const vi = valueIteration(env, { gamma: 0.99 }); return evaluatePolicy(env, (s) => vi.policy[s], { episodes: 2000 }).success; }, [env]);
  useLoop(() => {
    const n = speed === 1 ? 1 : speed === 2 ? 40 : 2000;
    for (let i = 0; i < n; i++) L.step();
    if (speed > 1 || L.episode % 20 === 0) {
      const g = L.greedy();
      const e = L.episode;
      if (!succ.length || e - succ.at(-1).e >= (speed === 3 ? 500 : 20)) setSucc((x) => [...x, { e, v: evaluatePolicy(env, (s) => g[s], { episodes: 300 }).success }]);
    }
    tick((t) => t + 1);
    if (L.episode >= 20000) { setOn(false); return false; }
    return true;
  }, on, { every: speed === 1 ? 6 : 2 });
  const sc = 1;
  return (
    <Card title="Tabular Q-learning on FrozenLake">
      <p className="text-xs text-slate-400">
        Gymnasium's 4x4 FrozenLake: reach ★ (+1) without falling through a hole (✕, 0); at most 100 steps. Slippery ice moves you in the intended direction
        only one time in three. The notes run exactly this pair of experiments.<Cite id="nalawade" />
      </p>
      <Eq note="the Q-learning update; the target uses the best next action, whatever the agent actually does next">Q(s, a) ← Q(s, a) + α [ r + γ maxₐ′ Q(s′, a′) − Q(s, a) ]</Eq>
      <div className="flex flex-wrap items-center gap-1">
        <Btn on={!slippery} onClick={() => setSlippery(false)}>deterministic</Btn>
        <Btn on={slippery} onClick={() => setSlippery(true)}>slippery</Btn>
        <Btn on={on} onClick={() => setOn(!on)}>{on ? 'Pause' : 'Learn'}</Btn>
        <Btn onClick={() => setSeed(seed + 1)}>Reset</Btn>
        {['watch', 'fast', 'fastest'].map((t, i) => <Btn key={t} on={speed === i + 1} onClick={() => setSpeed(i + 1)}>{t}</Btn>)}
      </div>
      <Slider label="step size α" value={alpha} set={setAlpha} min={0.01} max={1} step={0.01} />
      <Slider label="ε (explore)" value={eps} set={setEps} min={0} max={1} step={0.05} />
      <div className="grid gap-3 sm:grid-cols-[minmax(0,1fr)_minmax(0,1fr)]">
        <GridView env={env} q={L.Q} scale={sc} policy={L.greedy()} agent={L.state} paths={[L.path]} size={60} showValues={false} />
        <div className="space-y-2">
          <Stat label="episodes" value={L.episode} />
          <Stat label="greedy policy success (300 tests)" value={succ.length ? `${(100 * succ.at(-1).v).toFixed(0)}%` : '-'} tone="warm" />
          <Stat label="best possible (value iteration)" value={`${(100 * optimal).toFixed(0)}%`} tone="dim" />
        </div>
      </div>
      {succ.length > 1 && <Chart series={[{ label: 'greedy success', y: succ.map((x) => 100 * x.v) }]} refLine={100 * optimal} refLabel="optimal" xLabel={`evaluations (every ${speed === 3 ? 500 : 20} episodes)`} yLabel="% success" yMin={0} yMax={100} height={120} fmt={(v) => `${v.toFixed(0)}%`} />}
      <p className="text-[11px] text-slate-400">
        Triangles show Q(s, a) for each action, darker meaning higher. On slippery ice even the best policy only succeeds about three times in four, and it
        often walks away from the goal so that slipping cannot drop it in a hole.
      </p>
    </Card>
  );
}

function Cliff() {
  const env = useMemo(() => gridOf('cliff'), []);
  const [eps, setEps] = useState(0.1);
  const [res, setRes] = useState(null);
  const [busy, setBusy] = useState(false);
  const run = async () => {
    setBusy(true);
    const out = await later(() => {
      const q = controlCurves(env, { method: 'q', eps, runs: 30, episodes: 500 }), s = controlCurves(env, { method: 'sarsa', eps, runs: 30, episodes: 500 });
      const Lq = learner(env, { method: 'q', eps, seed: 3 }); Lq.runEpisodes(500);
      const Ls = learner(env, { method: 'sarsa', eps, seed: 3 }); Ls.runEpisodes(500);
      return { q, s, pq: greedyPath(env, Lq.Q), ps: greedyPath(env, Ls.Q) };
    });
    setRes(out); setBusy(false);
  };
  const tail = (a) => (a.slice(400).reduce((x, y) => x + y, 0) / 100).toFixed(1);
  return (
    <Card title="Off-policy vs on-policy on the cliff">
      <p className="text-xs text-slate-400">
        Supplementary (Sutton & Barto, Example 6.6).<Cite id="sutton2018" /> Neither course source writes out <Term k="SARSA" />, the on-policy version that bootstraps
        from the action actually taken next: r + γ Q(s′, a′). Every step costs 1; stepping on the cliff costs 100 and sends you back.
      </p>
      <Slider label="ε" value={eps} set={setEps} min={0.01} max={0.3} step={0.01} />
      <Btn on onClick={run} disabled={busy}>{busy ? 'Running 60 learners...' : 'Run Q-learning and SARSA (30 runs, 500 episodes each)'}</Btn>
      {res && (
        <>
          <GridView env={env} paths={[Object.assign([...res.pq], { color: 'var(--color-warm)' }), Object.assign([...res.ps], { color: 'var(--color-cool)', dash: '5 3' })]} size={32} showValues={false} />
          <div className="flex flex-wrap gap-2">
            <Stat label="Q-learning: greedy path, reward per episode while learning" value={`${res.pq.length - 1} steps, ${tail(res.q)}`} tone="warm" />
            <Stat label="SARSA" value={`${res.ps.length - 1} steps, ${tail(res.s)}`} tone="cool" />
          </div>
          <Chart series={[{ label: 'Q-learning', y: movingAvg(res.q, 10), color: 'var(--color-warm)' }, { label: 'SARSA', y: movingAvg(res.s, 10), color: 'var(--color-cool)' }]} xLabel="episode" yLabel="sum of rewards" yMin={-120} yMax={0} height={130} fmt={(v) => v.toFixed(0)} />
        </>
      )}
      <p className="text-[11px] text-slate-400">
        Q-learning learns the value of the greedy policy, so it finds the shortest path along the edge, then keeps falling off while it explores. SARSA learns
        the value of its own exploring behaviour, so it takes the long safe way and earns more while learning. Set ε near 0 and the two agree.
      </p>
    </Card>
  );
}

function Overestimation() {
  const m = useMemo(() => maxBias(), []);
  const [res, setRes] = useState(null);
  const [busy, setBusy] = useState(false);
  const run = async () => { setBusy(true); const r = await later(() => ({ q: maxBiasMDP({ method: 'q' }), d: maxBiasMDP({ method: 'double' }) })); setRes(r); setBusy(false); };
  return (
    <Card title="The max of noisy estimates is too high">
      <p className="text-xs text-slate-400">
        Ten actions, each truly worth 0, each estimated from 5 noisy samples. The biggest estimate averages {m.single.toFixed(2)}, not 0. That is the
        <Term k="overestimation"> overestimation</Term> lecture 6 warns about: E[max(X₁, X₂)] ≥ max(E X₁, E X₂). Double Q-learning picks with one set of
        estimates and scores with an independent one: {m.double.toFixed(3)}.<Cite id="vanhasselt2010" />
      </p>
      <Eq note="double Q target, as lecture 6 writes it for DQN: select with φ, evaluate with φ′">y = r + γ Q_φ′(s′, argmaxₐ′ Q_φ(s′, a′))</Eq>
      <Btn on onClick={run} disabled={busy}>{busy ? 'Running 2000 learners...' : 'Run Sutton & Barto Example 6.7'}</Btn>
      {res && <Chart series={[{ label: 'Q-learning', y: res.q.map((v) => 100 * v), color: 'var(--color-warm)' }, { label: 'double Q-learning', y: res.d.map((v) => 100 * v), color: 'var(--color-cool)' }]} refLine={5} refLabel="optimal with ε = 0.1" xLabel="episode" yLabel="% choosing left" yMin={0} yMax={100} height={130} fmt={(v) => `${v.toFixed(0)}%`} />}
      <p className="text-[11px] text-slate-400">
        From A, going left reaches B, whose ten actions each pay N(−0.1, 1): worse than going right (0). Q-learning's max over B's noisy estimates makes left
        look good at first.<Cite id="sutton2018" />
      </p>
    </Card>
  );
}

function DQN() {
  const [replay, setReplay] = useState(true);
  const [target, setTarget] = useState(true);
  const [double, setDouble] = useState(false);
  const [seed, setSeed] = useState(1);
  const D = useMemo(() => makeDQN({ replay, target, double, seed }), [replay, target, double, seed]);
  const [snap, setSnap] = useState(null);
  const [on, setOn] = useState(false);
  useEffect(() => { setSnap(null); setOn(false); }, [D]);
  useLoop(() => {
    const r = D.run(400);
    setSnap({ ...r, episodes: [...r.episodes], qStart: [...r.qStart] });
    if (r.steps >= 25000) { setOn(false); return false; }
    return true;
  }, on, { every: 1 });
  const ep = snap?.episodes ?? [];
  return (
    <Card title="DQN with its two stabilisers as switches">
      <p className="text-xs text-slate-400">
        A 4-32-2 Q-network on CartPole. Lecture 6: a <Term k="replay buffer" /> breaks the correlation between consecutive samples, and a
        <Term k="target network"> target network</Term> freezes the labels so each inner fit is plain supervised regression.<Cite id="mnih2015" /> Here φ′ is copied every 250
        steps; ε decays from 1 to 0.05 over 4000 steps.
      </p>
      <Eq note="DQN loss (the notes' form, with the done mask)">L(φ) = ( r + γ (1 − done) maxₐ′ Q_φ′(s′, a′) − Q_φ(s, a) )²</Eq>
      <div className="flex flex-wrap gap-1">
        <Btn on={replay} onClick={() => setReplay(!replay)}>replay buffer</Btn>
        <Btn on={target} onClick={() => setTarget(!target)}>target network</Btn>
        <Btn on={double} onClick={() => setDouble(!double)}>double DQN</Btn>
        <Btn on={on} onClick={() => setOn(!on)}>{on ? 'Pause' : snap ? 'Continue' : 'Train 25,000 steps'}</Btn>
        <Btn onClick={() => setSeed(seed + 1)}>New seed</Btn>
      </div>
      {snap && <CartPoleView s={snap.current} label={`step ${snap.steps}, ε ${snap.eps.toFixed(2)}`} height={110} />}
      {ep.length > 1 && <Chart series={[{ label: 'episode length', y: ep, opacity: 0.35, width: 1 }, { label: 'last 20 episodes', y: movingAvg(ep, 20) }]} xLabel="episode" yLabel="steps" yMin={0} yMax={500} height={130} fmt={(v) => v.toFixed(0)} />}
      {ep.length > 1 && <Chart series={[{ label: 'max Q at the upright start state', y: snap.qStart, color: 'var(--color-warm)' }]} refLine={(1 - 0.99 ** 500) / 0.01} refLabel="a perfect policy's discounted return" xLabel="episode" yLabel="Q" yMin={0} height={110} fmt={(v) => v.toFixed(1)} />}
      <p className="text-[11px] text-slate-400">
        Measured over two seeds: with both switches on, the last 20 episodes average about 450 to 465 steps. Without the replay buffer, learning is faster per
        step but far less reliable. Q at the start can only be checked against the best possible return, about 99.3 with γ = 0.99.
      </p>
      <NotShown>Atari from pixels, where these tricks were introduced; n-step DQN targets, which lecture 6 notes are only correct on-policy.</NotShown>
    </Card>
  );
}

export default function QLearning() {
  return (
    <div className="grid gap-4">
      <Frozen />
      <Cliff />
      <Overestimation />
      <DQN />
    </div>
  );
}
