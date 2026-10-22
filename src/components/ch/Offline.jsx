// Chapter 9 (CS224R lecture 7): offline RL. Stitching on the 3x3 example, out-of-distribution
// overestimation in a 1D action space and in a gridworld, and the expectile behind IQL.
import React, { useMemo, useState } from 'react';
import { Card, Btn, Slider, Term, Cite, Eq, NotShown, Stat } from '../ui.jsx';
import GridView from '../GridView.jsx';
import Chart from '../Chart.jsx';
import { STITCH, stitchPolicies, stitchRollout, oodBandit, expectile, collect, fittedQ } from '../../lib/offline.js';
import { makeGrid } from '../../lib/grid.js';
import { valueIteration } from '../../lib/dp.js';

const METHODS = [
  ['bc', 'behaviour cloning'],
  ['fbc', 'filtered BC (top third)'],
  ['awr', 'AWR (Monte Carlo values)'],
  ['iql', 'IQL (TD + expectile)'],
];
const pos = (s) => [40 + ((s - 1) % 3) * 90, 35 + Math.floor((s - 1) / 3) * 75];
const COLORS = { tau1: 'var(--color-cool)', tau2: 'var(--color-glow)', tau3: '#8a5cc7' };

function Stitching() {
  const [method, setMethod] = useState('iql');
  const [tau, setTau] = useState(0.9);
  const [alpha, setAlpha] = useState(0.3);
  const r = useMemo(() => stitchPolicies({ tau, alpha }), [tau, alpha]);
  const pi = r[method];
  const roll = stitchRollout(pi);
  const V = method === 'iql' ? r.V : r.vmc;
  return (
    <Card title="Stitching: better than any trajectory in the log">
      <p className="text-xs text-slate-400">
        Lecture 7's example, laid out as a 3x3 grid: +1 for reaching s9, −1 for s6. Three logged trajectories: τ1 starts well (s1 down to s4) and ends in s6;
        τ2 finishes well (s4 down to s7 and on to s9); τ3 goes s1 right to s3 and nowhere. No trajectory goes from s1 to s9.<Cite id="cs224r" />
      </p>
      <svg viewBox="0 0 300 220" className="w-full max-w-md">
        {Array.from({ length: 9 }, (_, i) => i + 1).map((s) => {
          const [x, y] = pos(s);
          return (
            <g key={s}>
              <rect x={x - 26} y={y - 22} width="52" height="44" rx="6" fill={s === 9 ? 'color-mix(in srgb, var(--color-warm) 35%, transparent)' : s === 6 ? 'color-mix(in srgb, var(--color-cool) 35%, transparent)' : 'var(--color-panel)'} stroke="var(--color-line)" />
              <text x={x} y={y - 6} fontSize="11" textAnchor="middle" fill="var(--ink)">s{s}</text>
              {V[s] != null && <text x={x} y={y + 12} fontSize="9" textAnchor="middle" fill="var(--color-dim)">V {V[s].toFixed(2)}</text>}
            </g>
          );
        })}
        {STITCH.trajs.map((t) => <polyline key={t.name} points={t.s.map((s) => pos(s).map((v, k) => v + (t.name === 'tau1' ? -5 : t.name === 'tau3' ? 5 : 0) * (k === 0 ? 0 : 1)).join(',')).join(' ')} fill="none" stroke={COLORS[t.name]} strokeWidth="2.5" opacity="0.8" strokeDasharray="5 3" />)}
        <polyline points={roll.path.map((s) => pos(s).join(',')).join(' ')} fill="none" stroke="var(--color-warm)" strokeWidth="4" opacity="0.85" />
        <circle cx={pos(roll.path.at(-1))[0]} cy={pos(roll.path.at(-1))[1]} r="7" fill="var(--color-warm)" />
      </svg>
      <div className="flex flex-wrap gap-2 text-[10px] text-dim">
        <span style={{ color: 'var(--color-cool)' }}>- - τ1 (return −1)</span><span style={{ color: 'var(--color-glow)' }}>- - τ2 (+1)</span><span style={{ color: '#8a5cc7' }}>- - τ3 (0)</span><span className="text-warm">━ the learned policy from s1</span>
      </div>
      <div className="flex flex-wrap gap-1">{METHODS.map(([k, t]) => <Btn key={k} on={method === k} onClick={() => setMethod(k)}>{t}</Btn>)}</div>
      <div className="text-[11px] text-slate-300">
        From s1 the policy goes {roll.path.map((s) => `s${s}`).join(' → ')} and earns <b className={roll.G > 0 ? 'text-warm' : 'text-dim'}>{roll.G}</b>.
        {roll.path.length === 1 && ' It has no data at s1 at all, because filtering kept only τ2.'}
      </div>
      <Slider label="IQL expectile τ" value={tau} set={setTau} min={0.5} max={0.99} step={0.01} fmt={(v) => v.toFixed(2)} />
      <Slider label="temperature α" value={alpha} set={setAlpha} min={0.05} max={2} step={0.05} fmt={(v) => v.toFixed(2)} />
      <ul className="list-disc space-y-0.5 pl-4 text-[11px] text-slate-300">
        <li><b>BC</b> copies everything, including τ1's walk into s6.</li>
        <li><b>Filtered BC</b> keeps only the best trajectories ("very primitive", the lecture says, and a good baseline), so it knows nothing about s1.</li>
        <li><b>AWR</b> weights each logged action by exp((return-to-go − V(s))/α) with Monte Carlo values. At s1 it compares τ1's −1 with τ3's 0 and picks τ3.</li>
        <li><b>IQL</b> learns V by an expectile over data actions and Q by TD, so s4's value from τ2 flows back to s1. It picks "down" and stitches s1 → s4 → s7 → s8 → s9.</li>
      </ul>
      <p className="text-[11px] text-slate-400">IQL is Kostrikov et al.<Cite id="kostrikov2022" /> and AWR is Peng et al.,<Cite id="peng2019" /> both as presented in lecture 7. At τ = 0.5 the expectile is a plain mean, and the stitch can fail.</p>
    </Card>
  );
}

function OOD() {
  const [deg, setDeg] = useState(5);
  const o = useMemo(() => oodBandit({ deg }), [deg]);
  const cap = (v) => Math.max(-3, Math.min(4, v));
  return (
    <Card title="Asking a Q-function about actions nobody took">
      <p className="text-xs text-slate-400">
        One state and a continuous action a in [−1, 1]. The true reward peaks at a = 0.5, but the logged data only tried a in [−0.2, 0.2]. A flexible model
        fits the data well and says anything at all elsewhere; the argmax goes looking for its errors.<Cite id="levine2020" />
      </p>
      <Slider label="model flexibility (degree)" value={deg} set={setDeg} min={1} max={8} step={1} />
      <Chart series={[
        { label: 'true reward', y: o.truth, color: 'var(--color-dim)', dash: '4 3' },
        { label: 'fitted Q (clipped at 4 for display)', y: o.pred.map(cap), color: 'var(--color-warm)' },
      ]} xLabel="action a, from −1 to 1" yLabel="value" yMin={-3} yMax={4} height={150} fmt={(v) => v.toFixed(2)} />
      <div className="flex flex-wrap gap-2">
        <Stat label="argmax over all actions" value={`a = ${o.best.toFixed(2)}: predicted ${o.predBest.toFixed(1)}, true ${o.trueBest.toFixed(2)}`} tone="warm" />
        <Stat label="argmax over data actions only" value={`a = ${o.bestIn.toFixed(2)}: predicted ${o.predIn.toFixed(2)}, true ${o.trueIn.toFixed(2)}`} />
      </div>
      <p className="text-[11px] text-slate-400">Online RL would try that action, see it is bad, and correct itself. Offline, nothing ever corrects it, so the method has to stay on the data.</p>
    </Card>
  );
}

function GridOffline() {
  const env = useMemo(() => makeGrid({ rows: ['S......', '.#####.', '.#...#.', '.#.L.#.', '...#..G'], step: -1, slip: 0, gamma: 0.95, rewards: { G: 10, L: -10 } }), []);
  const [episodes, setEpisodes] = useState(20);
  const [eps, setEps] = useState(0.1);
  const [noise, setNoise] = useState(2);
  const [method, setMethod] = useState('naive');
  const D = useMemo(() => collect(env, valueIteration(env).policy, { episodes, eps }), [env, episodes, eps]);
  const fit = useMemo(() => fittedQ(env, D, { method, noise }), [env, D, method, noise]);
  const best = useMemo(() => valueIteration(env).V[env.start], [env]);
  const seenCount = fit.seen.map((a) => a.filter(Boolean).length);
  return (
    <Card title="Fitted Q-iteration on a fixed log">
      <p className="text-xs text-slate-400">
        A noisy expert logged {D.length} transitions. Q-values for (state, action) pairs the log never contains start as noise of size {noise}, standing in for
        a network's arbitrary outputs off the data. Naive fitted Q bootstraps from max over all actions; the constrained version maxes only over actions the
        log tried in that state (in the spirit of BCQ).<Cite id="fujimoto2019" />
      </p>
      <div className="flex gap-1">{[['naive', 'naive max'], ['constrained', 'max over data actions']].map(([k, t]) => <Btn key={k} on={method === k} onClick={() => setMethod(k)}>{t}</Btn>)}</div>
      <Slider label="logged episodes" value={episodes} set={setEpisodes} min={2} max={60} step={1} />
      <Slider label="expert noise ε" value={eps} set={setEps} min={0} max={0.6} step={0.05} />
      <Slider label="off-data noise" value={noise} set={setNoise} min={0} max={4} step={0.25} />
      <GridView env={env} values={seenCount.map((c) => c / 4)} scale={1} policy={fit.pi} paths={[Object.assign(greedyLine(env, fit.pi), { color: 'var(--color-warm)' })]} size={44} showValues={false} />
      <p className="text-[10px] text-dim">Shade: how many of the four actions the log contains in each cell. Arrows: the learned greedy policy.</p>
      <div className="flex flex-wrap gap-2">
        <Stat label="Q's own estimate at the start" value={fit.predicted.toFixed(2)} tone="warm" />
        <Stat label="what the policy actually earns" value={fit.actual.toFixed(2)} />
        <Stat label="best possible" value={best.toFixed(2)} tone="dim" />
      </div>
      <p className="text-[11px] text-slate-400">
        A return of −20 means the policy never finishes (−1 per step forever at γ = 0.95). The naive learner is confident and wrong; the constrained one
        cannot overrate actions it has no data for.
      </p>
      <NotShown>CQL, which pushes Q down on all actions and up on data actions;<Cite id="kumar2020" /> AWAC, which homework 3 of the course implements.</NotShown>
    </Card>
  );
}

function greedyLine(env, pi) {
  let s = env.start;
  const out = [s];
  for (let t = 0; t < 40 && !env.isTerminal(s) && pi[s] >= 0; t++) {
    const tr = env.trans(s, pi[s]);
    if (!tr.length) break;
    s = tr[0].s2; out.push(s);
    if (out.filter((x) => x === s).length > 2) break;
  }
  return out;
}

function Expectile() {
  const xs = [0, 0.2, 0.3, 1, 2.5];
  const [tau, setTau] = useState(0.7);
  const e = expectile(xs, tau);
  const X = (v) => 20 + (v / 3) * 360;
  return (
    <Card title="The expectile behind IQL">
      <Eq note="IQL's value loss, in the paper's form; lecture 7 writes the same with λ = 1 − τ on V − Q">L(V) = E [ |τ − 1(Q − V &lt; 0)| (Q − V)² ]</Eq>
      <Slider label="τ" value={tau} set={setTau} min={0.5} max={0.99} step={0.01} fmt={(v) => v.toFixed(2)} />
      <svg viewBox="0 0 400 60" className="w-full">
        <line x1="20" x2="380" y1="35" y2="35" stroke="var(--color-line)" />
        {xs.map((x, i) => <circle key={i} cx={X(x)} cy="35" r="5" fill="var(--color-glow)" />)}
        <line x1={X(e)} x2={X(e)} y1="12" y2="52" stroke="var(--color-warm)" strokeWidth="2" />
        <text x={X(e)} y="10" fontSize="10" textAnchor="middle" fill="var(--color-warm)">{e.toFixed(2)}</text>
      </svg>
      <p className="text-[11px] text-slate-400">
        Five Q-values of actions seen in one state. At τ = 0.5 the expectile is their mean ({expectile(xs, 0.5).toFixed(2)}); as τ → 1 it approaches the best of
        them (2.5). So V(s) estimates "the value of the better actions in the data" without ever evaluating an action outside it.
      </p>
    </Card>
  );
}

export default function Offline() {
  return (
    <div className="grid gap-4">
      <Stitching />
      <OOD />
      <GridOffline />
      <Expectile />
    </div>
  );
}
