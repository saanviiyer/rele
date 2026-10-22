// Chapter 1: trial and error with slot machines, then agents on the 10-armed testbed.
import React, { useMemo, useState } from 'react';
import { Card, Btn, Slider, Term, Cite, Eq, NotShown, Stat } from '../ui.jsx';
import Chart from '../Chart.jsx';
import { rng, argmax } from '../../lib/rng.js';
import { AGENTS, runBandits, gaussianBandit, bernoulliBandit } from '../../lib/bandit.js';
import { useLoop, later } from '../../lib/useLoop.js';

const BUDGET = 30;

function machinesFor(seed) {
  const r = rng(seed * 31 + 5);
  return Array.from({ length: 5 }, () => Math.round((0.15 + 0.7 * r()) * 100) / 100);
}

function YourTurn() {
  const [game, setGame] = useState(1);
  const p = useMemo(() => machinesFor(game), [game]);
  const [log, setLog] = useState([]);
  const rnd = useMemo(() => rng(game * 977), [game]);
  const pulls = p.map((_, i) => log.filter((x) => x.a === i).length);
  const wins = p.map((_, i) => log.filter((x) => x.a === i && x.r).length);
  const done = log.length >= BUDGET;
  const total = log.reduce((a, x) => a + x.r, 0);
  // the same budget for Thompson sampling, averaged over many plays of these machines
  const agent = useMemo(() => {
    if (!done) return null;
    let s = 0;
    const runs = 400;
    for (let k = 0; k < runs; k++) {
      const r = rng(k + 1);
      const ag = AGENTS.thompson.make(5, { kind: 'bernoulli' }, r);
      const b = bernoulliBandit(p);
      for (let t = 0; t < BUDGET; t++) { const a = ag.act(); const x = b.pull(a, r); ag.learn(a, x); s += x; }
    }
    return s / runs;
  }, [done, p]);
  const best = Math.max(...p);
  return (
    <Card title="Your turn: five slot machines, 30 pulls">
      <p className="text-xs leading-snug text-slate-400">
        Each machine pays 1 with its own hidden probability. Every pull either teaches you something (explore) or cashes in on what you know (exploit).
        There is no state and nothing you do changes the machines, so this is reinforcement learning with only the exploration problem left.
      </p>
      <div className="grid grid-cols-5 gap-2">
        {p.map((pi, i) => (
          <button key={i} disabled={done} onClick={() => setLog([...log, { a: i, r: rnd() < pi ? 1 : 0 }])}
            className="rounded-lg border border-line bg-panel p-2 text-center hover:border-warm/60 disabled:opacity-70">
            <div className="text-2xl">🎰</div>
            <div className="text-[11px] text-slate-200">machine {i + 1}</div>
            <div className="text-[10px] tabular-nums text-dim">{wins[i]} / {pulls[i]}</div>
            {done && <div className="mt-0.5 text-[10px] text-warm">p = {pi.toFixed(2)}</div>}
          </button>
        ))}
      </div>
      <div className="flex flex-wrap items-center gap-2 text-xs">
        <Stat label="pulls left" value={BUDGET - log.length} />
        <Stat label="you won" value={total} tone="warm" />
        {done && <Stat label="Thompson sampling, same machines (mean of 400 plays)" value={agent.toFixed(1)} />}
        {done && <Stat label="always the best machine (expected)" value={(best * BUDGET).toFixed(1)} tone="dim" />}
        <Btn onClick={() => { setGame(game + 1); setLog([]); }}>New machines</Btn>
      </div>
      {log.length > 0 && (
        <div className="flex flex-wrap gap-0.5">
          {log.map((x, i) => <span key={i} className={`inline-block h-3 w-3 rounded-sm text-center text-[8px] leading-3 ${x.r ? 'bg-warm/80 text-black' : 'bg-line text-dim'}`}>{x.a + 1}</span>)}
        </div>
      )}
      {done && <p className="text-xs text-slate-300">Your <Term k="regret">regret</Term>, in expectation: {(best * BUDGET - log.reduce((a, x) => a + p[x.a], 0)).toFixed(1)} wins lost to not always pulling machine {argmax(p) + 1}.</p>}
    </Card>
  );
}

const PICK = ['greedy', 'eps', 'optimistic', 'ucb', 'thompson', 'gradient'];

function Testbed() {
  const [chosen, setChosen] = useState(['greedy', 'eps', 'ucb', 'thompson']);
  const [eps, setEps] = useState(0.1);
  const [c, setC] = useState(2);
  const [res, setRes] = useState(null);
  const [busy, setBusy] = useState(false);
  const run = async () => {
    setBusy(true);
    const out = await later(() => chosen.map((a) => ({ a, ...runBandits({ agent: a, opts: { eps, c, q0: 5 }, runs: 200, steps: 1000 }) })));
    setRes(out); setBusy(false);
  };
  const lab = (a) => (a === 'eps' ? `epsilon-greedy (${eps})` : a === 'ucb' ? `UCB (c = ${c})` : AGENTS[a].label);
  return (
    <Card title="Six agents on the 10-armed testbed">
      <p className="text-xs leading-snug text-slate-400">
        Ten arms with true means drawn from N(0, 1); each pull pays the arm's mean plus N(0, 1) noise. Curves average 200 independent testbeds, 1000 pulls each
        (Sutton & Barto, ch. 2).<Cite id="sutton2018" />
      </p>
      <div className="flex flex-wrap gap-1">
        {PICK.map((a) => <Btn key={a} on={chosen.includes(a)} onClick={() => setChosen(chosen.includes(a) ? chosen.filter((x) => x !== a) : [...chosen, a])}>{AGENTS[a].label}</Btn>)}
      </div>
      <Slider label="epsilon" value={eps} set={setEps} min={0} max={0.5} step={0.01} />
      <Slider label="UCB c" value={c} set={setC} min={0} max={4} step={0.1} />
      <Btn on onClick={run} disabled={busy || !chosen.length}>{busy ? 'Running 200 x 1000 pulls each...' : 'Run'}</Btn>
      {res && (
        <>
          <Chart series={res.map((r) => ({ label: lab(r.a), y: r.reward }))} smooth={10} xLabel="pull" yLabel="average reward" />
          <Chart series={res.map((r) => ({ label: lab(r.a), y: r.optimal.map((v) => 100 * v) }))} smooth={10} xLabel="pull" yLabel="% best arm" yMin={0} yMax={100} fmt={(v) => `${v.toFixed(0)}%`} />
        </>
      )}
      <ul className="list-disc space-y-0.5 pl-4 text-[11px] text-slate-300">
        <li><b>Greedy</b> locks onto whichever arm looked good first.</li>
        <li><b>Epsilon-greedy</b> explores at random a fixed share of the time, forever.</li>
        <li><b>Optimistic greedy</b> starts every estimate at 5, so disappointment drives early exploration.</li>
        <li><b>UCB</b> adds a bonus that shrinks as an arm is tried.<Cite id="auer2002" /></li>
        <li><b>Thompson sampling</b> draws a plausible value for each arm from its posterior and plays the best draw.<Cite id="thompson1933" /></li>
        <li><b>Gradient bandit</b> learns preferences and samples from their softmax: the simplest policy gradient, the subject of chapter 5.</li>
      </ul>
    </Card>
  );
}

function OneAgent() {
  const [agentKey, setAgentKey] = useState('ucb');
  const [seed, setSeed] = useState(3);
  const [on, setOn] = useState(false);
  const env = useMemo(() => gaussianBandit(10, seed), [seed]);
  const [state, setState] = useState(() => ({ t: 0 }));
  const sim = useMemo(() => {
    const r = rng(seed + 100);
    return { r, ag: AGENTS[agentKey].make(10, { eps: 0.1, c: 2, q0: 5 }, r), t: 0, last: null };
  }, [agentKey, seed]); // eslint-disable-line
  const step = () => {
    const a = sim.ag.act(sim.t);
    const x = env.pull(a, sim.r);
    sim.ag.learn(a, x);
    sim.t++; sim.last = { a, x };
    setState({ t: sim.t });
    return sim.t < 500;
  };
  useLoop(step, on, { every: 4 });
  const lo = -3, hi = 3, Y = (v) => 100 - ((Math.max(lo, Math.min(hi, v)) - lo) / (hi - lo)) * 100;
  const spread = sim.ag.spread();
  const pol = sim.ag.policy?.();
  return (
    <Card title="Inside one agent">
      <div className="flex flex-wrap items-center gap-1">
        {['eps', 'ucb', 'thompson', 'gradient'].map((k) => <Btn key={k} on={agentKey === k} onClick={() => { setAgentKey(k); setOn(false); }}>{AGENTS[k].label}</Btn>)}
        <Btn onClick={() => setOn(!on)}>{on ? 'Pause' : 'Run 500 pulls'}</Btn>
        <Btn onClick={() => { setSeed(seed + 1); setOn(false); }}>New testbed</Btn>
      </div>
      <svg viewBox="0 0 400 120" className="w-full">
        <line x1="0" x2="400" y1={Y(0)} y2={Y(0)} stroke="var(--color-line)" />
        {env.q.map((q, a) => {
          const x = 20 + a * 38;
          const est = sim.ag.Q[a], sp = spread?.[a];
          return (
            <g key={a}>
              <line x1={x - 12} x2={x + 12} y1={Y(q)} y2={Y(q)} stroke="var(--color-dim)" strokeWidth="2" strokeDasharray="3 2" />
              {sp && <rect x={x - 3} y={Y(sp[1])} width="6" height={Math.max(1, Y(sp[0]) - Y(sp[1]))} fill="var(--color-cool)" opacity="0.35" />}
              {agentKey !== 'gradient' && <circle cx={x} cy={Y(est)} r="4" fill={sim.last?.a === a ? 'var(--color-warm)' : 'var(--color-glow)'} />}
              {pol && <rect x={x - 8} y={100 - pol[a] * 90} width="16" height={pol[a] * 90} fill={sim.last?.a === a ? 'var(--color-warm)' : 'var(--color-glow)'} opacity="0.6" />}
              <text x={x} y="116" fontSize="9" textAnchor="middle" fill="var(--color-dim)">{a + 1}, {sim.ag.N[a]}</text>
            </g>
          );
        })}
      </svg>
      <p className="text-[11px] text-slate-400">
        Dashed: each arm's true mean. {agentKey === 'gradient' ? 'Bars: the policy\'s probability of each arm.' : 'Dots: the agent\'s estimate; shaded: the range it still considers plausible.'} Under each arm: its number and pull count. Pull {state.t}
        {sim.last && <>; last pulled arm {sim.last.a + 1} and got {sim.last.x.toFixed(2)}</>}.
      </p>
    </Card>
  );
}

export default function Bandits() {
  return (
    <div className="grid gap-4">
      <YourTurn />
      <OneAgent />
      <Testbed />
      <Card title="What carries forward">
        <Eq note="the incremental mean, used by most agents above">Q(a) ← Q(a) + (1/N(a)) [r − Q(a)]</Eq>
        <p className="text-xs leading-relaxed text-slate-300">
          Every later chapter keeps this shape: an estimate, nudged toward a target by a step size. What changes is the target. In the next chapters
          actions also move you between states, so a choice now changes what you can earn later.
        </p>
        <NotShown>Contextual bandits, where the best arm depends on an observed context. They sit between this chapter and full RL.</NotShown>
      </Card>
    </div>
  );
}
