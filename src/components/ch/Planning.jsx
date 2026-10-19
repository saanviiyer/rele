// Chapter 3 (the notes, section 1; Bellman 1957; Howard 1960): when the model is known, plan.
// An editable gridworld, value iteration sweep by sweep, policy iteration, and the Bellman backup
// of any cell under the mouse.
import React, { useMemo, useState } from 'react';
import { Card, Btn, Slider, Select, Term, Cite, Eq, NotShown } from '../ui.jsx';
import GridView from '../GridView.jsx';
import Terrain from '../Terrain.jsx';
import Chart from '../Chart.jsx';
import { PRESETS, makeGrid, ACTIONS, ARROWS } from '../../lib/grid.js';
import { viSweep, greedy, backupTerms, qFromV, policyIteration, evaluate, rollout } from '../../lib/dp.js';
import { useLoop } from '../../lib/useLoop.js';

const PAINT = [['.', 'floor'], ['#', 'wall'], ['G', 'goal'], ['L', 'pit'], ['S', 'start']];

export default function Planning() {
  const [preset, setPreset] = useState('eight');
  const [rows, setRows] = useState(PRESETS.eight.rows);
  const [gamma, setGamma] = useState(PRESETS.eight.gamma);
  const [slip, setSlip] = useState(PRESETS.eight.slip);
  const [step, setStep] = useState(PRESETS.eight.step);
  const [brush, setBrush] = useState(null);
  const rewards = PRESETS[preset].rewards;
  const env = useMemo(() => makeGrid({ rows, step, slip, gamma, rewards }), [rows, step, slip, gamma, rewards]);
  const [V, setV] = useState(() => new Float64Array(env.N));
  const [sweeps, setSweeps] = useState(0);
  const [deltas, setDeltas] = useState([]);
  const [on, setOn] = useState(false);
  const [hover, setHover] = useState(null);
  const [view, setView] = useState('2d');
  const [pi, setPi] = useState(null);
  const reset = () => { setV(new Float64Array(env.N)); setSweeps(0); setDeltas([]); setOn(false); setPi(null); };
  // any change to the world restarts the planner
  React.useEffect(reset, [env]); // eslint-disable-line
  const sweep = () => {
    const r = viSweep(env, V, gamma);
    setV(r.V); setSweeps((k) => k + 1); setDeltas((d) => [...d, r.delta]);
    const more = r.delta > 1e-6 && sweeps < 400;
    if (!more) setOn(false);
    return more;
  };
  useLoop(sweep, on, { every: 6 });
  const policy = useMemo(() => greedy(env, V, gamma), [env, V, gamma]);
  const converged = deltas.length > 0 && deltas.at(-1) < 1e-6;
  const path = useMemo(() => (sweeps ? rollout(env, policy) : null), [env, policy, sweeps]);
  const loadPreset = (k) => { setPreset(k); setRows(PRESETS[k].rows); setGamma(PRESETS[k].gamma); setSlip(PRESETS[k].slip); setStep(PRESETS[k].step); };
  const paint = (s) => {
    if (!brush) return;
    const r = Math.floor(s / env.W), c = s % env.W;
    setRows(rows.map((row, i) => {
      let line = row;
      if (brush === 'S') line = line.replace('S', '.');
      return i === r ? line.slice(0, c) + brush + line.slice(c + 1) : line;
    }));
  };
  const runPI = () => { const r = policyIteration(env, { gamma }); setPi(r); };
  const terms = hover != null && !env.isTerminal(hover) && !env.isWall(hover) ? [0, 1, 2, 3].map((a) => ({ a, q: qFromV(env, V, hover, a, gamma), t: backupTerms(env, V, hover, a, gamma) })) : null;
  const bestA = terms ? terms.reduce((m, x) => (x.q > m.q ? x : m)) : null;

  return (
    <div className="grid gap-4 xl:grid-cols-[minmax(0,1.15fr)_minmax(0,1fr)]">
      <Card title="Value iteration on a world you can edit">
        <div className="flex flex-wrap items-center gap-1">
          <Select label="world" value={preset} set={loadPreset} options={Object.entries(PRESETS).filter(([k]) => k !== 'corridor').map(([k, p]) => [k, p.label])} />
        </div>
        <div className="flex flex-wrap items-center gap-1 text-[11px] text-dim">
          paint:
          {PAINT.map(([c, t]) => <Btn key={c} on={brush === c} onClick={() => setBrush(brush === c ? null : c)}>{t}</Btn>)}
          <span className="ml-1">{brush ? 'click or drag on the grid' : 'hover a cell to see its backup'}</span>
        </div>
        <Slider label="discount gamma" value={gamma} set={setGamma} min={0} max={1} step={0.01} fmt={(v) => v.toFixed(2)} />
        <Slider label="slip chance" value={slip} set={setSlip} min={0} max={0.8} step={0.05} fmt={(v) => v.toFixed(2)} title="probability the move goes sideways instead" />
        <Slider label="step reward" value={step} set={setStep} min={-1} max={0.2} step={0.01} fmt={(v) => v.toFixed(2)} />
        <div className="flex flex-wrap items-center gap-1">
          <Btn onClick={sweep}>One sweep</Btn>
          <Btn on={on} onClick={() => setOn(!on)}>{on ? 'Pause' : 'Sweep to convergence'}</Btn>
          <Btn onClick={reset}>Reset V = 0</Btn>
          <Btn on={view === '3d'} onClick={() => setView(view === '3d' ? '2d' : '3d')}>{view === '3d' ? '2D view' : '3D terrain'}</Btn>
          <span className="text-[11px] tabular-nums text-dim">sweep {sweeps}{deltas.length ? `, largest change ${deltas.at(-1).toExponential(1)}` : ''}{converged ? ', converged' : ''}</span>
        </div>
        {view === '2d' ? (
          <GridView env={env} values={V} policy={sweeps ? policy : null} onHover={setHover} onPaint={brush ? paint : null} mark={hover} paths={path ? [Object.assign(path.path, { dash: '4 3' })] : []} size={Math.min(56, Math.floor(520 / env.W))} />
        ) : (
          <Terrain env={env} V={V} policy={sweeps ? policy : null} agent={env.start} />
        )}
        {path?.loops && <p className="text-[11px] text-warm">The greedy policy circles instead of ending. With γ close to 1 and no step cost, staying put can be worth as much as finishing.</p>}
        {sweeps > 0 && gamma < 1 && converged && (
          <p className="text-[11px] text-slate-400">Shade is V*(s); arrows are the greedy policy; the dashed line follows it from the start, ignoring slips.</p>
        )}
      </Card>

      <div className="space-y-4">
        <Card title={hover != null && terms ? `Bellman backup at cell (${env.rc(hover).join(', ')})` : 'The Bellman backup'}>
          <Eq note="value iteration: one backup per state per sweep, then extract the greedy policy">V(s) ← maxₐ Σ_s′ p(s′ | s, a) [ r + γ V(s′) ]</Eq>
          {terms ? (
            <div className="space-y-1 text-[11px] tabular-nums text-slate-300">
              {terms.map(({ a, q, t }) => (
                <div key={a} className={`rounded px-1.5 py-0.5 ${a === bestA.a ? 'bg-glow/10' : ''}`}>
                  <span className="inline-block w-14">{ARROWS[a]} {ACTIONS[a]}</span>
                  {t.map((x, i) => <span key={i} className="mr-2 text-dim">{x.p.toFixed(2)}×({x.r.toFixed(2)} + {gamma.toFixed(2)}×{x.v.toFixed(2)})</span>)}
                  <span className={a === bestA.a ? 'text-glow' : ''}>= {q.toFixed(3)}</span>
                </div>
              ))}
              <div className="text-dim">The new V is the largest line, {bestA.q.toFixed(3)}. With slip, each action's line mixes three landing cells.</div>
            </div>
          ) : <p className="text-xs text-slate-400">Hover a cell on the grid to see the four sums that compete for its value.</p>}
        </Card>
        <Card title="How fast it settles">
          {deltas.length > 1 ? (
            <Chart series={[{ label: 'largest change in V per sweep (log10)', y: deltas.map((d) => Math.log10(Math.max(d, 1e-12))) }]} xLabel="sweep" yLabel="log10 change" fmt={(v) => v.toFixed(1)} />
          ) : <p className="text-xs text-slate-400">Run a few sweeps to see the change shrink. With γ &lt; 1 each sweep shrinks the error by at least a factor γ, so smaller γ settles faster.</p>}
        </Card>
        <Card title="Policy iteration">
          <p className="text-xs text-slate-400">
            Evaluate the current policy exactly (repeat the backup without the max), then make it greedy; repeat until nothing changes.<Cite id="howard1960" /> It usually needs
            only a handful of improvements, each one expensive.
          </p>
          <Btn onClick={runPI}>Run policy iteration</Btn>
          {pi && (
            <div className="text-[11px] tabular-nums text-slate-300">
              {pi.steps.map((s, i) => <div key={i}>improvement {i + 1}: evaluation took {s.sweeps} sweeps; {s.changed} states changed action</div>)}
              <div className="text-glow">Done after {pi.steps.length} improvements.</div>
            </div>
          )}
        </Card>
        <ThoughtExperiment env={env} gamma={gamma} />
      </div>
    </div>
  );
}

// Lecture 6's thought experiment: start from a fixed policy, evaluate it, act greedily on its Q.
function ThoughtExperiment({ env, gamma }) {
  const [k, setK] = useState(0);
  const pis = useMemo(() => {
    const live = env.states.filter((s) => !env.isTerminal(s));
    let pi = new Int8Array(env.N).fill(-1);
    live.forEach((s) => (pi[s] = 1));
    const out = [];
    for (let i = 0; i < 6; i++) {
      const { V } = evaluate(env, pi, { gamma: Math.min(gamma, 0.99), maxSweeps: 3000 });
      out.push({ pi: Int8Array.from(pi), V });
      pi = greedy(env, V, Math.min(gamma, 0.99));
    }
    return out;
  }, [env, gamma]);
  const cur = pis[Math.min(k, pis.length - 1)];
  return (
    <Card title="Is acting greedily on Q^π better than π?">
      <p className="text-xs text-slate-400">
        Lecture 6 starts from a policy that always goes right, asks what happens if you pick argmaxₐ <Term k="Q-function">Q^π(s, a)</Term> instead, and whether the
        result is optimal.<Cite id="cs224r" /> Step through: each step evaluates the current policy, then switches to the greedy one.
      </p>
      <div className="flex items-center gap-1">
        <Btn disabled={k === 0} onClick={() => setK(k - 1)}>Back</Btn>
        <Btn disabled={k >= pis.length - 1} onClick={() => setK(k + 1)}>Improve once</Btn>
        <span className="text-[11px] tabular-nums text-dim">policy {k}: V(start) = {cur.V[env.start].toFixed(3)}</span>
      </div>
      <GridView env={env} values={cur.V} policy={cur.pi} size={Math.min(34, Math.floor(330 / env.W))} showValues={false} />
      <p className="text-[11px] text-slate-400">
        The value at every state never goes down from one step to the next (the policy improvement theorem), and the first greedy step already helps a lot.
        It is not optimal after one step in general; repeating it is policy iteration.
      </p>
      <NotShown>Learning the model p(s′ | s, a) from data before planning (model-based RL, lecture 11).</NotShown>
    </Card>
  );
}
