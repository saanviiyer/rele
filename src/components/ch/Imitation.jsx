// Chapter 4 (CS224R lecture 2): imitation learning. Multimodal demonstrations break an l2
// policy; a cloned policy compounds its errors off the expert's states; DAgger fixes that.
import React, { useMemo, useState } from 'react';
import { Card, Btn, Slider, Term, Cite, Eq, NotShown, Stat } from '../ui.jsx';
import GridView from '../GridView.jsx';
import Chart from '../Chart.jsx';
import { carDemos, carFit, carRollout, CAR, dagger, bcMatched, expertOf, evaluatePolicy } from '../../lib/imitation.js';
import { gridOf } from '../../lib/grid.js';
import { rng } from '../../lib/rng.js';

function CarPlot({ demos, rolls, kind }) {
  const W = 420, H = 170, X = (x) => 10 + (x / CAR.xEnd) * (W - 20), Y = (y) => H / 2 - y * 40;
  const o = CAR.obs;
  return (
    <svg viewBox={`0 0 ${W} ${H}`} className="w-full">
      <rect x="0" y="0" width={W} height={H} fill="var(--color-panel)" />
      <rect x={X(o.x0)} y={Y(o.half)} width={X(o.x1) - X(o.x0)} height={Y(-o.half) - Y(o.half)} fill="var(--color-dim)" opacity="0.5" rx="3" />
      <text x={X((o.x0 + o.x1) / 2)} y={H / 2 + 3} fontSize="9" textAnchor="middle" fill="var(--color-white)">obstacle</text>
      {demos.map((d, i) => <polyline key={`d${i}`} points={d.pts.map((p) => `${X(p.x)},${Y(p.y)}`).join(' ')} fill="none" stroke={d.side > 0 ? 'var(--color-cool)' : 'var(--color-glow)'} strokeWidth="1" opacity="0.35" />)}
      {rolls.map((r, i) => (
        <g key={`r${i}`}>
          <polyline points={r.pts.map((p) => `${X(p.x)},${Y(p.y)}`).join(' ')} fill="none" stroke="var(--color-warm)" strokeWidth="1.6" opacity="0.8" />
          {r.crashed && <text x={X(r.pts.at(-1).x)} y={Y(r.pts.at(-1).y) + 4} fontSize="11" textAnchor="middle" fill="var(--color-warm)">✕</text>}
        </g>
      ))}
      <text x="8" y="14" fontSize="10" fill="var(--color-dim)">{kind === 'l2' ? 'l2 policy: predicts the mean action' : 'categorical policy: samples one of the demonstrated actions'}</text>
    </svg>
  );
}

function Multimodal() {
  const [n, setN] = useState(20);
  const [seed, setSeed] = useState(3);
  const demos = useMemo(() => carDemos(n, seed), [n, seed]);
  const table = useMemo(() => carFit(demos), [demos]);
  const rolls = (kind) => { const r = rng(seed + 50 + (kind === 'l2' ? 0 : 1)); return Array.from({ length: 12 }, () => carRollout(table, kind, r)); };
  const l2 = useMemo(() => rolls('l2'), [table]); // eslint-disable-line
  const cat = useMemo(() => rolls('cat'), [table]); // eslint-disable-line
  const crash = (rs) => rs.filter((r) => r.crashed).length;
  return (
    <Card title="Two right answers, one wrong average">
      <p className="text-xs text-slate-400">
        Half the drivers pass the obstacle above (blue), half below (teal). Lecture 2's point: an l2-trained policy regresses to the mean of the data, which
        happens "all the time" when several people collect it, and here the mean is straight ahead.<Cite id="cs224r" />
      </p>
      <Eq note="version 0 (l2) and version 1 (maximum likelihood with an expressive distribution)">min_θ Σ ||a − π_θ(s)||²   vs   min_θ −E log π_θ(a | s)</Eq>
      <CarPlot demos={demos} rolls={l2} kind="l2" />
      <CarPlot demos={demos} rolls={cat} kind="cat" />
      <div className="flex flex-wrap items-center gap-2">
        <Stat label="l2 policy crashes" value={`${crash(l2)} / ${l2.length}`} tone="warm" />
        <Stat label="categorical policy crashes" value={`${crash(cat)} / ${cat.length}`} />
        <Btn onClick={() => setSeed(seed + 1)}>New demonstrations</Btn>
      </div>
      <Slider label="demonstrations" value={n} set={setN} min={2} max={60} step={2} />
      <p className="text-[11px] text-slate-400">
        Both policies are lookup tables over position; the only difference is the output distribution. Network expressivity and distribution expressivity are
        different things, the lecture stresses: a huge network with an l2 loss still outputs the mean.
      </p>
      <NotShown>The expressive policy classes the lecture compares in practice: mixtures of Gaussians, discretized autoregressive actions, and diffusion or flow-matching policies, plus action chunking.</NotShown>
    </Card>
  );
}

function Dagger() {
  const [slip, setSlip] = useState(0.2);
  const [demosN, setDemosN] = useState(3);
  const [k, setK] = useState(0);
  const env = useMemo(() => gridOf('corridor', { slip }), [slip]);
  const run = useMemo(() => dagger(env, { demosN, iters: 6, rolloutsPerIter: 3 }), [env, demosN]);
  const it = run.iters[Math.min(k, run.iters.length - 1)];
  const expert = useMemo(() => evaluatePolicy(env, (s) => run.expert[s]).success, [env, run]);
  const matched = useMemo(() => run.iters.map((x) => bcMatched(env, x.labels).success), [env, run]);
  const labelled = it.counts.map((c) => (c[0] + c[1] + c[2] + c[3] > 0 ? 1 : 0));
  const visits = Array.from(it.visits).map((v) => v / Math.max(...it.visits));
  return (
    <Card title="Compounding errors, and DAgger">
      <p className="text-xs text-slate-400">
        The expert walks straight down the middle row to the goal. The ice is slippery, so any walker is sometimes pushed sideways. A
        <Term k="behaviour cloning"> cloned policy</Term> only knows what to do in states the demos reached; elsewhere it guesses at random, drifts further, and
        eventually falls in a pit. That is <Term k="compounding errors" /> (covariate shift: p_expert(s) ≠ p_π(s)).<Cite id="cs224r" />
      </p>
      <Eq note="DAgger (Ross et al. 2011): label the learner's own states, aggregate, retrain">D ← D ∪ {'{'}(s′, a*) : s′ visited by π_θ, a* = π_expert(s′){'}'}</Eq>
      <Slider label="slip chance" value={slip} set={setSlip} min={0} max={0.4} step={0.05} fmt={(v) => v.toFixed(2)} />
      <Slider label="expert demos" value={demosN} set={setDemosN} min={1} max={10} step={1} />
      <div className="flex items-center gap-1">
        <span className="text-[11px] text-dim">DAgger iteration</span>
        {run.iters.map((x, i) => <Btn key={i} on={k === i} onClick={() => setK(i)}>{i === 0 ? 'BC' : i}</Btn>)}
      </div>
      <GridView env={env} values={visits} scale={1} q={null} policy={it.counts.map((c, s) => (labelled[s] ? c.indexOf(Math.max(...c)) : -1))} size={34} showValues={false} />
      <p className="text-[10px] text-dim">Shade: how often the learner visits each cell over 200 runs. Arrows: states with an expert label, and the action learned there; cells without an arrow have no data.</p>
      <div className="flex flex-wrap gap-2">
        <Stat label="expert success" value={`${(expert * 100).toFixed(0)}%`} tone="dim" />
        <Stat label={k === 0 ? 'behaviour cloning success' : `DAgger iteration ${k} success`} value={`${(it.success * 100).toFixed(0)}%`} tone="warm" />
        <Stat label="expert labels" value={it.labels} />
        <Stat label="states with a label" value={it.covered} />
      </div>
      <Chart series={[
        { label: 'DAgger', y: run.iters.map((x) => 100 * x.success) },
        { label: 'BC with the same number of expert labels', y: matched.map((v) => 100 * v), dash: '4 3' },
        { label: 'expert', y: run.iters.map(() => 100 * expert), color: 'var(--color-dim)', dash: '2 3' },
      ]} xLabel="iteration (labels grow along the axis)" yLabel="% reaching the goal" yMin={0} yMax={100} height={140} fmt={(v) => `${v.toFixed(0)}%`} />
      <p className="text-[11px] text-slate-400">
        Measured honestly: in this small table world, plain BC given as many expert labels as DAgger used does about as well, because extra demos also slip and
        cover the side rows. DAgger's advantage is that its labels land where the learner goes; with a learned function approximator and long horizons that
        matters much more. Its cost, per the lecture, is an expert who can be queried while the agent is in control.
      </p>
      <NotShown>HG-DAgger, where the expert takes over when the policy errs; and learning from human videos, limited by the embodiment gap.</NotShown>
    </Card>
  );
}

export default function Imitation() {
  return (
    <div className="grid gap-4">
      <Multimodal />
      <Dagger />
    </div>
  );
}
