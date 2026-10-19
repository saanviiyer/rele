// Chapter 6 (CS224R lecture 4; the notes, section 3): critics. Monte Carlo vs TD targets, the
// shared-state example, n-step returns, GAE, and actor-critic on CartPole.
import React, { useMemo, useState } from 'react';
import { Card, Btn, Slider, Term, Cite, Eq, NotShown, Stat } from '../ui.jsx';
import Chart, { bandOf } from '../Chart.jsx';
import { walkCurves, walkEpisode, tdUpdate, mcUpdate, WALK_TRUE, targetSpread } from '../../lib/td.js';
import { gae, nStepTargets, mcVsTdShared } from '../../lib/toys.js';
import { pgCurves } from '../../lib/pg.js';
import { rng } from '../../lib/rng.js';
import { useLoop, later } from '../../lib/useLoop.js';

function RandomWalk() {
  const [alphaTD, setAlphaTD] = useState(0.1);
  const [alphaMC, setAlphaMC] = useState(0.02);
  const curves = useMemo(() => ({ td: walkCurves({ method: 'td', alpha: alphaTD }), mc: walkCurves({ method: 'mc', alpha: alphaMC }) }), [alphaTD, alphaMC]);
  // one live run of each, episode by episode
  const [ep, setEp] = useState(0);
  const [on, setOn] = useState(false);
  const live = useMemo(() => ({ r: rng(42), td: [0, 0.5, 0.5, 0.5, 0.5, 0.5, 0], mc: [0, 0.5, 0.5, 0.5, 0.5, 0.5, 0] }), [alphaTD, alphaMC]); // eslint-disable-line
  useLoop(() => {
    const e = walkEpisode(live.r);
    tdUpdate(live.td, e, alphaTD); mcUpdate(live.mc, e, alphaMC);
    setEp((k) => k + 1);
    if (ep >= 99) { setOn(false); return false; }
    return true;
  }, on, { every: 6 });
  const X = (i) => 30 + i * 70, Y = (v) => 110 - v * 95;
  return (
    <Card title="Monte Carlo or temporal difference?">
      <p className="text-xs text-slate-400">
        Five states A to E in a row; start in C, step left or right at random, +1 for leaving on the right, 0 on the left. The true values are 1/6 to 5/6
        (Sutton & Barto, Example 6.2).<Cite id="sutton2018" /> Both methods start at 0.5 and see the same episodes.
      </p>
      <Eq note="Monte Carlo: wait for the whole return">V(sₜ) ← V(sₜ) + α [ Gₜ − V(sₜ) ]</Eq>
      <Eq note="TD(0): bootstrap from the next state's estimate">V(sₜ) ← V(sₜ) + α [ rₜ + γ V(sₜ₊₁) − V(sₜ) ]</Eq>
      <svg viewBox="0 0 400 125" className="w-full">
        <polyline points={WALK_TRUE.map((v, i) => `${X(i)},${Y(v)}`).join(' ')} fill="none" stroke="var(--color-dim)" strokeDasharray="4 3" />
        <polyline points={[1, 2, 3, 4, 5].map((s, i) => `${X(i)},${Y(live.td[s])}`).join(' ')} fill="none" stroke="var(--color-glow)" strokeWidth="2" />
        <polyline points={[1, 2, 3, 4, 5].map((s, i) => `${X(i)},${Y(live.mc[s])}`).join(' ')} fill="none" stroke="var(--color-warm)" strokeWidth="2" />
        {'ABCDE'.split('').map((c, i) => <text key={c} x={X(i)} y="122" fontSize="10" textAnchor="middle" fill="var(--color-dim)">{c}</text>)}
      </svg>
      <div className="flex items-center gap-2 text-[11px] text-dim">
        <Btn on={on} onClick={() => { if (ep >= 100) { setEp(0); live.td.splice(0, 7, 0, 0.5, 0.5, 0.5, 0.5, 0.5, 0); live.mc.splice(0, 7, 0, 0.5, 0.5, 0.5, 0.5, 0.5, 0); } setOn(!on); }}>{on ? 'Pause' : 'Play 100 episodes'}</Btn>
        episode {ep}; <span className="text-glow">TD</span> <span className="text-warm">MC</span> <span>dashed: truth</span>
      </div>
      <Slider label="TD step size" value={alphaTD} set={setAlphaTD} min={0.01} max={0.3} step={0.01} />
      <Slider label="MC step size" value={alphaMC} set={setAlphaMC} min={0.01} max={0.1} step={0.01} />
      <Chart series={[{ label: `TD(0), α = ${alphaTD}`, y: curves.td }, { label: `Monte Carlo, α = ${alphaMC}`, y: curves.mc }]} xLabel="episode" yLabel="RMS error vs truth" yMin={0} height={140} fmt={(v) => v.toFixed(3)} />
      <p className="text-[11px] text-slate-400">Averaged over 100 runs. On this problem TD reaches lower error for any step size here; MC's targets (0 or 1) are far noisier.</p>
    </Card>
  );
}

function Spread() {
  const V = useMemo(() => [0, ...WALK_TRUE.map((v, i) => v + [0.15, -0.1, 0.12, -0.08, 0.1][i]), 0], []);
  const s = useMemo(() => targetSpread(V, 3), [V]);
  return (
    <Card title="Bias against variance, in one state">
      <p className="text-xs text-slate-400">From C, with a critic whose estimates are a little off, here are the two targets over 2000 visits:</p>
      <div className="flex flex-wrap gap-2">
        <Stat label="truth V(C)" value={s.truth.toFixed(3)} tone="dim" />
        <Stat label="MC target: mean, spread" value={`${s.mc.mean.toFixed(3)}, ${s.mc.sd.toFixed(3)}`} tone="warm" />
        <Stat label="TD target: mean, spread" value={`${s.td.mean.toFixed(3)}, ${s.td.sd.toFixed(3)}`} />
      </div>
      <p className="text-[11px] text-slate-400">
        The Monte Carlo target is right on average and swings between 0 and 1. The TD target barely moves, but its average is wrong by exactly as much as the
        critic's estimates of B and D are. The notes summarise it as MC unbiased with high variance, TD biased with low variance.<Cite id="nalawade" />
      </p>
    </Card>
  );
}

function Predictor() {
  const episodes = [
    { states: ['A', 'B'], rewards: [0, 0] },
    ...Array.from({ length: 6 }, () => ({ states: ['B'], rewards: [1] })),
    { states: ['B'], rewards: [0] },
  ];
  const r = mcVsTdShared({ episodes });
  return (
    <Card title="Eight episodes, two answers for V(A)">
      <p className="text-xs text-slate-400">
        Lecture 4 poses a think-pair-share with this shape; the version below is Sutton & Barto's Example 6.4. Logged: A, 0, B, 0, end; then six times B, 1, end;
        then B, 0, end.<Cite id="sutton2018" />
      </p>
      <div className="flex flex-wrap gap-2">
        <Stat label="V(B), both methods" value={r.mc.B.toFixed(2)} />
        <Stat label="V(A), Monte Carlo" value={r.mc.A.toFixed(2)} tone="warm" />
        <Stat label="V(A), batch TD" value={r.td.A.toFixed(2)} />
      </div>
      <p className="text-[11px] text-slate-400">
        Monte Carlo only remembers that A's one episode returned 0. TD says A leads to B, and B is worth 0.75 from everything seen there. TD shares information
        across trajectories through the states they have in common; offline RL (chapter 9) depends on exactly that.
      </p>
    </Card>
  );
}

function GAECalc() {
  const [lambda, setLambda] = useState(0.95);
  const [gamma, setGamma] = useState(0.99);
  const [n, setN] = useState(1);
  const r = [0, 0, 1, 0, 2], V = [0.5, 0.4, 0.9, 0.3, 1.1];
  const g = gae(r, V, { gamma, lambda });
  const ys = nStepTargets(r, V, n, gamma);
  const cell = 'w-14 text-right tabular-nums';
  return (
    <Card title="GAE and n-step targets on one trajectory">
      <Eq note="the TD error, and generalized advantage estimation (Schulman et al. 2016)">δₜ = rₜ + γ V(sₜ₊₁) − V(sₜ)      Âₜ = Σₗ (γλ)ˡ δₜ₊ₗ</Eq>
      <Slider label="λ" value={lambda} set={setLambda} min={0} max={1} step={0.05} fmt={(v) => v.toFixed(2)} />
      <Slider label="γ" value={gamma} set={setGamma} min={0.5} max={1} step={0.01} fmt={(v) => v.toFixed(2)} />
      <Slider label="n (n-step)" value={n} set={setN} min={1} max={5} step={1} />
      <div className="overflow-x-auto text-[11px] text-slate-300">
        {[['t', [1, 2, 3, 4, 5]], ['rₜ', r], ['V(sₜ)', V], ['δₜ', g.delta], ['Âₜ (GAE)', g.A], ['Gₜ − V(sₜ)', g.mcAdv], [`n-step target`, ys]].map(([lab, row]) => (
          <div key={lab} className="flex"><span className="w-24 text-dim">{lab}</span>{row.map((v, i) => <span key={i} className={`${cell} ${lab.startsWith('Â') ? 'text-glow' : ''}`}>{typeof v === 'number' ? (lab === 't' ? v : v.toFixed(3)) : v}</span>)}</div>
        ))}
      </div>
      <p className="text-[11px] text-slate-400">
        At λ = 0 the GAE row equals δₜ (pure TD); at λ = 1 it equals Gₜ − V(sₜ) (Monte Carlo). The n-step target is TD at n = 1 and the full return at n = 5. The
        lecture notes that n between 1 and T often works best.<Cite id="cs224r" /> The episode ends after t = 5.
      </p>
    </Card>
  );
}

function LambdaSweep() {
  const [res, setRes] = useState(null);
  const [busy, setBusy] = useState(false);
  const run = async () => {
    setBusy(true);
    const out = await later(() => [0, 0.5, 0.95, 1].map((lambda) => ({ lambda, ...bandOf(pgCurves('gae', { seeds: 3, iters: 50, lambda })) })));
    setRes(out); setBusy(false);
  };
  return (
    <Card title="Actor-critic on CartPole: sweep λ">
      <p className="text-xs text-slate-400">
        The same CartPole policy as chapter 5, now with a critic V_φ (a 4-16-1 network fitted to the returns) and GAE advantages. λ trades the critic's bias
        against the returns' noise.<Cite id="schulman2016" />
      </p>
      <Eq note="the actor-critic loop of lecture 4">sample → fit V_φ → Â = r + γ V_φ(s′) − V_φ(s) (or GAE) → θ ← θ + α Σ ∇ log π_θ(a | s) Â</Eq>
      <Btn on onClick={run} disabled={busy}>{busy ? 'Training 12 actor-critics...' : 'Run λ = 0, 0.5, 0.95, 1 (three seeds each)'}</Btn>
      {res && <Chart series={res.map((r) => ({ label: `λ = ${r.lambda}`, y: r.y, band: r.band }))} xLabel="iteration" yLabel="mean episode length" yMin={0} yMax={500} fmt={(v) => v.toFixed(0)} />}
      <p className="text-[11px] text-slate-400">Read the bands, not just the lines: with three seeds, differences between neighbouring λ values are within noise. λ = 0 leans entirely on a critic that starts out wrong.</p>
      <NotShown>The off-policy actor-critic of lecture 4's second half (a replay buffer with a Q-critic, as in SAC).</NotShown>
    </Card>
  );
}

export default function ActorCritic() {
  return (
    <div className="grid gap-4 xl:grid-cols-2">
      <RandomWalk />
      <div className="space-y-4"><Spread /><Predictor /></div>
      <GAECalc />
      <LambdaSweep />
    </div>
  );
}
