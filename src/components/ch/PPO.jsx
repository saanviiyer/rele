// Chapter 7 (CS224R lecture 5; the notes, section 4): PPO. The clipped surrogate, what it does
// to the gradient, and PPO on CartPole against plain REINFORCE.
import React, { useEffect, useMemo, useState } from 'react';
import { Card, Btn, Slider, Term, Cite, Eq, NotShown, Stat } from '../ui.jsx';
import Chart, { bandOf } from '../Chart.jsx';
import { clippedObjective, makeTrainer, pgCurves } from '../../lib/pg.js';
import { useLoop, later } from '../../lib/useLoop.js';

function ClipExplorer() {
  const [eps, setEps] = useState(0.2);
  const [ratio, setRatio] = useState(1.3);
  const W = 400, H = 170, P = 30;
  const X = (r) => P + ((r - 0) / 2) * (W - 2 * P), Y = (v) => H / 2 - v * 32;
  const curve = (A) => Array.from({ length: 201 }, (_, i) => { const r = i / 100; return `${X(r)},${Y(clippedObjective(r, A, eps))}`; }).join(' ');
  const flat = (A, r) => (A >= 0 ? r > 1 + eps : r < 1 - eps);
  const cases = [[1.5, 1], [0.5, 1], [0.5, -1], [1.5, -1]];
  return (
    <Card title="The clipped surrogate, as a function of the ratio">
      <Eq note="the final PPO surrogate (lecture 5, trick 2; the notes' L^CLIP)">L(θ) = E [ min( ρₜ Âₜ , clip(ρₜ, 1−ε, 1+ε) Âₜ ) ],   ρₜ = π_θ(aₜ | sₜ) / π_θ_old(aₜ | sₜ)</Eq>
      <Slider label="ε" value={eps} set={setEps} min={0.05} max={0.5} step={0.01} fmt={(v) => v.toFixed(2)} />
      <Slider label="ratio ρ" value={ratio} set={setRatio} min={0} max={2} step={0.01} fmt={(v) => v.toFixed(2)} />
      <svg viewBox={`0 0 ${W} ${H}`} className="w-full">
        <line x1={P} x2={W - P} y1={Y(0)} y2={Y(0)} stroke="var(--color-line)" />
        <rect x={X(1 - eps)} y="0" width={X(1 + eps) - X(1 - eps)} height={H} fill="var(--color-glow)" opacity="0.07" />
        <line x1={X(1)} x2={X(1)} y1="0" y2={H} stroke="var(--color-dim)" strokeDasharray="3 3" />
        <polyline points={curve(1)} fill="none" stroke="var(--color-warm)" strokeWidth="2.2" />
        <polyline points={curve(-1)} fill="none" stroke="var(--color-cool)" strokeWidth="2.2" />
        {[1, -1].map((A) => <circle key={A} cx={X(ratio)} cy={Y(clippedObjective(ratio, A, eps))} r="5" fill={A > 0 ? 'var(--color-warm)' : 'var(--color-cool)'} stroke="var(--color-panel)" />)}
        {[0, 0.5, 1, 1.5, 2].map((r) => <text key={r} x={X(r)} y={H - 4} fontSize="9" textAnchor="middle" fill="var(--color-dim)">{r}</text>)}
        <text x={W - P} y={Y(1.6)} fontSize="10" textAnchor="end" fill="var(--color-warm)">Â = +1</text>
        <text x={W - P} y={Y(-1.6) + 8} fontSize="10" textAnchor="end" fill="var(--color-cool)">Â = −1</text>
      </svg>
      <div className="text-[11px] text-slate-300">
        At ρ = {ratio.toFixed(2)}: with Â = +1 the objective is {clippedObjective(ratio, 1, eps).toFixed(2)} and its gradient is {flat(1, ratio) ? <b className="text-warm">zero (clipped)</b> : 'live'}; with Â = −1 it is {clippedObjective(ratio, -1, eps).toFixed(2)}, gradient {flat(-1, ratio) ? <b className="text-cool">zero (clipped)</b> : 'live'}.
      </div>
      <div className="grid grid-cols-2 gap-1 text-[10.5px] tabular-nums text-slate-300">
        {cases.map(([r, A]) => <div key={`${r}${A}`} className="rounded border border-line px-1.5 py-0.5">ρ {r}, Â {A > 0 ? '+1' : '−1'}: {clippedObjective(r, A, eps).toFixed(2)}, {flat(A, r) ? 'no gradient' : 'gradient flows'}</div>)}
      </div>
      <p className="text-[11px] text-slate-400">
        A good action (Â &gt; 0) earns nothing for pushing its probability ratio past 1 + ε, and a bad one earns nothing for pushing past 1 − ε. The min
        keeps the pessimistic term, so a step that made things worse is still corrected: at ρ = 1.5 with Â = −1 the gradient still flows.
      </p>
    </Card>
  );
}

function LivePPO() {
  const [eps, setEps] = useState(0.2);
  const [epochs, setEpochs] = useState(4);
  const [seed, setSeed] = useState(1);
  const T = useMemo(() => makeTrainer({ variant: 'ppo', clip: eps, epochs, seed }), [eps, epochs, seed]);
  const [hist, setHist] = useState([]);
  const [on, setOn] = useState(false);
  useEffect(() => { setHist([]); setOn(false); }, [T]);
  useLoop(() => { T.iterate(); setHist([...T.history]); if (T.history.length >= 60) { setOn(false); return false; } return true; }, on, { every: 2 });
  return (
    <Card title="PPO on CartPole, live">
      <p className="text-xs text-slate-400">
        Same network and batch (8 episodes) as chapter 5, plus a critic for GAE advantages. Each batch is reused for several epochs; the clip fraction is the
        share of samples whose gradient the clip switched off in the last epoch.<Cite id="schulman2017" />
      </p>
      <Slider label="ε" value={eps} set={setEps} min={0.05} max={0.5} step={0.05} fmt={(v) => v.toFixed(2)} />
      <Slider label="epochs per batch" value={epochs} set={setEpochs} min={1} max={10} step={1} />
      <div className="flex items-center gap-1">
        <Btn on={on} onClick={() => setOn(!on)}>{on ? 'Pause' : hist.length ? 'Continue' : 'Train 60 iterations'}</Btn>
        <Btn onClick={() => setSeed(seed + 1)}>New seed</Btn>
        {hist.length > 0 && <Stat label="episode length" value={hist.at(-1).mean.toFixed(0)} />}
        {hist.length > 0 && <Stat label="clip fraction" value={`${(100 * hist.at(-1).clipFrac).toFixed(0)}%`} tone="warm" />}
      </div>
      {hist.length > 1 && (
        <>
          <Chart series={[{ label: 'mean episode length', y: hist.map((h) => h.mean), band: [hist.map((h) => h.min), hist.map((h) => h.max)] }]} xLabel="iteration" yLabel="steps" yMin={0} yMax={500} height={130} fmt={(v) => v.toFixed(0)} />
          <Chart series={[{ label: 'clip fraction', y: hist.map((h) => 100 * h.clipFrac), color: 'var(--color-warm)' }]} xLabel="iteration" yLabel="% clipped" yMin={0} height={100} fmt={(v) => `${v.toFixed(0)}%`} />
        </>
      )}
    </Card>
  );
}

function Versus() {
  const [res, setRes] = useState(null);
  const [busy, setBusy] = useState(false);
  const run = async () => {
    setBusy(true);
    const out = await later(() => [['rtg', 'REINFORCE, reward-to-go'], ['ppo', 'PPO (4 epochs, ε 0.2)']].map(([v, label]) => ({ label, ...bandOf(pgCurves(v, { seeds: 3, iters: 50 })) })));
    setRes(out); setBusy(false);
  };
  return (
    <Card title="PPO against vanilla policy gradient">
      <p className="text-xs text-slate-400">Both collect exactly the same amount of data per iteration; PPO squeezes more gradient steps out of it.</p>
      <Btn on onClick={run} disabled={busy}>{busy ? 'Training 6 policies...' : 'Compare (three seeds each)'}</Btn>
      {res && <Chart series={res.map((r) => ({ label: r.label, y: r.y, band: r.band }))} xLabel="iteration (8 episodes each)" yLabel="mean episode length" yMin={0} yMax={500} fmt={(v) => v.toFixed(0)} />}
      <p className="text-[11px] text-slate-400">
        Lecture 5's example settings: about 2000 timesteps per batch, about 10 epochs, minibatches of 64, so about {Math.round((2000 * 10) / 64)} gradient steps per
        batch; ε = 0.2.<Cite id="cs224r" /> This page uses far smaller batches so it runs in a second.
      </p>
    </Card>
  );
}

export default function PPO() {
  return (
    <div className="grid gap-4 xl:grid-cols-2">
      <ClipExplorer />
      <LivePPO />
      <Versus />
      <Card title="Why not just take many steps on one batch?">
        <p className="text-xs leading-relaxed text-slate-300">
          After a few steps the policy is "incentivized to differ significantly from the old policy" (lecture 5), and the advantages were computed for the old one.
          Two fixes appear in the lecture: constrain the KL divergence between new and old policy (TRPO; it returns in chapter 11 as the KL penalty of RLHF),
          or bound the importance weight itself, which is PPO.<Cite id="schulman2015" />
        </p>
        <Eq note="the notes' full objective: clipped policy loss, value loss, entropy bonus">L_PPO = L^CLIP − c₁ L^VF + c₂ H[π_θ]</Eq>
        <p className="text-[11px] text-slate-400">
          The lecture's guidelines: PPO is on-policy in spirit, stable and "plug & play", and data-inefficient; off-policy methods with replay buffers (SAC) are
          more data-efficient and harder to tune.<Cite id="cs224r" />
        </p>
        <NotShown>SAC and the reparameterization trick for continuous actions.</NotShown>
      </Card>
    </div>
  );
}
