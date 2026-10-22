// Chapter 5 (CS224R lecture 3; the notes, section 3): the policy gradient, what it pushes on,
// causality and baselines, REINFORCE on CartPole, and why importance weights blow up.
import React, { useEffect, useMemo, useRef, useState } from 'react';
import { Card, Btn, Slider, Term, Cite, Eq, NotShown, Stat, Select } from '../ui.jsx';
import Chart, { bandOf } from '../Chart.jsx';
import CartPoleView from '../CartPoleView.jsx';
import { HUMANOID, pgWeights, importanceSpread } from '../../lib/toys.js';
import { makeTrainer, pgCurves, rolloutCP, VARIANTS } from '../../lib/pg.js';
import { cpReset, cpStep, CP } from '../../lib/cartpole.js';
import { rng } from '../../lib/rng.js';
import { useLoop, later } from '../../lib/useLoop.js';

function Pushes() {
  const [set, setSet] = useState('set1');
  const [causality, setCausality] = useState(false);
  const [baseline, setBaseline] = useState(false);
  const trajs = HUMANOID[set];
  const W = pgWeights(trajs, { causality, baseline });
  const m = Math.max(1e-9, ...W.flat().map(Math.abs));
  return (
    <Card title="Which way does the gradient push?">
      <p className="text-xs text-slate-400">
        Lecture 3's simulated humanoid, rewarded for forward velocity. The policy gradient raises the log-probability of every action in a trajectory, in
        proportion to the weight next to it: "do more of the good stuff, less of the bad stuff".<Cite id="cs224r" /> The per-step rewards are illustrative; the
        slides show videos, and the signs follow their descriptions.
      </p>
      <Eq note="the weight on ∇ log π(aₜ | sₜ): R(τ), or the reward-to-go, minus an optional baseline b">∇_θ J ≈ (1/N) Σᵢ Σₜ ∇_θ log π_θ(aᵢ,ₜ | sᵢ,ₜ) ( Σ_t′≥t r(sᵢ,t′, aᵢ,t′) − b )</Eq>
      <div className="flex flex-wrap gap-1">
        <Btn on={set === 'set1'} onClick={() => setSet('set1')}>Set 1</Btn>
        <Btn on={set === 'set2'} onClick={() => setSet('set2')}>Set 2 (all returns positive)</Btn>
        <Btn on={causality} onClick={() => setCausality(!causality)}>causality (reward-to-go)</Btn>
        <Btn on={baseline} onClick={() => setBaseline(!baseline)}>subtract a baseline</Btn>
      </div>
      <div className="space-y-1.5">
        {trajs.map((tr, i) => (
          <div key={i} className="text-[11px] text-slate-300">
            <div className="flex justify-between"><span>{tr.label}</span><span className="tabular-nums text-dim">R(τ) = {tr.r.reduce((a, b) => a + b, 0).toFixed(1)}</span></div>
            <div className="flex gap-1">
              {W[i].map((w, t) => (
                <div key={t} className="flex h-7 flex-1 items-center justify-center rounded text-[10px] tabular-nums" style={{ background: w >= 0 ? `color-mix(in srgb, var(--color-warm) ${Math.round((Math.abs(w) / m) * 80)}%, transparent)` : `color-mix(in srgb, var(--color-cool) ${Math.round((Math.abs(w) / m) * 80)}%, transparent)` }}>
                  t{t + 1}: r {tr.r[t].toFixed(1)} → {w >= 0 ? '+' : ''}{w.toFixed(2)}
                </div>
              ))}
            </div>
          </div>
        ))}
      </div>
      <ul className="list-disc space-y-0.5 pl-4 text-[11px] text-slate-300">
        <li>Set 1 without causality: falling forwards wins, so the gradient teaches falling forward. The big backward step still drags down the forward steps after it.</li>
        <li>Causality: an action at time t cannot change earlier rewards, so weight it by the reward-to-go. The forward steps at t2 to t4 of the last trajectory turn positive.</li>
        <li>Set 2 without a baseline: every weight is positive, so stumbling gets reinforced too, just less. Subtracting the average makes below-average behaviour negative. It does not change the expected gradient.</li>
      </ul>
    </Card>
  );
}

function CartPole() {
  const [variant, setVariant] = useState('rtg');
  const [seed, setSeed] = useState(1);
  const T = useMemo(() => makeTrainer({ variant, seed }), [variant, seed]);
  const [hist, setHist] = useState([]);
  const [on, setOn] = useState(false);
  const [show, setShow] = useState(null);
  const frame = useRef(0);
  useEffect(() => { setHist([]); setOn(false); }, [T]);
  useLoop(() => {
    const h = T.iterate();
    setHist([...T.history]);
    // replay one episode of the current policy every few iterations
    if (h.iter % 3 === 1) setShow(rolloutCP(T.policy, rng(h.iter), { maxSteps: CP.maxSteps }).states);
    if (h.iter >= 100) { setOn(false); return false; }
    return true;
  }, on, { every: 2 });
  const [t, setT] = useState(0);
  useLoop(() => { if (!show) return true; frame.current = (frame.current + 2) % show.length; setT(frame.current); return true; }, !!show, { every: 1 });
  return (
    <Card title="REINFORCE balancing a pole">
      <p className="text-xs text-slate-400">
        CartPole: push the cart left or right; +1 for every step the pole stays within 12° and the cart on the track, up to 500.<Cite id="barto1983" /> The
        policy is a 4-16-2 network with a softmax; each iteration plays 8 episodes, then takes one gradient step. The notes run this comparison and find vanilla
        fails, reward-to-go works, and a baseline is smoother.<Cite id="nalawade" />
      </p>
      <Select label="Ψₜ" value={variant} set={setVariant} options={['vanilla', 'rtg', 'baseline'].map((k) => [k, VARIANTS[k]])} />
      <div className="flex flex-wrap items-center gap-1">
        <Btn on={on} onClick={() => setOn(!on)}>{on ? 'Pause' : hist.length ? 'Continue' : 'Train 100 iterations'}</Btn>
        <Btn onClick={() => setSeed(seed + 1)}>New seed</Btn>
        <span className="text-[11px] tabular-nums text-dim">iteration {hist.length}{hist.length ? `, episode length ${hist.at(-1).mean.toFixed(0)}` : ''}</span>
      </div>
      <CartPoleView s={show?.[t]} label={show ? `latest policy, step ${t} of ${show.length - 1}` : 'untrained'} />
      {hist.length > 1 && <Chart series={[{ label: 'mean episode length', y: hist.map((h) => h.mean), band: [hist.map((h) => h.min), hist.map((h) => h.max)] }]} xLabel="iteration" yLabel="steps balanced" yMin={0} yMax={500} refLine={500} refLabel="max 500" fmt={(v) => v.toFixed(0)} />}
    </Card>
  );
}

function Compare() {
  const [res, setRes] = useState(null);
  const [busy, setBusy] = useState(false);
  const run = async () => {
    setBusy(true);
    const out = await later(() => ['vanilla', 'rtg', 'baseline'].map((v) => ({ v, ...bandOf(pgCurves(v, { seeds: 4, iters: 60 })) })));
    setRes(out); setBusy(false);
  };
  return (
    <Card title="Three choices of Ψₜ, four seeds each">
      <Btn on onClick={run} disabled={busy}>{busy ? 'Training 12 policies...' : 'Run the comparison (a few seconds)'}</Btn>
      {res && <Chart series={res.map((r) => ({ label: VARIANTS[r.v], y: r.y, band: r.band }))} xLabel="iteration" yLabel="mean episode length" yMin={0} yMax={500} fmt={(v) => v.toFixed(0)} />}
      <p className="text-[11px] text-slate-400">
        Lines are the mean over seeds; bands span the best and worst seed. Same network, learning rate and batch for all three; only the weight changes.
        Whole-trajectory weights give every action of a long episode the same credit, so the signal is buried in noise.
      </p>
    </Card>
  );
}

function YouDrive() {
  const [s, setS] = useState(null);
  const [steps, setSteps] = useState(0);
  const [best, setBest] = useState(0);
  const key = useRef(null);
  useEffect(() => {
    const d = (e) => { if (e.key === 'ArrowLeft') key.current = 0; if (e.key === 'ArrowRight') key.current = 1; };
    const u = () => { key.current = null; };
    window.addEventListener('keydown', d); window.addEventListener('keyup', u);
    return () => { window.removeEventListener('keydown', d); window.removeEventListener('keyup', u); };
  }, []);
  useLoop(() => {
    if (!s) return false;
    // with no key held, keep pushing the way the pole leans less (alternating), as a neutral default
    const a = key.current ?? (steps % 2);
    const o = cpStep(s, a);
    if (o.done || steps >= CP.maxSteps) { setBest(Math.max(best, steps)); setS(null); return false; }
    setS(o.s2); setSteps(steps + 1);
    return true;
  }, !!s, { every: 2 });
  return (
    <Card title="Your turn: balance it yourself">
      <p className="text-xs text-slate-400">Hold the left and right arrow keys. Each step is 0.02 s of simulated time, shown at about half speed. With no key held, the cart alternates.</p>
      <div className="flex items-center gap-2">
        <Btn on onClick={() => { setS(cpReset(rng(Date.now() % 1000))); setSteps(0); }}>{s ? 'Restart' : 'Start'}</Btn>
        <Stat label="steps" value={steps} />
        <Stat label="your best" value={best} tone="warm" />
      </div>
      <CartPoleView s={s ?? [0, 0, 0, 0]} action={key.current} />
    </Card>
  );
}

function Importance() {
  const [T, setT] = useState(10);
  const [pNew, setPNew] = useState(0.6);
  const r = useMemo(() => importanceSpread({ T, pNew, pOld: 0.5 }), [T, pNew]);
  const curve = useMemo(() => Array.from({ length: 40 }, (_, k) => importanceSpread({ T: k + 1, pNew, pOld: 0.5, samples: 1500 }).ess), [pNew]);
  return (
    <Card title="Off-policy data and exploding importance weights">
      <p className="text-xs text-slate-400">
        Policy gradients are <Term k="on-policy" />: after each step, the old samples came from a different policy. <Term k="importance sampling" /> can
        reweight them, but the weight for a whole trajectory is a product over time steps, and products spread out fast.<Cite id="cs224r" />
      </p>
      <Eq note="trajectory-level weight; lecture 3 then approximates it per time step by one action ratio">p_θ′(τ) / p_θ(τ) = ∏ₜ π_θ′(aₜ | sₜ) / π_θ(aₜ | sₜ)</Eq>
      <Slider label="horizon T" value={T} set={setT} min={1} max={100} step={1} />
      <Slider label="π_new(a=0)" value={pNew} set={setPNew} min={0.5} max={0.9} step={0.05} fmt={(v) => v.toFixed(2)} title="the old policy picks a=0 with probability 0.5" />
      <div className="flex flex-wrap gap-2">
        <Stat label="mean weight (should be 1)" value={r.mean.toFixed(2)} />
        <Stat label="median weight" value={r.median.toFixed(3)} tone="warm" />
        <Stat label="95th percentile" value={r.p95.toFixed(2)} />
        <Stat label="effective sample share" value={`${(100 * r.ess).toFixed(0)}%`} tone="warm" />
      </div>
      <Chart series={[{ label: 'effective sample share', y: curve.map((v) => 100 * v) }]} xLabel="horizon T" x0={1} yLabel="% of samples that count" yMin={0} yMax={100} height={120} fmt={(v) => `${v.toFixed(0)}%`} />
      <p className="text-[11px] text-slate-400">
        The average weight stays 1, so the estimate is still unbiased, but the typical weight shrinks toward 0 while a few samples get enormous weights. The
        lecture's way out, a per-step ratio plus a limit on how far the policy moves, is chapter 7.
      </p>
    </Card>
  );
}

export default function PolicyGradient() {
  return (
    <div className="grid gap-4">
      <Pushes />
      <CartPole />
      <Compare />
      <YouDrive />
      <Importance />
      <Card title="The derivation in three lines">
        <Eq note="the log-derivative identity (lecture 3)">∇_θ p_θ(τ) = p_θ(τ) ∇_θ log p_θ(τ)</Eq>
        <Eq>∇_θ J(θ) = E_τ∼p_θ [ ∇_θ log p_θ(τ) r(τ) ]</Eq>
        <Eq note="the dynamics terms of log p_θ(τ) do not depend on θ and drop out">∇_θ log p_θ(τ) = Σₜ ∇_θ log π_θ(aₜ | sₜ)</Eq>
        <p className="text-[11px] text-slate-400">
          In code, you never write the gradient. You build the surrogate (1/N) Σ log π_θ(aₜ | sₜ) Ψₜ, a "weighted maximum likelihood", and let autodiff take one
          backward pass.<Cite id="cs224r" />
        </p>
        <NotShown>Continuous actions with a Gaussian policy, where the same surrogate becomes a weighted squared error.</NotShown>
      </Card>
    </div>
  );
}
