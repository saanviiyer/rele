// Chapter 11 (CS224R lecture 9; the notes, sections 5-6): RLHF and DPO on a "language model"
// with 30 candidate responses to one prompt.
import React, { useMemo, useState } from 'react';
import { Card, Btn, Slider, Term, Cite, Eq, NotShown, Stat } from '../ui.jsx';
import Chart from '../Chart.jsx';
import { makeLM, preferencePairs, fitRewardModel, klOptimal, kl, expect, overoptimisation, dpo } from '../../lib/prefs.js';
import { later } from '../../lib/useLoop.js';

function Scatter({ lm, p, rm }) {
  const S = 300, H = 170, X = (q) => 20 + q * (S - 30), Y = (l) => H - 18 - l * (H - 30);
  const m = Math.max(...p);
  return (
    <svg viewBox={`0 0 ${S} ${H}`} className="w-full">
      <line x1="20" x2={S - 10} y1={H - 18} y2={H - 18} stroke="var(--color-line)" />
      <line x1="20" x2="20" y1="8" y2={H - 18} stroke="var(--color-line)" />
      {lm.resp.map((r, i) => (
        <circle key={i} cx={X(r.q)} cy={Y(r.len)} r={2 + 14 * Math.sqrt(p[i] / m)} fill="var(--color-warm)" opacity={0.25 + 0.6 * (p[i] / m)} stroke="var(--color-warm)" />
      ))}
      <text x={S - 10} y={H - 4} fontSize="9" textAnchor="end" fill="var(--color-dim)">true quality →</text>
      <text x="24" y="14" fontSize="9" fill="var(--color-dim)">↑ length</text>
      {rm && <text x={S - 10} y="14" fontSize="9" textAnchor="end" fill="var(--color-dim)">r_φ = {rm.w[0].toFixed(1)}×quality + {rm.w[1].toFixed(1)}×length</text>}
    </svg>
  );
}

function RLHFCard({ lm, pairs, rm }) {
  const [logBeta, setLogBeta] = useState(0);
  const beta = 10 ** logBeta;
  const p = useMemo(() => klOptimal(lm.ref, rm.r, beta), [lm, rm, beta]);
  const q = lm.resp.map((r) => r.q), len = lm.resp.map((r) => r.len);
  const sweep = useMemo(() => overoptimisation(lm, rm), [lm, rm]);
  const peak = sweep.reduce((m, o) => (o.truth > m.truth ? o : m));
  return (
    <Card title="RLHF: a reward model, then a KL-regularised optimum">
      <p className="text-xs text-slate-400">
        One prompt, 30 possible responses (circles; size = probability). Simulated annotators compared {pairs.length} pairs drawn from π_ref. They mostly judge
        quality, but they also like longer answers, so the reward model r_φ, fitted by Bradley-Terry, gives length credit it should not.<Cite id="ouyang2022" />
      </p>
      <Eq note="the RLHF objective and its closed-form optimum (lecture 9; the notes, section 6)">max_π E_y∼π [ r_φ(x, y) ] − β KL(π ‖ π_ref)   ⇒   π*(y | x) = π_ref(y | x) exp(r_φ(x, y) / β) / Z(x)</Eq>
      <Slider label="β" value={logBeta} set={setLogBeta} min={-3} max={1} step={0.05} fmt={() => beta.toFixed(beta < 0.01 ? 4 : 3)} />
      <Scatter lm={lm} p={p} rm={rm} />
      <div className="flex flex-wrap gap-2">
        <Stat label="KL(π ‖ π_ref)" value={kl(p, lm.ref).toFixed(2)} />
        <Stat label="reward model's score" value={expect(p, rm.r).toFixed(2)} tone="warm" />
        <Stat label="true quality" value={expect(p, q).toFixed(3)} />
        <Stat label="mean length" value={expect(p, len).toFixed(2)} tone="dim" />
      </div>
      <Chart series={[
        { label: 'true quality', y: sweep.map((o) => o.truth) },
        { label: 'reward model score, rescaled', y: sweep.map((o) => (o.proxy - sweep[0].proxy) / (sweep.at(-1).proxy - sweep[0].proxy || 1) * (peak.truth - sweep[0].truth) + sweep[0].truth), color: 'var(--color-warm)', dash: '4 3' },
      ]} xLabel="β from 10 down to 0.0005 (more optimisation to the right)" yLabel="score" height={140} fmt={(v) => v.toFixed(3)} />
      <p className="text-[11px] text-slate-400">
        As β falls, the policy moves further from π_ref and the reward model's score keeps climbing. True quality peaks at {peak.truth.toFixed(3)} around KL{' '}
        {peak.kl.toFixed(1)}, then falls as the policy buys length the reward model overvalues: <Term k="reward hacking" />, or reward-model over-optimisation
        (Gao et al. 2023).<Cite id="gao2023" /> The β term is what holds it back.
      </p>
    </Card>
  );
}

function DPOCard({ lm, pairs, rm }) {
  const [beta, setBeta] = useState(0.5);
  const [res, setRes] = useState(null);
  const [busy, setBusy] = useState(false);
  const run = async () => { setBusy(true); setRes(await later(() => dpo(lm, pairs, { beta }))); setBusy(false); };
  const q = lm.resp.map((r) => r.q), len = lm.resp.map((r) => r.len);
  const rlhf = useMemo(() => klOptimal(lm.ref, rm.r, beta), [lm, rm, beta]);
  return (
    <Card title="DPO: skip the reward model">
      <p className="text-xs text-slate-400">
        Rearranging the closed form gives r(x, y) = β log(π*(y | x) / π_ref(y | x)) + β log Z(x). Put that into the Bradley-Terry loss and Z cancels, leaving
        a classification loss on the policy itself.<Cite id="rafailov2023" />
      </p>
      <Eq note="the DPO loss (lecture 9; the notes, section 6)">L_DPO = −E log σ( β log π_θ(y_w)/π_ref(y_w) − β log π_θ(y_l)/π_ref(y_l) )</Eq>
      <Slider label="β" value={beta} set={setBeta} min={0.05} max={2} step={0.05} fmt={(v) => v.toFixed(2)} />
      <Btn on onClick={run} disabled={busy}>{busy ? 'Training...' : 'Train DPO on the same pairs'}</Btn>
      {res && (
        <>
          <Chart series={[{ label: 'DPO loss', y: res.hist.map((h) => h.loss) }]} refLine={Math.log(2)} refLabel="log 2: the loss at π = π_ref" xLabel="every 20 steps" yLabel="loss" height={110} fmt={(v) => v.toFixed(4)} />
          <div className="grid grid-cols-2 gap-2 text-[11px] text-slate-300">
            <div className="rounded border border-line p-2"><div className="text-dim">DPO policy</div>quality {expect(res.pi, q).toFixed(3)}, length {expect(res.pi, len).toFixed(2)}, KL {kl(res.pi, lm.ref).toFixed(2)}</div>
            <div className="rounded border border-line p-2"><div className="text-dim">RLHF optimum, same β</div>quality {expect(rlhf, q).toFixed(3)}, length {expect(rlhf, len).toFixed(2)}, KL {kl(rlhf, lm.ref).toFixed(2)}</div>
          </div>
          <p className="text-[11px] text-slate-400">
            The loss starts at exactly log 2 ≈ 0.6931, since the policy and reference agree on every pair. After {(res.hist.length - 1) * 20} steps DPO has
            moved less far from π_ref than the RLHF optimum (KL {kl(res.pi, lm.ref).toFixed(2)} against {kl(rlhf, lm.ref).toFixed(2)}): it fits a free score for each
            of the 30 responses from {pairs.length} noisy comparisons, while the reward model only had a quality-and-length rule to fit. Both answer longer than
            π_ref does ({expect(lm.ref, len).toFixed(2)}), because the annotators liked length.
          </p>
        </>
      )}
      <p className="text-[11px] text-slate-400">
        What DPO gives up, per lecture 9: it learns only from the fixed preference data, never from its own new samples. Classic RLHF needs the policy, the
        reference and the reward model in memory at once.<Cite id="nalawade" />
      </p>
      <NotShown>Real token sequences: a response here is one choice, so log π(y) has no sum over tokens.</NotShown>
    </Card>
  );
}

export default function RLHF() {
  const lm = useMemo(() => makeLM(), []);
  const pairs = useMemo(() => preferencePairs(lm), [lm]);
  const rm = useMemo(() => fitRewardModel(lm, pairs), [lm, pairs]);
  return (
    <div className="grid gap-4">
      <RLHFCard lm={lm} pairs={pairs} rm={rm} />
      <DPOCard lm={lm} pairs={pairs} rm={rm} />
    </div>
  );
}
