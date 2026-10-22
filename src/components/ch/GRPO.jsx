// Chapter 12 (the notes, sections 7-9): GRPO, Dr GRPO and DAPO. Group-relative advantages, the
// length and difficulty biases, dynamic sampling, and clip-higher.
import React, { useMemo, useState } from 'react';
import { Card, Btn, Slider, Term, Cite, Eq, NotShown, Stat } from '../ui.jsx';
import Chart from '../Chart.jsx';
import { groupAdvantages, k3, difficultyWeight, zeroGroupShare, lengthBiasRun, KINDS } from '../../lib/prefs.js';
import { later } from '../../lib/useLoop.js';

function Advantages() {
  const [r, setR] = useState([1, 0, 0, 1, 0, 0, 0, 1]);
  const A = groupAdvantages(r), Adr = groupAdvantages(r, { std: false });
  const mean = r.reduce((a, b) => a + b, 0) / r.length;
  const sd = Math.sqrt(r.reduce((a, b) => a + (b - mean) ** 2, 0) / r.length);
  return (
    <Card title="Group-relative advantages">
      <p className="text-xs text-slate-400">
        For one prompt, sample G answers and check each (1 = right, 0 = wrong). Each answer's advantage is its reward relative to the group: no value network,
        the group is the baseline.<Cite id="shao2024" /> Click an answer to flip it.
      </p>
      <Eq note="GRPO (the notes use the population std; Dr GRPO drops the division)">Âᵢ = ( rᵢ − mean(r) ) / std(r)</Eq>
      <div className="flex flex-wrap gap-1">
        {r.map((x, i) => (
          <button key={i} onClick={() => setR(r.map((y, j) => (j === i ? 1 - y : y)))} className={`w-16 rounded-md border px-1 py-1 text-center text-[11px] ${x ? 'border-glow/60 bg-glow/10 text-glow' : 'border-line bg-panel text-dim'}`}>
            <div>answer {i + 1}</div><div>{x ? 'right' : 'wrong'}</div>
          </button>
        ))}
      </div>
      <div className="flex flex-wrap gap-1">
        <Btn onClick={() => setR(r.length > 2 ? r.slice(0, -1) : r)}>fewer</Btn>
        <Btn onClick={() => setR(r.length < 16 ? [...r, 0] : r)}>more</Btn>
        <Btn onClick={() => setR([1, 0, 0, 1])}>the notes' example [1, 0, 0, 1]</Btn>
      </div>
      <div className="text-[11px] tabular-nums text-slate-300">
        <div>mean {mean.toFixed(3)}, std {sd.toFixed(3)}</div>
        <div>GRPO Â: {A.map((a) => a.toFixed(2)).join(', ')}</div>
        <div>Dr GRPO Â: {Adr.map((a) => a.toFixed(2)).join(', ')}</div>
      </div>
      {sd === 0 && <p className="text-[11px] text-warm">Every answer is the same, so every advantage is 0 and this prompt contributes no gradient at all (std = 0 is guarded).</p>}
      <Eq note="the k3 estimator of the KL to the reference, from one sample; kept outside the advantage">k3 = π_ref/π_θ − log(π_ref/π_θ) − 1 ≥ 0</Eq>
      <div className="text-[11px] tabular-nums text-dim">k3 at ratios 0.5, 1, 2: {[0.5, 1, 2].map((x) => k3(x).toFixed(3)).join(', ')}</div>
    </Card>
  );
}

function LengthBias() {
  const [res, setRes] = useState(null);
  const [busy, setBusy] = useState(false);
  const run = async () => { setBusy(true); setRes(await later(() => ({ g: lengthBiasRun({ norm: 'grpo' }), d: lengthBiasRun({ norm: 'dr' }) }))); setBusy(false); };
  return (
    <Card title="The length bias of 1/|oᵢ|">
      <p className="text-xs text-slate-400">
        GRPO averages each answer's token losses, a factor 1/|oᵢ|. For the same negative advantage a 40-token wrong answer is pushed down a quarter as hard as
        a 10-token one, and a short right answer is pushed up harder than a long one. Dr GRPO replaces 1/|oᵢ| with a constant.<Cite id="liu2025" />
      </p>
      <Eq note="GRPO objective (the notes, section 7)">(1/G) Σᵢ (1/|oᵢ|) Σₜ [ min(ρ Âᵢ, clip(ρ, 1−ε, 1+ε) Âᵢ) − β KL ]</Eq>
      <p className="text-[11px] text-slate-400">
        Toy: 24 prompts of varying difficulty; for each, the policy chooses one of four answers ({KINDS.map((k) => k.label).join('; ')}), with 10 or 40 tokens.
        Each update samples 8 answers per prompt.
      </p>
      <Btn on onClick={run} disabled={busy}>{busy ? 'Training both...' : 'Train with GRPO and with Dr GRPO (150 updates)'}</Btn>
      {res && (
        <>
          <Chart series={[{ label: 'GRPO', y: res.g.map((h) => h.lenWrong), color: 'var(--color-warm)' }, { label: 'Dr GRPO', y: res.d.map((h) => h.lenWrong), color: 'var(--color-cool)' }]} xLabel="update" yLabel="tokens" height={120} fmt={(v) => v.toFixed(1)} />
          <p className="text-[10px] text-dim">Mean length of wrong answers.</p>
          <Chart series={[{ label: 'GRPO', y: res.g.map((h) => 100 * h.acc), color: 'var(--color-warm)' }, { label: 'Dr GRPO', y: res.d.map((h) => 100 * h.acc), color: 'var(--color-cool)' }]} xLabel="update" yLabel="% right" yMin={0} yMax={100} height={110} fmt={(v) => `${v.toFixed(0)}%`} />
          <div className="flex flex-wrap gap-2">
            <Stat label="GRPO: wrong answers, start → end" value={`${res.g[0].lenWrong.toFixed(1)} → ${res.g.at(-1).lenWrong.toFixed(1)} tokens`} tone="warm" />
            <Stat label="Dr GRPO" value={`${res.d[0].lenWrong.toFixed(1)} → ${res.d.at(-1).lenWrong.toFixed(1)} tokens`} tone="cool" />
          </div>
          <p className="text-[11px] text-slate-400">
            Both learn to answer correctly; under GRPO the remaining wrong answers get longer, since long ones are punished less. In this toy GRPO also gets
            right faster, because the same 1/|oᵢ| gives short right answers bigger steps; the scale of each step is part of what the normalisation changes.
          </p>
        </>
      )}
    </Card>
  );
}

function Difficulty() {
  const [G, setG] = useState(8);
  const ps = Array.from({ length: 49 }, (_, i) => 0.02 + i * 0.02);
  const g = ps.map((p) => difficultyWeight(p, G, true)), d = ps.map((p) => difficultyWeight(p, G, false));
  const mid = ps.findIndex((p) => Math.abs(p - 0.5) < 1e-9);
  return (
    <Card title="Difficulty bias, and DAPO's dynamic sampling">
      <p className="text-xs text-slate-400">
        Expected total |advantage| one prompt contributes, as a function of how often the policy already gets it right, each curve scaled to 1 at 50%. Dividing by
        the group's std inflates prompts that are nearly always right or nearly always wrong, relative to Dr GRPO.<Cite id="liu2025" />
      </p>
      <Slider label="group size G" value={G} set={setG} min={2} max={32} step={1} />
      <Chart series={[{ label: 'GRPO (÷ std)', y: g.map((v) => v / g[mid]), color: 'var(--color-warm)' }, { label: 'Dr GRPO', y: d.map((v) => v / d[mid]), color: 'var(--color-cool)' }]} xLabel="prompt accuracy, 2% to 98%" yLabel="relative weight" yMin={0} height={120} fmt={(v) => v.toFixed(2)} />
      <Chart series={[{ label: 'share of groups that are all right or all wrong', y: ps.map((p) => 100 * zeroGroupShare(p, G)) }]} xLabel="prompt accuracy, 2% to 98%" yLabel="% zero-gradient groups" yMin={0} yMax={100} height={110} fmt={(v) => `${v.toFixed(0)}%`} />
      <p className="text-[11px] text-slate-400">
        At 95% accuracy with G = {G}, {(100 * zeroGroupShare(0.95, G)).toFixed(0)}% of groups are all right and contribute nothing. DAPO's dynamic sampling drops
        those groups and samples more prompts until the batch is full of groups that teach something.<Cite id="yu2025" />
      </p>
    </Card>
  );
}

function ClipHigher() {
  const [p, setP] = useState(0.01);
  const [lo, setLo] = useState(0.2);
  const [hi, setHi] = useState(0.28);
  const sym = p * (1 + lo), asym = p * (1 + hi);
  return (
    <Card title="Clip-higher">
      <p className="text-xs text-slate-400">
        The clip bounds the probability ratio, so a token can rise at most to p_old × (1 + ε) in one update. That is a big move for a common token and almost
        nothing for a rare one, so rare tokens, the ones exploration needs, stay rare and entropy collapses. DAPO raises only the upper bound, ε_high &gt;
        ε_low.<Cite id="yu2025" />
      </p>
      <Slider label="p_old of a token" value={p} set={setP} min={0.001} max={0.95} step={0.001} fmt={(v) => v.toFixed(3)} />
      <Slider label="ε_low" value={lo} set={setLo} min={0.05} max={0.4} step={0.01} fmt={(v) => v.toFixed(2)} />
      <Slider label="ε_high" value={hi} set={setHi} min={0.05} max={0.5} step={0.01} fmt={(v) => v.toFixed(2)} />
      <div className="flex flex-wrap gap-2">
        <Stat label="ceiling with symmetric ε" value={sym > 1 ? `${sym.toFixed(3)} (ratio bound; probability caps at 1)` : sym.toFixed(4)} />
        <Stat label="ceiling with ε_high" value={asym > 1 ? `${asym.toFixed(3)} (caps at 1)` : asym.toFixed(4)} tone="warm" />
        <Stat label="floor" value={(p * (1 - lo)).toFixed(4)} tone="dim" />
      </div>
      <p className="text-[11px] text-slate-400">
        The notes' example: at ε = 0.2 a token at 0.01 can reach 0.012, while one at 0.9 gets a bound of 1.08. That 1.08 is a bound on the ratio, not a
        reachable probability. With ε_high = 0.28, the rare token's ceiling is 0.0128.
      </p>
      <ul className="list-disc space-y-0.5 pl-4 text-[11px] text-slate-300">
        <li><b>Token-level loss:</b> DAPO divides by the total tokens in the batch, Σ|oᵢ|, instead of averaging per answer, so long answers count per token.</li>
        <li><b>Overlong answers:</b> truncated answers are masked out, and a soft penalty grows as an answer nears the length limit.</li>
        <li><b>No KL term:</b> DAPO drops it for reasoning training, where the policy is meant to move far from the reference.</li>
      </ul>
      <NotShown>The verifiable-reward pipeline itself (sampling real token sequences and checking math answers); here each answer is a single choice.</NotShown>
    </Card>
  );
}

export default function GRPO() {
  return (
    <div className="grid gap-4">
      <Advantages />
      <LengthBias />
      <Difficulty />
      <ClipHigher />
    </div>
  );
}
