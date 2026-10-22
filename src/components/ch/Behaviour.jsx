// Chapter 2 (CS224R lecture 1): behaviour as data. States, actions, trajectories, rewards, the
// trajectory distribution p_theta(tau), and the RL objective.
import React, { useMemo, useState } from 'react';
import { Card, Btn, Slider, Term, Cite, Eq } from '../ui.jsx';
import { enumerateTrajectories } from '../../lib/toys.js';
import { rng, sample } from '../../lib/rng.js';

const EXAMPLES = [
  { name: 'Robot hangs a towel', s: 'camera images plus joint positions and velocities', a: 'the next commanded joint positions', tau: '10 seconds at 20 Hz, so T = 200', r: '1 if the towel is on the hook, else 0', from: 'lecture 1' },
  { name: 'Chatbot', s: 'o = the user\'s latest message (the bot never sees the whole "state" of the person)', a: 'the bot\'s reply', tau: 'one conversation, of varying length', r: '1 for an upvote, -10 for a downvote, 0 otherwise', from: 'lecture 1' },
  { name: 'Chess (the notes\' example)', s: 'the board', a: 'a legal move', tau: 'one game', r: '+1 win, -1 loss, 0 draw, at the end only', from: 'the notes' },
];
const PAIR = [
  { name: 'Autonomous driving', ans: 's: camera, lidar and map; a: steering and acceleration; tau: one drive; r: progress minus penalties for collisions and discomfort. Partly observed: other drivers\' intentions are hidden.' },
  { name: 'Web agent', ans: 's or o: the page (screenshot or HTML) and the task; a: click, type or scroll; tau: one task attempt; r: 1 if the task succeeded, often only at the end.' },
  { name: 'Poker player', ans: 'o: your cards, the board and the betting so far (opponents\' cards are hidden, so this is an observation, not the state); a: fold, call or raise; tau: one hand; r: chips won or lost.' },
];

function Examples() {
  const [open, setOpen] = useState({});
  return (
    <Card title="Write it as an RL problem">
      <p className="text-xs text-slate-400">The lecture's examples, then the think-pair-share: try writing <Term k="state">s</Term> (or <Term k="observation">o</Term>), <Term k="action">a</Term>, <Term k="trajectory">tau</Term> and <Term k="reward">r</Term> yourself before revealing a suggested answer.<Cite id="cs224r" /></p>
      <div className="space-y-2">
        {EXAMPLES.map((e) => (
          <div key={e.name} className="rounded-md border border-line p-2 text-[11px] text-slate-300">
            <div className="mb-0.5 text-xs text-slate-100">{e.name} <span className="text-dim">({e.from})</span></div>
            <div><b className="text-glow">s</b> {e.s}</div>
            <div><b className="text-glow">a</b> {e.a}</div>
            <div><b className="text-glow">tau</b> {e.tau}</div>
            <div><b className="text-glow">r</b> {e.r}</div>
          </div>
        ))}
        {PAIR.map((e, i) => (
          <div key={e.name} className="rounded-md border border-dashed border-line p-2 text-[11px] text-slate-300">
            <div className="flex items-center justify-between"><span className="text-xs text-slate-100">{e.name}</span><Btn onClick={() => setOpen({ ...open, [i]: !open[i] })}>{open[i] ? 'Hide' : 'Reveal one answer'}</Btn></div>
            {open[i] && <div className="mt-1">{e.ans}</div>}
          </div>
        ))}
      </div>
      <p className="text-[11px] text-slate-400">
        Unlike supervised learning, nobody labels the right action, and the data is not independent: the actions you take decide which states you see next.
      </p>
    </Card>
  );
}

function Product() {
  const [p0, setP0] = useState(0.7); // pi(a=0 | s=0)
  const [p1, setP1] = useState(0.2); // pi(a=0 | s=1)
  const [seed, setSeed] = useState(1);
  const pi = [[p0, 1 - p0], [p1, 1 - p1]];
  const E = useMemo(() => enumerateTrajectories({ pi }), [p0, p1]); // eslint-disable-line
  // sample one trajectory with the same model
  const drawn = useMemo(() => {
    const r = rng(seed);
    let u = r(), acc = 0;
    for (const t of E.trajectories) { acc += t.prob; if (u < acc) return t; }
    return E.trajectories.at(-1);
  }, [E, seed]);
  const name = (f) => (f.kind === 'p1' ? `p(s1=${f.s})` : f.kind === 'pi' ? `pi(a=${f.a}|s=${f.s})` : `p(s'=${f.s2}|s=${f.s},a=${f.a})`);
  return (
    <Card title="p_theta(tau) is a product of small terms">
      <Eq note="the trajectory distribution (lecture 1)">p_θ(τ) = p(s₁) ∏ₜ π_θ(aₜ | sₜ) p(sₜ₊₁ | sₜ, aₜ)</Eq>
      <Eq note="the RL objective: an expectation, because both the world and the policy are random">θ* = argmax_θ E_τ∼p_θ(τ) [ Σₜ r(sₜ, aₜ) ]</Eq>
      <p className="text-xs text-slate-400">
        A world with two states and two actions, horizon T = 2. Rewards: r(0, a=1) = 1, r(1, a=0) = 2, otherwise 0. The policy's two numbers are the only
        knobs; the dynamics are fixed. All {E.trajectories.length} trajectories are listed with their probabilities.
      </p>
      <Slider label="pi(a=0 | s=0)" value={p0} set={setP0} min={0} max={1} step={0.05} fmt={(v) => v.toFixed(2)} />
      <Slider label="pi(a=0 | s=1)" value={p1} set={setP1} min={0} max={1} step={0.05} fmt={(v) => v.toFixed(2)} />
      <div className="grid grid-cols-2 gap-x-3 text-[10.5px] tabular-nums text-slate-300 sm:grid-cols-2">
        {E.trajectories.map((t, i) => (
          <div key={i} className={`flex justify-between rounded px-1 ${t === drawn ? 'bg-warm/15' : ''}`}>
            <span>{t.steps.map((s) => `s${s.s}a${s.a}`).join(' ')}</span>
            <span>p {t.prob.toFixed(3)}, R {t.ret}</span>
          </div>
        ))}
      </div>
      <div className="text-xs text-slate-300">
        Sum of probabilities <b className="text-glow">{E.total.toFixed(6)}</b>; expected return J(θ) = <b className="text-warm">{E.J.toFixed(3)}</b>.
      </div>
      <div className="rounded-md border border-line p-2 text-[11px] text-slate-300">
        <div className="flex items-center justify-between"><span>One sampled trajectory, as its factors:</span><Btn onClick={() => setSeed(seed + 1)}>Sample another</Btn></div>
        <div className="mt-1 font-mono text-[10.5px]">{drawn.factors.map((f) => `${name(f)} ${f.p.toFixed(2)}`).join(' × ')} = {drawn.prob.toFixed(4)}</div>
      </div>
      <p className="text-[11px] text-slate-400">
        Only the policy terms depend on θ. That is why the policy gradient (chapter 5) never needs to know the dynamics: their log disappears when you
        differentiate log p_θ(τ).
      </p>
    </Card>
  );
}

function Loop() {
  const boxes = [
    { t: 'Run the policy to collect a batch of data', c: 'every online method' },
    { t: 'Fit a model to estimate expected return', c: 'V, Q or A (chapters 6, 8), or a model of the world (chapter 3)' },
    { t: 'Improve the policy', c: 'a gradient step (5, 6, 7), an argmax (3, 8), or imitation (4, 9)' },
  ];
  return (
    <Card title="One loop behind every method">
      <p className="text-xs text-slate-400">Lecture 4 draws all of reinforcement learning as one cycle; the algorithm families differ in which box does the work.<Cite id="cs224r" /></p>
      <svg viewBox="0 0 420 150" className="w-full">
        {boxes.map((b, i) => {
          const x = [10, 150, 290][i];
          return (
            <g key={i}>
              <rect x={x} y="20" width="120" height="70" rx="8" fill="var(--color-panel)" stroke="var(--color-glow)" />
              <foreignObject x={x + 4} y="24" width="112" height="64"><div style={{ fontSize: 10, color: 'var(--ink)', lineHeight: 1.25 }}>{b.t}</div></foreignObject>
              <text x={x + 60} y="108" fontSize="8.5" textAnchor="middle" fill="var(--color-dim)">{b.c.length > 38 ? b.c.slice(0, 38) + '…' : b.c}</text>
            </g>
          );
        })}
        <path d="M130 55 L150 55 M270 55 L290 55" stroke="var(--color-warm)" strokeWidth="2" markerEnd="url(#ah)" />
        <path d="M350 92 C350 140, 70 140, 70 92" fill="none" stroke="var(--color-warm)" strokeWidth="2" markerEnd="url(#ah)" />
        <defs><marker id="ah" markerWidth="8" markerHeight="8" refX="6" refY="4" orient="auto"><path d="M0,0 L8,4 L0,8 z" fill="var(--color-warm)" /></marker></defs>
      </svg>
      <ul className="list-disc space-y-0.5 pl-4 text-[11px] text-slate-300">
        <li><b>Imitation learning</b> (chapter 4): skip the reward entirely and copy demonstrations.</li>
        <li><b>Policy gradients</b> (5): estimate return by summing rewards, then push up good actions.</li>
        <li><b>Actor-critic</b> (6, 7): fit a value function, use it to judge actions.</li>
        <li><b>Value-based</b> (8): fit Q*, act by argmax.</li>
        <li><b>Model-based</b> (3 is its simplest case): know or learn p(s′ | s, a) and plan.</li>
      </ul>
      <p className="text-[11px] text-slate-400">
        Why use a stochastic policy at all? It explores, and it can represent several good behaviours at once (chapter 4 shows what goes wrong when a policy
        cannot).
      </p>
    </Card>
  );
}

function Markov() {
  const [seed, setSeed] = useState(2);
  // a tiny weather chain to show that the next state needs only the current one
  const P = { sun: { sun: 0.8, rain: 0.2 }, rain: { sun: 0.4, rain: 0.6 } };
  const chain = useMemo(() => {
    const r = rng(seed);
    let s = 'sun';
    const out = [s];
    for (let t = 0; t < 30; t++) { s = sample([P[s].sun, P[s].rain], r) === 0 ? 'sun' : 'rain'; out.push(s); }
    return out;
  }, [seed]); // eslint-disable-line
  return (
    <Card title="The Markov property">
      <Eq note="the next state depends on the present, not the path that led to it">p(sₜ₊₁ | sₜ, aₜ, sₜ₋₁, ...) = p(sₜ₊₁ | sₜ, aₜ)</Eq>
      <p className="text-xs text-slate-400">
        A two-state weather chain (sun stays sun with 0.8, rain stays rain with 0.6). Knowing the last 10 days adds nothing to tomorrow's forecast
        beyond today. Lecture 1's warning: with observations instead of states, p(oₜ₊₁ | o₁:ₜ, a₁:ₜ) ≠ p(oₜ₊₁ | oₜ, aₜ), so a policy may need memory.<Cite id="cs224r" />
      </p>
      <div className="flex flex-wrap gap-0.5 text-sm">{chain.map((s, i) => <span key={i}>{s === 'sun' ? '☀️' : '🌧️'}</span>)}</div>
      <Btn onClick={() => setSeed(seed + 1)}>Another 30 days</Btn>
    </Card>
  );
}

export default function Behaviour() {
  return (
    <div className="grid gap-4">
      <Examples />
      <Product />
      <Loop />
      <Markov />
    </div>
  );
}
