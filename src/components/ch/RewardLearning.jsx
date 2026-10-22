// Chapter 10 (CS224R lecture 8): learning the reward. A goal classifier gets exploited when
// frozen and is fixed by adding the agent's states as negatives; Bradley-Terry recovers a reward
// from pairwise preferences.
import React, { useMemo, useState } from 'react';
import { Card, Btn, Slider, Term, Cite, Eq, NotShown, Stat } from '../ui.jsx';
import Chart from '../Chart.jsx';
import { goalClassifier, score, GOAL, inGoal, bradleyTerry } from '../../lib/prefs.js';
import { useTheme } from '../../lib/theme.js';

function Plane({ w, pos, neg, path }) {
  const theme = useTheme();
  const N = 40, S = 300, X = (x) => ((x + 1) / 2) * S, Y = (y) => S - ((y + 1) / 2) * S;
  const cells = [];
  for (let i = 0; i < N; i++) for (let j = 0; j < N; j++) {
    const x = -1 + (i + 0.5) * (2 / N), y = -1 + (j + 0.5) * (2 / N);
    const p = score(w, [x, y]);
    cells.push(<rect key={`${i}-${j}`} x={X(x - 1 / N)} y={Y(y + 1 / N)} width={S / N + 0.5} height={S / N + 0.5} fill={theme === 'light' ? `rgba(180,95,14,${0.7 * p})` : `rgba(255,181,71,${0.6 * p})`} />);
  }
  return (
    <svg viewBox={`0 0 ${S} ${S}`} className="w-full max-w-xs rounded-md border border-line" style={{ background: 'var(--color-panel)' }}>
      {cells}
      <circle cx={X(GOAL.cx)} cy={Y(GOAL.cy)} r={(GOAL.r / 2) * S} fill="none" stroke="var(--color-glow)" strokeWidth="2" strokeDasharray="4 3" />
      {pos.map((p, i) => <circle key={`p${i}`} cx={X(p[0])} cy={Y(p[1])} r="2.2" fill="var(--color-glow)" />)}
      {neg.map((p, i) => <circle key={`n${i}`} cx={X(p[0])} cy={Y(p[1])} r="1.8" fill="var(--color-cool)" opacity="0.6" />)}
      {path && <polyline points={path.map((p) => `${X(p[0])},${Y(p[1])}`).join(' ')} fill="none" stroke="var(--ink)" strokeWidth="2" />}
      {path && <circle cx={X(path.at(-1)[0])} cy={Y(path.at(-1)[1])} r="5" fill={inGoal(path.at(-1)) ? 'var(--color-glow)' : 'var(--color-warm)'} stroke="var(--ink)" />}
    </svg>
  );
}

function GoalClassifier() {
  const g = useMemo(() => goalClassifier(), []);
  const [round, setRound] = useState(0);
  const cur = round === 0 ? { w: g.w0, path: g.frozen } : g.vice[round - 1];
  const negs = round === 0 ? g.neg0 : g.neg0.concat(...g.vice.slice(0, round - 1).map((v) => v.path));
  const end = cur.path.at(-1);
  const succ = g.vice.map((v) => (v.success ? 1 : 0));
  return (
    <Card title="A goal classifier as the reward, and how RL fools it">
      <p className="text-xs text-slate-400">
        The task: reach the dashed disk. Instead of writing a reward, train a classifier p(success | s) on example goal states (teal) and some ordinary states
        (blue), and use it as the reward; the episode ends when it says "success" (score above 0.5). Shading is the classifier's score. The agent heads
        for whatever the classifier likes best, as RL eventually would.<Cite id="cs224r" />
      </p>
      <Plane w={cur.w} pos={g.pos} neg={negs} path={cur.path} />
      <Slider label={round === 0 ? 'frozen classifier' : `VICE round ${round}`} value={round} set={setRound} min={0} max={g.vice.length} step={1} />
      <div className="flex flex-wrap gap-2">
        <Stat label="classifier's score where the agent stopped" value={score(cur.w, end).toFixed(2)} tone="warm" />
        <Stat label="actually in the goal?" value={inGoal(end) ? 'yes' : 'no'} />
        <Stat label="negatives so far" value={negs.length} tone="dim" />
      </div>
      <Chart series={[{ label: 'in the goal at each VICE round', y: succ.map((v) => 100 * v) }]} xLabel="round" x0={1} yLabel="success" yMin={0} yMax={100} height={90} fmt={(v) => (v ? 'yes' : 'no')} />
      <p className="text-[11px] text-slate-400">
        Frozen, the classifier has only learned "up and to the right is good", and the agent claims success far from the goal. The fix from lecture 8 (VICE, Fu
        et al. 2018<Cite id="fu2018" />): keep adding the states the policy reaches as negatives, with balanced batches, so the boundary closes in on what success really looks
        like. It takes a dozen or more rounds here, and it is an adversarial game: on other seeds it can overshoot before settling.
      </p>
    </Card>
  );
}

function BT() {
  const [pairs, setPairs] = useState(300);
  const [k, setK] = useState(4);
  const r = useMemo(() => bradleyTerry({ pairs }), [pairs]);
  return (
    <Card title="Rewards from comparisons: Bradley-Terry">
      <p className="text-xs text-slate-400">
        Scoring an outcome on an absolute scale is hard and inconsistent; saying which of two is better is easier.<Cite id="christiano2017" /> Here 40 outcomes have
        three features and a hidden reward w* = (1.5, −1, 0.5). A simulated person picks a over b with probability σ(r(a) − r(b)).<Cite id="bradley1952" />
      </p>
      <Eq note="the preference model and its loss (lecture 8)">P(τ_a ≻ τ_b) = σ(r_θ(τ_a) − r_θ(τ_b))      max_θ E log σ(r_θ(τ_w) − r_θ(τ_l))</Eq>
      <Slider label="comparisons" value={pairs} set={setPairs} min={10} max={1000} step={10} />
      <div className="flex flex-wrap gap-2">
        <Stat label="true w*" value={r.wTrue.map((v) => v.toFixed(2)).join(', ')} tone="dim" />
        <Stat label="recovered w" value={r.w.map((v) => v.toFixed(2)).join(', ')} tone="warm" />
        <Stat label="cosine similarity" value={r.cos.toFixed(3)} />
      </div>
      <Chart series={[{ label: 'training loss', y: r.losses }]} xLabel="gradient step" yLabel="−log σ" height={110} fmt={(v) => v.toFixed(3)} />
      <Slider label="k ranked per batch" value={k} set={setK} min={2} max={9} step={1} />
      <p className="text-[11px] text-slate-400">
        Ranking k outcomes at once gives C(k, 2) = {(k * (k - 1)) / 2} pairs, which is how the lecture's loop gets many comparisons per human query. Only
        differences of reward are identified, so any constant can be added to r without changing a single prediction.
      </p>
      <NotShown>Active query selection, where the system asks for the comparisons it is least sure about (Sadigh et al. 2017).<Cite id="sadigh2017" /></NotShown>
    </Card>
  );
}

export default function RewardLearning() {
  return (
    <div className="grid gap-4">
      <GoalClassifier />
      <BT />
    </div>
  );
}
