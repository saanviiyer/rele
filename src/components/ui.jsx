import React, { useState } from 'react';
import { GLOSSARY } from '../lib/glossary.js';
import { REFS, REF_ORDER } from '../lib/refs.js';

// The name: "ReLe" underlined; hover or focus it to read what it stands for.
export function ReLe({ className = '' }) {
  return (
    <span className={`group relative inline-block ${className}`} tabIndex={0}>
      <span className="cursor-help underline decoration-glow decoration-2 underline-offset-4">ReLe</span>
      <span className="pointer-events-none absolute left-1/2 top-full z-[60] mt-2 hidden -translate-x-1/2 whitespace-nowrap rounded-md border border-line bg-panel px-2.5 py-1 text-xs font-normal normal-case tracking-normal text-slate-200 shadow-xl group-hover:block group-focus:block">
        reinforcement learning
      </span>
    </span>
  );
}

export function Title({ className = '' }) {
  return (
    <span className={`font-semibold text-slate-100 ${className}`}>
      it's <ReLe /> that simple?
    </span>
  );
}

// Hover (or focus) a term to get its one-sentence definition.
export function Term({ k, children }) {
  const def = GLOSSARY[k] ?? GLOSSARY[k?.toLowerCase()];
  return (
    <span className="group relative inline" tabIndex={0}>
      <span className="cursor-help border-b border-dotted border-glow/60">{children ?? k}</span>
      <span className="pointer-events-none absolute left-0 top-full z-50 mt-1 hidden w-64 rounded-md border border-line bg-panel px-3 py-2 text-xs leading-snug text-slate-200 shadow-xl group-hover:block group-focus:block">
        <span className="font-semibold text-glow">{k}.</span> {def}
      </span>
    </span>
  );
}

export function Cite({ id }) {
  const r = REFS[id];
  const n = REF_ORDER.indexOf(id) + 1;
  if (!r) return <sup className="text-red-500">[?{id}]</sup>;
  return (
    <span className="group relative inline">
      <a href={r.url ?? undefined} target="_blank" rel="noreferrer" className="ml-0.5 align-super text-[9px] text-glow/80 hover:text-glow">
        [{r.short}]
      </a>
      <span className="pointer-events-none absolute bottom-full left-0 z-50 mb-1 hidden w-72 rounded-md border border-line bg-panel px-3 py-2 text-[11px] leading-snug text-slate-200 shadow-xl group-hover:block">
        {n}. {r.full}
      </span>
    </span>
  );
}

export function NotShown({ children }) {
  return (
    <div className="rounded border border-dashed border-line px-2 py-1 text-[11px] text-dim">
      <span className="mr-1 rounded bg-line px-1 py-px text-[10px] uppercase tracking-wide text-slate-300">not shown here</span>
      {children}
    </div>
  );
}

export function Btn({ on, children, className = '', ...p }) {
  return (
    <button
      {...p}
      className={`rounded-md border px-2.5 py-1 text-xs transition disabled:opacity-40 ${
        on ? 'border-glow/70 bg-glow/15 text-glow' : 'border-line bg-panel text-slate-200 hover:border-glow/40'
      } ${className}`}
    >
      {children}
    </button>
  );
}

export function Kbd({ children }) {
  return <kbd className="rounded border border-line bg-panel px-1 text-[10px] text-slate-300">{children}</kbd>;
}

// A labelled slider. fmt formats the shown value.
export function Slider({ label, value, set, min, max, step, fmt = (v) => v, title }) {
  return (
    <label className="flex items-center gap-2 text-xs text-slate-300" title={title}>
      <span className="w-24 shrink-0 text-dim">{label}</span>
      <input type="range" min={min} max={max} step={step} value={value} onChange={(e) => set(+e.target.value)} className="min-w-0 flex-1 accent-[var(--color-warm)]" />
      <span className="w-12 text-right tabular-nums text-warm">{fmt(value)}</span>
    </label>
  );
}

export function Select({ label, value, set, options }) {
  return (
    <label className="flex items-center gap-2 text-xs text-slate-300">
      <span className="w-24 shrink-0 text-dim">{label}</span>
      <select value={value} onChange={(e) => set(e.target.value)} className="min-w-0 flex-1 rounded border border-line bg-panel px-1.5 py-0.5 text-xs text-slate-200">
        {options.map(([k, t]) => <option key={k} value={k}>{t}</option>)}
      </select>
    </label>
  );
}

// A boxed equation or rule, set in a mono face so subscripts read plainly.
export function Eq({ children, note }) {
  return (
    <div className="my-1.5 rounded-md border border-line bg-panel px-3 py-2">
      <div className="overflow-x-auto whitespace-nowrap font-mono text-[12px] text-slate-100">{children}</div>
      {note && <div className="mt-1 text-[10px] text-dim">{note}</div>}
    </div>
  );
}

export function Card({ title, children, className = '' }) {
  return (
    <div className={`space-y-2 rounded-lg border border-line p-3 ${className}`}>
      {title && <div className="text-sm text-slate-100">{title}</div>}
      {children}
    </div>
  );
}

export function Takeaway({ children }) {
  return <div className="rounded-lg border border-glow/25 bg-glow/5 p-3 text-xs leading-relaxed text-slate-200">{children}</div>;
}

export function Stat({ label, value, tone = 'glow' }) {
  return (
    <div className="rounded-md border border-line px-2 py-1">
      <div className="text-[10px] text-dim">{label}</div>
      <div className={`text-sm tabular-nums text-${tone}`}>{value}</div>
    </div>
  );
}

export function Sources({ onClose }) {
  return (
    <div className="fixed inset-0 z-[100] flex items-center justify-center bg-black/70 p-4" onClick={onClose}>
      <div className="scroll-thin max-h-[80vh] w-full max-w-2xl overflow-y-auto rounded-lg border border-line bg-panel p-5" onClick={(e) => e.stopPropagation()}>
        <div className="mb-3 flex items-center justify-between">
          <h2 className="text-sm font-semibold tracking-wide text-glow">Sources</h2>
          <Btn onClick={onClose}>Close</Btn>
        </div>
        <ol className="space-y-2 text-xs text-slate-300">
          {REF_ORDER.map((k, i) => (
            <li key={k}>
              {i + 1}. {REFS[k].full}{' '}
              {REFS[k].url && <a className="text-glow/80 hover:text-glow" href={REFS[k].url} target="_blank" rel="noreferrer">link</a>}
            </li>
          ))}
        </ol>
      </div>
    </div>
  );
}

export class Boundary extends React.Component {
  state = { err: null };
  static getDerivedStateFromError(err) { return { err }; }
  render() {
    if (this.state.err) return <div className="rounded border border-red-900 p-2 text-xs text-red-500">This panel hit an error: {String(this.state.err.message)}</div>;
    return this.props.children;
  }
}

export function useToggle(v = false) {
  const [x, set] = useState(v);
  return [x, () => set((y) => !y), set];
}
