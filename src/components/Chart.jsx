// A small SVG line chart: several series, optional shaded bands (min to max over seeds), a
// reference line, and a hover readout. Colors come from CSS tokens so both themes work.
import React, { useMemo, useState } from 'react';

export const SERIES = ['var(--color-glow)', 'var(--color-warm)', 'var(--color-cool)', '#8a5cc7', '#2f8f5b', '#b0413e'];

// series: [{ label, y: number[], band?: [lo[], hi[]], color?, dash? }]
export default function Chart({ series, height = 160, xLabel = '', yLabel = '', yMin, yMax, refLine, refLabel, smooth = 1, x0 = 0, xStep = 1, fmt = (v) => v.toFixed(2) }) {
  const W = 420, H = height, P = { l: 38, r: 8, t: 8, b: 26 };
  const sm = (y) => {
    if (smooth <= 1) return y;
    return y.map((_, i) => { const a = Math.max(0, i - smooth + 1); let s = 0; for (let j = a; j <= i; j++) s += y[j]; return s / (i - a + 1); });
  };
  const S = useMemo(() => series.map((s) => ({ ...s, ys: sm(s.y) })), [series, smooth]); // eslint-disable-line
  const n = Math.max(1, ...S.map((s) => s.ys.length));
  const all = S.flatMap((s) => [...s.ys, ...(s.band ? [...s.band[0], ...s.band[1]] : [])]).filter(Number.isFinite);
  if (refLine != null) all.push(refLine);
  const lo = yMin ?? Math.min(...all, 0), hi = yMax ?? Math.max(...all, lo + 1e-9);
  const X = (i) => P.l + (i / Math.max(1, n - 1)) * (W - P.l - P.r);
  const Y = (v) => P.t + (1 - (v - lo) / (hi - lo || 1)) * (H - P.t - P.b);
  const path = (ys) => ys.map((v, i) => `${i ? 'L' : 'M'}${X(i).toFixed(1)},${Y(Math.max(lo, Math.min(hi, v))).toFixed(1)}`).join('');
  const [hov, setHov] = useState(null);
  const ticks = [lo, (lo + hi) / 2, hi];
  return (
    <div className="relative">
      <svg viewBox={`0 0 ${W} ${H}`} className="w-full select-none" style={{ height }}
        onMouseMove={(e) => { const r = e.currentTarget.getBoundingClientRect(); const i = Math.round((((e.clientX - r.left) / r.width) * W - P.l) / ((W - P.l - P.r) / Math.max(1, n - 1))); setHov(i >= 0 && i < n ? i : null); }}
        onMouseLeave={() => setHov(null)}>
        <rect x={P.l} y={P.t} width={W - P.l - P.r} height={H - P.t - P.b} fill="none" stroke="var(--color-line)" />
        {ticks.map((t, i) => (
          <g key={i}>
            <line x1={P.l} x2={W - P.r} y1={Y(t)} y2={Y(t)} stroke="var(--color-line)" strokeDasharray="2 3" />
            <text x={P.l - 4} y={Y(t) + 3} textAnchor="end" fontSize="9" fill="var(--color-dim)">{Math.abs(t) >= 100 ? t.toFixed(0) : t.toFixed(Math.abs(t) >= 10 ? 0 : 2)}</text>
          </g>
        ))}
        {refLine != null && <line x1={P.l} x2={W - P.r} y1={Y(refLine)} y2={Y(refLine)} stroke="var(--color-dim)" strokeDasharray="5 3" />}
        {refLine != null && refLabel && <text x={W - P.r - 2} y={Y(refLine) - 3} textAnchor="end" fontSize="9" fill="var(--color-dim)">{refLabel}</text>}
        {S.map((s, k) => s.band && (
          <path key={`b${k}`} d={`${path(sm(s.band[1]))}L${[...sm(s.band[0])].map((v, i) => [i, v]).reverse().map(([i, v]) => `${X(i).toFixed(1)},${Y(Math.max(lo, Math.min(hi, v))).toFixed(1)}`).join('L')}Z`} fill={s.color ?? SERIES[k % SERIES.length]} opacity="0.13" />
        ))}
        {S.map((s, k) => (
          <path key={k} d={path(s.ys)} fill="none" stroke={s.color ?? SERIES[k % SERIES.length]} strokeWidth={s.width ?? 1.8} strokeDasharray={s.dash} opacity={s.opacity ?? 1} />
        ))}
        {hov != null && <line x1={X(hov)} x2={X(hov)} y1={P.t} y2={H - P.b} stroke="var(--color-dim)" strokeWidth="0.5" />}
        <text x={(W + P.l) / 2} y={H - 6} textAnchor="middle" fontSize="9.5" fill="var(--color-dim)">{xLabel}</text>
        <text x={10} y={(H - P.b) / 2} textAnchor="middle" fontSize="9.5" fill="var(--color-dim)" transform={`rotate(-90 10 ${(H - P.b) / 2})`}>{yLabel}</text>
      </svg>
      <div className="mt-0.5 flex flex-wrap gap-x-3 gap-y-0.5 text-[10px] text-dim">
        {S.filter((s) => s.label).map((s, k) => (
          <span key={k} className="flex items-center gap-1">
            <span className="inline-block h-0.5 w-3" style={{ background: s.color ?? SERIES[series.indexOf(s) % SERIES.length] }} />
            {s.label}
            {hov != null && s.ys[hov] != null && <span className="tabular-nums text-slate-300">{fmt(s.ys[hov])}</span>}
          </span>
        ))}
        {hov != null && <span className="tabular-nums">at {x0 + hov * xStep}</span>}
      </div>
    </div>
  );
}

// Several runs of the same thing: mean line plus min-max band.
export function bandOf(runs) {
  const n = Math.max(...runs.map((r) => r.length));
  const at = (i) => runs.map((r) => r[i]).filter((v) => v != null);
  return {
    y: Array.from({ length: n }, (_, i) => { const a = at(i); return a.reduce((s, v) => s + v, 0) / a.length; }),
    band: [Array.from({ length: n }, (_, i) => Math.min(...at(i))), Array.from({ length: n }, (_, i) => Math.max(...at(i)))],
  };
}
