// An SVG gridworld: cells shaded by value, optional Q-value triangles, policy arrows (or arrow
// widths for a stochastic policy), the agent, paths, and hover/click/paint callbacks.
import React, { useState } from 'react';
import { useTheme } from '../lib/theme.js';

const PAPER = { light: [244, 238, 226], dark: [12, 16, 26] };
const POS = { light: [180, 95, 14], dark: [255, 181, 71] };
const NEG = { light: [47, 95, 191], dark: [91, 140, 255] };
const mix = (a, b, t) => a.map((x, i) => Math.round(x + (b[i] - x) * t));
export function valueColor(v, scale, theme) {
  const t = Math.max(-1, Math.min(1, v / (scale || 1)));
  const c = t >= 0 ? mix(PAPER[theme], POS[theme], t) : mix(PAPER[theme], NEG[theme], -t);
  return `rgb(${c.join(',')})`;
}

const ICON = { G: '★', L: '✕', C: '', P: '◆', S: 'S' };
const CELLFILL = { '#': 'var(--color-line)' };

// policy: Int8Array of actions (-1 none) or array of 4-probability arrays
export default function GridView({ env, values, scale, q, policy, agent, paths = [], mark, onHover, onClick, onPaint, size = 44, showValues = true, labelFmt = (v) => v.toFixed(2), dim }) {
  const theme = useTheme();
  const [drag, setDrag] = useState(false);
  const { W, H, cells } = env;
  const cs = size, w = W * cs, h = H * cs;
  const sc = scale ?? Math.max(1e-9, ...(values ? Array.from(values).map(Math.abs) : [1]));
  const cx = (s) => (s % W) * cs + cs / 2, cy = (s) => Math.floor(s / W) * cs + cs / 2;
  const tri = (s, a) => {
    const x = (s % W) * cs, y = Math.floor(s / W) * cs, m = [x + cs / 2, y + cs / 2];
    const corners = [[[x, y], [x + cs, y]], [[x + cs, y], [x + cs, y + cs]], [[x + cs, y + cs], [x, y + cs]], [[x, y + cs], [x, y]]][a];
    return `${m.join(',')} ${corners[0].join(',')} ${corners[1].join(',')}`;
  };
  const arrow = (s, a, len, width, color, key) => {
    const dx = [0, 1, 0, -1][a], dy = [-1, 0, 1, 0][a];
    const x0 = cx(s), y0 = cy(s), L = (cs / 2 - 6) * len;
    const x1 = x0 + dx * L, y1 = y0 + dy * L;
    return (
      <g key={key} opacity={0.95}>
        <line x1={x0} y1={y0} x2={x1} y2={y1} stroke={color} strokeWidth={width} strokeLinecap="round" />
        <polygon points={`${x1 + dx * 4},${y1 + dy * 4} ${x1 - dy * 3.5},${y1 + dx * 3.5} ${x1 + dy * 3.5},${y1 - dx * 3.5}`} fill={color} />
      </g>
    );
  };
  const at = (e) => {
    const r = e.currentTarget.getBoundingClientRect();
    const c = Math.floor(((e.clientX - r.left) / r.width) * W), rr = Math.floor(((e.clientY - r.top) / r.height) * H);
    return c >= 0 && c < W && rr >= 0 && rr < H ? rr * W + c : null;
  };
  return (
    <svg viewBox={`-1 -1 ${w + 2} ${h + 2}`} className="w-full touch-none select-none" style={{ maxWidth: w + 2, opacity: dim ? 0.5 : 1 }}
      onPointerMove={(e) => { const s = at(e); onHover?.(s); if (drag && s != null) onPaint?.(s); }}
      onPointerLeave={() => { onHover?.(null); setDrag(false); }}
      onPointerDown={(e) => { const s = at(e); if (s == null) return; if (onPaint) { setDrag(true); onPaint(s); } onClick?.(s); }}
      onPointerUp={() => setDrag(false)}>
      {cells.map((ch, s) => {
        const x = (s % W) * cs, y = Math.floor(s / W) * cs;
        const fill = CELLFILL[ch] ?? (ch === 'C' ? valueColor(-1, 1, theme) : values && !env.isTerminal(s) ? valueColor(values[s], sc, theme) : ch === 'G' ? valueColor(0.7, 1, theme) : ch === 'L' ? valueColor(-0.7, 1, theme) : PAPER[theme] && `rgb(${PAPER[theme].join(',')})`);
        return (
          <g key={s}>
            <rect x={x} y={y} width={cs} height={cs} fill={fill} stroke="var(--color-line)" strokeWidth={0.8} />
            {q && !env.isTerminal(s) && ch !== '#' && [0, 1, 2, 3].map((a) => (
              <polygon key={a} points={tri(s, a)} fill={valueColor(q[s][a], sc, theme)} stroke="var(--color-line)" strokeWidth={0.4} />
            ))}
            {ICON[ch] && <text x={x + 4} y={y + 12} fontSize={10} fill="var(--color-dim)">{ICON[ch]}</text>}
            {ch === 'C' && <text x={x + cs / 2} y={y + cs / 2 + 3} fontSize={8} textAnchor="middle" fill="var(--color-white)" opacity={0.7}>cliff</text>}
            {showValues && values && ch !== '#' && !env.isTerminal(s) && ch !== 'C' && cs >= 34 && (
              <text x={x + cs - 3} y={y + cs - 4} fontSize={cs >= 48 ? 9.5 : 8} textAnchor="end" fill="var(--color-slate-200)" className="tabular-nums">{labelFmt(values[s])}</text>
            )}
            {env.isTerminal(s) && <text x={x + cs / 2} y={y + cs / 2 + 4} fontSize={10} textAnchor="middle" fill="var(--color-slate-100)">{env.rewards[ch] > 0 ? `+${env.rewards[ch]}` : env.rewards[ch]}</text>}
          </g>
        );
      })}
      {policy && cells.map((ch, s) => {
        if (ch === '#' || env.isTerminal(s) || ch === 'C') return null;
        const p = policy[s];
        if (p == null || p === -1) return null;
        if (typeof p === 'number') return arrow(s, p, 0.75, 2, 'var(--color-slate-100)', `p${s}`);
        return p.map((pa, a) => (pa > 0.04 ? arrow(s, a, 0.25 + 0.6 * pa, 0.6 + 3 * pa, 'var(--color-slate-100)', `p${s}${a}`) : null));
      })}
      {paths.map((pth, k) => pth && pth.length > 1 && (
        <polyline key={k} points={pth.map((s) => `${cx(s)},${cy(s)}`).join(' ')} fill="none" stroke={pth.color ?? 'var(--color-warm)'} strokeWidth={pth.width ?? 2.5} strokeOpacity={pth.opacity ?? 0.8} strokeLinejoin="round" strokeDasharray={pth.dash} />
      ))}
      {mark != null && <rect x={(mark % W) * cs + 1.5} y={Math.floor(mark / W) * cs + 1.5} width={cs - 3} height={cs - 3} fill="none" stroke="var(--color-glow)" strokeWidth={2.5} />}
      {agent != null && <circle cx={cx(agent)} cy={cy(agent)} r={cs * 0.22} fill="var(--color-warm)" stroke="var(--color-panel)" strokeWidth={2} />}
    </svg>
  );
}
