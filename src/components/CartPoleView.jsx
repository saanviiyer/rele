// Draws a CartPole state: track, cart, pole, and the failure limits.
import React from 'react';
import { CP } from '../lib/cartpole.js';

export default function CartPoleView({ s, action, height = 130, label }) {
  const W = 420, H = height, scale = W / (2 * CP.xMax * 1.15);
  const cx = W / 2 + (s?.[0] ?? 0) * scale, cy = H - 34;
  const len = 2 * CP.l * scale * 1.6;
  const th = s?.[2] ?? 0;
  const px = cx + Math.sin(th) * len, py = cy - Math.cos(th) * len;
  const lim = (x) => W / 2 + x * scale;
  return (
    <svg viewBox={`0 0 ${W} ${H}`} className="w-full rounded-md border border-line" style={{ background: 'var(--color-panel)' }}>
      <line x1="0" x2={W} y1={cy + 12} y2={cy + 12} stroke="var(--color-dim)" />
      <line x1={lim(-CP.xMax)} x2={lim(-CP.xMax)} y1={cy - 8} y2={cy + 16} stroke="var(--color-warm)" strokeDasharray="3 2" />
      <line x1={lim(CP.xMax)} x2={lim(CP.xMax)} y1={cy - 8} y2={cy + 16} stroke="var(--color-warm)" strokeDasharray="3 2" />
      <rect x={cx - 24} y={cy - 8} width="48" height="20" rx="3" fill="var(--color-cool)" />
      <line x1={cx} y1={cy - 6} x2={px} y2={py} stroke="var(--color-warm)" strokeWidth="6" strokeLinecap="round" />
      <circle cx={cx} cy={cy - 6} r="3.5" fill="var(--color-slate-100)" />
      {action != null && <text x={cx + (action ? 34 : -34)} y={cy + 6} fontSize="16" textAnchor="middle" fill="var(--color-glow)">{action ? '→' : '←'}</text>}
      {label && <text x="8" y="16" fontSize="11" fill="var(--color-dim)">{label}</text>}
    </svg>
  );
}
