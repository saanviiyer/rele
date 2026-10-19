// The value function as a landscape: one column per cell, its height the state's value, walls
// as dark blocks, the greedy policy as small arrows on top. Drag to orbit.
import React, { useMemo } from 'react';
import { Canvas } from '@react-three/fiber';
import { OrbitControls, Html } from '@react-three/drei';
import { useTheme, sceneHex } from '../lib/theme.js';
import { valueColor } from './GridView.jsx';

export default function Terrain({ env, V, policy, agent, height = 300 }) {
  const theme = useTheme();
  const { W, H } = env;
  const sc = Math.max(1e-9, ...Array.from(V).map(Math.abs));
  const cols = useMemo(() => env.cells.map((ch, s) => {
    const x = (s % W) - (W - 1) / 2, z = Math.floor(s / W) - (H - 1) / 2;
    const v = ch === '#' ? 0.6 : env.isTerminal(s) ? ((env.rewards[ch] ?? 0) / Math.max(Math.abs(env.rewards[ch] ?? 1), sc)) * 2.4 : (V[s] / sc) * 2.4;
    return { s, x, z, h: Math.max(0.03, Math.abs(v)), up: v >= 0, ch };
  }), [env, V, sc, W, H]);
  return (
    <div style={{ height }} className="overflow-hidden rounded-md border border-line">
      <Canvas camera={{ position: [W * 0.35, Math.max(W, H) * 0.7, H * 0.95], fov: 45 }} dpr={[1, 2]}>
        <color attach="background" args={[sceneHex(theme).bg]} />
        <ambientLight intensity={0.8} />
        <directionalLight position={[4, 8, 5]} intensity={0.9} />
        {cols.map((c) => (
          <mesh key={c.s} position={[c.x, c.up ? c.h / 2 : -c.h / 2, c.z]}>
            <boxGeometry args={[0.92, c.h, 0.92]} />
            <meshStandardMaterial color={c.ch === '#' ? (theme === 'light' ? '#8a7f6e' : '#2a3246') : valueColor(c.up ? c.h : -c.h, 2.4, theme)} />
          </mesh>
        ))}
        {policy && cols.map((c) => {
          const a = policy[c.s];
          if (a == null || a < 0 || c.ch === '#') return null;
          const rot = [0, -Math.PI / 2, Math.PI, Math.PI / 2][a];
          return (
            <mesh key={`a${c.s}`} position={[c.x, (c.up ? c.h : 0) + 0.06, c.z]} rotation={[-Math.PI / 2, 0, rot]}>
              <coneGeometry args={[0.13, 0.36, 3]} />
              <meshBasicMaterial color={sceneHex(theme).ink} />
            </mesh>
          );
        })}
        {agent != null && (
          <mesh position={[(agent % W) - (W - 1) / 2, cols[agent].up ? cols[agent].h + 0.25 : 0.25, Math.floor(agent / W) - (H - 1) / 2]}>
            <sphereGeometry args={[0.22, 20, 14]} />
            <meshStandardMaterial color={sceneHex(theme).accent} />
          </mesh>
        )}
        {env.cells.map((ch, s) => (env.isTerminal(s) ? (
          <Html key={`t${s}`} position={[(s % W) - (W - 1) / 2, 0.9, Math.floor(s / W) - (H - 1) / 2]} center zIndexRange={[10, 0]} style={{ pointerEvents: 'none' }}>
            <div className="text-[11px] font-semibold" style={{ color: sceneHex(theme).ink }}>{ch === 'G' ? '★' : '✕'}</div>
          </Html>
        ) : null))}
        <OrbitControls enablePan={false} target={[0, 0, 0]} />
      </Canvas>
    </div>
  );
}
