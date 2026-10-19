import React, { useEffect, useState } from 'react';
import { Title, Cite, Sources, Boundary, Takeaway, Btn } from './components/ui.jsx';
import { useTheme, setTheme } from './lib/theme.js';
import { CHAPTERS } from './chapters.jsx';

function About({ onClose }) {
  return (
    <div className="fixed inset-0 z-[100] flex items-center justify-center bg-black/70 p-4" onClick={onClose}>
      <div className="w-full max-w-xl space-y-3 rounded-lg border border-line bg-panel p-5 text-sm leading-relaxed text-slate-300" onClick={(e) => e.stopPropagation()}>
        <h1 className="text-lg"><Title /></h1>
        <p>
          ReLe is a set of small, live simulations for learning reinforcement learning, from one slot machine to the methods used to post-train language models.
          Each chapter follows a lecture of Stanford's CS 224R (Deep Reinforcement Learning, Spring 2026)<Cite id="cs224r" /> and the matching section of
          Aaditya Nalawade's notes<Cite id="nalawade" />, and turns its main idea into something you can run, break and measure.
        </p>
        <p>
          Every agent here learns in your browser while you watch, with seeded randomness, so the same settings give the same curves. Every number the page
          states is checked by a test (<code className="text-xs">npm test</code>) against a textbook value or an exact identity. Where an example goes beyond
          the two sources, the chapter says so.
        </p>
        <div className="text-right"><Btn onClick={onClose}>Close</Btn></div>
      </div>
    </div>
  );
}

const fromHash = () => {
  const n = parseInt((typeof location !== 'undefined' ? location.hash : '').replace('#', ''), 10);
  return n >= 1 && n <= CHAPTERS.length ? n : 1;
};

export default function App() {
  const [n, setN] = useState(fromHash);
  const [sources, setSources] = useState(false);
  const [about, setAbout] = useState(false);
  const theme = useTheme();
  useEffect(() => { try { history.replaceState(null, '', `${location.pathname}${location.search}#${n}`); } catch { /* ignore */ } }, [n]);
  useEffect(() => {
    const f = () => setN(fromHash());
    window.addEventListener('hashchange', f);
    return () => window.removeEventListener('hashchange', f);
  }, []);
  const ch = CHAPTERS[n - 1];
  const Body = ch.Body;
  return (
    <div className="flex h-full flex-col">
      <header className="flex flex-wrap items-center gap-x-4 gap-y-2 border-b border-line px-4 py-2">
        <button onClick={() => setAbout(true)} className="text-left text-sm" title="About this project"><Title /></button>
        <nav className="flex flex-1 flex-wrap gap-1">
          {CHAPTERS.map((c, i) => (
            <button key={c.name} onClick={() => setN(i + 1)} title={c.source}
              className={`rounded px-2 py-1 text-xs ${n === i + 1 ? 'bg-glow/15 text-glow' : 'text-dim hover:text-slate-200'}`}>
              {i + 1}. {c.name}
            </button>
          ))}
        </nav>
        <div className="flex items-center gap-3 text-xs">
          <button className="text-glow/80 hover:text-glow" onClick={() => setAbout(true)}>About</button>
          <button className="text-glow/80 hover:text-glow" onClick={() => setSources(true)}>Sources</button>
          <button className="rounded-md border border-line bg-panel px-2 py-0.5 text-slate-300 hover:border-glow/40" onClick={() => setTheme(theme === 'light' ? 'dark' : 'light')} title="switch between the light and dark themes">
            {theme === 'light' ? 'Dark mode' : 'Light mode'}
          </button>
        </div>
      </header>

      <main className="flex min-h-0 flex-1 flex-col overflow-y-auto lg:flex-row lg:overflow-hidden">
        <aside className="scroll-thin w-full shrink-0 space-y-3 border-line p-4 lg:w-80 lg:overflow-y-auto lg:border-r">
          <div>
            <p className="text-xs uppercase tracking-[0.2em] text-dim">Chapter {n} · {ch.name}</p>
            <p className="mt-2 text-sm leading-relaxed text-slate-100">{ch.intro}</p>
          </div>
          <div className="rounded-lg border border-line p-3 text-xs leading-relaxed text-slate-300">
            <div className="mb-1 text-[10px] uppercase tracking-[0.18em] text-dim">In the sources</div>
            {ch.sources}
          </div>
          {ch.notation && (
            <div className="rounded-lg border border-line p-3 text-xs leading-relaxed text-slate-300">
              <div className="mb-1 text-[10px] uppercase tracking-[0.18em] text-dim">Notation</div>
              {ch.notation}
            </div>
          )}
          <Takeaway>{ch.takeaway}</Takeaway>
          <div className="flex justify-between">
            <Btn disabled={n === 1} onClick={() => setN(n - 1)}>Previous</Btn>
            <Btn disabled={n === CHAPTERS.length} onClick={() => setN(n + 1)}>Next chapter</Btn>
          </div>
        </aside>
        <section className="scroll-thin min-w-0 flex-1 p-4 lg:overflow-y-auto">
          <Boundary key={n}><Body /></Boundary>
        </section>
      </main>
      <footer className="border-t border-line px-4 py-1.5 text-[10px] text-dim">
        Following Stanford CS 224R<Cite id="cs224r" /> and Nalawade's RL notes<Cite id="nalawade" />. Everything runs in your browser; every stated number is checked by a test.
      </footer>
      {sources && <Sources onClose={() => setSources(false)} />}
      {about && <About onClose={() => setAbout(false)} />}
    </div>
  );
}
