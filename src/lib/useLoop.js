// Run a step function about 60 times a second while `on` is true. The step returns false to stop.
// A timer rather than requestAnimationFrame, so training keeps going in a pane the browser
// considers hidden.
import { useEffect, useRef } from 'react';

export function useLoop(step, on, { every = 1 } = {}) {
  const ref = useRef(step);
  ref.current = step;
  useEffect(() => {
    if (!on) return undefined;
    let id, alive = true;
    const tick = () => {
      if (!alive) return;
      if (ref.current() === false) return;
      id = setTimeout(tick, 16 * every);
    };
    id = setTimeout(tick, 16 * every);
    return () => { alive = false; clearTimeout(id); };
  }, [on, every]);
}

// Defer a heavy computation to after paint, so a button press shows "working" first.
export function later(fn) {
  return new Promise((res) => setTimeout(() => res(fn()), 30));
}
