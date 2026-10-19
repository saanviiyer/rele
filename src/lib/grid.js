// Gridworlds: the Markov decision processes every chapter after the bandits runs on.
// A world is a list of strings, one character per cell:
//   .  empty        #  wall           S  start
//   G  goal (ends the episode)        L  pit (ends the episode)
//   C  cliff (sends you back to start) P  bonus pad (pays and stays open)
// Entering a cell pays rewards[cell] if it has one, else the step reward.
// Actions: 0 up, 1 right, 2 down, 3 left. With slip > 0 the move goes sideways with
// probability slip/2 to each side (Russell & Norvig 2020, section 17.1). Gymnasium's slippery
// FrozenLake is slip = 2/3: each of the intended and two sideways moves has probability 1/3.

export const ACTIONS = ['up', 'right', 'down', 'left'];
export const ARROWS = ['↑', '→', '↓', '←'];
const DR = [-1, 0, 1, 0], DC = [0, 1, 0, -1];

export const PRESETS = {
  classic: {
    label: '4x3 world (Russell & Norvig)',
    rows: ['...G', '.#.L', 'S...'],
    step: -0.04, slip: 0.2, gamma: 1, rewards: { G: 1, L: -1 },
  },
  cliff: {
    label: 'cliff walk (Sutton & Barto)',
    rows: ['............', '............', '............', 'SCCCCCCCCCCG'],
    step: -1, slip: 0, gamma: 1, rewards: { G: 0, C: -100 },
  },
  maze: {
    label: 'small maze',
    rows: ['S..#....', '.#.#.##.', '.#...#..', '.####.#.', '......#G'],
    step: -1, slip: 0, gamma: 0.95, rewards: { G: 10 },
  },
  frozen: {
    label: 'FrozenLake 4x4 (Gymnasium)',
    rows: ['S...', '.L.L', '...L', 'L..G'],
    step: 0, slip: 0, gamma: 0.99, rewards: { G: 1, L: 0 },
  },
  eight: {
    label: '8x8 world with a +1 and a -1 (the notes)',
    rows: ['S.......', '........', '...#....', '...#..L.', '...#....', '........', '.....#..', '.......G'],
    step: 0, slip: 0, gamma: 0.9, rewards: { G: 1, L: -1 },
  },
  corridor: {
    label: 'imitation corridor between pits',
    rows: ['LLLLLLLLLLLL', '............', '............', 'S..........G', '............', '............', 'LLLLLLLLLLLL'],
    step: -0.05, slip: 0.2, gamma: 0.97, rewards: { G: 5, L: -5 },
  },
  windy: {
    label: 'slippery room with pits',
    rows: ['S....', '.L.L.', '.....', 'L.L.G'],
    step: -0.1, slip: 0.3, gamma: 0.95, rewards: { G: 5, L: -5 },
  },
  pad: {
    label: 'race with a bonus pad',
    rows: ['S.P...G'],
    step: 0, slip: 0, gamma: 0.99, rewards: { G: 10, P: 1 },
  },
};

export function makeGrid({ rows, step = -0.04, slip = 0, gamma = 0.95, rewards = {} }) {
  const H = rows.length, W = Math.max(...rows.map((r) => r.length));
  const cells = [];
  for (let r = 0; r < H; r++) for (let c = 0; c < W; c++) cells.push(rows[r][c] ?? '.');
  const N = W * H;
  const start = Math.max(0, cells.indexOf('S'));
  const isTerminal = (s) => cells[s] === 'G' || cells[s] === 'L';
  const isWall = (s) => cells[s] === '#';
  const move = (s, a) => {
    const r = Math.floor(s / W) + DR[a], c = (s % W) + DC[a];
    if (r < 0 || r >= H || c < 0 || c >= W) return s;
    const s2 = r * W + c;
    return isWall(s2) ? s : s2;
  };
  const rewardOf = (s2) => (cells[s2] in rewards ? rewards[cells[s2]] : step);
  // where a move lands once the cell's own effect is applied (cliff: back to start)
  const outcome = (s2) => ({ s2: cells[s2] === 'C' ? start : s2, r: rewardOf(s2), done: isTerminal(s2) });
  const trans = (s, a) => {
    if (isTerminal(s) || isWall(s)) return [];
    const opts = slip > 0 ? [[a, 1 - slip], [(a + 1) % 4, slip / 2], [(a + 3) % 4, slip / 2]] : [[a, 1]];
    const out = new Map();
    for (const [d, p] of opts) {
      const o = outcome(move(s, d));
      const key = `${o.s2}:${o.r}:${o.done}`;
      if (out.has(key)) out.get(key).p += p; else out.set(key, { ...o, p });
    }
    return [...out.values()];
  };
  const stepEnv = (s, a, rnd) => {
    let d = a;
    if (slip > 0) { const u = rnd(); if (u < slip / 2) d = (a + 1) % 4; else if (u < slip) d = (a + 3) % 4; }
    return { ...outcome(move(s, d)), dir: d };
  };
  const states = Array.from({ length: N }, (_, s) => s).filter((s) => !isWall(s));
  return {
    rows, W, H, N, cells, start, step, slip, gamma, rewards, states,
    isTerminal, isWall, trans, step_: stepEnv, stepEnv, rewardOf,
    rc: (s) => [Math.floor(s / W), s % W],
  };
}

export const gridOf = (key, over = {}) => makeGrid({ ...PRESETS[key], ...over });
