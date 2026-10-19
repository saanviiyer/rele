// DQN on CartPole (Mnih et al. 2015), with its two stabilisers as switches: a replay buffer
// (sample old transitions at random, breaking the correlation of consecutive steps) and a target
// network (regress on labels from a frozen copy phi', so the target does not move every step).
// Loss: (r + gamma (1 - done) max_a' Q_phi'(s', a') - Q_phi(s, a))^2. With double = true the
// action is picked by Q_phi and scored by Q_phi' (van Hasselt et al. 2016).
import { rng, argmax } from './rng.js';
import { mlp, forward, backward, zeroGrads, adam, copyNet } from './nn.js';
import { cpReset, cpStep, cpFeatures, CP } from './cartpole.js';

export function makeDQN({ replay = true, target = true, double = false, gamma = 0.99, lr = 1e-3, hidden = 32, bufferSize = 10000, batch = 32, targetEvery = 250, epsStart = 1, epsEnd = 0.05, epsSteps = 4000, seed = 1, warmup = 500 } = {}) {
  const rnd = rng(seed);
  const net = mlp([4, hidden, 2], seed, 0.1);
  let tgt = copyNet(net);
  const buf = [];
  let s = cpReset(rnd), t = 0, epLen = 0, steps = 0;
  const episodes = [], losses = [], qStart = [];
  const Q = (n, phi) => forward(n, phi).out;
  const eps = () => Math.max(epsEnd, epsStart - (steps / epsSteps) * (epsStart - epsEnd));

  function trainStep() {
    const B = replay ? Array.from({ length: batch }, () => buf[rnd.int(buf.length)]) : [buf[buf.length - 1]];
    const g = zeroGrads(net);
    const tn = target ? tgt : net;
    let L = 0;
    for (const tr of B) {
      const f = forward(net, tr.phi);
      let next = 0;
      if (!tr.done) {
        const qn = Q(tn, tr.phi2);
        next = double ? qn[argmax(Q(net, tr.phi2))] : Math.max(...qn);
      }
      const y = tr.r + gamma * next;
      const err = f.out[tr.a] - y;
      L += err * err;
      const gOut = [0, 0];
      gOut[tr.a] = 2 * err;
      backward(net, f, gOut, g);
    }
    adam(net, g, { lr, n: B.length, clip: 10 });
    losses.push(L / B.length);
  }

  // Advance n environment steps (each followed by one gradient step once warmed up).
  function run(n) {
    for (let i = 0; i < n; i++) {
      const phi = cpFeatures(s);
      const a = rnd() < eps() ? rnd.int(2) : argmax(Q(net, phi));
      const { s2, r, done } = cpStep(s, a);
      epLen++; steps++;
      const timeout = epLen >= CP.maxSteps;
      buf.push({ phi, a, r, phi2: cpFeatures(s2), done });
      if (buf.length > (replay ? bufferSize : 1)) buf.shift();
      s = s2;
      if (done || timeout) { episodes.push(epLen); qStart.push(Math.max(...Q(net, cpFeatures([0, 0, 0, 0])))); s = cpReset(rnd); epLen = 0; }
      if (steps >= (replay ? warmup : 1)) trainStep();
      if (target && steps % targetEvery === 0) tgt = copyNet(net);
      t++;
    }
    return { steps, episodes, losses, qStart, eps: eps(), current: s };
  }
  return { run, net, get steps() { return steps; }, episodes, qStart };
}
