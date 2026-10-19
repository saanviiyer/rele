// CartPole (Barto, Sutton & Anderson 1983), with the constants and limits of Gymnasium's
// CartPole-v1: push the cart left or right to keep the pole up; +1 per step, at most 500.

export const CP = { g: 9.8, mc: 1.0, mp: 0.1, l: 0.5, F: 10, tau: 0.02, xMax: 2.4, thMax: (12 * Math.PI) / 180, maxSteps: 500 };

export function cpReset(rnd) {
  return [0, 0, 0, 0].map(() => (rnd() - 0.5) * 0.1); // x, xdot, theta, thetadot in [-0.05, 0.05]
}

export function cpStep(s, a) {
  const [x, xd, th, thd] = s;
  const f = a === 1 ? CP.F : -CP.F;
  const cos = Math.cos(th), sin = Math.sin(th), M = CP.mc + CP.mp, pml = CP.mp * CP.l;
  const temp = (f + pml * thd * thd * sin) / M;
  const thacc = (CP.g * sin - cos * temp) / (CP.l * (4 / 3 - (CP.mp * cos * cos) / M));
  const xacc = temp - (pml * thacc * cos) / M;
  const s2 = [x + CP.tau * xd, xd + CP.tau * xacc, th + CP.tau * thd, thd + CP.tau * thacc];
  const done = Math.abs(s2[0]) > CP.xMax || Math.abs(s2[2]) > CP.thMax;
  return { s2, r: 1, done };
}

// Scale the state to roughly unit size for a policy network.
export const cpFeatures = (s) => [s[0] / 2.4, s[1] / 3, s[2] / CP.thMax, s[3] / 3];
