# it's ReLe that simple?

Interactive simulations for learning reinforcement learning, from one slot machine to the methods used to post-train language models. Every agent learns live in your browser.

**Live:** https://saanviiyer.github.io/rele/

The chapters follow Stanford's [CS 224R: Deep Reinforcement Learning](https://cs224r.stanford.edu/) (Chelsea Finn, Spring 2026) lecture by lecture. Alongside them run the matching sections of Aaditya Nalawade's [Reinforcement Learning notes](https://www.aadityan.com/blog/reinforcement-learning). Where an example goes beyond those two sources (bandits, SARSA, the cliff walk), the chapter says so and cites Sutton & Barto.

## Chapters

1. **Trial and error** (supplementary). You play five slot machines against Thompson sampling. Six agents (greedy, epsilon-greedy, optimistic, UCB, Thompson, gradient bandit) run on the 10-armed testbed.
2. **Behaviour as data** (lecture 1). s, a, tau and r for the lecture's towel robot and chatbot, plus think-pair-share prompts. Every trajectory of a 2-state MDP is listed with p_theta(tau) written as a product of factors. Also: the one RL loop behind every method, and the Markov property.
3. **Planning with a model** (notes, section 1). Value iteration, sweep by sweep, on an editable gridworld: the notes' 8x8 world, Russell & Norvig's 4x3, FrozenLake and others. Hover any cell to see its Bellman backup. Includes a 3D value terrain, policy iteration, and lecture 6's "is greedy on Q^pi better" thought experiment.
4. **Imitation** (lecture 2). With two-mode demonstrations, an l2 policy drives into the obstacle and a categorical one does not. Behaviour cloning compounds its errors on slippery ice, and DAgger is shown against BC given the same number of labels.
5. **Policy gradients** (lecture 3, notes section 3). The lecture's humanoid trajectories show which way the gradient pushes, with causality and baseline toggles. REINFORCE learns CartPole live with the notes' three choices of Psi_t. You can balance the pole yourself, and a demo shows how importance weights fall apart with horizon.
6. **Actor-critic** (lecture 4). Monte Carlo vs TD(0) on the random walk, and the bias and variance of each target. Sutton & Barto's "you are the predictor". A GAE and n-step calculator, and a lambda sweep on CartPole.
7. **PPO** (lecture 5, notes section 4). A clipped-surrogate explorer, live PPO with a clip-fraction readout, and PPO against REINFORCE.
8. **Q-learning** (lecture 6, notes section 2). FrozenLake, deterministic and slippery, learned live. Q-learning vs SARSA on the cliff, maximisation bias with double Q, and DQN with switches for the replay buffer, target network and double DQN.
9. **Offline RL** (lecture 7). The stitching example compares BC, filtered BC, AWR and IQL. Also: out-of-distribution overestimation in a 1D action space, naive vs data-constrained fitted Q on a logged gridworld, and the expectile.
10. **Reward learning** (lecture 8). A frozen goal classifier gets fooled and VICE-style negatives fix it. Bradley-Terry recovers a hidden reward from comparisons.
11. **RLHF and DPO** (lecture 9, notes sections 5-6). A 30-response "language model" with length-biased annotators. Shows the KL-regularised optimum, reward-model over-optimisation as beta falls, and DPO trained on the same pairs.
12. **GRPO and DAPO** (notes sections 7-9). Group-relative advantages and the k3 KL estimate. Covers Dr GRPO's length and difficulty biases, DAPO's dynamic sampling, and clip-higher.

## Checked numbers

`npm test` runs `scripts/test_rl.mjs` (46 checks, a few seconds). Among them:

- **Value iteration:** reproduces Russell & Norvig's 4x3 utilities to three decimals (0.812, 0.868, 0.918 / 0.762, 0.660 / 0.705, 0.655, 0.611, 0.388). It agrees with policy iteration, and gives gamma^(d-1) on a corridor.
- **FrozenLake:** Q-learning reaches 100% on the deterministic map. On the slippery one the optimal policy succeeds 75.0% of the time within 100 steps.
- **Cliff walk:** SARSA earns about -30 per episode while learning and Q-learning about -49, as in Sutton & Barto Fig. 6.4. Q-learning's greedy path hugs the cliff in 13 steps.
- **Random walk:** TD(0) beats Monte Carlo in RMS error after 100 episodes.
- **Policy gradient:**
  - Each step's weights average to 0 with the mean baseline.
  - The effective sample share of importance weights falls from 0.92 at T = 2 to 0.12 at T = 50.
  - GAE equals the TD error at lambda = 0 and the Monte Carlo advantage at lambda = 1.
  - The PPO surrogate gives 1.2 / 0.5 / -0.8 / -1.5 at the lecture's four cases.
- **CartPole:** REINFORCE with reward-to-go passes 300 steps within 60 iterations. DQN with replay and a target network averages about 450 over its last 20 episodes.
- **Imitation:** with two-mode demos, the l2 policy crashes 100% and the categorical one 0%. DAgger lifts success from 0.57 to 0.89.
- **Offline:** only IQL stitches s1 to s9. Naive fitted Q predicts 4.06 at the start and earns -20 (it never finishes); constrained fitted Q predicts and earns -1.09.
- **Preferences:**
  - Bradley-Terry recovers the hidden reward direction (cosine 1.000).
  - The KL-regularised optimum is pi_ref as beta goes to infinity and greedy as beta goes to 0.
  - True quality peaks at about 0.86, then falls as KL grows.
  - The DPO loss starts at log 2.
  - GRPO advantages for [1, 0, 0, 1] are [1, -1, -1, 1].
  - Under 1/|o| normalisation, wrong answers grow from 25 to 32.6 tokens.

## Notes on the sources

- **Notation:** this site follows CS 224R (s_t, a_t, r(s_t, a_t), J(theta)). For PPO and the language-model chapters it switches to theta_old and pi_ref, as the papers and the notes do. Each chapter has a notation box saying where the two sources differ.
- **Corrections to the notes:**
  - They attribute DAPO to Alibaba; the paper is from ByteDance Seed and Tsinghua AIR.
  - Their DQN text calls the target network a moving average, while their code copies the weights. This site copies.
  - Their clip-higher example's "1.08" is a bound on the probability ratio, not a reachable probability.
- **Toy numbers:** the per-step rewards in the humanoid widget are illustrative, because the slides show videos. The page says so.

## Run locally

```
npm install
npm run dev
npm test
```

Vite, React, Tailwind and react-three-fiber. There is no backend and nothing is downloaded at runtime.
