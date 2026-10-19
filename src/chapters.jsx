// The chapters, in the order of CS 224R's lectures, with the notes' sections alongside.
import React from 'react';
import { Cite } from './components/ui.jsx';
import Bandits from './components/ch/Bandits.jsx';
import Behaviour from './components/ch/Behaviour.jsx';
import Planning from './components/ch/Planning.jsx';
import Imitation from './components/ch/Imitation.jsx';
import PolicyGradient from './components/ch/PolicyGradient.jsx';
import ActorCritic from './components/ch/ActorCritic.jsx';
import PPO from './components/ch/PPO.jsx';
import QLearning from './components/ch/QLearning.jsx';
import Offline from './components/ch/Offline.jsx';
import RewardLearning from './components/ch/RewardLearning.jsx';
import RLHF from './components/ch/RLHF.jsx';
import GRPO from './components/ch/GRPO.jsx';

const L = (n, t) => <><b className="text-slate-100">CS 224R lecture {n}</b>, {t}<Cite id="cs224r" /></>;
const N = (t) => <><b className="text-slate-100">Nalawade's notes</b>, {t}<Cite id="nalawade" /></>;

export const CHAPTERS = [
  {
    name: 'Trial and error',
    source: 'supplementary: Sutton & Barto ch. 2',
    Body: Bandits,
    intro: <>Before states and plans, the smallest version of the problem: you can only learn what a choice is worth by trying it.</>,
    sources: <>Neither course source covers bandits, so this chapter follows Sutton & Barto, chapter 2.<Cite id="sutton2018" /> It is here because exploration comes back in every later chapter: epsilon-greedy in Q-learning, the entropy bonus in PPO, clip-higher in DAPO.</>,
    notation: <>A bandit has one state, so Q(a) is just the expected payout of arm a.</>,
    takeaway: <>Never trying anything new locks in an early guess; trying at random forever wastes pulls. Methods that explore where they are unsure (UCB, Thompson sampling) do better than both.</>,
  },
  {
    name: 'Behaviour as data',
    source: 'CS 224R lecture 1; notes section 1',
    Body: Behaviour,
    intro: <>Describe a decision problem as states, actions, trajectories and rewards, then write down what "good behaviour" means as one number to maximize.</>,
    sources: <>{L(1, 'course intro and modeling behavior.')} {N('section 1, the agent-environment setup and the Markov assumption.')}</>,
    notation: <>This site follows the lecture: sₜ, aₜ, r(sₜ, aₜ), t = 1..T, trajectory τ, objective J(θ). The notes write the MDP as (A, S, R, P, γ) and the return as Gₜ = Σₖ γᵏ Rₜ₊ₖ₊₁; here Gₜ means the reward-to-go Σ_t′≥t γ^(t′−t) r_t′.</>,
    takeaway: <>RL learns behaviour π(a | s) from indirect feedback on data its own actions produce. The objective is an expectation over trajectories, and only the policy's terms depend on θ.</>,
  },
  {
    name: 'Planning with a model',
    source: 'notes section 1 (MDPs, policy and value iteration)',
    Body: Planning,
    intro: <>If you know exactly how the world responds, you can compute the best behaviour without trying anything, by repeatedly asking each state what it is worth.</>,
    sources: <>{N('section 1: policy evaluation, policy iteration and value iteration on an 8x8 gridworld with a +1 and a -1 cell.')} {L(6, 'the greedy-on-Q thought experiment and the Bellman optimality equation.')} The 4x3 world is Russell & Norvig's.<Cite id="russell2020" /></>,
    notation: <>V^π(s): expected return from s under π. Q^π(s, a): the same after taking a first. A^π = Q^π − V^π. The notes write the reward as R(s) in places and R(s′) in others; here the reward is paid on entering a cell.</>,
    takeaway: <>The Bellman backup turns "what is this state worth" into a local calculation. Value iteration repeats it with a max until nothing changes; policy iteration alternates exact evaluation and greedy improvement. Both need the model p(s′ | s, a), which the rest of RL does without.</>,
  },
  {
    name: 'Imitation',
    source: 'CS 224R lecture 2',
    Body: Imitation,
    intro: <>The simplest way to get behaviour: copy an expert's actions with supervised learning. It works, until the copy makes one mistake.</>,
    sources: <>{L(2, 'imitation learning: expressive policies, compounding errors, DAgger.')} Behaviour cloning goes back to ALVINN<Cite id="pomerleau1989" />; DAgger is Ross et al.<Cite id="ross2011" /></>,
    notation: <>Demonstrations D = {'{'}(s₁, a₁, ..., s_T){'}'} from an unknown π_expert. Version 0: min_θ Σ ||a − π_θ(s)||². Version 1: min_θ −E log π_θ(a | s).</>,
    takeaway: <>A policy that predicts the mean action fails on data with two right answers. A cloned policy that drifts off the expert's states has no data there, and its mistakes compound. DAgger fixes the second by labelling the states the learner itself visits.</>,
  },
  {
    name: 'Policy gradients',
    source: 'CS 224R lecture 3; notes section 3',
    Body: PolicyGradient,
    intro: <>Turn trial and error into calculus: make the actions in good trajectories more likely and the actions in bad ones less likely.</>,
    sources: <>{L(3, 'policy gradients: REINFORCE, causality, baselines, importance sampling.')} {N('section 3: REINFORCE on CartPole with three choices of Ψₜ.')} REINFORCE is Williams 1992<Cite id="williams1992" />; CartPole is Barto, Sutton & Anderson 1983.<Cite id="barto1983" /></>,
    notation: <>∇_θ J(θ) ≈ (1/N) Σᵢ Σₜ ∇_θ log π_θ(aᵢ,ₜ | sᵢ,ₜ) Ψᵢ,ₜ. The notes call the weight Ψₜ: the whole return R(τ), the reward-to-go, or reward-to-go minus a baseline.</>,
    takeaway: <>The policy gradient is an imitation gradient weighted by reward. Using only the rewards that come after an action (causality) and subtracting a baseline both keep it unbiased and make it far less noisy.</>,
  },
  {
    name: 'Actor-critic',
    source: 'CS 224R lecture 4; notes section 3',
    Body: ActorCritic,
    intro: <>Instead of waiting for a whole episode to judge an action, learn a value function and judge it after one step.</>,
    sources: <>{L(4, 'actor-critic: Monte Carlo vs bootstrapped value targets, n-step returns, the advantage.')} {N('section 3: actor-critic and GAE.')} GAE is Schulman et al. 2016.<Cite id="schulman2016" /> The random walk is Sutton & Barto's Example 6.2.<Cite id="sutton2018" /></>,
    notation: <>Critic V_φ. TD error δₜ = rₜ + γ V_φ(sₜ₊₁) − V_φ(sₜ). GAE: Âₜ = Σₗ (γλ)ˡ δₜ₊ₗ.</>,
    takeaway: <>Monte Carlo targets are unbiased and noisy; bootstrapped (TD) targets are steady but inherit the critic's errors. n-step returns and GAE's λ slide between the two.</>,
  },
  {
    name: 'PPO',
    source: 'CS 224R lecture 5; notes section 4',
    Body: PPO,
    intro: <>Reuse each batch of experience for many gradient steps without letting the policy run away from the data that justified them.</>,
    sources: <>{L(5, 'off-policy actor-critic: importance weights, the clipped surrogate, GAE.')} {N('section 4: PPO with value loss and entropy bonus.')} PPO is Schulman et al. 2017<Cite id="schulman2017" />, after TRPO.<Cite id="schulman2015" /></>,
    notation: <>Ratio ρₜ(θ) = π_θ(aₜ | sₜ) / π_θ_old(aₜ | sₜ). Lecture 5 writes the new policy as θ′ and the data policy as θ; this site uses θ and θ_old, as the paper and the notes do.</>,
    takeaway: <>Clipping the ratio removes any reward for pushing an action's probability beyond 1 ± ε of where it was, so several epochs on one batch stay safe. It is stable and simple, and it still needs fresh data every iteration.</>,
  },
  {
    name: 'Q-learning',
    source: 'CS 224R lecture 6; notes section 2',
    Body: QLearning,
    intro: <>Drop the policy network: learn the value of the best action directly, and act by taking the argmax.</>,
    sources: <>{L(6, 'Q-learning: the Bellman optimality target, replay buffers, target networks, overestimation.')} {N('section 2: tabular Q-learning on FrozenLake and DQN.')} Q-learning is Watkins<Cite id="watkins1992" />; DQN is Mnih et al.<Cite id="mnih2015" /> The cliff and SARSA are supplementary, from Sutton & Barto.<Cite id="sutton2018" /></>,
    notation: <>Target yᵢ = rᵢ + γ maxₐ′ Q_φ′(s′ᵢ, a′) with target parameters φ′. The notes' DQN text calls the target network a moving average, but their code copies the weights every 5000 steps; this page copies.</>,
    takeaway: <>Q-learning is off-policy: it learns the greedy policy's value from any experience. With a table it converges; with a network it needs a replay buffer and a target network to stay stable, and its max still overrates noisy estimates.</>,
  },
  {
    name: 'Offline RL',
    source: 'CS 224R lecture 7',
    Body: Offline,
    intro: <>Learn from a fixed log of someone else's behaviour, with no chance to try anything new.</>,
    sources: <>{L(7, 'offline RL: distribution shift, filtered BC, AWR, IQL and the stitching example.')} See also Levine et al.'s tutorial<Cite id="levine2020" />, BCQ<Cite id="fujimoto2019" /> and CQL.<Cite id="kumar2020" /></>,
    notation: <>Behaviour policy π_β. IQL here uses the paper's form, an expectile τ &gt; 0.5 on Q − V; lecture 7 writes the same loss on V − Q with a small λ.</>,
    takeaway: <>Asking a Q-function about actions the data never took gets you confident nonsense, and a greedy policy goes looking for it. Offline methods stay on the data's actions, and the TD-based ones can still stitch pieces of different trajectories into something better than any of them.</>,
  },
  {
    name: 'Reward learning',
    source: 'CS 224R lecture 8',
    Body: RewardLearning,
    intro: <>Rewards are not given by nature. Learn one from examples of success, or from people comparing two outcomes.</>,
    sources: <>{L(8, 'reward learning: goal classifiers and their exploitation, preferences with the Bradley-Terry model.')} Preferences for deep RL are Christiano et al. 2017<Cite id="christiano2017" />; the model is Bradley & Terry 1952.<Cite id="bradley1952" /></>,
    notation: <>P(τ_a ≻ τ_b) = σ(r_θ(τ_a) − r_θ(τ_b)); fit by maximizing E log σ(r_θ(τ_w) − r_θ(τ_l)).</>,
    takeaway: <>RL optimizes the reward you give it, including its mistakes. A classifier frozen as the reward gets fooled; keep training it on what the policy actually reaches. Pairwise preferences recover a reward up to its scale.</>,
  },
  {
    name: 'RLHF and DPO',
    source: 'CS 224R lecture 9; notes sections 5-6',
    Body: RLHF,
    intro: <>Post-training a language model: learn a reward from human preferences, then optimize it without drifting too far from the original model.</>,
    sources: <>{L(9, 'RLHF and DPO.')} {N('sections 5-6: reward models and DPO.')} InstructGPT<Cite id="ouyang2022" />, DPO<Cite id="rafailov2023" />, and reward-model over-optimisation.<Cite id="gao2023" /></>,
    notation: <>Prompt x, response y, reference π_ref, reward model r_φ, preferred y_w and dispreferred y_l. Lecture 9 writes p^PT, p^RL_θ and RM_φ for the same things.</>,
    takeaway: <>The KL-regularised objective has a closed-form optimum, π_ref · exp(r/β) / Z. Push β too low and the policy exploits the reward model's errors: the learned reward keeps rising while true quality falls. DPO fits the same optimum directly from preferences.</>,
  },
  {
    name: 'GRPO and DAPO',
    source: 'notes sections 7-9',
    Body: GRPO,
    intro: <>Reasoning models are post-trained against rewards that can be checked, like a right answer, using the group of answers to one prompt as its own baseline.</>,
    sources: <>{N('sections 7-9: GRPO, Dr GRPO and DAPO.')} GRPO is from DeepSeekMath<Cite id="shao2024" />, Dr GRPO from Liu et al.<Cite id="liu2025" />, and DAPO from ByteDance Seed and Tsinghua AIR<Cite id="yu2025" /> (the notes attribute DAPO to Alibaba). CS 224R's lecture 10 covers reasoning; its slides were not used here.</>,
    notation: <>Âᵢ = (rᵢ − mean(r)) / std(r) over G answers to one prompt, with the population std as in the notes' code. Clip range ε, split into ε_low and ε_high by DAPO.</>,
    takeaway: <>Group-relative advantages remove the value network. Each normalisation choice has a side effect: dividing by length lets wrong answers grow long, dividing by the group's spread reweights easy and hard prompts, and a symmetric clip starves rare tokens.</>,
  },
];
