// Every claim on the page cites one of these. The two course sources come first.
export const REFS = {
  cs224r: {
    short: 'CS224R 2026',
    full: 'C. Finn. CS 224R: Deep Reinforcement Learning, Stanford University, Spring 2026. Lecture slides.',
    url: 'https://cs224r.stanford.edu/',
  },
  nalawade: {
    short: 'Nalawade, RL notes',
    full: 'A. Nalawade. Reinforcement Learning (notes from MDPs to PPO, RLHF, GRPO and DAPO).',
    url: 'https://www.aadityan.com/blog/reinforcement-learning',
  },
  sutton2018: {
    short: 'Sutton & Barto 2018',
    full: 'R. S. Sutton, A. G. Barto. Reinforcement Learning: An Introduction, 2nd ed. MIT Press, 2018.',
    url: 'http://incompleteideas.net/book/the-book-2nd.html',
  },
  russell2020: {
    short: 'Russell & Norvig 2020',
    full: 'S. Russell, P. Norvig. Artificial Intelligence: A Modern Approach, 4th ed., ch. 17. Pearson, 2020. (the 4x3 world)',
    url: 'https://aima.cs.berkeley.edu/',
  },
  bellman1957: {
    short: 'Bellman 1957',
    full: 'R. Bellman. Dynamic Programming. Princeton University Press, 1957.',
    url: null,
  },
  howard1960: {
    short: 'Howard 1960',
    full: 'R. A. Howard. Dynamic Programming and Markov Processes. MIT Press, 1960. (policy iteration)',
    url: null,
  },
  thompson1933: {
    short: 'Thompson 1933',
    full: 'W. R. Thompson. On the likelihood that one unknown probability exceeds another in view of the evidence of two samples. Biometrika 25, 1933.',
    url: 'https://doi.org/10.2307/2332286',
  },
  auer2002: {
    short: 'Auer et al. 2002',
    full: 'P. Auer, N. Cesa-Bianchi, P. Fischer. Finite-time analysis of the multiarmed bandit problem. Machine Learning 47, 2002. (UCB1)',
    url: 'https://doi.org/10.1023/A:1013689704352',
  },
  watkins1992: {
    short: 'Watkins & Dayan 1992',
    full: 'C. J. C. H. Watkins, P. Dayan. Q-learning. Machine Learning 8, 1992.',
    url: 'https://doi.org/10.1007/BF00992698',
  },
  rummery1994: {
    short: 'Rummery & Niranjan 1994',
    full: 'G. A. Rummery, M. Niranjan. On-line Q-learning using connectionist systems. Tech. report CUED/F-INFENG/TR 166, 1994. (SARSA)',
    url: null,
  },
  vanhasselt2010: {
    short: 'van Hasselt 2010',
    full: 'H. van Hasselt. Double Q-learning. NeurIPS, 2010.',
    url: 'https://papers.nips.cc/paper/3964-double-q-learning',
  },
  mnih2015: {
    short: 'Mnih et al. 2015',
    full: 'V. Mnih et al. Human-level control through deep reinforcement learning. Nature 518, 2015. (DQN: replay buffer, target network)',
    url: 'https://doi.org/10.1038/nature14236',
  },
  williams1992: {
    short: 'Williams 1992',
    full: 'R. J. Williams. Simple statistical gradient-following algorithms for connectionist reinforcement learning. Machine Learning 8, 1992. (REINFORCE)',
    url: 'https://doi.org/10.1007/BF00992696',
  },
  barto1983: {
    short: 'Barto et al. 1983',
    full: 'A. G. Barto, R. S. Sutton, C. W. Anderson. Neuronlike adaptive elements that can solve difficult learning control problems. IEEE Trans. SMC 13, 1983. (CartPole)',
    url: 'https://doi.org/10.1109/TSMC.1983.6313077',
  },
  schulman2016: {
    short: 'Schulman et al. 2016',
    full: 'J. Schulman, P. Moritz, S. Levine, M. Jordan, P. Abbeel. High-dimensional continuous control using generalized advantage estimation. ICLR, 2016.',
    url: 'https://arxiv.org/abs/1506.02438',
  },
  schulman2015: {
    short: 'Schulman et al. 2015',
    full: 'J. Schulman, S. Levine, P. Moritz, M. Jordan, P. Abbeel. Trust region policy optimization. ICML, 2015.',
    url: 'https://arxiv.org/abs/1502.05477',
  },
  schulman2017: {
    short: 'Schulman et al. 2017',
    full: 'J. Schulman, F. Wolski, P. Dhariwal, A. Radford, O. Klimov. Proximal policy optimization algorithms. arXiv:1707.06347, 2017.',
    url: 'https://arxiv.org/abs/1707.06347',
  },
  ross2011: {
    short: 'Ross et al. 2011',
    full: 'S. Ross, G. Gordon, D. Bagnell. A reduction of imitation learning and structured prediction to no-regret online learning. AISTATS, 2011. (DAgger)',
    url: 'https://arxiv.org/abs/1011.0686',
  },
  pomerleau1989: {
    short: 'Pomerleau 1989',
    full: 'D. A. Pomerleau. ALVINN: An autonomous land vehicle in a neural network. NeurIPS, 1989. (behaviour cloning)',
    url: null,
  },
  levine2020: {
    short: 'Levine et al. 2020',
    full: 'S. Levine, A. Kumar, G. Tucker, J. Fu. Offline reinforcement learning: tutorial, review, and perspectives on open problems. arXiv:2005.01643, 2020.',
    url: 'https://arxiv.org/abs/2005.01643',
  },
  kumar2020: {
    short: 'Kumar et al. 2020',
    full: 'A. Kumar, A. Zhou, G. Tucker, S. Levine. Conservative Q-learning for offline reinforcement learning. NeurIPS, 2020.',
    url: 'https://arxiv.org/abs/2006.04779',
  },
  fujimoto2019: {
    short: 'Fujimoto et al. 2019',
    full: 'S. Fujimoto, D. Meger, D. Precup. Off-policy deep reinforcement learning without exploration. ICML, 2019. (BCQ)',
    url: 'https://arxiv.org/abs/1812.02900',
  },
  peng2019: {
    short: 'Peng et al. 2019',
    full: 'X. B. Peng, A. Kumar, G. Zhang, S. Levine. Advantage-weighted regression: simple and scalable off-policy reinforcement learning. arXiv:1910.00177, 2019.',
    url: 'https://arxiv.org/abs/1910.00177',
  },
  kostrikov2022: {
    short: 'Kostrikov et al. 2022',
    full: 'I. Kostrikov, A. Nair, S. Levine. Offline reinforcement learning with implicit Q-learning. ICLR, 2022.',
    url: 'https://arxiv.org/abs/2110.06169',
  },
  fu2018: {
    short: 'Fu et al. 2018',
    full: 'J. Fu, A. Singh, D. Ghosh, L. Yang, S. Levine. Variational inverse control with events: a general framework for data-driven reward definition. NeurIPS, 2018. (VICE)',
    url: 'https://arxiv.org/abs/1805.11686',
  },
  sadigh2017: {
    short: 'Sadigh et al. 2017',
    full: 'D. Sadigh, A. D. Dragan, S. Sastry, S. A. Seshia. Active preference-based learning of reward functions. RSS, 2017.',
    url: 'https://www.roboticsproceedings.org/rss13/p53.pdf',
  },
  christiano2017: {
    short: 'Christiano et al. 2017',
    full: 'P. Christiano, J. Leike, T. Brown, M. Martic, S. Legg, D. Amodei. Deep reinforcement learning from human preferences. NeurIPS, 2017.',
    url: 'https://arxiv.org/abs/1706.03741',
  },
  bradley1952: {
    short: 'Bradley & Terry 1952',
    full: 'R. A. Bradley, M. E. Terry. Rank analysis of incomplete block designs I: the method of paired comparisons. Biometrika 39, 1952.',
    url: 'https://doi.org/10.2307/2334029',
  },
  ouyang2022: {
    short: 'Ouyang et al. 2022',
    full: 'L. Ouyang et al. Training language models to follow instructions with human feedback. NeurIPS, 2022. (InstructGPT)',
    url: 'https://arxiv.org/abs/2203.02155',
  },
  rafailov2023: {
    short: 'Rafailov et al. 2023',
    full: 'R. Rafailov, A. Sharma, E. Mitchell, S. Ermon, C. D. Manning, C. Finn. Direct preference optimization: your language model is secretly a reward model. NeurIPS, 2023.',
    url: 'https://arxiv.org/abs/2305.18290',
  },
  gao2023: {
    short: 'Gao et al. 2023',
    full: 'L. Gao, J. Schulman, J. Hilton. Scaling laws for reward model overoptimization. ICML, 2023.',
    url: 'https://arxiv.org/abs/2210.10760',
  },
  shao2024: {
    short: 'Shao et al. 2024',
    full: 'Z. Shao et al. DeepSeekMath: pushing the limits of mathematical reasoning in open language models. arXiv:2402.03300, 2024. (GRPO)',
    url: 'https://arxiv.org/abs/2402.03300',
  },
  liu2025: {
    short: 'Liu et al. 2025',
    full: 'Z. Liu et al. Understanding R1-Zero-like training: a critical perspective. arXiv:2503.20783, 2025. (Dr. GRPO)',
    url: 'https://arxiv.org/abs/2503.20783',
  },
  yu2025: {
    short: 'Yu et al. 2025',
    full: 'Q. Yu et al. DAPO: an open-source LLM reinforcement learning system at scale. arXiv:2503.14476, 2025.',
    url: 'https://arxiv.org/abs/2503.14476',
  },
  ng1999: {
    short: 'Ng et al. 1999',
    full: 'A. Y. Ng, D. Harada, S. Russell. Policy invariance under reward transformations: theory and application to reward shaping. ICML, 1999.',
    url: null,
  },
  krakovna2020: {
    short: 'Krakovna et al. 2020',
    full: 'V. Krakovna et al. Specification gaming: the flip side of AI ingenuity. DeepMind blog, 2020.',
    url: 'https://deepmind.google/discover/blog/specification-gaming-the-flip-side-of-ai-ingenuity/',
  },
};

export const REF_ORDER = Object.keys(REFS);
