# Multi-Agent Reinforcement Learning (MARL) — Foundations
(More readable and simplified version than the .py file, but without the sub-pseudocodes)

These are my summary notes of the excellent https://www.marl-book.com and lectures by Stefano V. Albrecht.


---

## Games: Models of Multi-Agent Interaction

- Partially Observable Stochastic Games (POSGs)
  - Stochastic Games (SGs)
    - Repeated Normal-form Games
    - Markov Decision Processes (MDPs)

### Normal-form games (building block)
- Zero-sum: r1 = −r2 (competitive)
- Common-reward: r1 = r2 (cooperative)
- General-sum: r1 + r2 unconstrained (mixed interests)

### Solution concepts
- Best Response: optimal policy given others’ policies.
- Minimax (zero-sum): minimize worst-case loss.
- Nash Equilibrium (NE): no unilateral profitable deviation.
- ε-NE: relax by ε.
- Correlated Equilibrium (CE): recommendation signal; no profitable deviation if followed.

Shortcomings: computing exact NE is hard; NE can be suboptimal, non-unique; coordination not guaranteed.

### Refinements
- Pareto optimality: no one can improve without hurting another.
- Welfare optimality: maximize sum of rewards.
- Fairness: reduce inequality.
- No-regret dynamics: converge to CE in repeated play.

### Core challenges in MARL
- Non-stationarity from simultaneous learning.
- Equilibrium selection (risk- vs reward-dominant).
- Credit assignment under joint actions.
- Scalability (joint action/state explosion).

---

## Deep MARL: Learning Setups

- Centralized Learning: a single controller πc maps global info to joint action; easy in cooperative tasks, hard in zero/general-sum; poor scalability; needs global state.
- Independent Learning: each agent learns πi(h_i) independently and treats others as part of the environment; scalable but non-stationary.

---

## Independent DQN (IDQN)

Minimal sketch:
- Each agent i has its own Q_i and target network, its own replay buffer.
- ε-greedy action per agent from its own observation history h_i.
- TD target uses target network; periodic target updates.

---

## CTDE: Policy Gradients and Centralized Critics

Policy Gradient intuition: agent i increases log π_i(a_i|h_i) proportional to joint value/advantage that reflects team outcome, holding others fixed in the value term.

Variance reduction:
- Baselines: A_i = Q_i − b_i(h)
- COMA: counterfactual baseline varying only a_i

CTDE principle:
- Centralized critic uses global info z (state, joint actions, histories) during training.
- Decentralized actors use only local info h_i at execution.

### Centralized A2C (sketch)
- K parallel envs; collect batched observations and centralized info.
- Compute advantages and critic targets; update actor and critic networks.

### Equilibrium selection and Pareto Actor-Critic
- Agents may converge to risk-dominant equilibria.
- In no-conflict games, optimize a joint objective, e.g., via a critic that conditions on joint action; train actors to improve team return.

---

## Value Decomposition in Common-Reward Games

Goal: learn a team value Q_tot for the joint action but build it from per-agent utilities Q_i(h_i, a_i). Train centrally with team reward; execute decentrally with greedy per-agent actions.

### VDN (Value Decomposition Networks)
- Mixer: Q_tot = Σ_i Q_i(h_i, a_i)
- Pros: simple; satisfies IGM property (see below).
- Cons: only additive cooperation; misses interaction terms (e.g., AND-style synergies).

### QMIX (Monotonic Value Decomposition)
- Mixer is a state-conditioned network: Q_tot = Mix_s(Q_1, …, Q_n).
- Weights produced by a hypernetwork f_hyper(s) and constrained to be non-negative to ensure monotonicity in each Q_i.
- More expressive than a sum: models context-dependent credit assignment while preserving decentralized greedy execution.

### Formal notes: IGM and QMIX monotonicity constraints

- Individual–Global–Max (IGM) property
  - Informal: if each agent greedily maximizes its own Q_i, the resulting joint action also maximizes the team value Q_tot.
  - One formal statement at a fixed state s and histories h:
    argmax_{a_1,…,a_n} Q_tot(h, a_1,…,a_n) = (argmax_{a_1} Q_1(h_1,a_1), …, argmax_{a_n} Q_n(h_n,a_n)).
  - Sufficient condition: Q_tot is monotone non-decreasing in each per-agent utility Q_i:
    ∂ Q_tot / ∂ Q_i ≥ 0 for all i. Under this, greedy per-agent argmaxes compose to a joint argmax.

- QMIX monotonic mixer constraints
  - Design Mix_s so that for any fixed s, Q_tot = Mix_s(Q_1,…,Q_n) is monotone in each Q_i.
  - Typical implementation:
    - Use a hypernetwork that outputs non-negative mixing weights w_i(s) (enforced by softplus/abs/ReLU) and unconstrained biases b(s).
    - Use only monotone operations with respect to inputs Q_i (e.g., weighted sums with w_i(s) ≥ 0 and monotone activations like ELU/ReLU).
  - This ensures the gradient conditions ∂Q_tot/∂Q_i ≥ 0, which guarantees IGM and hence valid decentralized greedy execution.

---

## Agent Modeling with Deep Learning

Learn models of other agents to reduce non-stationarity and enable best-response reasoning.
- Predict point actions or distributions for others from observations/history.
- Optionally use a joint critic Q(s, a_1,…,a_n) evaluating own action with predicted others’ actions.
- Parameter sharing can reduce the naive O(n^2) modeling burden.

Minimal training sketch:
1) Update agent models to maximize likelihood of observed actions of others.
2) Update critic with targets that plug in predicted others’ actions (or samples from their predicted distributions).

---

## Compact Policy Representations (Encoder–Decoder)

- Encode high-dimensional histories into compact vectors z_i; decode into distributions over others’ actions.
- Loss sums over others’ true actions.
- Benefits: fewer parameters, better generalization across roles/tasks, more stable training with shared structure.

---

## Parameter and Experience Sharing, SEAC

- Parameter sharing: a shared actor/critic with agent-ID/role embeddings; optionally grouped by role.
- Experience sharing: a common replay buffer with agent identifiers; stabilizes shared models.

SEAC (Shared Experience Actor–Critic):
- Each agent updates on its own data plus other agents’ data reweighted by importance ratios ρ_ij = π_i(a_j|o_j) / π_j(a_j|o_j).
- Policy gradient ≈ self-update + λ Σ_{j≠i} ρ_ij × (their policy-gradient terms), with a centralized value function for CTDE.

---

## Policy Self-Play and MCTS

- Train against current/past versions of self.
- MCTS loop: selection → expansion → evaluation → backpropagation; AlphaZero-style PUCT for balancing exploration/exploitation.

---

## Population-Based Training (PBT)

- Maintain a population with differing hyperparameters.
- Periodically evaluate; exploit (copy top performers) and explore (perturb hyperparameters), then continue training.

---

## Policy Space Response Oracles (PSRO)

- Maintain a small policy population Π per player.
- Play all-vs-all to build a payoff matrix M.
- Solve meta-strategies (e.g., approximate Nash or regret matching) over Π.
- Train best responses (oracles) to these meta-strategies; add to Π; iterate.
