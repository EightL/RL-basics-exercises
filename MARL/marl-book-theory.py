# MULTI-AGENT REINFORCEMENT LEARNING (MARL) - FOUNDATIONS
# These are my summary notes of the amazing https://www.marl-book.com + lectures by Stefano V. Albrecht.
# Warning: Some of my notes are not 100% accurate, but the goal is understanding the key concepts.

# Games: Models of Multi-Agent Interaction

# Classification:
# [ Partially Observable Stochastic Games (n agents, m states - partially observable) 
#       [ Stochastic Games (n agents, m states) 
#           [Repeated Normal-Form Games (n agents, 1 state)] [Markov Decision Processes (1 agent, m states)] 
#       ] 
# ]

# Normal Form Games - Building block of all games
# Classification: Zero-sum: A1 loses, A2 wins: r_A1 = -x, r_A2 = x <- competitive
#                 Common-Reward: r_A1 = r_A2 = x <- cooperative
#                 General-Sum: r_A1 + r_A2 = x <- mixed


# Solution Concepts for games
# MARL Problem = Game Model + Solution Concept

# In Single RL, goal is maximize expected return E[R]
# In MARL, maximizing each agents E[R] could interfere with each other (non-stationary), leading to suboptimal outcomes.
# Consider: Joint policies, expected return now depends on other agents' actions combined: Ri = Ri(s, a1, a2, ..., an)

# Observation History: agents dont have access to the full state, only their own observations, we also map actions of other agents to it.
# in MARL, we map history -> action, not just current state -> action (especially in partially observable settings).

# History Based Expected Return: expected sum of discounted rewards given the agent's history - this is the concept
# Recursive Expected Return: Like history-based Bellman equation - this is how we compute it. 

# Joint policy + history-aware recursive returns. Is the key

# Equilibrium Solution Concepts

# Best Response: Given the other agents' policies, a best response is the policy that maximizes an agent's expected return.
# Minimax (zero-sum): A strategy that minimizes the possible loss for a worst-case scenario.
# Nash Equilibrium: A set of strategies where no agent can improve its payoff by unilaterally changing its strategy.
# e-NE: relaxed version, no agent can gain more than e by deviating.
# Correlated Equilibrium: some central policy, agents can follow this recommended policy or deviate, if none deviate central policy is CE.

# Shortcomings of equilibrium concepts:
# Computational Complexity: Exact NE is NP-hard - exponential complexity
# Sub-optimality: NE do not always maximize E[R]
# Non-uniqueness: There can be multiple NE's with different E[R]
# Coordination problems: NE doesnt specify how to reach equilibrium, we just know every game has one

# Refinement Concepts - how good a solution is beyond equilibrium
# Pareto Optimality: An outcome is Pareto optimal if no agent can be made better off without making another agent worse off.
#                    ex. agents splitting 10 points: 1: (5,5) - pareto optimal, 2: (7,3) - also pareto optimal
#                    its about efficiency not fairness

# Welfare Optimality: Maximize sum of all agents' rewards - highest total reward, utilitarian
# Fairness Optimality: Distributes rewards fairly, minimizes inequality
# No-Regret: In repeated games, an agent has no regret if, over time, it doesn't wish it had played another strategy more often.
#            this converges to correlated equilibrium

# Challenges of MARL
# 1. Non-stationarity: The environment is constantly changing due to the actions of multiple agents.
# 2. Equilibrium Selection: With multiple NE's, some solutions converge to risk dominant (safer) rather than reward dominant equilibria
# 3. Credit Assignment: Difficulties in attributing outcomes to individual agent actions due to joint action effects.
# 4. Scalability: As the number of agents increases, the complexity of the joint action space grows exponentially.


# DEEP MULTI-AGENT REINFORCEMENT LEARNING - ALGORITHMS
# Prerequisites: Deep Single Agent RL

# Two Main ways of learning:
# Central Learning: learn single pi_c which recieves all agent observations and selects an action for each agent (joint action(a1,a2,...,an))
#                   - transform joint reward -> single scalar reward
#                   - easy in common rewards but hard in zero-sum or general-sum games
#                   - doesn't scale with large n of agents
#                   - requires a centralized controller and global state

# Independent Learning: from each agent's perspective, we have a transition function.
#                   - each agent learns their own policy pi_i independently, no global state, treats other agents as part of environment


# INDEPENDENT DQN
# Pseudocode (minimal gist of the algorithm):
# ------------------------------------------------------------------
def train_idqn(env, n_agents, num_episodes, gamma=0.99, batch_size=64, target_update=500, eps_start=1.0, eps_end=0.05, eps_decay=0.995):
    # Initialize n value networks with random parameters
    Q <- {build_q_network() for i in range(n_agents)}
    # Initialize n target networks with random parameters
    target_Q <- {build_q_network() for i in range(n_agents)}
    # Init replay buffer for each agent
    replay_buffer <- {ReplayBuffer(capacity=100_000) for i in range(n_agents)}
    
    # for time step t = 0,1,2,.. do
    for t in range(num_episodes):
        # collect current observations o_1(t), o_2(t), ..., o_n(t)
        o_1(t), o_2(t), ..., o_n(t) <- env.reset()

        # for agent i = 1,...,n do
        for i in range(n_agents):
            # epsilon-greedy per agent
            if np.random.random() < eps_start:
                a_i(t) <- random_action()
            else:
                a_i(t) <- argmax_a Q_i(h_i(t), a) # h = observation history
        # Apply actions (a_1(t), a_2(t), ..., a_n(t)), collect rewards, and next observations
        o_1(t+1), o_2(t+1), ..., o_n(t+1), r_1(t), r_2(t), ..., r_n(t) <- env.step((a_1(t), a_2(t), ..., a_n(t)))

        # for agent i = 1,2,...,n do
        for i in range(n_agents):
            # Store transition in replay buffer
            replay_buffer[i].add((h_i(t), a_i(t), r_i(t), h_i(t+1)))
            # Sample random mini-batch from replay buffers D_i
            batch <- replay_buffer[i].sample(batch_size)
            # if s^(k+1) is terminal then:
            if s^(k+1) is terminal then:
                target_Q_value <- r_i(t)
            else:
                target_Q_value <- r_i(t) + gamma * max_a target_Q(h_i(t+1), a)
            
            # compute loss
            loss <- F.mse_loss(Q_i(h_i(t), a_i(t)), target_Q_value)
            # update parameters by minimizing loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # in a set interval, update target network params
            if t % target_update == 0:
                target_Q <- Q
# ------------------------------------------------------------------


# ---- CENTRALIZED TRAINING DECENTRALIZED EXECUTION ----

# POLICY GRADIENT THEOREM
# Defines an expectation over policies of all agents
# Intuitively: Each agent increases the propability of its chosen action proportional to how much that action helped the team outcome, 
#               holding the other's actions fixed in the value term.

# Middle School analogy:
# Team game. Each player tries moves. If the team scores, each player is more likely to repeat their move from that moment; if the team fails, they're less likely
# You only change your own habits, but you judge your move by how well the whole team did when everyone's moves came together.

# Gradient of performance ≈ expectation of: joint value of the taken joint action × how much agent i “turns up” the probability of its own chosen action.
# In words: grad_phi_i J ≈ E over trajectories and joint actions [ Q_i(h, a1..an) * grad_phi_i log pi_i(ai | h_i) ]

# Variance reduction:
# - replace Q_i with an advantage A_i = Q_i - b_i(h) (baseline independent of ai)
# - COMA: counterfactual baseline holds others fixed and varies ai to better isolate i's contribution

# CTDE:
# - Centralized critic uses global info to estimate Q_I or A_i during training
# - Decentralized actors use only h_i at execution

# Caveats:
# - Non-stationarity from simultaneous learning.
# - Credit assignment: Q_i entangles others’ actions; good critics/baselines are crucial.
# - Off-policy corrections (importance sampling) are often high variance in MARL.

# Centralized Critics - critic can be centralized with info 'z' (global states, joint action a, join h,...)

# agent i's history        h_i [] \ 
#                                  -> NN -> V(h_i,z; theta_i)
# extra conditioning info    z [] / 

# Centralized Action-Value Critics
# agent i's history        h_i [] \ 
#                                   -> NN -> Q(h_i,z, <a_(i,1), a(-i)>; theta_i)
# extra conditioning info    z []  / 
# join action           a_(-i) [] /

# Centralized A2C with synchronous environments
# Pseudocode - please see official pseudocode in the MARL book
# ------------------------------------------------------------------
# Init n actor networks with random parameters
# Init n critic networks with random parameters
# init K parallel environments
# for time step t = 0... do: (crazy matrixes)
    # batch of observations for each agent and environment <o_1(t), o_2(t), ..., o_n(t)> 
    # batch of centralized infor for each env <z_1(t), z_2(t), ..., z_n(t)>
    # sample actions from each agent's policy
    # apply actions; collect rewards, observations, and cetralized information

    # for agent i = 1, ..., n do:
    # if s^(t+1) is terminal then:
        # advantage(h,z,a) <- r - V(h,z,theta)
        # critic target y <- r
    # else:
        # advantage(h,z,a) <- r + gamma * V(h^(t+1),z^(t+1),theta) - V(h,z,theta)
        # critic target y <- r + gamma * V(h^(t+1),z^(t+1),theta)
    # compute actor loss
    # compute critic loss
    # update actor networks, by minimizing actor loss
    # update critic networks, by minimizing critic loss
# ------------------------------------------------------------------

# EQUILIBRIUM SELECTION PROBLEM
# its hard for all agents to agree and converge into single equilibrium
# agents often converge to risk dominant equilibria, even tho there are better reward dominant equilibria

# Pareto Actor-Critic Equilibrium Selection
# for no-conflict games, we can assume that agents 'agree' on optimal policy (each of them would like to strive for the best for their team)
# the critic recieves join action as input
# during training, optimize pi by minimizing loss


# VALUE DECOMPOSITION IN COMMON-REWARD GAMES
# Learn one team value for the join action, but build it from per-agent pieces.
# Each agent i has its own small scorer Q_i (h_i, a_i)
# A mixer combines these into the team value Q_tot.
# Training (centralized): use team reward to fit Q_tot with TD targets; gradients flow back into each Q_i through the mixer
# Execution (decentralized): no mixer/critic; each agent picks argmax_a Q_i(h_i, a)

# Individual-Global-Max (IGM) Property: if each agent independenty maximizes its own Q_i, the team's combined Q_tot is also the best for the team value

# Mixers:
# Goal - combine per-agent Q_i into a team value Q_tot during training, design the mixer so IGM holds

# VDN (Value Decomposition Networks):
# Mixer: Q_tot = sum_i Q_i(h_i, a_i)
# simple, satisfies IGM property
# Limitation: only captures additive cooperation; can miss interactions like "only good if both do X"

# Pseudocode - please see official pseudocode in the MARL book:
# ------------------------------------------------------------------

# Init n utility networks with random parameters
# Init n target networks with random parameters
# Init a shared replay buffer D
# for time step t = 0,1,2,... do:
    # collect current observations o_1(t), o_2(t), ..., o_n(t)
    # for agent i = 1,...,n do:
        # epsilon-greedy per agent
        if np.random.random() < eps_start:
            a_i(t) <- random_action()
        else:
            a_i(t) <- argmax_a Q_i(h_i(t), a)
    # Apply actions (a_1(t), a_2(t), ..., a_n(t)), collect rewards, and next observations
    o_1(t+1), o_2(t+1), ..., o_n(t+1), r_1(t), r_2(t), ..., r_n(t) <- env.step((a_1(t), a_2(t), ..., a_n(t)))
    # Store transitions in replay buffer
    replay_buffer.add((h_1(t), a_1(t), r_1(t), h_1(t+1)))
    # Sample random mini-batch from replay buffer D
    batch <- replay_buffer.sample(batch_size)
    # if s^(k+1) is terminal then:
    if s^(k+1) is terminal then:
        y_t = r_team(t)
    else:
        y_t = r_team(t) + gamma * sum_i max_a Q_target_i(h_i(t+1), a)
    # compute loss
    Q_tot_t = sum_i Q_i(h_i(t), a_i(t))
    loss = (y_t - Q_tot_t)**2
    # update parameters by minimizing loss
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    # in a set interval, update target network params
    if t % target_update == 0:
        target_Q <- Q
# ------------------------------------------------------------------

# Monotonic Value Decomposition (QMIX)
# Mixer is a learned, state-conditioned NN: Q_tot = Mix_s(Q_1, Q_2, ..., Q_n)
# It uses a hypernetwork f_hyper - ensures positive weights
# The mixer's weights are generated by a hypernetwork conditioned on the global state s, its monotonic in each Q_i, so IGM stands
# More expressive than a sum, it models context-dependent credit assignment and interactions, while still allowing decentralized greedy execution


# Pseudocode:
# ------------------------------------------------------------------
# inits..
# init mixer NN
# ...
# Team value computation: Q_tot = Mix_s(Q_1(h_1,a_1), ..., Q_n(h_n,a_n))
# TD Target: y = r_team + γ * Mix_{s'}(max_a Q_target_1(h_1', a), ..., max_a Q_target_n(h_n', a))
# Loss = (y - Mix_s(Q_1(...), ..., Q_n(...)))^2
# (Gradients also flow through the mixer and its hypernetwork)
# ------------------------------------------------------------------


# AGENT MODELING WITH DEEP LEARNING
# learning agent models to predict the behavior of other agents to reduce non-stationarity and enable best-response reasoning
# Observations -> [agent model] -> prediction
# We learn a small network that predicts what other agents do next from observations/history
# what the predictor outputs: either single likely action for others (point prediction), or prop. distribution over their actions
# each agent i has a model of each agent j -> n^2 models = n^2 NNs combined in training, we can reduce this with parameter sharing, etc.
# With a join critic Q(s, a_1,...,a_n): evaluate your action together with predicted others' actions (or their dist)

# Minimal pseudocode (JAL + agent models):
# for each update step:
#   batch ← replay.sample(B)
#   # 1) Update agent model(s)
#   minimize NLL of π_hat_{-i} on observed a_{-i}
#   # 2) Update critic
#   produce a_hat'_{-i} (or sample from π_hat at s')
#   y = r + γ*(1-done) * max_{a_i'} Q_target(s', a_i', a_hat'_{-i})
#   L_Q = (y - Q(s, a))^2
#   step optimizers; periodically update targets

# LEARNING COMPACT REPRESENTATIONS OF AGENT POLICIES
# Goal: Reduce parameters/memory for many agents; Improve generalization across roles/tasks; Stabilize MARL by sharing structure and lowering non-stability

# Encoder-Decoder architecture: 

# Core-idea: Encode high-dimensional history into compact representation, then decode it into action
# Encoder turns i's history h_i^t into compact vector z_i^t
# Decoder takes z_i^t and returns a propability for each possible action of each other agent j.
# Loss: sums over j!=i of -log prob the model gave to j's true action at time t

# Parameter and Experience Sharing
# Training agents is slow and unstable because too many parameters -> parameter, experience sharing
# Parameter Sharing: Agents use the same core policy/critic, with small adjustments based on their unique ID or role.
                    # A single actor (and optionally critic) is used for all agents, with agent-specific information (like ID) fed in.
                    # Reduces number of parameters, speeds up learning, and leverages similarites between agents.
                    # Group PS: share within roles, not across all agents.

# Experience Sharing: Data from all agents is collected and stored together in a single replay buffer.
                    # All agent trajectories are added to one common buffer, often including agent-specific indentifiers.
                    # It provides more varied data for each learning update, helps stabilize training for shared models.

# Shared Experience Actor-Critic for cooperative MARL (SEAC):
# Idea: Each agent updates not only from its own trajectories, but also from other agents' trajectories, corrected with Importance Sampling

# Each agent has its own actor π_i(a|o); A centralized critic V(s)(CTDE) estimates value using global info; Replay/batch contains per-agent tuples (o_j, a_j, r, o'_j, s, s', done) for all agents j
# For updating agent i:
    # Use its own data (standart A2C/PPO update)    
    # Plus use other agents' data j!=i with an importance ratio ρ_ij = π_i(a_j | o_j) / π_j(a_j | o_j)
        # "How likely would agent i have taken agent j's action under j's observation?"
    # Update rule: Policy gradient = self-update + λ × sum_over_others( ρ_ij × their-policy-gradient-terms )


# POLICY SELF-PLAY IN ZERO-SUM GAMES
# Train by playing against current or past versions of yourself
# Monte Carlo Tree Search (MCTS): 
    # Build a search tree online from current state by simulating many playouts
    # Four steps: 
        # Selection: traverse from root choosing child via an upper-confidence rule
        # Expansion: add one or more new children when a leaf with untried actions is hit
        # Evaluation: estimate value of the leaf by running a rollout or value network
        # Backpropagation: propagate the result back up, updating visit counts and values

# POPULATION-BASED TRAINING: Self-play for general-sum games
# Jointly optimize policies and their hyperparameters by evolving a population online
# Loop: 
    # Train K agents in parrallel for a short window with different hyperparams
    # Evaluate each agent on validation metric (win-rate/return)
    # Exploit: replace low-performers' weights with a copy of a top-performer
    # Explore: perturb the copied hyperparams
    # Repeat while continuing training

# Policy Space Response Oracles (PSRO): 
# Iteratively compute approximate best responces to the current meta-strategy over opponents, add them to the population, then recompute the meta-strategy
# Core loop:
    # Build a small population Π for each player.
    # Play all-vs-all to fill a payoff matrix M between policies.
    # Meta-solve M to get a mixed strategy σ over each population (e.g., Nash/REGRET).
    # Train a new policy (oracle) as a best response to opponents sampled from σ.
    # Add the oracle to Π and repeat.