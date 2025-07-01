# Mountain Car Problem using REINFORCE algorithm

import gym
import numpy as np
import torch as T
import torch.nn as nn
import matplotlib.pyplot as plt
import argparse


env = gym.make('MountainCarContinuous-v0')
print(env.observation_space)
print(env.action_space)

# state, _ = env.reset()
# done = False

# while not done:
#     action = env.action_space.sample()
#     state, reward, done, truncated, info = env.step(action)
#     if done or truncated:
#         break

# env.close()


# instead of value function, we learn a POLICY function, that maps states to actions.
# for continous we model it as propability distribution over actions.
# we will use Gaussian (Normal) distribution - where we have mean: the average action, and std: how much to explore.
class PolicyNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(2, 16) # input state
        self.fc2 = nn.Linear(16, 16) # hidden layer
        self.mean = nn.Linear(16, 1) # output mean
        self.log_std = nn.Linear(16, 1) # output std, just the scalar values, later we will plug in N(mean, std)

    def forward(self, state):
        x = T.relu(self.fc1(state)) # 2D -> 16D
        x = T.relu(self.fc2(x)) # 16D -> 16D
        mean = self.mean(x) # 16D -> 1D
        log_std = self.log_std(x) # 16D -> 1D, why log? because we want to make sure std is positive.
        return mean, log_std

import torch.distributions as D


# Testing the policy network ------------------------------------------------------------

# state = T.tensor([0.0, 0.0], dtype=T.float32)  # Example state

# mean, log_std = policy(state)
# std = T.exp(log_std)  # Convert log_std to std

# dist = D.Normal(mean, std) # N(mean, std)
# action = dist.sample()  # Sample an action
# log_prob = dist.log_prob(action)  # Log probability of the action

# print("Mean:", mean.item())
# print("Std:", std.item())
# print("Sampled action:", action.item())
# print("Log probability:", log_prob.item())

# -----------------------------------------------------------------------------------------

def run_episode(env, policy, render=False):
    state, _ = env.reset()
    done = False
    total_reward = 0
    while not done:
        if render:
            env.render()
        state_tensor = T.tensor(state, dtype=T.float32)
        mean, log_std = policy(state_tensor)
        std = T.exp(log_std)
        dist = D.Normal(mean, std)
        action = dist.sample()
        next_state, reward, done, truncated, info = env.step([action.item()])
        total_reward += reward
        state = next_state
        if done or truncated:
            break
    return total_reward

def compute_returns(rewards, gamma=0.99):
    returns = []
    G = 0
    for r in reversed(rewards):
        G = r + gamma * G
        returns.insert(0, G)
    return returns

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', nargs='?', default='train', choices=['train', 'visualize'], help="Mode: 'train' or 'visualize'")
    parser.add_argument('--model-path', default='mountaincar_policy.pt', help='Path to save/load the policy model')
    args = parser.parse_args()

    if args.mode == 'visualize':
        policy = PolicyNetwork()
        policy.load_state_dict(T.load(args.model_path))
        policy.eval()
        env_vis = gym.make('MountainCarContinuous-v0', render_mode="human")
        run_episode(env_vis, policy, render=True)
        env_vis.close()
        return

    # Training mode
    env = gym.make('MountainCarContinuous-v0')
    print(env.observation_space)
    print(env.action_space)
    policy = PolicyNetwork()
    optimizer = T.optim.Adam(policy.parameters(), lr=3e-4)
    num_episodes = 1000
    rewards_history = []
    for episode in range(num_episodes):
        states = []
        actions = []
        rewards = []
        log_probs = []
        state, _ = env.reset()
        done = False
        total_reward = 0
        while not done:
            state_tensor = T.tensor(state, dtype=T.float32)
            mean, log_std = policy(state_tensor)
            std = T.exp(log_std)
            dist = D.Normal(mean, std)
            action = dist.sample()
            log_prob = dist.log_prob(action)
            next_state, reward, done, truncated, info = env.step([action.item()])
            states.append(state)
            actions.append(action)
            rewards.append(reward)
            log_probs.append(log_prob)
            total_reward += reward
            state = next_state
            if done or truncated:
                break
        returns = compute_returns(rewards, gamma=0.99)
        returns = T.tensor(returns, dtype=T.float32)
        returns = (returns - returns.mean()) / (returns.std() + 1e-8)
        policy_loss = []
        baseline = returns.mean()
        for log_prob, G in zip(log_probs, returns):
            policy_loss.append(-log_prob * (G - baseline))
        policy_loss = T.stack(policy_loss).sum()
        optimizer.zero_grad()
        policy_loss.backward()
        optimizer.step()
        if episode % 10 == 0:
            print(f"Episode {episode} - Policy loss: {policy_loss.item()}")
        rewards_history.append(total_reward)
        if episode % 500 == 0 and episode > 0:
            T.save(policy.state_dict(), args.model_path)
            print(f"Saved model at episode {episode} to {args.model_path}")
            env_vis = gym.make('MountainCarContinuous-v0', render_mode="human")
            run_episode(env_vis, policy, render=True)
            env_vis.close()
    # Save final model
    T.save(policy.state_dict(), args.model_path)
    print(f"Training complete. Model saved to {args.model_path}")
    plt.plot(rewards_history)
    plt.xlabel("Episode")
    plt.ylabel("Total Reward")
    plt.title("REINFORCE on MountainCarContinuous")
    plt.show()
    env.close()

if __name__ == "__main__":
    main()

