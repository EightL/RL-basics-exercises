import torch # main pytorch library
import torch.nn as nn # neural network module
import torch.nn.functional as F # activation functions

# Fix for numpy bool8 compatibility issue
import numpy as np
if not hasattr(np, 'bool8'):
    np.bool8 = np.bool_
# Fix for numpy float_ compatibility issue
if not hasattr(np, 'float_'):
    np.float_ = np.float64

class QNetwork(nn.Module):
    def __init__(self, state_size, action_size):
        super(QNetwork, self).__init__()
        # Improved architecture from Kaggle notebook
        self.fc1 = nn.Linear(state_size, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 128)
        self.fc4 = nn.Linear(128, action_size)
        
        # Initialize weights for better training
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.xavier_uniform_(module.weight)
            module.bias.data.fill_(0.01)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        return self.fc4(x)
    
import random
from collections import deque 

# Whats is a replay buffer?
# A replay buffer is a buffer that stores the experiences of the agent.
# without it: unstable learning, forgetting, correlated data

# it is NOT a model, it only stores the past experiences, it doesnt predict, model env, or simulate future scenarios

class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return states, actions, rewards, next_states, dones
    
    def __len__(self):
        return len(self.buffer)

import gym
import numpy as np
import time
from gym.wrappers import TimeLimit

# Create environment with shorter time limit
env = gym.make('Acrobot-v1')
env = TimeLimit(env, max_episode_steps=250)  # Limit to 250 steps

state_size = env.observation_space.shape[0] # 6 state variables for Acrobot
action_size = env.action_space.n # 3 possible actions for Acrobot

# Create main Q-network and target network
q_network = QNetwork(state_size, action_size)
target_network = QNetwork(state_size, action_size)
target_network.load_state_dict(q_network.state_dict())  # Copy weights

# Larger replay buffer and lower learning rate
optimizer = torch.optim.Adam(q_network.parameters(), lr=1e-4)
replay_buffer = ReplayBuffer(100_000)

epsilon = 1.0
epsilon_decay = 0.9999  # Slower decay
epsilon_min = 0.01

gamma = 0.99
batch_size = 32  # Smaller batch size for stability
target_update_freq = 20  # More frequent target network updates

def render_episodes(q_network, env_name, num_episodes=1, use_policy=True):
    # Create rendering environment with same time limit
    render_env = gym.make(env_name, render_mode='human')
    render_env = TimeLimit(render_env, max_episode_steps=250)
    
    for ep in range(num_episodes):
        state = render_env.reset()
        if isinstance(state, tuple):
            state = state[0]
        state = np.array(state, dtype=np.float32)
        state = torch.FloatTensor(state)
        total_reward = 0
        done = False
        while not done:
            if use_policy:
                with torch.no_grad():
                    q_values = q_network(state)
                    action = q_values.argmax().item()
            else:
                action = render_env.action_space.sample()
            result = render_env.step(action)
            if len(result) == 4:
                next_state, reward, done, _ = result
            else:
                next_state, reward, terminated, truncated, _ = result
                done = terminated or truncated
            next_state = np.array(next_state, dtype=np.float32)
            next_state = torch.FloatTensor(next_state)
            state = next_state
            total_reward += reward
            time.sleep(0.02)
        print(f"Episode {ep+1}: Total reward = {total_reward}")
    render_env.close()


# Training with improvements from Kaggle notebook
episode_rewards = []  # Track rewards for plotting

def get_tip_height(state):
    # state: [cos(theta1), sin(theta1), cos(theta2), sin(theta2), thetaDot1, thetaDot2]
    cos1, sin1, cos2, sin2, _, _ = state
    theta1 = np.arctan2(sin1, cos1)
    theta2 = np.arctan2(sin2, cos2)
    # Link lengths are 1.0 each
    y = -np.cos(theta1) - np.cos(theta1 + theta2)
    return y

best_avg_reward = float('-inf')  # Track best average reward

for episode in range(1000):
    state = env.reset()
    if isinstance(state, tuple):
        state = state[0]
    state = np.array(state, dtype=np.float32)
    state = torch.FloatTensor(state)
    total_reward = 0

    done = False
    while not done:
        if np.random.rand() < epsilon:
            action = env.action_space.sample()
        else:
            with torch.no_grad():
                q_values = q_network(state)
                action = q_values.argmax().item()

        result = env.step(action)
        if len(result) == 4:
            next_state, reward, done, _ = result
        else:
            next_state, reward, terminated, truncated, _ = result
            done = terminated or truncated
        
        next_state = np.array(next_state, dtype=np.float32)
        next_state = torch.FloatTensor(next_state)
        replay_buffer.push(state, action, reward, next_state, done)

        state = next_state
        total_reward += reward

        if len(replay_buffer) > batch_size:
            states, actions, rewards, next_states, dones = replay_buffer.sample(batch_size)

            states = torch.stack(states)
            actions = torch.tensor(actions)
            rewards = torch.tensor(rewards)
            next_states = torch.stack(next_states)
            dones = torch.tensor(dones, dtype=torch.float32)

            # Use target network for stability (Double DQN)
            current_q = q_network(states).gather(1, actions.unsqueeze(1)).squeeze(1)
            with torch.no_grad():
                next_actions = q_network(next_states).argmax(1)
                next_q = target_network(next_states).gather(1, next_actions.unsqueeze(1)).squeeze(1)
                expected_q = rewards + gamma * next_q * (1 - dones)

            loss = F.mse_loss(current_q, expected_q)

            optimizer.zero_grad()
            loss.backward()
            # Gradient clipping for stability
            torch.nn.utils.clip_grad_norm_(q_network.parameters(), 1.0)
            optimizer.step()

            epsilon = max(epsilon_min, epsilon * epsilon_decay)

        episode_rewards.append(total_reward)
        
        # Update target network more frequently
        if episode % target_update_freq == 0:
            target_network.load_state_dict(q_network.state_dict())
        
        # Save best model
        if episode >= 100:
            avg_reward = np.mean(episode_rewards[-100:])
            if avg_reward > best_avg_reward:
                best_avg_reward = avg_reward
                torch.save(q_network.state_dict(), 'best_acrobot_dqn.pth')
        
        # Print progress
        if episode % 100 == 0:
            avg_reward = np.mean(episode_rewards[-100:])
            print(f"Episode {episode} - Avg reward (last 100): {avg_reward:.2f} - Epsilon: {epsilon:.3f}")

# Load best model for evaluation
q_network.load_state_dict(torch.load('best_acrobot_dqn.pth'))

print("Showing last 1 episode (trained agent):")
render_episodes(q_network, 'Acrobot-v1', num_episodes=1, use_policy=True)

# Plot learning curve
import matplotlib.pyplot as plt

plt.figure(figsize=(10, 6))
plt.plot(episode_rewards)
plt.title('Acrobot DQN Learning Curve')
plt.xlabel('Episode')
plt.ylabel('Total Reward')
plt.grid(True)
plt.show()

# Print final statistics
final_avg = np.mean(episode_rewards[-100:])
print(f"\nFinal average reward (last 100 episodes): {final_avg:.2f}")
print(f"Best episode reward: {max(episode_rewards)}")
print(f"Episodes with positive reward: {sum(1 for r in episode_rewards if r > -250)}")
