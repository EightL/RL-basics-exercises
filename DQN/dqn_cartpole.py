import torch # main pytorch library
import torch.nn as nn # neural network module
import torch.nn.functional as F # activation functions

# Fix for numpy bool8 compatibility issue
import numpy as np
if not hasattr(np, 'bool8'):
    np.bool8 = np.bool_

class QNetwork(nn.Module):
    def __init__(self, state_size, action_size):
        super(QNetwork, self).__init__()
        self.fc1 = nn.Linear(state_size, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, action_size)

    def forward(self, x): # x is the current state [cart position, cart velocity, pole angle, pole velocity]
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)
    
import random
from collections import deque 
import pickle
import os

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
        batch = random.sample(self.buffer, batch_size) # sample a random batch of experiences from the buffer
        states, actions, rewards, next_states, dones = zip(*batch) # zip is used to unpack the batch into separate lists
        return states, actions, rewards, next_states, dones # return the batch of experiences
    
    def __len__(self):
        return len(self.buffer)

class EpisodeRecorder:
    def __init__(self, max_saved_episodes=10):
        self.max_saved_episodes = max_saved_episodes
        self.best_episodes = []  # List of (reward, episode_data) tuples
        
    def record_episode(self, episode_data, total_reward):
        """Record an episode if it's among the best"""
        episode_info = {
            'frames': episode_data['frames'],
            'actions': episode_data['actions'],
            'rewards': episode_data['rewards'],
            'total_reward': total_reward,
            'episode_length': len(episode_data['frames'])
        }
        
        # Add to best episodes if we have room or if it's better than worst
        if len(self.best_episodes) < self.max_saved_episodes:
            self.best_episodes.append((total_reward, episode_info))
            self.best_episodes.sort(key=lambda x: x[0], reverse=True)  # Sort by reward
        elif total_reward > self.best_episodes[-1][0]:
            # Replace worst episode
            self.best_episodes[-1] = (total_reward, episode_info)
            self.best_episodes.sort(key=lambda x: x[0], reverse=True)
    
    def save_episodes(self, filename='best_episodes.pkl'):
        """Save best episodes to file"""
        with open(filename, 'wb') as f:
            pickle.dump(self.best_episodes, f)
        print(f"Saved {len(self.best_episodes)} best episodes to {filename}")
    
    def load_episodes(self, filename='best_episodes.pkl'):
        """Load episodes from file"""
        if os.path.exists(filename):
            with open(filename, 'rb') as f:
                self.best_episodes = pickle.load(f)
            print(f"Loaded {len(self.best_episodes)} episodes from {filename}")
        else:
            print(f"No saved episodes found at {filename}")

import gym
import numpy as np
import time
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# Create environment with rgb_array render mode for proper frame capture
env = gym.make('CartPole-v1', render_mode='rgb_array')

state_size = env.observation_space.shape[0] # 4 state variables
action_size = env.action_space.n # 2 possible actions

q_network = QNetwork(state_size, action_size) # this creates our 4 inputs -> 64 -> 64 -> 2 outputs

optimizer = torch.optim.Adam(q_network.parameters(), lr=1e-3) # default learning rate
# what is optimizer? - it is used to update the weights of the network, to minimize the loss (error)
# Adam solves simple gradient descent problems

# adam gives each of the weights a different learning rate

# Each update depends on previous momentum + current gradient
# momentum = 0.9 * previous_momentum + 0.1 * current_gradient
# weight_update = -learning_rate * momentum
# -> this results in smooth gradient descent, and prevents overshooting the minimum (like walking vs skiing downhill)

replay_buffer = ReplayBuffer(10000)
episode_recorder = EpisodeRecorder(max_saved_episodes=5)  # Save top 5 episodes

epsilon = 1.0 # initial epsilon
epsilon_decay = 0.995 # decay rate
epsilon_min = 0.01 # minimum epsilon

gamma = 0.99 # discount factor
batch_size = 64

# --- Define render_episodes function first ---
def render_episodes(q_network, env_name, num_episodes=5, use_policy=True):
    env = gym.make(env_name, render_mode='human')
    for ep in range(num_episodes):
        state = env.reset()
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
                action = env.action_space.sample()
            result = env.step(action)
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
    env.close()

def visualize_saved_episodes(episode_recorder, num_episodes_to_show=3):
    """Visualize the best saved episodes"""
    if not episode_recorder.best_episodes:
        print("No episodes to visualize!")
        return
    
    # Show top episodes
    episodes_to_show = min(num_episodes_to_show, len(episode_recorder.best_episodes))
    
    for i in range(episodes_to_show):
        reward, episode_data = episode_recorder.best_episodes[i]
        frames = episode_data['frames']
        actions = episode_data['actions']
        total_reward = episode_data['total_reward']
        
        print(f"\n=== Episode {i+1} (Reward: {total_reward}) ===")
        
        # Create animation
        fig, ax = plt.subplots(figsize=(8, 6))
        plt.axis('off')
        
        if frames and len(frames) > 0:
            # Ensure frame is a numpy array
            first_frame = np.array(frames[0])
            im = ax.imshow(first_frame)
            ax.set_title(f'Best Episode {i+1} - Total Reward: {total_reward}')
            
            def update(frame_idx):
                if frame_idx < len(frames):
                    frame = np.array(frames[frame_idx])
                    im.set_data(frame)
                return [im]
            
            ani = FuncAnimation(fig, update, frames=len(frames), 
                              interval=100, blit=True, repeat=True)
            plt.show()
        else:
            print("No valid frames to display!")
        
        # Wait a bit between episodes
        time.sleep(2)

# --- Now you can call it before training ---
print("Showing first 5 episodes (untrained agent):")
render_episodes(q_network, 'CartPole-v1', num_episodes=5, use_policy=False)

# training loop
for episode in range(100):
    state = env.reset()
    if isinstance(state, tuple):
        state = state[0]  # handle new gym API that returns (state, info)
    state = np.array(state, dtype=np.float32)  # convert to numpy array first
    state = torch.FloatTensor(state)  # then convert to tensor
    total_reward = 0

    # Record episode data
    episode_data = {
        'frames': [],
        'actions': [],
        'rewards': []
    }

    done = False
    while not done:
        if np.random.rand() < epsilon:
            action = env.action_space.sample() # random action (exploration)
        else:
            with torch.no_grad(): # we dont need gradience, we are just predicting
                q_values = q_network(state) # get q values for each action
                action = q_values.argmax().item() # choose the action with the highest q value

        result = env.step(action) # take action, get next state, reward, and done flag
        if len(result) == 4:
            next_state, reward, done, _ = result  # old gym API
        else:
            next_state, reward, terminated, truncated, _ = result  # new gym API
            done = terminated or truncated
            
        next_state = np.array(next_state, dtype=np.float32)  # convert to numpy array first
        next_state = torch.FloatTensor(next_state) # convert next state to tensor
        replay_buffer.push(state, action, reward, next_state, done) # store experience in buffer

        # Record frame and action - use rgb_array mode for proper frame capture
        try:
            frame = env.render()
            if frame is not None:
                episode_data['frames'].append(frame)
        except:
            # If render fails, skip frame recording
            pass
        episode_data['actions'].append(action)
        episode_data['rewards'].append(reward)

        state = next_state
        total_reward += reward

        if len(replay_buffer) > batch_size: # if buffer has enough experiences, train the network
            states, actions, rewards, next_states, dones = replay_buffer.sample(batch_size) # sample 64 random experiences from buffer

            states = torch.stack(states) # stack 64 states into a single tensor
            actions = torch.tensor(actions) # convert actions to tensor
            rewards = torch.tensor(rewards) # convert rewards to tensor
            next_states = torch.stack(next_states) # stack 64 next states into a single tensor
            dones = torch.tensor(dones, dtype=torch.float32) # convert dones to tensor

            current_q = q_network(states).gather(1, actions.unsqueeze(1)).squeeze(1) # get q values for current state and action
            # unsqueeze(1) adds a dimension to the actions tensor, so it can be used in the gather operation
            # gather(1, actions.unsqueeze(1)) gets the q values for the actions in the batch
            # squeeze(1) removes the dimension added by unsqueeze

            next_q = q_network(next_states).max(1)[0] # get max q value for next state
            expected_q = rewards + gamma * next_q * (1 - dones) # bellman equation, if done is true, no need to discount

            loss = F.mse_loss(current_q, expected_q) # mean squared error loss

            optimizer.zero_grad() # clear gradients
            loss.backward() # backpropagate the loss
            optimizer.step() # update the weights

            epsilon = max(epsilon_min, epsilon * epsilon_decay) # decay epsilon

    # Record this episode if it's among the best
    episode_recorder.record_episode(episode_data, total_reward)
    
    if episode % 50 == 0:  # Print every 50 episodes
        print(f"Episode {episode} - Total reward: {total_reward} - Epsilon: {epsilon:.3f}")

# Save the best episodes
episode_recorder.save_episodes('best_cartpole_episodes.pkl')

# Save the trained model for later use
torch.save(q_network.state_dict(), 'best_cartpole_dqn.pth')
print("Saved trained model to best_cartpole_dqn.pth")

# --- And after training ---
print("Showing last 5 episodes (trained agent):")
render_episodes(q_network, 'CartPole-v1', num_episodes=5, use_policy=True)

# Visualize the best saved episodes
print("\n=== Visualizing Best Episodes ===")
visualize_saved_episodes(episode_recorder, num_episodes_to_show=3)

# Load the trained model
q_network = QNetwork(4, 2)
q_network.load_state_dict(torch.load('best_cartpole_dqn.pth'))
q_network.eval()

# Run evaluation
env = gym.make('CartPole-v1', render_mode='human')
for ep in range(3):
    state = env.reset()
    if isinstance(state, tuple):
        state = state[0]
    state = np.array(state, dtype=np.float32)
    state = torch.FloatTensor(state)
    done = False
    total_reward = 0
    while not done:
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
        state = torch.FloatTensor(next_state)
        total_reward += reward
        time.sleep(0.02)
    print(f"Episode {ep+1}: Total reward = {total_reward}")
env.close()
