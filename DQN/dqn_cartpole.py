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

import gym
import numpy as np
import time

env = gym.make('CartPole-v1')

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

# --- Now you can call it before training ---
print("Showing first 5 episodes (untrained agent):")
render_episodes(q_network, 'CartPole-v1', num_episodes=5, use_policy=False)

# training loop
for episode in range(500):
    state = env.reset()
    if isinstance(state, tuple):
        state = state[0]  # handle new gym API that returns (state, info)
    state = np.array(state, dtype=np.float32)  # convert to numpy array first
    state = torch.FloatTensor(state)  # then convert to tensor
    total_reward = 0

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

            print(f"Episode {episode} - Total reward: {total_reward} - Epsilon: {epsilon:.3f}")

# --- And after training ---
print("Showing last 5 episodes (trained agent):")
render_episodes(q_network, 'CartPole-v1', num_episodes=5, use_policy=True)
