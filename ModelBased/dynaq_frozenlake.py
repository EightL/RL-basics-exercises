import numpy as np
import gym
import random
from collections import defaultdict

# Create FrozenLake environment (4x4, slippery by default, ansi render mode)
env = gym.make('FrozenLake-v1', is_slippery=True, render_mode='ansi')

n_states = env.observation_space.n
n_actions = env.action_space.n

print(f"States: {n_states}, Actions: {n_actions}")

# Display the initial environment
state = env.reset()
if isinstance(state, tuple):
    state = state[0]
print("Initial state:", state)
print(env.render())

input("Press Enter to continue...")
