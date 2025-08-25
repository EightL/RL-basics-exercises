import gym
import numpy as np
import matplotlib.pyplot as plt
import sys
from collections import defaultdict
import time

# Compatibility for numpy bool8
if not hasattr(np, 'bool8'):
    np.bool8 = np.bool_

# Create the Frozen Lake environment
env = gym.make('FrozenLake-v1', is_slippery=True)

print(f"Environment: {env.spec.id}")
print(f"Action space: {env.action_space}")
print(f"Observation space: {env.observation_space}")
print(f"Number of states: {env.observation_space.n}")
print(f"Number of actions: {env.action_space.n}")

def make_epsilon_greedy_policy(Q, epsilon, nA):
    def policy_fn(observation):
        A = np.ones(nA, dtype=float) * epsilon / nA
        best_action = np.argmax(Q[observation])
        A[best_action] += (1.0 - epsilon)
        return A
    return policy_fn

def q_learning(env, num_episodes, discount_factor=1.0, alpha=0.5, epsilon_start=1.0, epsilon_end=0.05, epsilon_decay=0.9995):
    Q = defaultdict(lambda: np.zeros(env.action_space.n))
    episode_rewards = []
    epsilon = epsilon_start
    
    for i_episode in range(num_episodes):
        if (i_episode + 1) % 100 == 0:
            print(f"\rEpisode {i_episode + 1}/{num_episodes}. Epsilon: {epsilon:.3f}", end="")
            sys.stdout.flush()
        
        state, _ = env.reset()
        done = False
        episode_reward = 0
        
        # Use current epsilon for this episode
        policy = make_epsilon_greedy_policy(Q, epsilon, env.action_space.n)
        
        while not done:
            action_probs = policy(state)
            action = np.random.choice(np.arange(env.action_space.n), p=action_probs)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            episode_reward += reward
            best_next_action_q_value = np.max(Q[next_state])
            Q[state][action] = Q[state][action] + alpha * (
                reward + discount_factor * best_next_action_q_value - Q[state][action]
            )
            state = next_state
        episode_rewards.append(episode_reward)
        # Decay epsilon
        epsilon = max(epsilon_end, epsilon * epsilon_decay)
    return Q, episode_rewards

def plot_rewards(episode_rewards, window=100):
    plt.figure(figsize=(10, 5))
    
    # Raw rewards
    plt.subplot(1, 2, 1)
    plt.plot(episode_rewards, alpha=0.6)
    plt.title('Episode Rewards')
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    
    # Smoothed rewards
    plt.subplot(1, 2, 2)
    if len(episode_rewards) >= window:
        smoothed = np.convolve(episode_rewards, np.ones(window)/window, mode='valid')
        plt.plot(range(window-1, len(episode_rewards)), smoothed)
    plt.title(f'Smoothed Rewards (window={window})')
    plt.xlabel('Episode')
    plt.ylabel('Average Reward')
    
    plt.tight_layout()
    plt.show()

def run_greedy_episodes_human(Q, n_episodes=5):
    render_env = gym.make('FrozenLake-v1', is_slippery=True, render_mode='human')
    for ep in range(n_episodes):
        state, _ = render_env.reset()
        done = False
        steps = 0
        total_reward = 0
        print(f"\nEpisode {ep+1}:")
        while not done and steps < 100:
            action = np.argmax(Q[state])
            next_state, reward, terminated, truncated, _ = render_env.step(action)
            done = terminated or truncated
            state = next_state
            total_reward += reward
            steps += 1
            time.sleep(0.3)  # Adjust for slower/faster animation
        print(f"  Episode finished in {steps} steps with total reward {total_reward}")
    render_env.close()

if __name__ == "__main__":
    print("Starting Q-Learning for Frozen Lake...")
    print("=" * 50)
    
    num_episodes = 100000
    Q, episode_rewards = q_learning(
        env, num_episodes,
        discount_factor=0.99, alpha=0.8,
        epsilon_start=1.0, epsilon_end=0.001, epsilon_decay=0.9995
    )
    
    print(f"\nQ-Learning completed after {num_episodes} episodes!")
    print("=" * 50)
    
    # Print stats
    success_rate = sum(1 for r in episode_rewards if r > 0) / len(episode_rewards)
    avg_reward = np.mean(episode_rewards)
    print(f"Success rate: {success_rate:.1%}")
    print(f"Average reward: {avg_reward:.3f}")
    
    # Show some Q-values to see if learning happened
    print("\nQ-values for key states:")
    print(f"State 0 (Start): {Q[0]}")
    print(f"State 5 (Hole): {Q[5]}")
    print(f"State 10 (Frozen): {Q[10]}")
    print(f"State 15 (Goal): {Q[15]}")
    
    # Plot
    plot_rewards(episode_rewards)
    
    print("Q-Learning implementation completed!")

    # Run greedy episodes for demonstration
    run_greedy_episodes_human(Q)
