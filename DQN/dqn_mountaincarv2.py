import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import gym
import random
import matplotlib.pyplot as plt
from collections import deque
import argparse
import os
import bisect

class DQNNetwork(nn.Module):
    """Deep Q-Network for MountainCarContinuous with discretized actions"""
    
    def __init__(self, state_size=2, action_size=21, hidden_size=128):
        super(DQNNetwork, self).__init__()
        self.fc1 = nn.Linear(state_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, hidden_size)
        self.fc4 = nn.Linear(hidden_size, action_size)
        
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        return self.fc4(x)

class PrioritizedReplayBuffer:
    """Prioritized Experience Replay Buffer (simple proportional version)"""
    def __init__(self, capacity=10000, alpha=0.6):
        self.capacity = capacity
        self.buffer = []
        self.priorities = []
        self.pos = 0
        self.alpha = alpha
        self.epsilon = 1e-5  # Small value to ensure nonzero priority

    def push(self, state, action, reward, next_state, done):
        max_priority = max(self.priorities, default=1.0)
        transition = (state, action, reward, next_state, done)
        if len(self.buffer) < self.capacity:
            self.buffer.append(transition)
            self.priorities.append(max_priority)
        else:
            self.buffer[self.pos] = transition
            self.priorities[self.pos] = max_priority
            self.pos = (self.pos + 1) % self.capacity

    def sample(self, batch_size, beta=0.4):
        if len(self.buffer) == self.capacity:
            priorities = np.array(self.priorities)
        else:
            priorities = np.array(self.priorities[:len(self.buffer)])
        probs = priorities ** self.alpha
        probs /= probs.sum()
        indices = np.random.choice(len(self.buffer), batch_size, p=probs)
        samples = [self.buffer[idx] for idx in indices]
        # Importance-sampling weights
        total = len(self.buffer)
        weights = (total * probs[indices]) ** (-beta)
        weights /= weights.max()  # Normalize
        state, action, reward, next_state, done = map(np.stack, zip(*samples))
        return state, action, reward, next_state, done, indices, weights.astype(np.float32)

    def update_priorities(self, indices, td_errors):
        for idx, td_error in zip(indices, td_errors):
            self.priorities[idx] = abs(td_error) + self.epsilon

    def __len__(self):
        return len(self.buffer)

class DQNAgent:
    """DQN Agent with experience replay and target network (with PER)"""
    def __init__(self, state_size=2, action_size=21, lr=1e-3, gamma=0.99, 
                 epsilon=1.0, epsilon_min=0.01, epsilon_decay=0.995,
                 buffer_size=10000, batch_size=32, target_update=100,
                 per_alpha=0.6, per_beta=0.4):
        self.state_size = state_size
        self.action_size = action_size
        self.lr = lr
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size
        self.target_update = target_update
        self.actions = np.linspace(-1, 1, action_size)
        self.q_network = DQNNetwork(state_size, action_size)
        self.target_network = DQNNetwork(state_size, action_size)
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=lr)
        self.target_network.load_state_dict(self.q_network.state_dict())
        self.memory = PrioritizedReplayBuffer(buffer_size, alpha=per_alpha)
        self.per_beta = per_beta
        self.step_count = 0
        self.losses = []
        self.initial_lr = lr
        self.performance_threshold = 85.0
    
    def get_action(self, state, training=True):
        if training and random.random() < self.epsilon:
            return random.randrange(self.action_size)
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        q_values = self.q_network(state_tensor)
        return q_values.argmax().item()
    
    def store_transition(self, state, action, reward, next_state, done):
        self.memory.push(state, action, reward, next_state, done)
    
    def replay(self):
        if len(self.memory) < self.batch_size:
            return
        states, actions, rewards, next_states, dones, indices, weights = self.memory.sample(self.batch_size, beta=self.per_beta)
        states = torch.FloatTensor(states)
        actions = torch.LongTensor(actions)
        rewards = torch.FloatTensor(rewards)
        next_states = torch.FloatTensor(next_states)
        dones = torch.BoolTensor(dones)
        weights = torch.FloatTensor(weights)
        current_q_values = self.q_network(states).gather(1, actions.unsqueeze(1)).squeeze(1)
        next_actions = self.q_network(next_states).argmax(1, keepdim=True)
        next_q_values_target = self.target_network(next_states).gather(1, next_actions).squeeze(1)
        target_q_values = rewards + (self.gamma * next_q_values_target * ~dones)
        td_errors = current_q_values.detach().cpu().numpy() - target_q_values.detach().cpu().numpy()
        # PER: use importance-sampling weights in loss
        loss = (weights * F.mse_loss(current_q_values, target_q_values, reduction='none')).mean()
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), max_norm=1.0)
        self.optimizer.step()
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
        self.step_count += 1
        if self.step_count % self.target_update == 0:
            self.target_network.load_state_dict(self.q_network.state_dict())
        self.losses.append(loss.item())
        # Update priorities
        self.memory.update_priorities(indices, td_errors)
    
    def adjust_learning_rate(self, avg_reward):
        """Reduce learning rate when performance is good to prevent catastrophic forgetting"""
        if avg_reward > self.performance_threshold:
            new_lr = self.initial_lr * 0.1  # Reduce to 10% of original
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = new_lr
    
    def save_model(self, filepath):
        """Save the trained model"""
        torch.save({
            'q_network_state_dict': self.q_network.state_dict(),
            'target_network_state_dict': self.target_network.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epsilon': self.epsilon,
            'step_count': self.step_count
        }, filepath)
        print(f"Model saved to {filepath}")
    
    def load_model(self, filepath):
        """Load a trained model"""
        checkpoint = torch.load(filepath)
        self.q_network.load_state_dict(checkpoint['q_network_state_dict'])
        self.target_network.load_state_dict(checkpoint['target_network_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.epsilon = checkpoint['epsilon']
        self.step_count = checkpoint['step_count']
        print(f"Model loaded from {filepath}")
    
    def save_checkpoint(self):
        """Save a checkpoint including replay buffer state"""
        checkpoint = {
            'q_network_state_dict': self.q_network.state_dict(),
            'target_network_state_dict': self.target_network.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epsilon': self.epsilon,
            'step_count': self.step_count,
            'replay_buffer': {
                'buffer': self.memory.buffer.copy(),
                'priorities': self.memory.priorities.copy(),
                'pos': self.memory.pos
            }
        }
        return checkpoint
    
    def restore_checkpoint(self, checkpoint):
        """Restore from a saved checkpoint"""
        self.q_network.load_state_dict(checkpoint['q_network_state_dict'])
        self.target_network.load_state_dict(checkpoint['target_network_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.epsilon = checkpoint['epsilon']
        self.step_count = checkpoint['step_count']
        
        # Restore replay buffer
        self.memory.buffer = checkpoint['replay_buffer']['buffer'].copy()
        self.memory.priorities = checkpoint['replay_buffer']['priorities'].copy()
        self.memory.pos = checkpoint['replay_buffer']['pos']
        print("🔄 Restored from checkpoint!")

def velocity_reward_shaping(state, action, reward, next_state, done):
    """
    Enhance reward with velocity-based shaping.
    Higher velocity (absolute value) gets better reward.
    """
    # Original reward from environment
    shaped_reward = reward
    
    # Velocity is the second component of the state
    velocity = next_state[1]
    position = next_state[0]
    
    # If we reached the goal, give the original reward
    if done and position >= 0.45:
        return reward  # This should be +100
    
    # Reward shaping: encourage higher velocity (movement)
    velocity_bonus = abs(velocity) * 5.0  # Reduced scale to prevent overshadowing main reward
    
    # Position-based reward: encourage moving towards the goal
    if position > -0.4:  # Only reward when getting out of the valley
        position_bonus = (position + 0.4) * 10.0
        shaped_reward += position_bonus
    
    # Add velocity bonus
    shaped_reward += velocity_bonus
    
    # Small penalty for staying still
    if abs(velocity) < 0.001:
        shaped_reward -= 0.5
    
    # Cap the shaped reward to prevent it from being too large
    shaped_reward = max(min(shaped_reward, 50.0), -10.0)
    
    return shaped_reward

def train_dqn(episodes=1000, save_every=500):
    """Train the DQN agent on MountainCarContinuous"""
    
    # Create environment
    env = gym.make('MountainCarContinuous-v0')
    
    # Create agent
    agent = DQNAgent(
        state_size=2,
        action_size=21,  # Discretize continuous action space
        lr=1e-4,  # Even lower learning rate to prevent catastrophic forgetting
        gamma=0.99,
        epsilon=1.0,
        epsilon_min=0.02,  # Lower minimum epsilon since we're learning fast
        epsilon_decay=0.999,  # Faster decay since we solve quickly
        buffer_size=100000,  # Larger buffer to remember more experiences
        batch_size=32,
        target_update=500,  # Much less frequent target updates for stability
        per_alpha=0.6,
        per_beta=0.4
    )
    
    # Training metrics
    episode_rewards = []
    episode_lengths = []
    avg_rewards = []
    success_episodes = []  # Track which episodes reached the goal
    first_success = None
    
    # Checkpoint/backup system
    best_avg_reward = -float('inf')
    best_checkpoint = None
    episodes_since_improvement = 0
    episodes_since_restore = 0
    performance_threshold = 5.0  # Restore if performance drops by this much
    patience_episodes = 100  # Wait this many episodes before considering restore
    exploration_boost_after_restore = 0.1  # Increase epsilon after restore
    
    print("Starting DQN training on MountainCarContinuous...")
    print(f"Action space discretized to {agent.action_size} actions: {agent.actions}")
    print("Goal: Reach position 0.45 to get +100 reward")
    
    for episode in range(episodes):
        state = env.reset()
        if isinstance(state, tuple):
            state = state[0]  # Handle gym API changes
        
        total_reward = 0
        steps = 0
        max_steps = 999
        
        while steps < max_steps:
            # Get action
            action_idx = agent.get_action(state, training=True)
            continuous_action = np.array([agent.actions[action_idx]])
            
            # Take step in environment
            step_result = env.step(continuous_action)
            if len(step_result) == 5:  # New gym API
                next_state, reward, terminated, truncated, info = step_result
                done = terminated or truncated
            else:  # Old gym API
                next_state, reward, done, info = step_result
            
            if len(next_state) > 2:  # Handle potential extra dimensions
                next_state = next_state[:2]
            
            # Apply reward shaping
            shaped_reward = velocity_reward_shaping(state, continuous_action, reward, next_state, done)
            
            # Store transition
            agent.store_transition(state, action_idx, shaped_reward, next_state, done)
            
            # Train agent (only after enough experiences)
            if len(agent.memory) > 1000:  # Wait for enough experiences before training
                agent.replay()
            
            state = next_state
            total_reward += reward  # Use original reward for logging
            steps += 1
            
            if done:
                if next_state[0] >= 0.45:  # Successfully reached the goal
                    success_episodes.append(episode)
                    if first_success is None:
                        first_success = episode
                        print(f"🎉 First successful episode: {episode} (reached goal in {steps} steps!)")
                break
        
        episode_rewards.append(total_reward)
        episode_lengths.append(steps)
        
        # Calculate running average
        if len(episode_rewards) >= 100:
            avg_reward = np.mean(episode_rewards[-100:])
            avg_rewards.append(avg_reward)
        else:
            avg_rewards.append(np.mean(episode_rewards))
        
        # Adjust learning rate based on performance
        if len(avg_rewards) >= 100:
            agent.adjust_learning_rate(avg_rewards[-1])
        
        # Logging
        if episode % 50 == 0:
            avg_reward = avg_rewards[-1]
            avg_length = np.mean(episode_lengths[-50:]) if len(episode_lengths) >= 50 else np.mean(episode_lengths)
            success_rate = len([e for e in success_episodes if e >= episode - 50]) / min(50, episode + 1) * 100
            current_lr = agent.optimizer.param_groups[0]['lr']
            print(f"Episode {episode:4d} | Avg Reward: {avg_reward:7.2f} | "
                  f"Avg Length: {avg_length:6.1f} | Success Rate: {success_rate:5.1f}% | Epsilon: {agent.epsilon:.3f} | LR: {current_lr:.2e}")
        
        # Save model periodically
        if episode % save_every == 0 and episode > 0:
            agent.save_model(f'mountaincar_dqn_episode_{episode}.pth')
        
        # Checkpoint/Restore System
        if len(avg_rewards) >= 100:
            current_avg = avg_rewards[-1]
            episodes_since_restore += 1
            
            # Save checkpoint when we find new best performance
            if current_avg > best_avg_reward + 0.5:  # Significant improvement
                best_avg_reward = current_avg
                best_checkpoint = agent.save_checkpoint()
                episodes_since_improvement = 0
                agent.save_model('mountaincar_dqn_best.pth')
                print(f"💾 New checkpoint saved! Avg reward: {best_avg_reward:.2f}")
            else:
                episodes_since_improvement += 1
            
            # Check if we should restore from checkpoint
            should_restore = (
                best_checkpoint is not None and  # We have a checkpoint
                episodes_since_restore > patience_episodes and  # Waited long enough
                current_avg < best_avg_reward - performance_threshold and  # Performance dropped
                episodes_since_improvement > 50  # No recent improvement
            )
            
            if should_restore:
                print(f"📉 Performance dropped from {best_avg_reward:.2f} to {current_avg:.2f}")
                print(f"🔄 Restoring checkpoint and boosting exploration...")
                
                # Restore the best checkpoint
                agent.restore_checkpoint(best_checkpoint)
                
                # Boost exploration to try new strategies
                agent.epsilon = min(0.3, agent.epsilon + exploration_boost_after_restore)
                
                # Reset counters
                episodes_since_restore = 0
                episodes_since_improvement = 0
                
                print(f"🎲 Exploration boosted to {agent.epsilon:.3f}")
            
            # Early stopping if consistently poor after multiple restores
            if episodes_since_improvement > 300:
                print(f"⚠️ No improvement for {episodes_since_improvement} episodes. Stopping training.")
                break
        
        # Check if environment is consistently solved (but don't stop training)
        if len(avg_rewards) >= 100 and avg_rewards[-1] > 90:
            if episode % 100 == 0:  # Only print this occasionally
                print(f"Environment consistently performing well! (100-episode avg: {avg_rewards[-1]:.2f})")
        
        # Only stop if we've achieved near-perfect performance for a long time
        if len(avg_rewards) >= 200 and avg_rewards[-1] > 95 and all(r > 90 for r in avg_rewards[-100:]):
            print(f"Optimal policy found after {episode} episodes! (Avg reward: {avg_rewards[-1]:.2f})")
            break
    
    env.close()
    
    # Print training summary
    print(f"\n=== Training Summary ===")
    print(f"Total episodes: {episode + 1}")
    if first_success is not None:
        print(f"First success at episode: {first_success}")
    print(f"Total successful episodes: {len(success_episodes)}")
    print(f"Best average reward achieved: {best_avg_reward:.2f}")
    print(f"Final 100-episode average reward: {avg_rewards[-1]:.2f}")
    final_success_rate = len([e for e in success_episodes if e >= episode - 99]) / min(100, episode + 1) * 100
    print(f"Final 100-episode success rate: {final_success_rate:.1f}%")
    if best_checkpoint is not None:
        print(f"✅ Checkpoint system was active - best performance preserved!")
    
    # Save final model
    agent.save_model('mountaincar_dqn_final.pth')
    
    # Plot training progress
    plt.figure(figsize=(15, 5))
    
    plt.subplot(1, 3, 1)
    plt.plot(episode_rewards, alpha=0.6, label='Episode Reward')
    plt.plot(avg_rewards, 'r-', linewidth=2, label='100-Episode Average')
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.title('Training Progress')
    plt.legend()
    plt.grid(True)
    
    plt.subplot(1, 3, 2)
    plt.plot(episode_lengths, alpha=0.6)
    plt.xlabel('Episode')
    plt.ylabel('Episode Length')
    plt.title('Episode Lengths')
    plt.grid(True)
    
    plt.subplot(1, 3, 3)
    if agent.losses:
        plt.plot(agent.losses, alpha=0.6)
        plt.xlabel('Training Step')
        plt.ylabel('Loss')
        plt.title('Training Loss')
        plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('mountaincar_dqn_training.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return agent, episode_rewards, avg_rewards

def visualize_agent(model_path='DQN/mountaincar_dqn_final.pth', num_episodes=5):
    """Visualize the trained agent"""
    
    # Create environment with rendering
    env = gym.make('MountainCarContinuous-v0', render_mode='human')
    
    # Create and load agent
    agent = DQNAgent()
    agent.load_model(model_path)
    agent.epsilon = 0  # No exploration during visualization
    
    print(f"Visualizing trained agent for {num_episodes} episodes...")
    
    for episode in range(num_episodes):
        state = env.reset()
        if isinstance(state, tuple):
            state = state[0]
        
        total_reward = 0
        steps = 0
        max_steps = 999
        
        print(f"\nEpisode {episode + 1}:")
        
        while steps < max_steps:
            # Get action (no exploration)
            action_idx = agent.get_action(state, training=False)
            continuous_action = np.array([agent.actions[action_idx]])
            
            # Take step
            step_result = env.step(continuous_action)
            if len(step_result) == 5:  # New gym API
                next_state, reward, terminated, truncated, info = step_result
                done = terminated or truncated
            else:  # Old gym API
                next_state, reward, done, info = step_result
            
            if len(next_state) > 2:
                next_state = next_state[:2]
            
            total_reward += reward
            steps += 1
            
            # Display current state
            position, velocity = next_state
            print(f"Step {steps:3d}: Pos={position:6.3f}, Vel={velocity:6.3f}, "
                  f"Action={continuous_action[0]:6.3f}, Reward={reward:6.2f}")
            
            state = next_state
            
            if done:
                print(f"Reached goal in {steps} steps!")
                break
        
        print(f"Episode {episode + 1} finished: Total Reward = {total_reward:.2f}, Steps = {steps}")
    
    env.close()

def main():
    parser = argparse.ArgumentParser(description='DQN for MountainCarContinuous with velocity reward shaping')
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'visualize'],
                        help='Mode: train the agent or visualize trained agent')
    parser.add_argument('--episodes', type=int, default=2000,
                        help='Number of training episodes')
    parser.add_argument('--model_path', type=str, default='mountaincar_dqn_final.pth',
                        help='Path to saved model for visualization')
    
    args = parser.parse_args()
    
    if args.mode == 'train':
        print("Training DQN agent on MountainCarContinuous with velocity reward shaping...")
        agent, rewards, avg_rewards = train_dqn(episodes=args.episodes)
        print(f"Training completed! Final average reward: {avg_rewards[-1]:.2f}")
        
    elif args.mode == 'visualize':
        if os.path.exists(args.model_path):
            visualize_agent(args.model_path)
        else:
            print(f"Model file {args.model_path} not found. Train the agent first!")

if __name__ == "__main__":
    main() 