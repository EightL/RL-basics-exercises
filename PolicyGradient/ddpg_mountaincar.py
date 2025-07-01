import gym
import argparse
from stable_baselines3 import DDPG
from stable_baselines3.common.noise import NormalActionNoise
import numpy as np
from stable_baselines3.common.callbacks import CheckpointCallback


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', nargs='?', default='train', choices=['train', 'visualize'], help="Mode: 'train' or 'visualize'")
    parser.add_argument('--model-path', default='ddpg_mountaincar', help='Path to save/load the DDPG model')
    parser.add_argument('--timesteps', type=int, default=100_000, help='Number of training timesteps')
    args = parser.parse_args()

    if args.mode == 'visualize':
        env = gym.make('MountainCarContinuous-v0', render_mode="human")
        model = DDPG.load(args.model_path)
        obs, _ = env.reset()
        done = False
        total_reward = 0
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, truncated, info = env.step(action)
            total_reward += reward
            if done or truncated:
                break
        print(f"Total reward: {total_reward}")
        env.close()
        return

    # Training mode
    env = gym.make('MountainCarContinuous-v0')
    n_actions = env.action_space.shape[-1]
    action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=0.1 * np.ones(n_actions))
    model = DDPG('MlpPolicy', env, action_noise=action_noise, verbose=1)

    # Save a checkpoint every 50,000 steps
    checkpoint_callback = CheckpointCallback(save_freq=50000, save_path='./checkpoints/',
                                             name_prefix='ddpg_mountaincar')

    model.learn(total_timesteps=args.timesteps, callback=checkpoint_callback)
    model.save(args.model_path)
    print(f"Training complete. Model saved to {args.model_path}")
    env.close()

if __name__ == "__main__":
    main() 