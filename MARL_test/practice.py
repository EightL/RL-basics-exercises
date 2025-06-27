import numpy as np
import matplotlib.pyplot as plt
from arena_env import ArenaEnv

# Improved heuristic policy for each agent
# Agents will move after resources are depleted, prioritize crafting, then attack boss

def simple_policy(agent, env):
    pos = tuple(agent['pos'])
    # 1. Craft if on bench and have resources
    if pos in env.benches and env.team_resources['wood'] >= 5 and env.team_resources['ore'] >= 5:
        return 7  # craft
    # 2. Attack if on boss
    if pos == tuple(env.boss['pos']):
        return 5  # attack
    # 3. Gather if on a resource and still need resources to craft
    if (pos in env.trees or pos in env.ores) and (env.team_resources['wood'] < 5 or env.team_resources['ore'] < 5):
        return 4  # gather
    # 4. If have enough to craft, move toward nearest bench
    if env.team_resources['wood'] >= 5 and env.team_resources['ore'] >= 5:
        targets = env.benches
    # 5. If not enough resources, move toward nearest resource
    elif env.trees or env.ores:
        targets = env.trees + env.ores
    # 6. If all resources are gone, move toward boss
    else:
        targets = [env.boss['pos']] if env.boss['alive'] else []
    if not targets:
        return 6  # idle
    dists = [np.linalg.norm(np.array(pos) - np.array(t)) for t in targets]
    nearest = targets[np.argmin(dists)]
    dx = nearest[0] - pos[0]
    dy = nearest[1] - pos[1]
    if abs(dx) > abs(dy):
        return 1 if dx > 0 else 0  # down or up
    elif dy != 0:
        return 3 if dy > 0 else 2  # right or left
    else:
        return 6  # idle

if __name__ == "__main__":
    env = ArenaEnv(cooperative=True)  # Set to False for competitive mode
    for episode in range(3):
        obs, _ = env.reset()
        done = False
        total_rewards = np.zeros(env.num_agents)
        frames = []
        while not done:
            actions = []
            for i, agent in enumerate(env.agents):
                if agent['alive']:
                    action = simple_policy(agent, env)
                else:
                    action = 6  # idle if dead
                actions.append(action)
            obs, rewards, done, _, info = env.step(actions)
            total_rewards += rewards
            env.render()  # Optional: keep text output
            frames.append(env.render_frame())
        print(f"Episode {episode+1} finished. Total rewards: {total_rewards}, Survival rate: {info['survival_rate']}")

        # --- Fast replay viewer ---
        fig, ax = plt.subplots(figsize=(6, 6))
        plt.axis('off')
        im = ax.imshow(frames[0])
        def update(frame):
            im.set_data(frame)
            return [im]
        from matplotlib.animation import FuncAnimation
        ani = FuncAnimation(fig, update, frames=frames, interval=80, blit=True, repeat=False)
        plt.show()
