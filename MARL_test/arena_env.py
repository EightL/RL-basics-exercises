import gymnasium as gym
from gymnasium import spaces
import numpy as np
import random
from collections import defaultdict
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
import matplotlib
matplotlib.use('Agg')

# Constants
GRID_SIZE = 10
NUM_AGENTS = 3
NUM_TREES = 10
NUM_ORE = 8
NUM_BENCHES = 3
NUM_MOBS = 4
MAX_STEPS = 200
BUFF_DURATION = 10
BOSS_HP = 30
MOB_HP = 5
AGENT_HP = 10
WOOD_PER_GATHER = 5
ORE_PER_GATHER = 5
WOOD_PER_CRAFT = 5
ORE_PER_CRAFT = 5
BUFF_ATTACK = 3
BUFF_HP = 5
ALPHA = 5   # Joint action bonus
BETA = 1    # Proximity bonus
GAMMA = 3   # Craft proximity bonus

RESOURCE_TREE = 1
RESOURCE_ORE = 2
BENCH = 3
MOB = 4
BOSS = 5

# Action space: 0=up, 1=down, 2=left, 3=right, 4=gather, 5=attack, 6=idle, 7=craft
ACTION_MEANINGS = ['up', 'down', 'left', 'right', 'gather', 'attack', 'idle', 'craft']

class ArenaEnv(gym.Env):
    """
    Multi-agent grid world with resources, mobs, boss, and crafting.
    Cooperative and competitive reward modes.
    """
    metadata = {"render_modes": ["human"], "render_fps": 4}

    def __init__(self, cooperative=True, seed=None):
        super().__init__()
        self.grid_size = GRID_SIZE
        self.num_agents = NUM_AGENTS
        self.max_steps = MAX_STEPS
        self.cooperative = cooperative
        self.alpha = ALPHA if cooperative else 0
        self.beta = BETA if cooperative else 0
        self.gamma = GAMMA if cooperative else 0
        self.seed(seed)
        self.action_space = spaces.MultiDiscrete([8] * self.num_agents)
        # Observation: for each agent: own pos, HP, buff, all resource/mob/bench/boss positions, team inventory
        obs_len = 2 + 1 + 1 + (NUM_TREES+NUM_ORE+NUM_BENCHES+NUM_MOBS)*2 + 2 + 2
        self.observation_space = spaces.Box(low=0, high=GRID_SIZE, shape=(self.num_agents, obs_len), dtype=np.float32)
        self.reset()

    def seed(self, seed=None):
        np.random.seed(seed)
        random.seed(seed)

    def reset(self, seed=None, options=None):
        self.steps = 0
        self.done = False
        self.team_resources = {'wood': 0, 'ore': 0}
        self.metrics = defaultdict(int)
        # Place agents
        self.agents = []
        for i in range(self.num_agents):
            self.agents.append({
                'pos': [0, i],
                'hp': AGENT_HP,
                'buff': 0,
                'alive': True,
                'buff_timer': 0
            })
        # Place resources
        self.trees = self._random_positions(NUM_TREES)
        self.ores = self._random_positions(NUM_ORE)
        self.benches = self._random_positions(NUM_BENCHES)
        self.mobs = [{'pos': p, 'hp': MOB_HP, 'alive': True} for p in self._random_positions(NUM_MOBS)]
        self.boss = {'pos': [self.grid_size//2, self.grid_size//2], 'hp': BOSS_HP, 'alive': True, 'aggressive': False}
        self._update_grid()
        obs = self._get_obs()
        return obs, {}

    def _random_positions(self, n):
        positions = set()
        while len(positions) < n:
            pos = (random.randint(0, self.grid_size-1), random.randint(0, self.grid_size-1))
            # Avoid center (boss) and agent spawns
            if pos == (self.grid_size//2, self.grid_size//2) or pos in [(0, i) for i in range(self.num_agents)]:
                continue
            positions.add(pos)
        return list(positions)

    def _update_grid(self):
        self.grid = np.zeros((self.grid_size, self.grid_size), dtype=int)
        for t in self.trees:
            self.grid[t[0], t[1]] = RESOURCE_TREE
        for o in self.ores:
            self.grid[o[0], o[1]] = RESOURCE_ORE
        for b in self.benches:
            self.grid[b[0], b[1]] = BENCH
        for m in self.mobs:
            if m['alive']:
                self.grid[m['pos'][0], m['pos'][1]] = MOB
        if self.boss['alive']:
            self.grid[self.boss['pos'][0], self.boss['pos'][1]] = BOSS

    def _get_obs(self):
        obs = []
        for agent in self.agents:
            # [x, y, hp, buff, ...positions..., team_wood, team_ore]
            o = [*agent['pos'], agent['hp'], agent['buff']]
            for t in self.trees:
                o.extend(t)
            for o_ in self.ores:
                o.extend(o_)
            for b in self.benches:
                o.extend(b)
            for m in self.mobs:
                o.extend(m['pos'])
            o.extend(self.boss['pos'])
            o.append(self.boss['hp'])
            o.append(self.team_resources['wood'])
            o.append(self.team_resources['ore'])
            obs.append(np.array(o, dtype=np.float32))
        return np.stack(obs)

    def step(self, actions):
        self.steps += 1
        rewards = [0 for _ in range(self.num_agents)]
        info = {'joint_gather': 0, 'joint_attack': 0, 'joint_attack_boss': 0, 'num_crafts': 0, 'survival_rate': 0}
        # 1. Move agents
        for i, (agent, action) in enumerate(zip(self.agents, actions)):
            if not agent['alive']:
                continue
            if action == 0:  # up
                agent['pos'][0] = max(0, agent['pos'][0]-1)
            elif action == 1:  # down
                agent['pos'][0] = min(self.grid_size-1, agent['pos'][0]+1)
            elif action == 2:  # left
                agent['pos'][1] = max(0, agent['pos'][1]-1)
            elif action == 3:  # right
                agent['pos'][1] = min(self.grid_size-1, agent['pos'][1]+1)
            # else: handled below
        # 2. Gather
        gather_events = defaultdict(list)  # pos -> [agent_idx]
        for i, (agent, action) in enumerate(zip(self.agents, actions)):
            if not agent['alive']:
                continue
            if action == 4:  # gather
                pos = tuple(agent['pos'])
                if pos in self.trees:
                    gather_events[pos].append(i)
                elif pos in self.ores:
                    gather_events[pos].append(i)
        # Remove resources after gathering
        trees_to_remove = []
        ores_to_remove = []
        for pos, idxs in gather_events.items():
            is_tree = pos in self.trees
            is_ore = pos in self.ores
            joint = len(idxs) > 1
            for i in idxs:
                if is_tree:
                    self.team_resources['wood'] += WOOD_PER_GATHER * (2 if joint else 1)
                    rewards[i] += 2 if joint else 1
                if is_ore:
                    self.team_resources['ore'] += ORE_PER_GATHER * (2 if joint else 1)
                    rewards[i] += 2 if joint else 1
                if joint:
                    rewards[i] += self.alpha
            if joint:
                info['joint_gather'] += 1
            # Mark resource for removal
            if is_tree:
                trees_to_remove.append(pos)
            if is_ore:
                ores_to_remove.append(pos)
        # Actually remove gathered resources
        self.trees = [t for t in self.trees if tuple(t) not in trees_to_remove]
        self.ores = [o for o in self.ores if tuple(o) not in ores_to_remove]
        # 3. Attack
        attack_events = defaultdict(list)  # pos -> [agent_idx]
        for i, (agent, action) in enumerate(zip(self.agents, actions)):
            if not agent['alive']:
                continue
            if action == 5:  # attack
                pos = tuple(agent['pos'])
                # Mob?
                for m in self.mobs:
                    if m['alive'] and tuple(m['pos']) == pos:
                        attack_events[pos].append(i)
                # Boss?
                if self.boss['alive'] and tuple(self.boss['pos']) == pos:
                    attack_events['boss'].append(i)
        # Mob attacks
        for pos, idxs in attack_events.items():
            if pos == 'boss':
                continue
            joint = len(idxs) > 1
            for i in idxs:
                for m in self.mobs:
                    if m['alive'] and tuple(m['pos']) == pos:
                        dmg = BUFF_ATTACK if self.agents[i]['buff'] else 1
                        m['hp'] -= dmg
                        rewards[i] += 2 if joint else 1
                        if joint:
                            rewards[i] += self.alpha
                        if m['hp'] <= 0:
                            m['alive'] = False
                            rewards[i] += 5
            if joint:
                info['joint_attack'] += 1
        # Boss attacks
        if 'boss' in attack_events:
            idxs = attack_events['boss']
            joint = len(idxs) > 1
            for i in idxs:
                dmg = BUFF_ATTACK if self.agents[i]['buff'] else 1
                self.boss['hp'] -= dmg
                rewards[i] += 3 if joint else 1
                if joint:
                    rewards[i] += self.alpha
                if self.boss['hp'] <= 0:
                    self.boss['alive'] = False
                    rewards[i] += 20
            if joint:
                info['joint_attack_boss'] += 1
        # 4. Craft
        for i, (agent, action) in enumerate(zip(self.agents, actions)):
            if not agent['alive']:
                continue
            if action == 7:  # craft
                pos = tuple(agent['pos'])
                if pos in self.benches and self.team_resources['wood'] >= WOOD_PER_CRAFT and self.team_resources['ore'] >= ORE_PER_CRAFT:
                    self.team_resources['wood'] -= WOOD_PER_CRAFT
                    self.team_resources['ore'] -= ORE_PER_CRAFT
                    agent['buff'] = 1
                    agent['buff_timer'] = BUFF_DURATION
                    rewards[i] += 10
                    info['num_crafts'] += 1
                    # Proximity bonus
                    if self.gamma > 0:
                        for j, other in enumerate(self.agents):
                            if j != i and other['alive'] and np.linalg.norm(np.array(agent['pos']) - np.array(other['pos'])) <= 1:
                                rewards[i] += self.gamma
        # 5. Buff timer decrement
        for agent in self.agents:
            if agent['buff']:
                agent['buff_timer'] -= 1
                if agent['buff_timer'] <= 0:
                    agent['buff'] = 0
        # 6. Boss aggression
        if not self.boss['aggressive'] and (self.steps > 20 or any(tuple(agent['pos']) == tuple(self.boss['pos']) for agent in self.agents)):
            self.boss['aggressive'] = True
        # 7. Boss attacks agents if aggressive
        if self.boss['alive'] and self.boss['aggressive']:
            for agent in self.agents:
                if agent['alive'] and np.linalg.norm(np.array(agent['pos']) - np.array(self.boss['pos'])) <= 1:
                    agent['hp'] -= 1
                    if agent['hp'] <= 0:
                        agent['alive'] = False
        # 8. Mobs attack agents if adjacent
        for m in self.mobs:
            if m['alive']:
                for agent in self.agents:
                    if agent['alive'] and np.linalg.norm(np.array(agent['pos']) - np.array(m['pos'])) <= 1:
                        agent['hp'] -= 0.5
                        if agent['hp'] <= 0:
                            agent['alive'] = False
        # 9. Metrics
        info['survival_rate'] = sum(agent['alive'] for agent in self.agents) / self.num_agents
        info['average_distance_between_agents'] = np.mean([
            np.linalg.norm(np.array(a1['pos']) - np.array(a2['pos']))
            for i, a1 in enumerate(self.agents) for j, a2 in enumerate(self.agents) if i < j
        ])
        # 10. Done?
        done = False
        victory = False
        if not self.boss['alive'] and any(agent['alive'] for agent in self.agents):
            done = True
            victory = True
        if all(not agent['alive'] for agent in self.agents):
            done = True
        if self.steps >= self.max_steps:
            done = True
        self.done = done
        self._update_grid()
        obs = self._get_obs()
        return obs, rewards, done, False, info

    def render(self, mode="human"):
        # Simple text render
        grid = np.full((self.grid_size, self.grid_size), '.', dtype=str)
        for t in self.trees:
            grid[t[0], t[1]] = 'T'
        for o in self.ores:
            grid[o[0], o[1]] = 'O'
        for b in self.benches:
            grid[b[0], b[1]] = 'C'
        for m in self.mobs:
            if m['alive']:
                grid[m['pos'][0], m['pos'][1]] = 'M'
        if self.boss['alive']:
            grid[self.boss['pos'][0], self.boss['pos'][1]] = 'B'
        for i, agent in enumerate(self.agents):
            if agent['alive']:
                grid[agent['pos'][0], agent['pos'][1]] = str(i+1)
        print("\n".join([" ".join(row) for row in grid]))
        print(f"Team resources: {self.team_resources}")
        for i, agent in enumerate(self.agents):
            print(f"Agent {i+1}: pos={agent['pos']} HP={agent['hp']} buff={agent['buff']} alive={agent['alive']}")
        print(f"Boss HP: {self.boss['hp']} Aggressive: {self.boss['aggressive']}")

    def render_frame(self):
        import matplotlib.pyplot as plt
        from matplotlib.patches import Rectangle
        from matplotlib.backends.backend_agg import FigureCanvasAgg

        color_map = {
            '.': (1, 1, 1),        # empty
            'T': (0.13, 0.55, 0.13),  # tree (forest green)
            'O': (0.6, 0.6, 0.6),  # ore (gray)
            'C': (1.0, 0.75, 0.0),  # bench (gold)
            'M': (0.8, 0.0, 0.0),  # mob (red)
            'B': (0.0, 0.2, 0.8),  # boss (blue)
            '1': (0.0, 0.0, 0.0),  # agent 1 (black)
            '2': (0.0, 0.7, 0.9),  # agent 2 (cyan)
            '3': (1.0, 0.5, 0.0),  # agent 3 (orange)
        }
        overlay_map = {
            'T': 'T', 'O': 'O', 'C': 'C', 'M': 'M', 'B': 'B', '1': '1', '2': '2', '3': '3'
        }
        grid = np.full((self.grid_size, self.grid_size), '.', dtype=str)
        for t in self.trees:
            grid[t[0], t[1]] = 'T'
        for o in self.ores:
            grid[o[0], o[1]] = 'O'
        for b in self.benches:
            grid[b[0], b[1]] = 'C'
        for m in self.mobs:
            if m['alive']:
                grid[m['pos'][0], m['pos'][1]] = 'M'
        if self.boss['alive']:
            grid[self.boss['pos'][0], self.boss['pos'][1]] = 'B'
        for i, agent in enumerate(self.agents):
            if agent['alive']:
                grid[agent['pos'][0], agent['pos'][1]] = str(i+1)
        # Draw with matplotlib for overlays and grid
        fig, ax = plt.subplots(figsize=(5, 5), dpi=64)
        ax.set_xlim(0, self.grid_size)
        ax.set_ylim(0, self.grid_size)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_aspect('equal')
        for i in range(self.grid_size):
            for j in range(self.grid_size):
                color = color_map.get(grid[i, j], (1, 1, 1))
                rect = Rectangle((j, self.grid_size-1-i), 1, 1, facecolor=color, edgecolor='black', linewidth=1)
                ax.add_patch(rect)
                if grid[i, j] in overlay_map:
                    ax.text(j+0.5, self.grid_size-1-i+0.5, overlay_map[grid[i, j]],
                            ha='center', va='center', fontsize=18, color='black' if color != (0,0,0) else 'white', fontweight='bold')
        plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
        # Use FigureCanvasAgg directly
        canvas = FigureCanvasAgg(fig)
        canvas.draw()
        img = np.frombuffer(canvas.buffer_rgba(), dtype='uint8')
        img = img.reshape(fig.canvas.get_width_height()[::-1] + (4,))
        img = img[..., :3]  # Drop alpha channel for RGB
        plt.close(fig)
        return img

def simple_policy(agent, env):
    # If on a resource, gather
    if tuple(agent['pos']) in env.trees or tuple(agent['pos']) in env.ores:
        return 4  # gather
    # If on a bench and have resources, craft
    if tuple(agent['pos']) in env.benches and env.team_resources['wood'] >= 2 and env.team_resources['ore'] >= 2:
        return 7  # craft
    # If on boss, attack
    if tuple(agent['pos']) == tuple(env.boss['pos']):
        return 5  # attack
    # Otherwise, move randomly
    return np.random.choice([0, 1, 2, 3])

if __name__ == "__main__":
    env = ArenaEnv(cooperative=True)  # Set to False for competitive mode
    for episode in range(3):
        obs, _ = env.reset()
        done = False
        total_rewards = np.zeros(env.num_agents)
        while not done:
            # Random actions for each agent
            actions = [env.action_space.sample()[i] for i in range(env.num_agents)]
            obs, rewards, done, _, info = env.step(actions)
            total_rewards += rewards
            env.render()
        print(f"Episode {episode+1} finished. Total rewards: {total_rewards}, Survival rate: {info['survival_rate']}") 