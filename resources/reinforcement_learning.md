# Reinforcement Learning Guide

Fundamentals, algorithms, and practical implementation reference for the AI Engineer path.

---

## Core Concepts

```
Agent ←→ Environment
  - Agent observes state s, takes action a
  - Environment returns reward r and next state s'
  - Goal: maximise cumulative discounted reward G_t = Σ γᵏ rₜ₊ₖ
```

| Term | Symbol | Definition |
|---|---|---|
| State | s ∈ S | Current situation of the environment |
| Action | a ∈ A | Choice made by the agent |
| Reward | r | Signal from environment (positive = good) |
| Policy | π(a\|s) | Probability of taking action a in state s |
| Value function | V^π(s) | Expected return from state s under policy π |
| Q-function | Q^π(s,a) | Expected return from taking action a in state s |
| Discount factor | γ ∈ [0,1] | How much future rewards are worth (0 = myopic, 1 = far-sighted) |

---

## Part 1: Q-Learning (Tabular)

```python
import numpy as np
import gymnasium as gym

env = gym.make("FrozenLake-v1", is_slippery=True)
n_states  = env.observation_space.n   # 16
n_actions = env.action_space.n        # 4

# Q-table: rows = states, cols = actions
Q = np.zeros((n_states, n_actions))

# Hyperparameters
alpha   = 0.1     # Learning rate
gamma   = 0.99    # Discount factor
epsilon = 1.0     # Starting exploration rate
eps_min = 0.01
eps_decay = 0.995
n_episodes = 10000

rewards_history = []

for episode in range(n_episodes):
    state, _ = env.reset()
    total_reward = 0

    while True:
        # ε-greedy action selection
        if np.random.random() < epsilon:
            action = env.action_space.sample()   # Explore
        else:
            action = np.argmax(Q[state])         # Exploit

        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated

        # Q-Learning update (Bellman equation)
        target = reward + gamma * np.max(Q[next_state]) * (not done)
        Q[state, action] += alpha * (target - Q[state, action])

        state = next_state
        total_reward += reward
        if done:
            break

    # Decay epsilon
    epsilon = max(eps_min, epsilon * eps_decay)
    rewards_history.append(total_reward)

    if (episode + 1) % 1000 == 0:
        avg = np.mean(rewards_history[-100:])
        print(f"Episode {episode+1:5d} | Avg Reward: {avg:.3f} | ε: {epsilon:.3f}")

print(f"Final policy win rate: {np.mean(rewards_history[-100:]):.2%}")
```

---

## Part 2: Deep Q-Network (DQN)

```python
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from collections import deque
import random

class QNetwork(nn.Module):
    def __init__(self, state_dim, action_dim, hidden=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden),    nn.ReLU(),
            nn.Linear(hidden, action_dim)
        )
    def forward(self, x): return self.net(x)

class ReplayBuffer:
    def __init__(self, capacity=100_000):
        self.buffer = deque(maxlen=capacity)
    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))
    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return (np.array(states), np.array(actions), np.array(rewards, dtype=np.float32),
                np.array(next_states), np.array(dones, dtype=np.float32))
    def __len__(self): return len(self.buffer)

class DQNAgent:
    def __init__(self, state_dim, action_dim, lr=1e-3, gamma=0.99,
                 epsilon=1.0, eps_min=0.01, eps_decay=0.995,
                 batch_size=64, target_update_freq=100):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.action_dim   = action_dim
        self.gamma        = gamma
        self.epsilon      = epsilon
        self.eps_min      = eps_min
        self.eps_decay    = eps_decay
        self.batch_size   = batch_size
        self.target_update_freq = target_update_freq
        self.step_count   = 0

        self.q_net     = QNetwork(state_dim, action_dim).to(self.device)
        self.target_net = QNetwork(state_dim, action_dim).to(self.device)
        self.target_net.load_state_dict(self.q_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.q_net.parameters(), lr=lr)
        self.buffer    = ReplayBuffer()
        self.loss_fn   = nn.MSELoss()

    def act(self, state):
        if np.random.random() < self.epsilon:
            return np.random.randint(self.action_dim)
        state_t = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            return self.q_net(state_t).argmax().item()

    def learn(self):
        if len(self.buffer) < self.batch_size:
            return
        states, actions, rewards, next_states, dones = self.buffer.sample(self.batch_size)

        states_t      = torch.FloatTensor(states).to(self.device)
        actions_t     = torch.LongTensor(actions).to(self.device)
        rewards_t     = torch.FloatTensor(rewards).to(self.device)
        next_states_t = torch.FloatTensor(next_states).to(self.device)
        dones_t       = torch.FloatTensor(dones).to(self.device)

        # Current Q values
        q_values = self.q_net(states_t).gather(1, actions_t.unsqueeze(1)).squeeze()

        # Target Q values (using target network — stable targets)
        with torch.no_grad():
            next_q = self.target_net(next_states_t).max(1)[0]
            targets = rewards_t + self.gamma * next_q * (1 - dones_t)

        loss = self.loss_fn(q_values, targets)
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q_net.parameters(), 10.0)
        self.optimizer.step()

        # Update target network periodically
        self.step_count += 1
        if self.step_count % self.target_update_freq == 0:
            self.target_net.load_state_dict(self.q_net.state_dict())

        # Decay epsilon
        self.epsilon = max(self.eps_min, self.epsilon * self.eps_decay)

# Training loop
env = gym.make("CartPole-v1")
agent = DQNAgent(state_dim=4, action_dim=2)

for episode in range(500):
    state, _ = env.reset()
    total_reward = 0
    while True:
        action = agent.act(state)
        next_state, reward, terminated, truncated, _ = env.step(action)
        agent.buffer.push(state, action, reward, next_state, terminated or truncated)
        agent.learn()
        state = next_state
        total_reward += reward
        if terminated or truncated: break

    if (episode + 1) % 50 == 0:
        print(f"Episode {episode+1} | Reward: {total_reward:.1f} | ε: {agent.epsilon:.3f}")
```

---

## Part 3: PPO with Stable-Baselines3

```python
from stable_baselines3 import PPO, SAC, TD3, A2C
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.callbacks import EvalCallback, StopTrainingOnRewardThreshold
import gymnasium as gym

# ─── CartPole with PPO ────────────────────────────────────────────────────────
vec_env = make_vec_env("CartPole-v1", n_envs=4)   # 4 parallel environments

model = PPO(
    "MlpPolicy",
    vec_env,
    learning_rate=3e-4,
    n_steps=2048,          # Steps per environment per update
    batch_size=64,
    n_epochs=10,
    gamma=0.99,
    gae_lambda=0.95,       # Generalised Advantage Estimation λ
    clip_range=0.2,        # PPO clipping parameter ε
    ent_coef=0.01,         # Entropy bonus (encourages exploration)
    verbose=1,
    tensorboard_log="./tb_logs/ppo_cartpole"
)

# Callbacks
eval_callback = EvalCallback(
    gym.make("CartPole-v1"),
    best_model_save_path="./models/best_ppo",
    eval_freq=10_000,
    deterministic=True,
    render=False
)

model.learn(total_timesteps=200_000, callback=eval_callback)

# Evaluate
mean_reward, std_reward = evaluate_policy(model, gym.make("CartPole-v1"), n_eval_episodes=20)
print(f"Mean reward: {mean_reward:.1f} ± {std_reward:.1f}")

# Save and load
model.save("ppo_cartpole")
model = PPO.load("ppo_cartpole", env=vec_env)

# ─── Continuous control: LunarLander with SAC ────────────────────────────────
env = gym.make("LunarLanderContinuous-v2")
sac = SAC("MlpPolicy", env, verbose=1)
sac.learn(total_timesteps=300_000)
```

---

## Algorithm Comparison

| Algorithm | Type | Action Space | Sample Efficiency | Stability | Use Case |
|---|---|---|---|---|---|
| Q-Learning | Value-based | Discrete only | Low | Medium | Tabular / simple |
| DQN | Value-based | Discrete | Medium | Medium | Atari-style games |
| Double DQN | Value-based | Discrete | Medium | Good | Less overestimation |
| REINFORCE | Policy Gradient | Both | Very Low | Poor | Simple policy gradient |
| A2C | Actor-Critic | Both | Medium | Good | Fast convergence |
| PPO | Actor-Critic | Both | Medium | Excellent | Most use cases — default |
| SAC | Actor-Critic | Continuous | High | Excellent | Robotics, continuous control |
| TD3 | Actor-Critic | Continuous | High | Good | Deterministic continuous |

---

## Custom Gymnasium Environment

```python
import gymnasium as gym
import numpy as np
from gymnasium import spaces

class InventoryEnv(gym.Env):
    """Simple inventory management environment."""

    metadata = {"render_modes": ["human"]}

    def __init__(self, max_inventory=100, max_demand=30, holding_cost=1.0, stockout_cost=5.0):
        super().__init__()
        self.max_inventory  = max_inventory
        self.max_demand     = max_demand
        self.holding_cost   = holding_cost
        self.stockout_cost  = stockout_cost

        # Action: how much to order (0 to 50 units)
        self.action_space = spaces.Discrete(51)

        # Observation: current inventory level
        self.observation_space = spaces.Box(
            low=0, high=max_inventory, shape=(1,), dtype=np.float32
        )

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.inventory = self.max_inventory // 2
        self.step_count = 0
        return np.array([self.inventory], dtype=np.float32), {}

    def step(self, action):
        order_qty = int(action)
        demand    = self.np_random.integers(0, self.max_demand + 1)

        # Receive order, then fulfil demand
        self.inventory = min(self.max_inventory, self.inventory + order_qty)
        unmet_demand   = max(0, demand - self.inventory)
        self.inventory = max(0, self.inventory - demand)

        # Reward = -(holding cost + stockout cost)
        reward = -(self.holding_cost * self.inventory + self.stockout_cost * unmet_demand)

        self.step_count += 1
        terminated = self.step_count >= 365   # 1 year

        return np.array([self.inventory], dtype=np.float32), reward, terminated, False, {}

# Train PPO on custom environment
env = InventoryEnv()
model = PPO("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=500_000)
```

---

## Quick Reference: Key Formulas

```
Bellman optimality:
  Q*(s,a) = r + γ max_{a'} Q*(s',a')

DQN loss:
  L(θ) = E[(r + γ max_{a'} Q(s',a';θ⁻) - Q(s,a;θ))²]
  θ⁻ = target network parameters (updated periodically)

PPO clipped objective:
  L^CLIP(θ) = E[min(r_t(θ)Â_t, clip(r_t(θ), 1-ε, 1+ε)Â_t)]
  r_t(θ) = π_θ(a_t|s_t) / π_θ_old(a_t|s_t)   (probability ratio)

SAC entropy-regularised objective:
  J(π) = E[Σ_t γ^t (r_t + α H(π(·|s_t)))]
  α = temperature (controls exploration vs. exploitation)
```
