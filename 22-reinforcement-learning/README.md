# Module 22: Reinforcement Learning

**Phase 10 — Advanced Specialized Topics** | Est. time: 1–1.5 months (full-time) · 2–3 months (part-time)

---

## Learning Objectives

By the end of this module, you will:
- Formulate real-world problems as Markov Decision Processes (MDPs)
- Implement Q-Learning and Deep Q-Networks (DQN) from scratch
- Apply policy gradient methods and actor-critic algorithms
- Train agents in OpenAI Gym / Gymnasium environments
- Understand multi-agent RL and hierarchical RL

---

## Prerequisites

- Module 10: Deep Learning Frameworks (PyTorch)
- Basic probability theory (Module 00)

---

## Topics Covered

### Foundations
- **Agent, Environment, State, Action, Reward, Policy**
- Markov Decision Processes (MDPs): states, transitions, rewards, discount factor γ
- Bellman equations: Bellman optimality, value iteration, policy iteration
- Exploration vs. exploitation: ε-greedy, UCB, Thompson sampling
- Return, cumulative discounted reward, value functions V(s) and Q(s,a)

### Dynamic Programming (Model-Based)
- Policy Evaluation: compute V^π iteratively
- Policy Improvement: make policy greedy w.r.t. V^π
- Policy Iteration and Value Iteration
- Limitation: requires full model of the environment

### Model-Free Methods
- **Monte Carlo**: learn from complete episodes; on-policy and off-policy
- **Temporal Difference (TD)**: learn from incomplete episodes; TD(0), n-step TD
- **Q-Learning**: off-policy TD; tabular Q-table updates
- **SARSA**: on-policy TD

### Deep Q-Networks (DQN)
- Neural network as function approximator for Q(s,a)
- Experience Replay: break correlation between samples
- Target Network: stable learning targets
- **Double DQN**: reduce Q-value overestimation
- **Dueling DQN**: separate value and advantage streams
- **Prioritized Experience Replay**: sample important transitions more

### Policy Gradient Methods
- REINFORCE: Monte Carlo policy gradient
- Baseline: reduce variance with value function baseline
- **TRPO** (Trust Region Policy Optimization): constrained policy updates
- **PPO** (Proximal Policy Optimization): clipped surrogate objective; most popular modern algorithm
- PPO training loop: collect rollouts → compute advantages → update policy

### Actor-Critic Methods
- **A2C** (Advantage Actor-Critic): synchronous parallel workers
- **A3C** (Asynchronous A3C): async workers; often outperformed by A2C
- **SAC** (Soft Actor-Critic): entropy regularization; sample efficient; continuous action spaces
- **TD3** (Twin Delayed DDPG): addresses overestimation in continuous control

### Advanced Topics
- **Hierarchical RL**: options framework, sub-goals
- **Imitation Learning**: behavioral cloning, DAGGER, inverse RL
- **Multi-Agent RL**: cooperative (MADDPG, QMIX), competitive, mixed
- **Model-Based RL**: Dyna-Q, World Models, MuZero

### Environments & Libraries
- `gymnasium` (OpenAI Gym successor): CartPole, LunarLander, Atari, MuJoCo
- `stable-baselines3`: production-ready PPO, SAC, TD3, A2C implementations
- `PettingZoo`: multi-agent environments
- Custom Gym environments: how to wrap your own problem

---

> **Note**: All learning content for this module is contained in this README. Additional notebooks and exercises can be added as you work through the material.


---

## Project Ideas

1. **CartPole with DQN**: train an agent to balance a pole (classic intro project)
2. **LunarLander with PPO**: train to land a rocket with PPO via stable-baselines3
3. **Custom environment**: wrap a business problem as a Gym env (pricing, inventory, routing)

---

## Further Reading

- [Reinforcement Learning: An Introduction](http://incompleteideas.net/book/the-book-2nd.html) — Sutton & Barto (free online)
- [Spinning Up in Deep RL](https://spinningup.openai.com/) — OpenAI
- [resources/reinforcement_learning.md](../resources/reinforcement_learning.md) — Quick reference guide

---

## Next Module

[Module 23: Graph Neural Networks →](../23-graph-neural-networks/README.md)
