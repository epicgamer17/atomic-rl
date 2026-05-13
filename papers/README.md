# RL Papers & Implementations

This directory maintains a mapping between the research papers in this folder and their corresponding implementations within the repository.

## DQN & Extensions

| Paper | PDF | Implementation | Location | Notes |
| :--- | :--- | :--- | :--- | :--- |
| **DQN** (Mnih et al., 2013/2015) | [dqn.pdf](dqn.pdf) | Done | `examples/dqn/dqn_cartpole.py` | Basic DQN with experience replay and target networks. |
| **Double DQN** (Van Hasselt et al., 2015) | [double_dqn.pdf](double_dqn.pdf) | Done | `examples/dqn/ddqn_cartpole.py` | Implements decoupled action selection and evaluation to reduce bias. |
| **Dueling DQN** (Wang et al., 2015) | [dueling_dqn.pdf](dueling_dqn.pdf) | Done | `examples/dqn/dueling_dqn_cartpole.py` | Uses separate Value and Advantage streams. |
| **Prioritized Experience Replay** (Schaul et al., 2015) | [per.pdf](per.pdf) | Done | `examples/dqn/prioritized_replay_dqn_cartpole.py` | Backed by a high-performance SumTree in `functional/replay_buffer.py`. |
| **Categorical DQN (C51)** (Bellemare et al., 2017) | [categorical_dqn.pdf](categorical_dqn.pdf) | Done | `examples/dqn/categorical_dqn_cartpole.py` | Implements distributional RL by predicting a discrete value distribution. |
| **Noisy DQN** (Fortunato et al., 2017) | [noisy_dqn.pdf](noisy_dqn.pdf) | Done | `examples/dqn/noisy_dqn_cartpole.py` | Replaces epsilon-greedy with learnable noise in Linear layers. |
| **Rainbow** (Hessel et al., 2017) | [rainbow_dqn.pdf](rainbow_dqn.pdf) | Done | `examples/dqn/rainbow_dqn_cartpole.py` | Combines all the above extensions into a single agent. |
| **Revisiting Rainbow** (Obando-Ceron et al., 2020) | [revisiting_rainbow.pdf](revisiting_rainbow.pdf) | Partial | `examples/dqn/` | Smaller-scale Rainbow experiments. |

### Missing / Future Work:
- **Distributional Rainbow (IQN/QR-DQN)**: While C51 is implemented, modern distributional methods like Implicit Quantile Networks (IQN) or Quantile Regression DQN (QR-DQN) are currently missing.

---

## Policy Gradient & Actor-Critic

| Paper | PDF | Implementation | Location | Notes |
| :--- | :--- | :--- | :--- | :--- |
| **REINFORCE** (Williams, 1992) | [reinforce.pdf](reinforce.pdf) | Done | `examples/reinforce/reinforce_cartpole.py` | Classic Monte Carlo policy gradient. |
| **A3C / A2C** (Mnih et al., 2016) | [a3c.pdf](a3c.pdf) | Done (A2C) | `examples/actor_critic/a2c_cartpole.py` | Synchronous version (A2C) implemented for stability on modern GPUs. |
| **PPO** (Schulman et al., 2017) | [ppo.pdf](ppo.pdf) | Done | `examples/ppo/` | Comprehensive implementations for Atari, MuJoCo, and LSTM variants. |

### Missing / Future Work:
- **TRPO** (Schulman et al., 2015): Folder exists in `examples/trpo/` but is currently empty.
- **SAC** (Haarnoja et al., 2018): Folder exists in `examples/sac/` but is currently empty.

---

## MuZero / AlphaZero

| Paper | PDF | Implementation | Location | Notes |
| :--- | :--- | :--- | :--- | :--- |
| **AlphaZero** (Silver et al., 2017) | [alphazero.pdf](alphazero.pdf) | Partial | `papers/alphazero_pseudocode.py` | Pseudocode provided; full runnable implementation is pending. |
| **MuZero** (Schrittwieser et al., 2019) | [muzero.pdf](muzero.pdf) | Partial | `functional/mcts.py` / `experiments/rainbowzero/` | Core MCTS logic is implemented. Agent experiments exist in `experiments/`. |
| **EfficientZero** (Ye et al., 2021) | [efficient_zero.pdf](efficient_zero.pdf) | Partial | `experiments/rainbowzero/` | Self-supervised consistency losses and MCTS improvements. |

### Missing / Future Work:
- **Gumbel MuZero**: Planned for `functional/mcts.py` to allow policy improvement without high simulation counts.
- **Stochastic MuZero**: Needed for handling environments with inherent randomness (like Catan).
- **Sampled MuZero**: For handling large or continuous action spaces.

---

## Continual Learning & Meta-Optimization (The Alberta Plan)

This repo heavily focuses on the **Alberta Plan** (Sutton et al., 2022) and the building blocks of continual, online learning.

| Paper | PDF | Implementation | Location | Notes |
| :--- | :--- | :--- | :--- | :--- |
| **Alberta Plan** (Sutton et al., 2022) | [AlbertaPlan.pdf](AlbertaPlan.pdf) / [Report](AlbertaPlanReport.pdf) | Building Blocks | `functional/plasticity.py` | Serves as the philosophical guide for the `functional/` module. |
| **SWR** (Selective Weight Reinit) | [selective_weight_reinitialization.pdf](selective_weight_reinitialization.pdf) | Done | `functional/plasticity.py` | Utility-based reinitialization. |
| **IDBD / Autostep** | [idbd_a.pdf](idbd_a.pdf) / [autostep.pdf](autostep.pdf) | Done | `functional/meta_optimization.py` | Meta-gradient learning rates. |
| **AdaGain** (Jacobsen et al., 2019) | [adagain.pdf](adagain.pdf) | Notes Only | `functional/meta_optimization.py` | Currently exists as detailed architectural notes. |
| **Continual Backprop** | [continualbackprop.pdf](continualbackprop.pdf) | Missing | `functional/plasticity.py` | Needs a cleaner, generic rewrite. |

---

## Distributed & Offline RL

| Paper | PDF | Implementation | Location | Notes |
| :--- | :--- | :--- | :--- | :--- |
| **Ape-X** (Horgan et al., 2018) | [ape-x.pdf](ape-x.pdf) | Done | `examples/dqn/ape_x/` | Parallel actors with central prioritized buffer. |
| **MuZero Unplugged** (2021) | [muzero_unplugged.pdf](muzero_unplugged.pdf) | Partial | `experiments/catan/` | Data collection and offline training patterns. |
| **Batch MCTS** | [batch_mcts.pdf](batch_mcts.pdf) | Included | `functional/mcts.py` | Vectorized MCTS for efficient batched inference. |

---

## Fundamentals

Core mathematical foundations used across multiple papers:

| Topic | PDF | Paper Ref | Location |
| :--- | :--- | :--- | :--- | :--- |
| **N-Step Returns** | [td_learning.pdf](td_learning.pdf) | Sutton (1988) | `functional/returns.py` |
| **GAE** | (No PDF) | Schulman et al. (2016) | `functional/returns.py` |
| **Temporal Credit Assignment** | (No PDF) | Various | `functional/returns.py` |
| **Loss Functions** | (No PDF) | Various | `functional/losses.py` |
| **Replay Buffers** | (No PDF) | Various | `functional/replay_buffer.py` |
