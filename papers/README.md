# RL Papers & Implementations

This directory maintains a mapping between the research papers in this folder and their corresponding implementations within the repository.

### Missing / Future Work:
- **Distributional Rainbow (IQN/QR-DQN)**: While C51 is implemented, modern distributional methods like Implicit Quantile Networks (IQN) or Quantile Regression DQN (QR-DQN) are currently missing.
- **DRQN / R2D2**: Recurrent DQN and R2D2. Core building blocks (recurrent unrolling, burn-in) are available in `functional/network.py`.
- **NGU / Agent 57**: Never Give Up and Agent 57.
- **IMPALA**: High-throughput distributed AC.
- **TRPO** (Schulman et al., 2015): Folder exists in `examples/trpo/` but is currently empty.
- **SAC** (Haarnoja et al., 2018): Folder exists in `examples/sac/` but is currently empty.
- **Batch MCTS**: [batch_mcts.pdf](batch_mcts.pdf). Vectorized MCTS for efficient batched inference. Different than vectorized envs, about searching mutliple branches at once for one state. 
- **Gumbel MuZero**: [gumbel_muzero.pdf](gumbel_muzero.pdf). Planned for `functional/mcts.py` to allow policy improvement without high simulation counts.
- **Stochastic MuZero**: [stochastic_muzero.pdf](stochastic_muzero.pdf). Needed for handling environments with inherent randomness (like Catan).
- **Sampled MuZero**: [sampled_muzero.pdf](sampled_muzero.pdf). For handling large or continuous action spaces.
- **EfficientZero**: [efficient_zero.pdf](efficient_zero.pdf). Sample efficient offline MuZero variant. Possible improvements on MuZero.
- **Efficient Zero V2**: [efficient_zero_v2.pdf](efficient_zero_v2.pdf). V2 of EfficientZero. 
- **MuZero Unplugged**: [muzero_unplugged.pdf](muzero_unplugged.pdf). Offline MuZero.
- **Sampled MuZero ** [sampled_muzero.pdf](sampled_muzero.pdf). MuZero for Continuous and complex action spaces.
- **AdaGain**: [adagain.pdf](adagain.pdf). An improvement on AutoStep (I think).
- **MetaOptimize**: [metaoptimize.pdf](metaoptimize.pdf).
- **ReDO**
- **The Primacy Bias in Deep RL**: Full periodic resets of weights, same replay buffer. May be worth recreating to see if CBP and SWR are better or worse. In some ways more of a good read than something to implement. (Maybe)
- **Horde** 
- **Developing a Predictive Approach to Knowledge** (important read)
- **Fast Gradient-Descent Methods for Temporal-Difference Learning with Linear Function Approximation** also see GTD/GTD2/TDC papers
- **True Online TD(lambda)** 
- **Tuning Free Step Size Adaptation**: introduces TIDBD (do i need this?)
- **NFSP** this largely goes against Alberta Plan as it requires an approximation of an INFINITE replay buffer. (Maybe)
- **JEPA** (Maybe)
- **GNN** 
- **ResNet**
- **Transformers** 
- **Vision Transformer** 
- **UniZero** 
- **OptionZero** 
- **Dyna**
- **World Models** 
- **Dreamer V1**
- **Dreamer V2** 
- **Dreamer V3** 
- **Dreamer V4** 
- **Flow Zero** (Maybe, read more)
- **Stochastic Gumbel MuZero**
- **OptionCritic**
- **GW-PCZero**  (Maybe read more)
- **ReZero** (Maybe read more)
- **Reinforcement Learning with Unsupervised Auxiliary Tasks**
- **deep learning fast and slow** (important read, maybe not implementation though)
- **AlphaStar** 
- **fractal MCTS**
- **Dynamics-Aware Unsupervised Discovery of Skills**
- **ROSMO** (Maybe)
- **ReBel** (Maybe)
- **CFR** (Maybe)
- **Learning World Graphs to Accelerate Hierarchical Reinforcement Learning** (Maybe)
- **I2A PAPER (Imagination-Augmented Agents for Deep Reinforcement Learning)** 
- **MAPPO** Multi Agent PPO 
- **Sarsa**
- **Soft Q Learning**
- **DDPG**
- **New Activations** GLU based and Squared ReLU, Dead ReLU fixes (is this meta learning related?)
- **Prenorm Dilution Fixes** - Attention Residuals and Full Attention Residuals and Block Attention Residuals
- **Attention is All You Need** Is this compatible with the Alberta Plan? (read for context of transformers)
- **Decision Transformers** and any other ways to use transformers in RL. Ideally without having to pass in full state history (like only s_t not s_1:t).
- **DenseNet** is this even good?
- **DenseFormer** is this even good?
- **DeepCrossAttention** is this even good?
- **Dino-v3** JEPA related?
- **V-JEPA**
- **Hyper Connections** with doubly stochastic matrices (The DeepSeek version of Hyper Connections) (is this meta learning related?)
- **POET**
- **Enhanced POET** 
- **CURL - Contrastive Unsupervised Representations for RL**
- **Datasets for Data-Driven Reinforcement Learning** 
- **Divide and Conquer Monte Carlo Tree Search for Goal Directed Planning**
- **Plan2Explore** 
- **Player of Games**





### Missing Features (TODO Find Papers):
For Agent 57, we are missing:
1. GRU (end of R2D2) 
2. Memory Networks 
3. Neural Episodic Control 
4. Transformers
6. Curiosity 
7. Intrinsic Motivation 
8. Density Models 
9. Hashing 
10. Random Network Distillation 
11. CoEx (what is this?) 
12. Reachability (end of Never Give Up) 
13. PBT 
14. Bandits 
15. Meta Gradients 
16. Adaptive Bandits (end of Agent 57)
17. Change Rainbow to predict Q values given s and a as input instead of value over all a given s (better for search maybe? not sure?)

- Successor Representations/Successor Features 
- Options 
- Changing network feature optimization (meta task)
- Learned Search
- DeepStack
- POMCP
- IS-MCTS 
- ABR
- Growing networks 
- MORE ON OPTIONS!
- V-Trace
- UPGO
- Hindsight Experience Replay (Against Alberta Plan)

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
| **Revisiting Rainbow** (Obando-Ceron et al., 2020) | [revisiting_rainbow.pdf](revisiting_rainbow.pdf) | Mostly a subset of Rainbow DQN | `examples/rainbow_dqn_cartpole/` | Smaller-scale Rainbow experiments. |

---

## Policy Gradient & Actor-Critic

| Paper | PDF | Implementation | Location | Notes |
| :--- | :--- | :--- | :--- | :--- |
| **REINFORCE** (Williams, 1992) | [reinforce.pdf](reinforce.pdf) | Done | `examples/reinforce/reinforce_cartpole.py` | Classic Monte Carlo policy gradient. |
| **A3C / A2C** (Mnih et al., 2016) | [a3c.pdf](a3c.pdf) | Done (A2C) | `examples/actor_critic/a2c_cartpole.py` | Synchronous version (A2C) implemented for stability on modern GPUs. |
| **PPO** (Schulman et al., 2017) | [ppo.pdf](ppo.pdf) | Done | `examples/ppo/` | Comprehensive implementations for Atari, MuJoCo, and MultiDiscrete. |
| **Recurrent PPO** (PPO + LSTM) | [ppo.pdf](ppo.pdf) | Mostly Done | `examples/ppo/ppo_lstm_cartpole.py` | Implementation of PPO with recurrent cells and unrolling. Some details needed (e.g. handling truncation, and more verification/testing perhaps) |

---

## MuZero / AlphaZero

| Paper | PDF | Implementation | Location | Notes |
| :--- | :--- | :--- | :--- | :--- |
| **AlphaZero** (Silver et al., 2017) | [alphazero.pdf](alphazero.pdf) | Partial | `papers/alphazero_pseudocode.py` | Pseudocode provided; full runnable implementation is pending. |
| **MuZero** (Schrittwieser et al., 2019) | [muzero.pdf](muzero.pdf) | Partial | `functional/mcts.py` | Core MCTS logic is implemented and vectorized. |

---

## Continual Learning & Meta-Optimization (The Alberta Plan)

This repo heavily focuses on the **Alberta Plan** (Sutton et al., 2022) and the building blocks of continual, online learning.

| Paper | PDF | Implementation | Location | Notes |
| :--- | :--- | :--- | :--- | :--- |
| **Alberta Plan** (Sutton et al., 2022) | [AlbertaPlan.pdf](AlbertaPlan.pdf) / [Report](AlbertaPlanReport.pdf) | Building Blocks | `functional/plasticity.py` | Serves as the philosophical guide for the `functional/` module. |
| **SWR** (Selective Weight Reinit) | [selective_weight_reinit.pdf](selective_weight_reinit.pdf) | Done | `functional/plasticity.py` | Utility-based reinitialization. Reproduced in `examples/plasticity/swr_permuted_mnist.py`. |
| **IDBD / Autostep** | [idbd_a.pdf](idbd_a.pdf) / [autostep.pdf](autostep.pdf) | Done | `functional/meta_optimization.py` | Meta-gradient learning rates. Reproduced in `examples/meta_optimization/`. |
| **K1 / K2 Algorithms** | [idbd_b.pdf](idbd_b.pdf) | Done | `functional/meta_optimization.py` | O(n) approximations of the Kalman Filter for adaptive step-sizes. |
| **Continual Backprop** | [cbp_1.pdf](cbp_1.pdf) / [cbp_2.pdf](cbp_2.pdf) | Done | `functional/plasticity.py` | Implements the generate-and-test plasticity mechanism. Reproduced in `examples/plasticity/cbp_permuted_mnist.py`. |



---

## Distributed & Offline RL

| Paper | PDF | Implementation | Location | Notes |
| :--- | :--- | :--- | :--- | :--- |
| **Ape-X** (Horgan et al., 2018) | [ape-x.pdf](ape-x.pdf) | Done | `examples/dqn/ape_x/` | Parallel actors with central prioritized buffer. |
| **MuZero Unplugged** (2021) | [muzero_unplugged.pdf](muzero_unplugged.pdf) | Partial | `experiments/catan/` | Data collection and offline training patterns. |
| **Batch MCTS** | [batch_mcts.pdf](batch_mcts.pdf) | Included | `functional/mcts.py` | Vectorized MCTS for efficient batched inference. |

---

## Other Important Papers (to be organized)

| Topic | PDF | Paper Ref | Location |
| :--- | :--- | :--- | :--- |
| **N-Step Returns** | [td_learning.pdf](td_learning.pdf) | Sutton (1988) | `functional/returns.py` |
| **GAE** | (No PDF) | Schulman et al. (2016) | `functional/returns.py` |
| **Temporal Credit Assignment** | (No PDF) | Various | `functional/returns.py` |
| **Loss Functions** | (No PDF) | Various | `functional/losses.py` |
| **Replay Buffers** | (No PDF) | Various | `functional/replay_buffer.py` |
| **TD learning** | [td_learning.pdf](td_learning.pdf) | Sutton (1988) | `functional/returns.py` |



### PRUNED (Not planned but were before)
- **MAML** - I think its episodic so not very applicable to Alberta Plan. Trying to make my Meta Learning many Alberta Plan based 
- **ANIL** - I think its episodic so not very applicable to Alberta Plan. Trying to make my Meta Learning many Alberta Plan based 
- **GoExplore** - Requires saving simulator states and teleporting back to them to explore. Again against Alberta Plan and my general philosophy.
