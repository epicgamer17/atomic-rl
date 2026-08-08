# RL Papers & Implementations

This directory maintains a mapping between the research papers in this folder and their corresponding implementations within the repository.

### Missing / Future Work:
- **Distributional Rainbow (IQN/QR-DQN)**: While C51 is implemented, modern distributional methods like Implicit Quantile Networks (IQN) or Quantile Regression DQN (QR-DQN) are currently missing.
- **R2D2**: [r2d2.pdf](r2d2.pdf). While the core building blocks (recurrent unrolling, burn-in) are available in `atomic_rl/network.py` and DRQN is implemented, a full R2D2 agent with prioritized sequences and overlapping burn-in is still pending.
- **NGU / Agent 57**: Never Give Up and Agent 57.
- **IMPALA**: High-throughput distributed AC.
- **TRPO** (Schulman et al., 2015): Folder exists in `examples/trpo/` but is currently empty.
- **SAC** (Haarnoja et al., 2018): Folder exists in `examples/sac/` but is currently empty.
- **Batch MCTS**: [batch_mcts.pdf](batch_mcts.pdf). Vectorized MCTS for efficient batched inference. Different than vectorized envs, about searching mutliple branches at once for one state. 
- **Gumbel MuZero**: [gumbel_muzero.pdf](gumbel_muzero.pdf). Planned for `atomic_rl/mcts.py` to allow policy improvement without high simulation counts.
- **Stochastic MuZero**: [stochastic_muzero.pdf](stochastic_muzero.pdf). Needed for handling environments with inherent randomness (like Catan).
- **EfficientZero**: [efficientzero.pdf](efficientzero.pdf). Sample efficient offline MuZero variant. Possible improvements on MuZero.
- **Efficient Zero V2**: [efficientzero_v2.pdf](efficientzero_v2.pdf). V2 of EfficientZero. 
- **MuZero Unplugged**: [muzero_unplugged.pdf](muzero_unplugged.pdf). Offline MuZero.
- **Sampled MuZero**: [sampled_muzero.pdf](sampled_muzero.pdf). MuZero for Continuous and complex action spaces.
- **SkyNet**: MuZero on partially observable information and multi agent uncertainty 
- **ROSMO**: [rosmo.pdf](rosmo.pdf). better muzero unplugged (for stochastic games I think)
- **Muesli**: Combining improvements in policy gradient methods.
- **AdaGain**: [adagain.pdf](adagain.pdf). An improvement on AutoStep (I think).
- **MetaOptimize**: [metaoptimize.pdf](metaoptimize.pdf).
- **ReDO**
- **The Primacy Bias in Deep RL**: Full periodic resets of weights, same replay buffer. May be worth recreating to see if CBP and SWR are better or worse. In some ways more of a good read than something to implement. (Maybe)
- **Developing a predictive approach to knowledge** (LONG)
- **Horde**: [AlbertaPlan.pdf](AlbertaPlan.pdf) related.
- **NFSP**: Implementation exists in `experiments/rainbow-nfsp/` but is not yet a standardized example.
- **Related Work From Stream RL Paper**: TODO: go through the related work section and add them to the list here. Then implement them. They discuss relevant IDBD methods, loss of plasticity methods, sparse representation methods, other stream RL related papers, etc. All useful to stream RL and the Alberta Plan.
- **Tuning Free Step Size Adaptation**: introduces TIDBD
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
- **Stream Deep RL Finally Works** (Mohamed Elsayed 2024): [stream_rl.pdf](stream_rl.pdf). Implemented in `examples/stream_rl/` and `atomic_rl/optimizer.py`.
- **Emphatic TD**
- **Temporal Abstraction in TD Networks** 
- **Between MDPs and Semi-MDPs: Learning, Planning, and Representing Knowledge at Multiple Temporal Scales** (Maybe read more) (LONG)
- **What's a good prediction? Challenges in evaluating an agent's knowledge** 
- **Learning Agent State Online with Recurrent Generate-and-Test**
- **Scalable Real-Time Recurrent Learning Using Columnar-Constructive Networks**
- **From eye-blinks to state construction: Diagnostic benchmarks for online representation learning**
- **Toward Generate-and-Test Algorithms for Continual Feature Discovery**
- **SwiftTD** 
- **Metatrace actor-critic: Online step-size tuning by meta-gradient descent for reinforcement learning control** (AC(lambda) variant), an incremental version of AC with meta optimization
- **Implementation Details of the TD( λ ) Procedure for the Case of Vector Predictions and Backpropagation** NonLinear TD/Backprop.
- **Reinforcement Learning an Introduction Second Edition**
    - Section 1.1 to 1.7: Introduction
    Tabular Methods
    - Section 2.1 to 2.10: Multi-armed Bandits
    - Section 3.1 to 3.8: Finite MDPs
    - Section 4.1 to 4.8: Dynamic Programming
    - Section 5.1 to 5.10: Monte Carlo Methods
    - Section 6.1 to 6.9: TD Learning
    - Section 7.1 to 7.6: N-step Bootstrapping
    - Section 8.1 to 8.13: Planning and Learning with Tabular Methods 
    Approximate Solution Methods
    - Section 9.1 to 9.12: On-policy Prediction with Approximation 
    - Section 10.1 to 10.6: On-policy Control with Approximation
    - Section 11.1 to 11.10: *Off-policy Methods with Approximation
    - Section 12.1 to 12.13: Eligibility Traces
    - Section 13.1 to 13.8: Policy Gradient Methods
    
    Skipped the Neuro and Psych

    - Section 16.1 to 16.8: Applications and Case-Studies
    - Section 17.1 to 17.6: Frontiers 

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
- Online Normalization like alberta plan ( $\tilde{x}_{t}^{i}\doteq\frac{x_{t}^{i}-\mu_{t}^{i}}{\sigma_{t}^{i}}$) $\mu_t^i$ and $\sigma_t^i$ must be tracking estimates—like an Exponential Moving Average (EMA)—that heavily discount the distant past so the normalization adapts quickly to new distributions

---

## DQN & Extensions

| Paper | PDF | Implementation | Location | Notes |
| :--- | :--- | :--- | :--- | :--- |
| **DQN** (Mnih et al., 2013/2015) | [dqn.pdf](dqn.pdf) | Done | `examples/dqn/dqn_cartpole.py` | Basic DQN with experience replay and target networks. |
| **Double DQN** (Van Hasselt et al., 2015) | [double_dqn.pdf](double_dqn.pdf) | Done | `examples/dqn/ddqn_cartpole.py` | Implements decoupled action selection and evaluation to reduce bias. |
| **Dueling DQN** (Wang et al., 2015) | [dueling_dqn.pdf](dueling_dqn.pdf) | Done | `examples/dqn/dueling_dqn_cartpole.py` | Uses separate Value and Advantage streams. |
| **Prioritized Experience Replay** (Schaul et al., 2015) | [per.pdf](per.pdf) | Done | `examples/dqn/prioritized_replay_dqn_cartpole.py` | Backed by a high-performance SumTree in `atomic_rl/replay_buffer.py`. |
| **Categorical DQN (C51)** (Bellemare et al., 2017) | [categorical_dqn.pdf](categorical_dqn.pdf) | Done | `examples/dqn/categorical_dqn_cartpole.py` | Implements distributional RL by predicting a discrete value distribution. |
| **Noisy DQN** (Fortunato et al., 2017) | [noisy_dqn.pdf](noisy_dqn.pdf) | Done | `examples/dqn/noisy_dqn_cartpole.py` | Replaces epsilon-greedy with learnable noise in Linear layers. |
| **N-Step DQN** | [td_learning.pdf](td_learning.pdf) | Done | `examples/dqn/n_step_dqn_cartpole.py` | Uses n-step returns to bootstrap future rewards. |
| **DRQN** (Hausknecht & Stone, 2015) | [drqn.pdf](drqn.pdf) | Done | `examples/dqn/drqn_cartpole.py` | Recurrent DQN using LSTM for POMDP environments. |
| **Ape-X** (Horgan et al., 2018) | [ape-x.pdf](ape-x.pdf) | Done | `examples/dqn/ape_x/` | Distributed prioritized experience replay using Ray. |
| **Rainbow** (Hessel et al., 2017) | [rainbow_dqn.pdf](rainbow_dqn.pdf) | Done | `examples/dqn/rainbow_dqn_cartpole.py` | Combines all the above extensions into a single agent. |
| **Revisiting Rainbow** (Obando-Ceron et al., 2020) | [revisiting_rainbow.pdf](revisiting_rainbow.pdf) | Done | `examples/dqn/rainbow_dqn_cartpole.py` | Smaller-scale Rainbow experiments. |

---

## Policy Gradient & Actor-Critic

| Paper | PDF | Implementation | Location | Notes |
| :--- | :--- | :--- | :--- | :--- |
| **REINFORCE** (Williams, 1992) | [reinforce.pdf](reinforce.pdf) | Done | `examples/reinforce/reinforce_cartpole.py` | Classic Monte Carlo policy gradient. |
| **VPG** (Vanilla Policy Gradient) | [reinforce.pdf](reinforce.pdf) | Done | `examples/reinforce/vpg_cartpole.py` | REINFORCE + learned value baseline for advantages. Also in Pendulum and MuJoCo. |
| **A3C / A2C** (Mnih et al., 2016) | [a3c.pdf](a3c.pdf) | Done (A2C) | `examples/actor_critic/` | Synchronous A2C for CartPole, Pendulum, and MuJoCo. Better for GPUs |
| **PPO** (Schulman et al., 2017) | [ppo.pdf](ppo.pdf) | Done | `examples/ppo/` | Comprehensive implementations for Atari, MuJoCo, Pendulum (continuous), and MultiDiscrete. |
| **Recurrent PPO** (PPO + LSTM) | [ppo.pdf](ppo.pdf) | Mostly Done | `examples/ppo/ppo_lstm_cartpole.py`, `ppo_lstm_atari.py` | PPO with recurrent cells and unrolling for CartPole and Atari. Some details needed (e.g. LSTM state handling on truncation). |

---

## MuZero & AlphaZero

| Paper | PDF | Implementation | Location | Notes |
| :--- | :--- | :--- | :--- | :--- |
| **AlphaZero** (Silver et al., 2017) | [alphazero.pdf](alphazero.pdf) | Done | `examples/alphazero/alphazero_tictactoe.py`, `atomic_rl/mcts.py` | Vectorized MCTS with PUCT action selection, Dirichlet noise, and policy-value head joint learning. |
| **MuZero** (Schrittwieser et al., 2019) | [muzero.pdf](muzero.pdf) | Done | `examples/muzero/muzero_tictactoe.py`, `examples/muzero/muzero_cartpole.py`, `atomic_rl/mcts.py` | Model-based RL using learned representation, dynamics, and prediction networks with recurrent MCTS unrolling. Work in progress at the moment, some details pending. |

---

## Stream RL, Continual Learning & Meta-Optimization (The Alberta Plan)

This repository heavily focuses on the **Alberta Plan** (Sutton et al., 2022) and streaming, non-stationary online reinforcement learning.

| Paper | PDF | Implementation | Location | Notes |
| :--- | :--- | :--- | :--- | :--- |
| **Alberta Plan** (Sutton et al., 2022) | [AlbertaPlan.pdf](AlbertaPlan.pdf) / [Report](AlbertaPlanReport.pdf) | Building Blocks | `atomic_rl/plasticity.py` | Serves as the philosophical guide for the `atomic_rl/` module. |
Step 1: ___ ?
| **Stream Deep RL** (Elsayed et al., 2024) | [stream_rl.pdf](stream_rl.pdf) | Done | `examples/stream_rl/`, `atomic_rl/optimizer.py` | Implements Stream AC(λ), Stream Q(λ), and Early Temporal Termination (ETT) with Adaptive ObGD optimizers. |
| **IDBD / Autostep** | [idbd_a.pdf](idbd_a.pdf) / [autostep.pdf](autostep.pdf) | Done | `atomic_rl/meta_optimization.py` | Meta-gradient learning rates. Reproduced in `examples/meta_optimization/`. |
| **K1 / K2 Algorithms** | [idbd_b.pdf](idbd_b.pdf) | Done | `atomic_rl/meta_optimization.py` | O(n) approximations of the Kalman Filter for adaptive step-sizes. |
Step 2: ___ ?
| **Continual Backprop** | [cbp_1.pdf](cbp_1.pdf) / [cbp_2.pdf](cbp_2.pdf) | Done | `atomic_rl/plasticity.py` | Implements the generate-and-test plasticity mechanism. Reproduced in `examples/plasticity/`. |
| **SWR** (Selective Weight Reinit) | [selective_weight_reinit.pdf](selective_weight_reinit.pdf) | Done | `atomic_rl/plasticity.py` | Utility-based reinitialization. Reproduced in `examples/plasticity/`. |
| **Alberta Plan Integration** | [AlbertaPlan.pdf](AlbertaPlan.pdf) | In Progress | `examples/alberta_plan/ablation_study.py` | Ablation study combining NN backbone + CBP + AutoStep/IDBD + True Online TD on a drifting random walk. |

---

## Temporal Difference Learning & Gradient TD

Core temporal credit assignment and gradient-based TD methods for linear and function approximation settings.

| Paper | PDF | Implementation | Location | Notes |
| :--- | :--- | :--- | :--- | :--- |
| **TD Learning** (Sutton, 1988) | [td_learning.pdf](td_learning.pdf) | Done | `atomic_rl/td.py` | Introduced TD(lambda) and eligibility traces. Reproduced in `examples/td_learning/`. |
| **True Online TD(lambda)** (Sutton & van Seijen, 2014) | [true_online_td.pdf](true_online_td.pdf) | Done | `atomic_rl/td.py`, `atomic_rl/traces.py` | True Online TD update with Dutch traces. Reproduced in `examples/td_learning/td_learning_true_online_random_walk.py`. |
| **GTD(0)** (Sutton et al., 2009) | [gtd.pdf](gtd.pdf) | Done | `atomic_rl/td.py` | Gradient TD method for off-policy learning stability. Not GTD2, there is no correction. |
| **Fast-GTD / TDC** (Sutton et al., 2009) | [fast_gtd.pdf](fast_gtd.pdf) | Done | `atomic_rl/td.py` | Temporal Difference with Gradient Correction (TDC). GTD2 is not implemented. |

---

## Distributed & Offline RL

| Paper | PDF | Implementation | Location | Notes |
| :--- | :--- | :--- | :--- | :--- |
| **Ape-X** (Horgan et al., 2018) | [ape-x.pdf](ape-x.pdf) | Done | `examples/dqn/ape_x/` | Parallel actors with central prioritized buffer using Ray. |

---

## Credits & Core Utilities

| Topic | PDF | Paper Ref | Location |
| :--- | :--- | :--- | :--- |
| **N-Step Returns** | [td_learning.pdf](td_learning.pdf) | Sutton (1988) | `atomic_rl/returns.py` |
| **GAE** | - | Schulman et al. (2016) | `atomic_rl/returns.py` |
| **TD(lambda) Returns** | [td_learning.pdf](td_learning.pdf) | Sutton (1988) | `atomic_rl/returns.py` |
| **Eligibility Traces** | [td_learning.pdf](td_learning.pdf) | Various | `atomic_rl/traces.py` |
| **True Online Traces** | [true_online_td.pdf](true_online_td.pdf) | Sutton & van Seijen (2014) | `atomic_rl/traces.py` |
| **Loss Functions** | - | Various | `atomic_rl/losses.py` |
| **Replay Buffers** | - | Various | `atomic_rl/replay_buffer.py` |
| **Optimizers (ObGD)** | [stream_rl.pdf](stream_rl.pdf) | Elsayed et al. (2024) | `atomic_rl/optimizer.py` |

---

### PRUNED (Not planned but were before)
- **MAML** - I think its episodic so not very applicable to Alberta Plan. Trying to make my Meta Learning many Alberta Plan based 
- **ANIL** - I think its episodic so not very applicable to Alberta Plan. Trying to make my Meta Learning many Alberta Plan based 
- **GoExplore** - Requires saving simulator states and teleporting back to them to explore. Again against Alberta Plan and my general philosophy.
- **Off-Policy Temporal-Difference Learning with Function Approximation** - Not technically implemented. Generally speaking n step importance sampling not used, and only 1 step is used along with TDC which is what we have.