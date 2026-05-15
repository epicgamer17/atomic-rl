remove shape bouncers using einops and move to asserts
remove shape manipulation in functions and move it to the imperative shells, functions should instead assert strict contracts. what shape should buffer keys have? like should rewards be B, 1 or B etc 

implement models and functions from https://coax.readthedocs.io/en/latest/index.html
look into possible coax like notation.

implement models and functions from RLAX 

figure out my naming convention for functions. sometimes i say compute_xyz and other times i just say xyz

potentially consider making functions for getting log probs for distributions (trouble is when doing multi discrete or multi continuous envs which require summing log probs), is this needed though? I handle this in the action selectors but do not handle it in the re-eval step cleanly at the moment. the user is expected to handle it instead.

Add examples on Atari for DQN and A2C
Add examples on Labyrinth (A3C Paper)
More testing/verification of implementations and examples (compared to 37 implementation details of PPO for example or APE-X)

Future models/examples: 
A2C + Trust Regions? (from PPO paper, what is this)
VPG (Adaptive)? (from PPO paper, what is this)
TRPO (from PPO paper, what is this)
MAPPO
...
R2D2
NGU
...
AlphaZero
Batch MCTS (different than vectorized MCTS)
MuZero (board game + atari)
MuZero Reanalyze
MuZero Unplugged
Sampled MuZero
Efficient Zero
Efficient Zero V2
Gumbel MuZero
VQ-VAE Paper (before Stochastic MuZero)
Stochastic MuZero
OptionZero
... 
Option Critic
... 
Sarsa
Soft Q Learning
... 
SAC
DDPG
... 
World Models Paper
Dreamer V1
Dreamer V2
Dreamer V3
Dreamer V4
... 
JEPA BASED? 
... 
Sutton Based Methods (linear value functions, average rewards, Dyna, etc)
- AdaGain (was too hard to implement? or maybe not, maybe I should just try again)
- MetaOptimize (also too hard? very optimizer specific, not a lot of freedom)
- Continual Back Prop (again very not friendly to general use, every new layer and architecture needs a new branch in the code, hence removed)
- Horde and GVF

ADD METRICS
PPO METRICS
Percent of Dead Units 
Weight Magnitude
Effective rank of representation layers

Experiments with Any RL agent with and without IDBD or variants and with or without CBP or SWR and with both. like an ablation of those. 
More generally examples that combine multiple ideas and methods from the Alberta Plan that I have made so far. Both on their own, on the stream problems, and on traditional RL problems (combined/enhancing the standard agents or frameworks). ie DQN with IDBD and SWR? or PPO with CBP and Autostep, or Stream DQN. Continue to explore and add as I add more Alberta Plan research and findings. 

Move data for permuted mnist out of examples folder and into a more general data folder.

IMPROVE MESSY LSTM CODE
IMPROVE MESSY EXAMPLES.

Consider bringing orchestration code back for TD losses like Q learning losses.