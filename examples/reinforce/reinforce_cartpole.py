"""
Notes on REINFORCE:

A foundational Algorithm in RL. The base for most policy based methods and actor critics methods.

Introduces policy gradient, simply put increase the probability of playing "good" actions. The hard part is figuring out what is a good action (credit assignment).

On-Policy Algorithm so it can't store old data. Simple places for inovation and changes, data efficiency, baseline methods/advantage computation, adding a value network, more efficient data collection, allowing for trajectories instead of only episodes, relying on Monte Carlo returns (high variance), allowing for offline data, among many others.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import gymnasium as gym
from typing import Tuple
import numpy as np
import random
import wandb
from functools import partial

from functional.action_selection import categorical_sampling_selector
from functional.optimizer import apply_gradients
from functional.returns import compute_mc_returns
from functional.losses import policy_gradient_loss
from functional.utils import exponential_moving_average

# Constants
LEARNING_RATE = 1e-3
MAX_EPISODES = 120_000
GAMMA = 0.99
SEED = 42

global_ema_baseline = 0.0  # if using ema for baseline

# Seeding for reproducibility
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)


class REINFORCE(nn.Module):
    def __init__(self, input_shape: Tuple, num_actions: int):
        super().__init__()
        self.l1 = nn.Linear(input_shape[0], 64)
        self.l2 = nn.Linear(64, 64)
        self.l3 = nn.Linear(64, num_actions)

    def forward(self, x):
        x = F.relu(self.l1(x))
        x = F.relu(self.l2(x))
        x = self.l3(x)
        return x


# --- 1. Initialization (Defining the State) ---
env = gym.make("CartPole-v1")
obs_shape = env.observation_space.shape
num_actions = env.action_space.n
device = torch.device("cpu")

model = REINFORCE(obs_shape, num_actions).to(device)

optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

obs, info = env.reset(seed=SEED)
terminated, truncated = False, False
stat_episode_return = 0.0
rng_key = torch.Generator(device=device)
rng_key.manual_seed(SEED)

# Initialize W&B
wandb.init(project="reinforce-cartpole", config={"lr": LEARNING_RATE, "gamma": GAMMA})

# Using full episodes
# NOTE: we use episode returns, but we could also use trajectories, etc
for episode in range(MAX_EPISODES):
    # NOTE: Since pure REINFORCE relys on MC returns, there is not a clean way to preallocate a buffer or use a circular buffer like we can do for PPO so we just use lists
    rewards = []
    log_probs = []
    while not (terminated or truncated):
        obs_tensor = torch.from_numpy(obs).float().unsqueeze(0).to(device)
        logits = model(obs_tensor)
        action = categorical_sampling_selector(logits, temperature=1.0)
        action = action.item()

        # 2. Step Env
        next_obs, reward, terminated, truncated, info = env.step(action)
        stat_episode_return += reward

        # 3. Add to "online" buffers
        log_probs.append(F.log_softmax(logits, dim=-1)[0, action])
        rewards.append(reward)

        # Update state for next tick
        obs = next_obs

    if terminated or truncated:
        wandb.log({"episode_return": stat_episode_return}, step=episode)
        obs, info = env.reset()
        terminated, truncated = False, False
        stat_episode_return = 0.0

    # --- 3. The Update Loop ---
    # NOTE: Again unlike PPO, we don't sample as a learning step always occurs at the end of an episode.

    # Calculate Loss & Gradients
    returns = compute_mc_returns(
        torch.tensor(rewards, dtype=torch.float32, device=device), GAMMA
    )

    # NOTE: optionally normalize returns

    # METHOD A: Mean baseline
    # if len(returns) > 1:
    #     advantages = (returns - returns.mean()) / (returns.std() + 1e-8)
    # else:
    #     advantages = 0.0
    # METHOD B: EMA baseline
    global_ema_baseline = exponential_moving_average(
        torch.tensor(global_ema_baseline, device=device), returns.mean(), alpha=0.01
    ).item()
    advantages = (returns - global_ema_baseline).detach()
    if len(advantages) > 1:
        advantages = advantages / (advantages.std() + 1e-8)
    else:
        advantages = torch.zeros_like(advantages)

    # Handle scaling
    # Others: learn a baseline with a neural network (advantage) (not done here)
    loss, info_dict = policy_gradient_loss(
        advantages=advantages,  # NOTE: calculate this outside
        log_probs=torch.stack(log_probs),
    )

    loss = loss.mean()

    # Apply Updates
    optimizer = apply_gradients(optimizer, loss)

    if episode % 100 == 0:
        # W&B handles scalars and histograms of tensors (like priorities) automatically.
        log_dict = info_dict.copy()
        log_dict.update({"loss": loss.item()})
        wandb.log(log_dict, step=episode)
