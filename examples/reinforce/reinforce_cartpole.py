"""
Notes on REINFORCE:

A foundational Algorithm in RL. The base for most policy based methods and actor critics methods.

Introduces policy gradient, simply put increase the probability of playing "good" actions. The hard part is figuring out what is a good action (credit assignment).

On-Policy Algorithm so it can't store old data. Simple places for inovation and changes, data efficiency, baseline methods/advantage computation, adding a value network, more efficient data collection, allowing for trajectories instead of only episodes, relying on Monte Carlo returns (high variance), allowing for offline data, among many others.
"""

from functional.initialization import layer_init, set_seed
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

from functional.action_selection import sample_distribution
from functional.optimizer import apply_gradients
from functional.returns import compute_mc_returns
from functional.losses import policy_gradient_loss
from functional.utils import (
    ema_update_,
    standardize_tensor,
    scale_tensor_by_std,
    to_tensor,
    to_numpy_action,
)

# Constants
LEARNING_RATE = 1e-3
MAX_EPISODES = 1_000
GAMMA = 0.99
SEED = 42

global_ema_baseline = 0.0  # if using ema for baseline

# Seeding for reproducibility
set_seed(SEED)


class Actor(nn.Module):
    def __init__(self, input_shape: Tuple, num_actions: int):
        super().__init__()
        self.l1 = layer_init(nn.Linear(input_shape[0], 64))
        self.l2 = layer_init(nn.Linear(64, 64))
        self.l3 = layer_init(nn.Linear(64, num_actions), std=0.01)

    def forward(self, x):
        x = F.relu(self.l1(x))
        x = F.relu(self.l2(x))
        x = self.l3(x)
        return x


# --- 1. Initialization (Defining the State) ---
env = gym.make("CartPole-v1")
env = gym.wrappers.RecordEpisodeStatistics(env)
obs_shape = env.observation_space.shape
num_actions = env.action_space.n
device = torch.device("cpu")

actor = Actor(obs_shape, num_actions).to(device)

optimizer = optim.Adam(actor.parameters(), lr=LEARNING_RATE)

obs, info = env.reset(seed=SEED)
terminated, truncated = False, False
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
    terminateds = []
    truncateds = []
    while not (terminated or truncated):
        obs_tensor = to_tensor(obs[None, ...], device=device)
        logits = actor(obs_tensor)
        dist = torch.distributions.Categorical(logits=logits)
        action, info_dict = sample_distribution(dist, explore=True)
        action_np = to_numpy_action(action)

        # 2. Step Env
        # Extract the scalar for a non-vectorized Gymnasium environment
        action_int = int(action_np.item())
        next_obs, reward, terminated, truncated, info = env.step(action_int)

        # 3. Add to "online" buffers
        rewards.append(reward)
        log_probs.append(info_dict["log_prob"].squeeze())
        terminateds.append(terminated)
        truncateds.append(truncated)

        # Update state for next tick
        obs = next_obs

    if terminated or truncated:
        if "episode" in info:
            wandb.log(
                {
                    "episode_return": info["episode"]["r"][0],
                    "episode_length": info["episode"]["l"][0],
                },
                step=episode,
            )
        obs, info = env.reset()
        terminated, truncated = False, False

    # --- 3. The Update Loop ---
    # NOTE: Again unlike PPO, we don't sample as a learning step always occurs at the end of an episode.

    # 1. Compute Returns (Algorithm Agnostic)
    returns = compute_mc_returns(
        rewards=torch.tensor([rewards], dtype=torch.float32, device=device),
        terminated=torch.tensor([terminateds], dtype=torch.float32, device=device),
        truncated=torch.tensor([truncateds], dtype=torch.float32, device=device),
        gamma=GAMMA,
    )[0]

    # 2. Define Baseline & Calculate Raw Advantage (Explicit Math)
    # METHOD A: Mean baseline
    # raw_advantages = returns - returns.mean()
    # advantages = standardize_tensor(raw_advantages)

    # METHOD B: EMA baseline (Standard for REINFORCE)
    global_ema_baseline = ema_update_(
        baseline=global_ema_baseline, value=returns.mean().item(), alpha=0.01
    )
    raw_advantages = returns - global_ema_baseline

    # 3. Optional Scaling
    advantages = scale_tensor_by_std(raw_advantages)

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
