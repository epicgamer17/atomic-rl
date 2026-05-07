"""
Notes on REINFORCE for Pendulum (Continuous Action Space):

REINFORCE adapted for continuous action spaces using a Gaussian policy.
The Actor outputs the mean (mu) and has a learnable log standard deviation (log_std).
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
from einops import rearrange

from functional.action_selection import gaussian_sampling_selector
from functional.optimizer import apply_gradients
from functional.returns import compute_mc_returns
from functional.losses import policy_gradient_loss
from functional.utils import exponential_moving_average
from functional.advantages import compute_mean_advantages, compute_ema_advantages

# Constants
LEARNING_RATE = 1e-3
MAX_EPISODES = 2_000
GAMMA = 0.99
SEED = 42
MAX_ACTION = 2.0
HIDDEN_SIZE = 64
INITIAL_LOG_STD = -0.5

# Seeding for reproducibility
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)


class Actor(nn.Module):
    def __init__(self, input_shape: Tuple, num_actions: int):
        super().__init__()
        self.l1 = nn.Linear(input_shape[0], HIDDEN_SIZE)
        self.l2 = nn.Linear(HIDDEN_SIZE, HIDDEN_SIZE)
        self.mu_head = nn.Linear(HIDDEN_SIZE, num_actions)
        # Learnable parameter for log_std, independent of state
        self.log_std = nn.Parameter(torch.full((1, num_actions), INITIAL_LOG_STD))

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass for the Actor network.

        Args:
            x (torch.Tensor): The input observation tensor. # [B, obs_dim]

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: The mean (mu) and standard deviation (std)
                of the Gaussian action distribution. # [B, num_actions], [B, num_actions]
        """
        x = F.relu(self.l1(x))  # [B, HIDDEN_SIZE]
        x = F.relu(self.l2(x))  # [B, HIDDEN_SIZE]
        mu = torch.tanh(self.mu_head(x)) * MAX_ACTION  # [B, num_actions]

        # Expand log_std to match batch size
        std = torch.exp(self.log_std).expand_as(mu)  # [B, num_actions]
        return mu, std


# --- 1. Initialization (Defining the State) ---
env = gym.make("Pendulum-v1")
obs_shape = env.observation_space.shape
num_actions = env.action_space.shape[0]
device = torch.device("cpu")

actor = Actor(obs_shape, num_actions).to(device)
optimizer = optim.Adam(actor.parameters(), lr=LEARNING_RATE)

obs, info = env.reset(seed=SEED)
terminated, truncated = False, False
stat_episode_return = 0.0
global_ema_baseline = 0.0

# Initialize W&B
wandb.init(project="reinforce-pendulum", config={"lr": LEARNING_RATE, "gamma": GAMMA})

for episode in range(MAX_EPISODES):
    rewards = []
    log_probs = []
    terminals = []

    while not (terminated or truncated):
        obs_tensor = torch.as_tensor(obs[None, ...], dtype=torch.float32, device=device)
        mu, std = actor(obs_tensor)

        # Sample action using Gaussian selector
        action_tensor, log_prob = gaussian_sampling_selector(mu, std, explore=True)

        # Pendulum expects a numpy array for actions
        action = action_tensor.detach().cpu().numpy().flatten()
        # Clip action to env bounds just in case, though mu is already scaled
        action = np.clip(action, -MAX_ACTION, MAX_ACTION)

        # 2. Step Env
        next_obs, reward, terminated, truncated, info = env.step(action)
        stat_episode_return += reward

        # 3. Add to buffers
        rewards.append(reward)
        log_probs.append(log_prob)
        terminals.append(terminated)

        # Update state
        obs = next_obs

    if terminated or truncated:
        wandb.log({"episode_return": stat_episode_return}, step=episode)
        obs, info = env.reset()
        terminated, truncated = False, False
        stat_episode_return = 0.0

    # --- 3. The Update Loop ---
    returns = compute_mc_returns(
        torch.tensor(rewards, dtype=torch.float32, device=device).unsqueeze(0),
        torch.tensor(terminals, dtype=torch.float32, device=device).unsqueeze(0),
        GAMMA,
    ).squeeze(0) # TODO: replace with rearrange

    # Use EMA baseline for variance reduction
    global_ema_baseline = exponential_moving_average(
        torch.tensor(global_ema_baseline, device=device), returns.mean(), alpha=0.01
    ).item()
    advantages = compute_ema_advantages(returns, global_ema_baseline).detach()

    loss, info_dict = policy_gradient_loss(
        advantages=advantages,
        log_probs=rearrange(torch.stack(log_probs), 't 1 1 -> t'),
    )
    loss = loss.mean()

    # Apply Updates
    optimizer = apply_gradients(optimizer, loss)

    if episode % 100 == 0:
        log_dict = info_dict.copy()
        log_dict.update({"loss": loss.item(), "std": torch.exp(actor.log_std).item()})
        wandb.log(log_dict, step=episode)
