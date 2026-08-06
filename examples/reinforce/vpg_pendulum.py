"""
Notes on Vanilla Policy Gradient for Pendulum (Continuous Action Space):

VPG (REINFORCE + Critic baseline) adapted for continuous action spaces.
The Actor outputs mean (mu) and has a learnable log standard deviation (log_std).
The Critic predicts state values to compute advantages.
NOTE: very similar to A2C/A3C
"""

from functional.initialization import layer_init
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import gymnasium as gym
from typing import Tuple
import numpy as np
import random
import wandb

from functional.action_selection import gaussian_sampling_selector
from functional.optimizer import apply_gradients
from functional.returns import compute_mc_returns
from functional.losses import policy_gradient_loss, mse_loss
from functional.utils import (
    standardize_tensor,
    to_tensor,
    to_numpy_action,
)
from functional.visualization import compute_explained_variance

# Constants
LEARNING_RATE = 1e-3
MAX_EPISODES = 2_000
GAMMA = 0.9
SEED = 42
CRITIC_COEFF = 0.5
MAX_ACTION = 2.0
HIDDEN_SIZE = 64
INITIAL_LOG_STD = 0.0

# Seeding for reproducibility
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)


class Actor(nn.Module):
    def __init__(self, input_shape: Tuple, num_actions: int):
        super().__init__()
        self.l1 = layer_init(nn.Linear(input_shape[0], HIDDEN_SIZE))
        self.l2 = layer_init(nn.Linear(HIDDEN_SIZE, HIDDEN_SIZE))
        self.mu_head = layer_init(nn.Linear(HIDDEN_SIZE, num_actions), std=0.01)
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
        x = torch.tanh(self.l1(x))  # [B, HIDDEN_SIZE]
        x = torch.tanh(self.l2(x))  # [B, HIDDEN_SIZE]
        mu = torch.tanh(self.mu_head(x)) * MAX_ACTION  # [B, num_actions]
        std = torch.exp(self.log_std).expand_as(mu)  # [B, num_actions]
        return mu, std


class Critic(nn.Module):
    def __init__(self, input_shape: Tuple):
        super().__init__()
        self.l1 = layer_init(nn.Linear(input_shape[0], HIDDEN_SIZE))
        self.l2 = layer_init(nn.Linear(HIDDEN_SIZE, HIDDEN_SIZE))
        self.l3 = layer_init(nn.Linear(HIDDEN_SIZE, 1), std=1.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for the Critic network.

        Args:
            x (torch.Tensor): The input observation tensor. # [B, obs_dim]

        Returns:
            torch.Tensor: The predicted state value. # [B, 1]
        """
        x = torch.tanh(self.l1(x))  # [B, HIDDEN_SIZE]
        x = torch.tanh(self.l2(x))  # [B, HIDDEN_SIZE]
        x = self.l3(x)  # [B, 1]
        return x


# --- 1. Initialization ---
env = gym.make("Pendulum-v1")
env = gym.wrappers.RecordEpisodeStatistics(env)
env = gym.wrappers.ClipAction(env)
obs_shape = env.observation_space.shape
num_actions = env.action_space.shape[0]
device = torch.device("cpu")

actor = Actor(obs_shape, num_actions).to(device)
critic = Critic(obs_shape).to(device)

optimizer = optim.Adam(
    list(actor.parameters()) + list(critic.parameters()), lr=LEARNING_RATE, eps=1e-5
)

obs, info = env.reset(seed=SEED)
terminated, truncated = False, False


# Initialize W&B
wandb.init(project="vpg-pendulum", config={"lr": LEARNING_RATE, "gamma": GAMMA})

for episode in range(MAX_EPISODES):
    rewards = []
    log_probs = []
    values = []
    terminateds = []
    truncateds = []

    while not (terminated or truncated):
        obs_tensor = to_tensor(obs[None, ...], device=device)
        mu, std = actor(obs_tensor)
        value = critic(obs_tensor)

        action_tensor, info_dict = gaussian_sampling_selector(mu, std, explore=True)
        action = to_numpy_action(action_tensor).flatten()  # remove batch dim of 1
        # Clip action to env bounds just in case, though mu is already scaled
        action = np.clip(action, -MAX_ACTION, MAX_ACTION)

        # 2. Step Env
        next_obs, reward, terminated, truncated, info = env.step(action)

        # 3. Add to buffers
        rewards.append(reward)
        log_probs.append(info_dict["log_prob"].sum(dim=-1))
        values.append(value)
        terminateds.append(terminated)
        truncateds.append(truncated)

        # Update state
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
    # 1. Compute Returns (Algorithm Agnostic)
    returns = compute_mc_returns(
        rewards=torch.tensor([rewards], dtype=torch.float32, device=device),
        terminated=torch.tensor([terminateds], dtype=torch.float32, device=device),
        truncated=torch.tensor([truncateds], dtype=torch.float32, device=device),
        gamma=GAMMA,
    )[0]

    values_tensor = torch.stack(values).flatten()

    # 2. Define Baseline & Calculate Advantage using the Critic
    raw_advantages = returns - values_tensor.detach()
    advantages = standardize_tensor(raw_advantages)

    pg_loss, info_dict = policy_gradient_loss(
        advantages=advantages,
        log_probs=torch.stack(log_probs).flatten(),
    )
    pg_loss = pg_loss.mean()

    critic_loss, _ = mse_loss(values_tensor, returns)
    critic_loss = critic_loss.mean()

    loss = pg_loss + CRITIC_COEFF * critic_loss

    # Apply Updates
    optimizer = apply_gradients(optimizer, loss)

    if episode % 100 == 0:
        explained_var = compute_explained_variance(
            returns.detach().cpu().numpy(), values_tensor.detach().cpu().numpy()
        )
        log_dict = info_dict.copy()
        log_dict.update(
            {
                "loss/total": loss.item(),
                "loss/critic": critic_loss.item(),
                "value/explained_variance": explained_var,
                "std": torch.exp(actor.log_std).item(),
            }
        )
        wandb.log(log_dict, step=episode)

wandb.finish()
