"""
Notes on Vanilla Policy Gradient:

Essentially REINFORCE + a value function for the baseline to compute advantages. Or another way of putting it is REINFORCE with learned state dependant baseline.
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
from functools import partial

from functional.action_selection import sample_distribution
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
MAX_EPISODES = 1_000
GAMMA = 0.99
SEED = 42
CRITIC_COEFF = 0.5

# Seeding for reproducibility
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

# NOTE: these can have a fused backbone and separate heads, for now we keep separate for simplicity


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


class Critic(nn.Module):
    def __init__(self, input_shape: Tuple):
        super().__init__()
        self.l1 = layer_init(nn.Linear(input_shape[0], 64))
        self.l2 = layer_init(nn.Linear(64, 64))
        self.l3 = layer_init(nn.Linear(64, 1), std=1.0)

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
critic = Critic(obs_shape).to(device)

# NOTE: we use a shared optimizer for both, but we could use separate ones
optimizer = optim.Adam(
    list(actor.parameters()) + list(critic.parameters()), lr=LEARNING_RATE
)

obs, info = env.reset(seed=SEED)
terminated, truncated = False, False
rng_key = torch.Generator(device=device)
rng_key.manual_seed(SEED)


# Initialize W&B
wandb.init(project="vpg-cartpole", config={"lr": LEARNING_RATE, "gamma": GAMMA})

# Using full episodes
# NOTE: we use episode returns, but we could also use trajectories, etc
for episode in range(MAX_EPISODES):
    # NOTE: Since pure Vanilla Policy Gradient relies on MC returns, there is not a clean way to preallocate a buffer or use a circular buffer like we can do for PPO so we just use lists
    rewards = []
    log_probs = []
    values = []
    terminateds = []
    truncateds = []
    while not (terminated or truncated):
        obs_tensor = to_tensor(obs[None, ...], device=device)
        logits = actor(obs_tensor)
        value = critic(obs_tensor)
        dist = torch.distributions.Categorical(logits=logits)
        action, info_dict = sample_distribution(dist, explore=True)
        action_np = to_numpy_action(action)

        # 2. Step Env
        # Extract the scalar for a non-vectorized Gymnasium environment
        action_int = int(action_np.item())
        next_obs, reward, terminated, truncated, info = env.step(action_int)

        # 3. Add to "online" buffers
        rewards.append(reward)
        log_probs.append(info_dict["log_prob"])
        values.append(value)
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

    values = torch.stack(values).flatten()

    # 2. Define Baseline & Calculate Advantage using the Critic
    # VPG traditionally uses the learned value function as the baseline
    raw_advantages = returns - values.detach()
    advantages = standardize_tensor(raw_advantages)

    pg_loss, info_dict = policy_gradient_loss(
        advantages=advantages,
        log_probs=torch.stack(log_probs).flatten(),
    )
    pg_loss = pg_loss.mean()

    # NOTE: i use my mse_loss from losses.py, but we don't need priorities here since it's not off policy.
    critic_loss, _ = mse_loss(values, returns)
    critic_loss = critic_loss.mean()

    loss = pg_loss + CRITIC_COEFF * critic_loss

    # Apply Updates
    optimizer = apply_gradients(optimizer, loss)

    if episode % 100 == 0:
        # Calculate explained variance
        explained_var = compute_explained_variance(
            returns.detach().cpu().numpy(), values.detach().cpu().numpy()
        )

        # W&B handles scalars and histograms of tensors (like priorities) automatically.
        log_dict = info_dict.copy()
        log_dict.update(
            {
                "loss/total": loss.item(),
                "loss/critic": critic_loss.mean().item(),
                "value/mean": values.mean().item(),
                "value/return_mean": returns.mean().item(),
                "value/explained_variance": explained_var,
                "advantages/mean": advantages.mean().item(),
                "advantages/std": advantages.std().item(),
            }
        )
        wandb.log(log_dict, step=episode)
