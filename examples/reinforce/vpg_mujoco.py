# Fully Generated

"""
Notes on Vanilla Policy Gradient for MuJoCo (Continuous Action Space):

This example demonstrates Vanilla Policy Gradient (VPG) with a Critic baseline applied to
the HalfCheetah-v4 MuJoCo environment.

VPG is the foundation for most policy-based methods but suffers from high variance and
poor sample efficiency on complex tasks like HalfCheetah. Without the clipping or
KL-divergence constraints of PPO, VPG updates can be too large, leading to instability.
"""

from atomic_rl.initialization import layer_init
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import gymnasium as gym
from typing import Tuple
import numpy as np
import random
import wandb

from atomic_rl.action_selection import sample_distribution
from atomic_rl.optimizer import apply_gradients
from atomic_rl.returns import compute_mc_returns
from atomic_rl.losses import policy_gradient_loss, mse_loss
from atomic_rl.utils import (
    standardize_tensor,
    to_tensor,
    to_numpy_action,
)
from atomic_rl.visualization import compute_explained_variance
from envs.wrappers import VecNormalize

# Constants
LEARNING_RATE = 3e-4
MAX_EPISODES = 5000  # Sufficient for benchmarking against PPO
GAMMA = 0.99
SEED = 42
CRITIC_COEFF = 0.5
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
        x = torch.tanh(self.l1(x))
        x = torch.tanh(self.l2(x))
        mu = self.mu_head(x)
        std = torch.exp(self.log_std).expand_as(mu)
        return mu, std


class Critic(nn.Module):
    def __init__(self, input_shape: Tuple):
        super().__init__()
        self.l1 = layer_init(nn.Linear(input_shape[0], HIDDEN_SIZE))
        self.l2 = layer_init(nn.Linear(HIDDEN_SIZE, HIDDEN_SIZE))
        self.l3 = layer_init(nn.Linear(HIDDEN_SIZE, 1), std=1.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.tanh(self.l1(x))
        x = torch.tanh(self.l2(x))
        x = self.l3(x)
        return x


# --- 1. Initialization ---
# For VPG, we use a single environment instance to collect full episodes
def make_single_env(env_id, seed):
    env = gym.make(env_id)
    env = gym.wrappers.RecordEpisodeStatistics(env)
    env = gym.wrappers.ClipAction(env)
    # Wrap in dummy vector env to keep downstream code consistent (using .single_observation_space)
    env = gym.vector.SyncVectorEnv([lambda: env])
    env = VecNormalize(
        env,
        norm_obs=True,
        norm_reward=True,
        clip_obs=10.0,
        clip_reward=10.0,
        gamma=GAMMA,
    )
    env.action_space.seed(seed)
    env.observation_space.seed(seed)
    return env


env = make_single_env("HalfCheetah-v4", SEED)
obs_shape = env.single_observation_space.shape
num_actions = env.single_action_space.shape[0]
device = torch.device("cpu")

actor = Actor(obs_shape, num_actions).to(device)
critic = Critic(obs_shape).to(device)

optimizer = optim.Adam(
    list(actor.parameters()) + list(critic.parameters()), lr=LEARNING_RATE, eps=1e-5
)

obs, info = env.reset(seed=SEED)
terminated, truncated = False, False


# Initialize W&B
wandb.init(
    project="vpg-mujoco",
    config={
        "lr": LEARNING_RATE,
        "gamma": GAMMA,
        "env": "HalfCheetah-v4",
    },
)
wandb.define_metric("*", step_metric="global_step")
global_step = 0

for episode in range(MAX_EPISODES):
    rewards = []
    log_probs = []
    values = []
    terminateds = []
    truncateds = []

    while not (terminated or truncated):
        obs_tensor = to_tensor(obs, device=device)
        mu, std = actor(obs_tensor)
        value = critic(obs_tensor)

        dist = torch.distributions.Normal(mu, std)
        action_tensor, info_dict = sample_distribution(dist, explore=True)
        action_np = to_numpy_action(action_tensor).flatten()

        # 2. Step Env
        next_obs, reward, terminated, truncated, info = env.step(action_np)
        global_step += 1

        # 3. Add to buffers
        rewards.append(reward[0])
        log_probs.append(info_dict["log_prob"].sum(dim=-1))
        values.append(value)
        terminateds.append(terminated[0])
        truncateds.append(truncated[0])

        # Update state
        obs = next_obs

    # End of episode
    if "final_info" in info:
        for item in info["final_info"]:
            if item is not None and "episode" in item:
                wandb.log(
                    {
                        "episode_return": item["episode"]["r"][0],
                        "episode_length": item["episode"]["l"][0],
                        "global_step": global_step,
                    }
                )
    obs, info = env.reset()
    terminated, truncated = False, False

    # --- 3. The Update Loop ---
    # 1. Compute Returns
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
                "std": torch.exp(actor.log_std).mean().item(),
                "global_step": global_step,
            }
        )
        wandb.log(log_dict)

wandb.finish()
