"""
Notes on A2C:

Essentially REINFORCE + a value function for the baseline to compute advantages. Or another way of putting it is REINFORCE with learned state dependant baseline.
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
from functools import partial

from functional.action_selection import categorical_sampling_selector
from functional.optimizer import apply_gradients
from functional.returns import compute_n_step_returns
from functional.losses import policy_gradient_loss, mse_loss, entropy_loss
from functional.visualization import compute_explained_variance
from functional.advantages import compute_critic_advantages
from functional.buffer import (
    init_rollout_buffer,
    store_rollout_step,
    flatten_rollout_buffer,
    record_truncations,
    get_rollout_next_values,
)
from tensordict import TensorDict
import pufferlib
import pufferlib.vector
import pufferlib.emulation

# Constants
LEARNING_RATE = 1e-3
MAX_ITERATIONS = 1_000
GAMMA = 0.99
N_STEP = 3
ENTROPY_COEFF = 0.01
CRITIC_COEFF = 0.5
STEPS_PER_ENV = 512
NUM_ENVS = 4
SEED = 42

# Seeding for reproducibility
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

# NOTE: these can have a fused backbone and separate heads, for now we keep separate for simplicity


class ActorCritic(nn.Module):
    def __init__(self, input_shape: Tuple, num_actions: int):
        super().__init__()
        self.l1 = nn.Linear(input_shape[0], 64)
        self.l2 = nn.Linear(64, 64)
        self.l3_actor = nn.Linear(64, num_actions)
        self.l3_critic = nn.Linear(64, 1)

    def forward(self, x):
        x = F.relu(self.l1(x))
        x = F.relu(self.l2(x))
        x_actor = self.l3_actor(x)
        x_critic = self.l3_critic(x)
        return x_actor, x_critic


# --- 1. Initialization (Defining the State) ---
def env_creator(**kwargs):
    # Create the standard Gym environment
    env = gym.make("CartPole-v1")

    # Wrap it, and pass along PufferLib's optimizations (buf, seed, etc.)
    return pufferlib.emulation.GymnasiumPufferEnv(env=env, **kwargs)


# PufferLib automatically handles vectorization and auto-resetting
envs = pufferlib.vector.make(
    env_creator, num_envs=NUM_ENVS, backend=pufferlib.vector.Serial
)  # Note: For heavy environments, you can swap to:
# envs = pufferlib.vector.Multiprocessing(env_creator, num_envs=NUM_ENVS)
obs_shape = envs.single_observation_space.shape
num_actions = envs.single_action_space.n
device = torch.device("cpu")

model = ActorCritic(obs_shape, num_actions).to(device)

optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

obs, info = envs.reset(seed=SEED)

# Pre-allocate rollout buffers using the new functional system
shapes = {
    "observations": obs_shape,
    "actions": (),
    "logprobs": (),
    "rewards": (),
    "terminated": (),
    "truncated": (),
    "values": (),
    "logits": (num_actions,),
}
buffer = init_rollout_buffer(
    steps_per_env=STEPS_PER_ENV,
    num_envs=NUM_ENVS,
    shapes=shapes,
    device=device,
)

# Track episodic returns
stat_episode_returns = np.zeros(NUM_ENVS)

rng_key = torch.Generator(device=device)
rng_key.manual_seed(SEED)

# Initialize W&B
wandb.init(project="a2c-cartpole", config={"lr": LEARNING_RATE, "gamma": GAMMA})

# Using full episodes
for iteration in range(MAX_ITERATIONS):
    # Clear the data
    # TODO: make this part of "online sampling" and put it in a function?, along with clearing the truncation records maybe?
    buffer.data.detach_()
    for step in range(STEPS_PER_ENV):
        obs_tensor = torch.as_tensor(obs, dtype=torch.float32, device=device)
        logits, value = model(obs_tensor)
        action, log_prob = categorical_sampling_selector(logits, temperature=1.0)
        action = (
            action.cpu().numpy().flatten().astype(np.int32)
        )  # NOTE: .item() is for python scalars (NOT for batches!)

        # 2. Step Env
        next_obs, reward, terminated, truncated, info = envs.step(action)

        # 3. Add to "online" buffers
        transition = TensorDict(
            {
                "observations": obs_tensor,
                "actions": torch.as_tensor(action, device=device),
                "logprobs": log_prob.squeeze(-1),
                "rewards": torch.as_tensor(reward, dtype=torch.float32, device=device),
                "terminated": torch.as_tensor(
                    terminated, dtype=torch.float32, device=device
                ),
                "truncated": torch.as_tensor(
                    truncated, dtype=torch.float32, device=device
                ),
                "values": value.squeeze(-1),
                "logits": logits,
            },
            batch_size=[NUM_ENVS],
        )
        store_rollout_step(buffer=buffer, step=step, transition=transition)
        record_truncations(buffer, step, info, truncated)
        # Update state for next tick

        stat_episode_returns += reward
        for i, (t, tr) in enumerate(zip(terminated, truncated)):
            if t or tr:
                wandb.log({"episode_return": stat_episode_returns[i]}, step=iteration)
                stat_episode_returns[i] = 0.0

        # TODO: vectorized envs like pufferlib auto reset so obs = next_obs is moved to here.
        obs = next_obs

    # --- 3. The Update Loop ---
    # TODO: handle einops reshaping

    last_obs_tensor = torch.as_tensor(obs, dtype=torch.float32, device=device)
    _, last_values = model(last_obs_tensor)

    # Calculate Loss & Gradients
    next_values = get_rollout_next_values(
        buffer, last_values, get_value_fn=lambda obs: model(obs)[1], device=device
    )

    # TODO: properly handle truncation and pass that to compute_n_step_returns

    returns = compute_n_step_returns(
        rewards=buffer.data["rewards"],
        terminated=buffer.data["terminated"],
        truncated=buffer.data["truncated"],
        values=buffer.data["values"],
        next_values=next_values,
        gamma=GAMMA,
        n=N_STEP,
    )

    advantages = compute_critic_advantages(
        returns=returns,
        values=buffer.data["values"],
    )

    # Flatten buffer for loss calculations
    flat_data = flatten_rollout_buffer(buffer)
    flat_advantages = rearrange(advantages, "t b -> (t b)")
    flat_returns = rearrange(returns, "t b -> (t b)")

    pg_loss, info_dict = policy_gradient_loss(
        advantages=flat_advantages.detach(),
        log_probs=flat_data["logprobs"],
    )

    ent_loss, _ = entropy_loss(logits=flat_data["logits"])
    ent_loss = ent_loss.mean()

    pg_loss = pg_loss.mean()

    # NOTE: i use my mse_loss from losses.py, but we don't need priorities here since it's not off policy.
    critic_loss, _ = mse_loss(
        predictions=flat_data["values"], targets=flat_returns.detach()
    )
    critic_loss = critic_loss.mean()

    loss = pg_loss + CRITIC_COEFF * critic_loss - ENTROPY_COEFF * ent_loss

    # Apply Updates
    optimizer = apply_gradients(optimizer, loss)

    # Clear truncation records for the next iteration
    buffer.truncation_records.clear()

    if iteration % 100 == 0:
        # Calculate explained variance
        explained_var = compute_explained_variance(
            returns.detach().cpu().numpy(), buffer.data["values"].detach().cpu().numpy()
        )

        # W&B handles scalars and histograms of tensors (like priorities) automatically.
        log_dict = info_dict.copy()
        log_dict.update(
            {
                "loss/total": loss.item(),
                "loss/critic": critic_loss.mean().item(),
                "value/mean": buffer.data["values"].mean().item(),
                "value/return_mean": returns.mean().item(),
                "value/explained_variance": explained_var,
                "advantages/mean": advantages.mean().item(),
                "advantages/std": advantages.std().item(),
            }
        )
        wandb.log(log_dict, step=iteration)
