"""
Notes on Noisy DQN:

This idea was introduced in Noisy Nets for Exploration. The idea is to add noise to the weights of the network to encourage exploration.

Unlike standard DQN where exploration is achieved by using epsilon-greedy action selection, NoisyDQN explores by adding noise to the weights of the network. To achieve this, the noisy linear layer parameterizes the noise as $\mu + \sigma \cdot \epsilon$, where $\mu$ and $\sigma$ are learnable parameters and $\epsilon$ is a random variable sampled from a noise distribution (usually a standard Gaussian or a discrete distribution). The key idea is that $\mu$ and $\sigma$ are learnable, so the network can learn the optimal level of exploration for each state. Generating a unique random Gaussian variable for every single weight in a massive linear layer during every forward pass is computationally brutal. The paper solves this by using Factorized Gaussian Noise, generating just two small vectors of noise (one for the input size, one for the output size) and taking their outer product. This makes the layer fast enough to actually use.

Noisy Nets provide a consistent, state-dependent local exploration strategy, meaning if the agent finds a promising path, the noise pushes it to explore variations of that path rather than just taking a random, suicidal action ($\epsilon$-greedy). However, it is not "curious", like intrinsic motivation methods.

This is useful for games like Montezuma's Revenge where exploration is very important and random exploration is not very efficient. In theory, noisy nets allows the agent to learn to get far enough in the game to achieve rewards while still selectively exploring when it is safe to do so. The agent can learn to focus its exploration on the areas of the state space where it is most uncertain.

This method is very generally applicable as a method of exploration and in the original paper is applied both to DQN and A2C. It could feasibly be applied to any other algorithm that uses neural networks to output actions, whether they are deterministic or stochastic.

Note: the below DQN implementation is not the same as used in the paper (it does not use dueling DQN, double DQN, etc).
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
from tensordict import TensorDict
from functools import partial

from functional.replay_buffer import init_buffer, circular_write_strategy, uniform_sample
from functional.losses import bellman_error, mse_loss
from functional.targets import scalar_td_target
from functional.action_selection import (
    argmax_selector,
)
from functional.optimizer import apply_gradients
from functional.network import hard_update_target_network
from networks.noisy_linear import NoisyLinear

# Constants
BATCH_SIZE = 128
GAMMA = 0.99
NOISY_SIGMA = 0.5
LEARNING_RATE = 1e-3
MAX_STEPS = 120_000
UPDATE_FREQ = 4
BUFFER_CAPACITY = 50000
MIN_BUFFER_SIZE = 500
TARGET_NET_UPDATE_FREQ = 100
SEED = 42

# Seeding for reproducibility
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)


class DQN(nn.Module):
    def __init__(self, input_shape: Tuple, num_actions: int):
        super().__init__()
        self.l1 = NoisyLinear(input_shape[0], 512, sigma_init=NOISY_SIGMA)
        self.l2 = NoisyLinear(512, 512, sigma_init=NOISY_SIGMA)
        self.l3 = NoisyLinear(512, num_actions, sigma_init=NOISY_SIGMA)

    def forward(self, x):
        x = F.relu(self.l1(x))
        x = F.relu(self.l2(x))
        x = self.l3(x)
        return x

    def reset_noise(self):
        """Propagates the noise reset to all NoisyLinear layers."""
        for module in self.children():
            if isinstance(module, NoisyLinear):
                module.reset_noise()


# --- 1. Initialization (Defining the State) ---
env = gym.make("CartPole-v1")
obs_shape = env.observation_space.shape
num_actions = env.action_space.n
device = torch.device("cpu")

model = DQN(obs_shape, num_actions).to(device)
target_model = DQN(obs_shape, num_actions).to(device)
target_model.load_state_dict(model.state_dict())

optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

buffer_state = init_buffer(
    capacity=BUFFER_CAPACITY,
    shapes={
        "obs": obs_shape,
        "action": (),
        "reward": (),
        "terminated": (),
        "truncated": (),
        "next_obs": obs_shape,
        "gamma": (),
    },
    device=device,
)

obs, info = env.reset(seed=SEED)
stat_episode_return = 0.0
rng_key = torch.Generator(device=device)
rng_key.manual_seed(SEED)

# Noisy DQN doesn't need a separate epsilon-greedy action_selector as exploration is handled by noise in the network.

# Initialize W&B
wandb.init(
    project="noisy-dqn-cartpole",
    config={
        "batch_size": BATCH_SIZE,
        "gamma": GAMMA,
        "learning_rate": LEARNING_RATE,
        "buffer_capacity": BUFFER_CAPACITY,
    },
)

# --- 2. The Monolithic Loop (The Imperative Shell) ---

for step in range(MAX_STEPS):
    # 2. Act (Pure function)
    with torch.inference_mode():
        obs_tensor = torch.as_tensor(obs[None, ...], dtype=torch.float32, device=device)

        # Resample noisy nets if using them!
        model.reset_noise()

        predictions = model(obs_tensor)
        action, _ = argmax_selector(predictions)
        action = action.item()

    # 2. Step Env
    next_obs, reward, terminated, truncated, info = env.step(action)
    stat_episode_return += reward

    # 3. Add to Buffer
    transition = {
        "obs": torch.as_tensor(obs, dtype=torch.float32),
        "action": torch.tensor(action, dtype=torch.long),
        "reward": torch.tensor(reward, dtype=torch.float32),
        "terminated": torch.tensor(terminated, dtype=torch.float32),
        "truncated": torch.tensor(truncated, dtype=torch.float32),
        "next_obs": torch.as_tensor(next_obs, dtype=torch.float32),
        "gamma": torch.tensor(GAMMA, dtype=torch.float32),
    }
    buffer_state, _ = circular_write_strategy(
        buffer_state, TensorDict(transition, batch_size=[]).unsqueeze(0)
    )

    # Update state for next tick
    obs = next_obs

    if terminated or truncated:
        wandb.log({"episode_return": stat_episode_return}, step=step)
        obs, info = env.reset()
        stat_episode_return = 0.0

    # --- 3. The Update Loop ---
    if step > MIN_BUFFER_SIZE and step % UPDATE_FREQ == 0:
        # Sample
        batch = uniform_sample(buffer_state, rng_key, BATCH_SIZE)

        # Resample noise before calculating loss to prevent overfitting to a single noise vector during the batch loss calculation
        model.reset_noise()
        target_model.reset_noise()

        # Calculate Loss & Gradients
        loss, info_dict = bellman_error(
            model,
            batch,
            target_model,
            partial(scalar_td_target, gamma=rearrange(batch["gamma"], "b -> b 1")),
            loss_fn=mse_loss,
        )
        loss = loss.mean()

        # Apply Updates
        optimizer = apply_gradients(optimizer, loss)

        if step % 100 == 0:
            # W&B handles scalars and histograms of tensors (like priorities) automatically.
            log_dict = info_dict.copy()
            log_dict.update({"loss": loss.item()})
            wandb.log(log_dict, step=step)

    # 4. Target Network Update
    if step % TARGET_NET_UPDATE_FREQ == 0:
        hard_update_target_network(model, target_model)
