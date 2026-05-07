"""
Notes on DQN:
DQN is a foundational algorithm in deep reinforcement learning that combines Q-learning with deep neural networks. It learns a policy that maximizes the expected cumulative reward by estimating Q-values for each state-action pair.

What made DQN revolutionary was that it was the first deep learning model to successfully learn control policies end-to-end directly from high-dimensional sensory input (raw pixels). Previous successes usually relied on hand-crafted features or low-dimensional state spaces. DQN proved that a Convolutional Neural Network (CNN) could act as the function approximator for raw video frames, and that Experience Replay could stabilize the normally chaotic process of training a neural network on correlated, non-stationary RL data

The paper also popularized the use of Experience Replay. That instead of learning from consecutive samples (which are correlated), we store past experiences in a buffer and sample randomly from it. This breaks the correlation between samples and improves learning stability.

Additionally, the paper introduced the concept of a separate target network to stabilize training. Instead of using the same network to calculate the target Q-values, a copy of the network is kept and updated only periodically. This prevents the network from chasing a moving target, which can lead to instability and divergence.

End to end learning is now essentially standard practice, and tabular methods are rarely used in deep RL. Additionally Experience Replay is used in MANY deep RL algorithms, and target networks are very commonly used (although not always, e.g. in actor-critic methods).

Note: DQN is fundamentally on off-policy algorithm and in theory works well with offline data. However, in practice, DQN can only be slightly off-policy without performance degradation. There are papers that improve DQNs ability with offline data.

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
from einops import rearrange

from functional.buffer import init_buffer, circular_write_strategy, uniform_sample
from functional.losses import bellman_error, mse_loss
from functional.targets import standard_td_target
from functional.action_selection import (
    argmax_selector,
    with_epsilon_greedy,
)
from functional.schedules import get_linear_schedule
from functional.optimizer import apply_gradients
from functional.network import hard_update_target_network

# Constants
BATCH_SIZE = 128
GAMMA = 0.99
EPS_START = 1.0
EPS_END = 0.01
EPS_DECAY_FRAMES = 50000
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
        self.l1 = nn.Linear(input_shape[0], 512)
        self.l2 = nn.Linear(512, 512)
        self.l3 = nn.Linear(512, num_actions)

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

action_selector = with_epsilon_greedy(argmax_selector)

# Initialize W&B
wandb.init(
    project="dqn-cartpole",
    config={
        "batch_size": BATCH_SIZE,
        "gamma": GAMMA,
        "learning_rate": LEARNING_RATE,
        "buffer_capacity": BUFFER_CAPACITY,
    },
)

for step in range(MAX_STEPS):

    # 1. Calculate Epsilon dynamically for this step
    current_epsilon = get_linear_schedule(step, EPS_START, EPS_END, EPS_DECAY_FRAMES)

    # 2. Act (Pure function)
    with torch.inference_mode():
        obs_tensor = torch.as_tensor(obs[None, ...], dtype=torch.float32, device=device)

        predictions = model(obs_tensor)
        action, rng_key = action_selector(
            predictions=predictions,
            epsilon=current_epsilon,
            num_actions=num_actions,
            generator=rng_key,
        )
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

        # Calculate Loss & Gradients
        loss, info_dict = bellman_error(
            model,
            batch,
            target_model,
            partial(standard_td_target, gamma=batch["gamma"]),
            loss_fn=mse_loss,
        )
        loss = loss.mean()

        # Apply Updates
        optimizer = apply_gradients(optimizer, loss)

        if step % 100 == 0:
            # W&B handles scalars and histograms of tensors (like priorities) automatically.
            log_dict = info_dict.copy()
            log_dict.update({"loss": loss.item(), "epsilon": current_epsilon})
            wandb.log(log_dict, step=step)

    # 4. Target Network Update
    if step % TARGET_NET_UPDATE_FREQ == 0:
        hard_update_target_network(model, target_model)
