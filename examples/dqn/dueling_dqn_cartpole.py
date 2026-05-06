"""
Notes on Dueling DQN:
The main idea is to separate the estimation of the value function and the advantage function. Instead of having a single Q function, we have a V function and an A function. These two functions are then combined to produce the Q function. The advantage function is then used to estimate the Q values for each action. The idea being that in many states all actions are good. We can split the learning into learning good states and learning good actions. This means for good states the network no longer has to learn that every action is good, it can simply learn that the value of the state is high.

Q = V + A wouldnt work as the network would not be able to tell what is V and what is A (the bias could appear in A instead of V). instead we do:
Q = V + (A - mean(A))
This means for a state to have all high Q values, it must have a high V value, and all A values must be 0 (relative to each other).

The specific Dueling architecture is mostly unique to algorithms that estimate Q-values (like DQN and its variants). However, the underlying concept of separating Value from Advantage is a fundamental pillar of modern RL. Actor-Critic methods (like PPO, TRPO, or A3C) rely heavily on computing advantages (often via Generalized Advantage Estimation) to update the policy network (the Actor), using a state-value network (the Critic) as a baseline. What makes the Dueling architecture unique is how it forces a single network to bottleneck and separate these two concepts internally before squashing them back together into an action-value ($Q$) output

Note this is implemented inline with common Rainbow Implementations and may not be in line with the original paper.
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


class DuelingDQN(nn.Module):
    def __init__(self, input_shape: Tuple, num_actions: int):
        super().__init__()
        self.l1 = nn.Linear(input_shape[0], 512)
        self.value_head = nn.Linear(512, 1)
        self.advantage_head = nn.Linear(512, num_actions)

    def forward(self, x):
        x = F.relu(self.l1(x))
        v = self.value_head(x)
        a = self.advantage_head(x)
        return v + a - a.mean(dim=1, keepdim=True)


# --- 1. Initialization (Defining the State) ---
env = gym.make("CartPole-v1")
obs_shape = env.observation_space.shape
num_actions = env.action_space.n
device = torch.device("cpu")

model = DuelingDQN(obs_shape, num_actions).to(device)
target_model = DuelingDQN(obs_shape, num_actions).to(device)
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
    project="dueling-dqn-cartpole",
    config={
        "batch_size": BATCH_SIZE,
        "gamma": GAMMA,
        "learning_rate": LEARNING_RATE,
        "buffer_capacity": BUFFER_CAPACITY,
    },
)

# --- 2. The Monolithic Loop (The Imperative Shell) ---

for step in range(MAX_STEPS):

    # 1. Calculate Epsilon dynamically for this step
    current_epsilon = get_linear_schedule(step, EPS_START, EPS_END, EPS_DECAY_FRAMES)

    # 2. Act (Pure function)
    with torch.inference_mode():
        obs_tensor = torch.from_numpy(obs).float().unsqueeze(0).to(device)

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
        "obs": torch.from_numpy(obs[None, ...]).float(),
        "action": torch.tensor([action], dtype=torch.long),
        "reward": torch.tensor([reward], dtype=torch.float32),
        "terminated": torch.tensor([terminated], dtype=torch.float32),
        "truncated": torch.tensor([truncated], dtype=torch.float32),
        "next_obs": torch.from_numpy(next_obs[None, ...]).float(),
        "gamma": torch.tensor([GAMMA], dtype=torch.float32),
    }
    buffer_state, _ = circular_write_strategy(buffer_state, TensorDict(transition, batch_size=[1]))

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
            partial(standard_td_target, gamma=batch["gamma"].to(device)),
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
