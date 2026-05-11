"""
Notes on Multi-step (N-step) DQN:
The idea is to use n-step returns to speed up learning. Instead of using the 1-step return $R_{t+1} + \gamma \max_{a} Q(s_{t+1}, a)$, we use the n-step return $\sum_{k=0}^{n-1} \gamma^k R_{t+k+1} + \gamma^n \max_{a} Q(s_{t+n}, a)$. This allows for faster propagation of rewards to earlier time steps. The argument for is that a single step (reward) affecting the n previous values instead of just the immediate previous one.

This method increases the variance of updates but reduces the bias. In a sense, it is a compromise between 1-step TD and Monte Carlo returns (which use the full episode return). By increasing n, we can trade decreased bias for increased variance (and vice versa). By using the n step return we rely less on our flawed guesses (reducing bias) but more on environmental randomness (increasing variance).

This idea of bootstrapping the value n steps ahead is foundational to Reinforcement Learning and extremely common. The overall concept is used in N-Step TD (TD(lambda)), GAE (for policy gradients), and Monte Carlo returns. Note, however, that TD(lambda) and GAE use an exponentially weighted sum of all possible n-step returns (all values of n). rather than a single value of n. So learning from n-step returns is a general concept that applies to many other algorithms.
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

from functional.replay_buffer import (
    init_buffer,
    circular_write_strategy,
    uniform_sample,
    make_n_step_accumulator,
)
from functional.losses import compute_q_td_loss, mse_loss
from functional.targets import scalar_td_target
from functional.action_selection import (
    argmax_selector,
    with_epsilon_greedy,
)
from functional.schedules import get_linear_schedule
from functional.optimizer import apply_gradients
from functional.network import hard_update_target_network, layer_init

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
N_STEPS = 3
SEED = 42

# Seeding for reproducibility
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)


class DQN(nn.Module):
    def __init__(self, input_shape: Tuple, num_actions: int):
        super().__init__()
        self.l1 = layer_init(nn.Linear(input_shape[0], 512))
        self.l2 = layer_init(nn.Linear(512, 512))
        self.l3 = layer_init(nn.Linear(512, num_actions), std=1.0)

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

# 1. Initialize the Accumulator before the loop
accumulate_n_step, reset_accumulator = make_n_step_accumulator(
    n_steps=N_STEPS, gamma=GAMMA
)

# Initialize W&B
wandb.init(
    project="n-step-dqn-cartpole",
    config={
        "batch_size": BATCH_SIZE,
        "gamma": GAMMA,
        "learning_rate": LEARNING_RATE,
        "n_steps": N_STEPS,
    },
)


# --- 2. The Monolithic Loop (The Imperative Shell) ---
for step in range(MAX_STEPS):

    # 1. Calculate Epsilon dynamically for this step
    current_epsilon = get_linear_schedule(step, EPS_START, EPS_END, EPS_DECAY_FRAMES)

    # 2. Act (Pure function)
    with torch.inference_mode():
        obs_tensor = torch.as_tensor(obs[None, ...], dtype=torch.float32, device=device)

        predictions = model(obs_tensor)
        action, info = action_selector(
            predictions=predictions,
            epsilon=current_epsilon,
            num_actions=num_actions,
            generator=rng_key,
        )
        rng_key = info["generator"]
        action = action.item()

    # 2. Step Env
    next_obs, reward, terminated, truncated, info = env.step(action)
    stat_episode_return += reward

    # 3. Add to Buffer
    n_step_transitions = accumulate_n_step(
        torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0),
        torch.tensor([action]),
        torch.tensor([reward], dtype=torch.float32),
        torch.as_tensor(next_obs, dtype=torch.float32).unsqueeze(0),
        torch.tensor([terminated], dtype=torch.float32),
        torch.tensor([truncated], dtype=torch.float32),
    )
    for transition in n_step_transitions:
        buffer_state, _ = circular_write_strategy(buffer_state, transition)

    # Update state for next tick
    obs = next_obs

    if terminated or truncated:
        wandb.log({"episode_return": stat_episode_return}, step=step)
        obs, info = env.reset()
        reset_accumulator()
        stat_episode_return = 0.0

    # --- 3. The Update Loop ---
    if step > MIN_BUFFER_SIZE and step % UPDATE_FREQ == 0:
        # Sample
        batch = uniform_sample(buffer_state, rng_key, BATCH_SIZE)

        # Calculate Loss & Gradients
        loss, info_dict = compute_q_td_loss(
            model,
            batch,
            model,
            partial(scalar_td_target, gamma=rearrange(batch["gamma"], "b -> b 1")),
            eval_model=target_model,
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
