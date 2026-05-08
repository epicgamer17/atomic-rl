"""
Notes on PER:

Prioritized Experience Replay (PER) is an optimization technique for experience replay in DQN. Instead of sampling transitions uniformly from the replay buffer, PER samples transitions based on their "priority" (commonly their TD error). Transitions with higher TD errors are sampled more frequently, allowing the agent to focus on learning from more "surprising" or informative experiences.

This method introduces a trade-off: we want to learn from important transitions, but sampling in a non uniform way introduces bias into the training process because the updates no longer match the underlying data distribution. To mitigate this, PER uses "Importance Sampling" (IS) weights. When we sample a transition, we weight its contribution to the loss by the inverse of its sampling probability. This corrects the bias introduced by non uniform sampling. IS weights scale the gradients so that the mathematical expectation matches uniform sampling. Note the IS weights must be normalized by $1/\max_i w_i$ so that they only scale gradient updates downwards.

The priority of a transition is updated after each training step. The new priority is typically set to the absolute difference between the predicted Q-value and the target Q-value, $p_i = |\delta_i| + \epsilon$ where epsilon is a small constant to ensure all transitions have a non-zero probability of being sampled. This ensures that transitions that are currently "surprising" are given higher priority for future sampling. You can also use a rank based method for priority $p_i = 1/\text{rank}(i)$, which is much more robust to outliers and spikes than the standard TD error based method, however is used much less often than the standard TD error based method. The authors of the PER paper actually expected the rank-based variant to be more robust because it ignores extreme outliers. However, they found that, surprisingly, both variants performed similarly in practice. They suspected this was because standard DQN heavily relies on clipping rewards and TD-errors (typically between -1 and 1), which naturally removes the extreme outliers that the rank-based method was designed to protect against. Because the proportional variant is slightly more straightforward to implement with a single sum-tree, it became the community standard. Addtionally, in many papers PER is used with distributional value prediction, and the cross entropy loss between the predition and target is used, this has similar effects to clipping the TD-error, hence why the TD approach is more popular in practice.

For more efficient sampling a sum tree and min tree can be used. This requires more memory but allows for $O(\log n)$ sampling and update times, as opposed to $O(n)$ for a naive implementation. For the rank based priority method a binary heap can be used for efficiency.

The $\alpha$ parameter controls how much weight is given to the priorities. A value of 0 means that all transitions are sampled uniformly. A value of 1 means that transitions are sampled strictly based on their priorities.

The $\beta$ parameter controls the degree of importance sampling. It is typically annealed from a starting value to 1 over time. This allows the agent to explore more initially and then focus on learning from important transitions later.

In summary, PER is a simple but effective technique that can significantly improve sample efficiency. PER is very generally applicable and can be used in many different RL algorithms that use replay buffers and experience replay.

Note: below we only implement the TD error based priority method for simplicity.
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

from functional.replay_buffer import (
    init_per_buffer,
    sample_per,
    update_priorities,
    circular_write_strategy,
    with_per_tracking,
)
from functional.schedules import get_linear_schedule
from functional.losses import bellman_error, with_per_weights, mse_loss
from functional.targets import scalar_td_target
from functional.action_selection import (
    argmax_selector,
    with_epsilon_greedy,
)
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

# PER Constants
ALPHA = 0.6
BETA_START = 0.4
BETA_FRAMES = 100000

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

buffer_state = init_per_buffer(
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
# Compose circular write with PER tracking
per_add_transition = with_per_tracking(circular_write_strategy)

obs, info = env.reset(seed=SEED)
stat_episode_return = 0.0
rng_key = torch.Generator(device=device)
rng_key.manual_seed(SEED)

action_selector = with_epsilon_greedy(argmax_selector)

# Initialize W&B
wandb.init(
    project="prioritized-replay-dqn-cartpole",
    config={
        "batch_size": BATCH_SIZE,
        "gamma": GAMMA,
        "learning_rate": LEARNING_RATE,
        "alpha": ALPHA,
    },
)

# --- 2. The Monolithic Loop (The Imperative Shell) ---
for step in range(MAX_STEPS):

    # 1. Calculate Epsilon dynamically for this step
    current_epsilon = get_linear_schedule(step, EPS_START, EPS_END, EPS_DECAY_FRAMES)

    # 2. Act
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
    transition = {
        "obs": torch.as_tensor(obs, dtype=torch.float32),
        "action": torch.tensor(action, dtype=torch.long),
        "reward": torch.tensor(reward, dtype=torch.float32),
        "terminated": torch.tensor(terminated, dtype=torch.float32),
        "truncated": torch.tensor(truncated, dtype=torch.float32),
        "next_obs": torch.as_tensor(next_obs, dtype=torch.float32),
        "gamma": torch.tensor(GAMMA, dtype=torch.float32),
    }
    buffer_state = per_add_transition(
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
        # Anneal Beta
        beta = get_linear_schedule(step, BETA_START, 1.0, BETA_FRAMES)
        beta_tensor = torch.tensor(beta, dtype=torch.float32, device=device)

        # Sample with PER
        batch, tree_indices, is_weights = sample_per(
            buffer_state, BATCH_SIZE, beta=beta_tensor
        )

        # Wrap loss function with PER weights
        per_loss_fn = with_per_weights(mse_loss, is_weights)

        # Calculate Loss & Gradients
        # bellman_error returns whatever loss_fn returns (loss, info)
        loss, info_dict = bellman_error(
            model,
            batch,
            target_model,
            partial(scalar_td_target, gamma=rearrange(batch["gamma"], "b -> b 1")),
            loss_fn=per_loss_fn,
        )

        # Apply Updates
        optimizer = apply_gradients(optimizer, loss)

        # Update PER Priorities
        buffer_state = update_priorities(
            buffer_state, tree_indices, info_dict["priorities"], alpha=ALPHA
        )

        if step % 100 == 0:
            # W&B handles scalars and histograms of tensors (like priorities) automatically.
            log_dict = info_dict.copy()
            log_dict.update(
                {"loss": loss.item(), "epsilon": current_epsilon, "beta": beta}
            )
            wandb.log(log_dict, step=step)

    # 4. Target Network Update
    if step % TARGET_NET_UPDATE_FREQ == 0:
        hard_update_target_network(model, target_model)
