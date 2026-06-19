"""
Notes on DQN:
DQN is a foundational algorithm in deep reinforcement learning that combines Q-learning with deep neural networks. It learns a policy that maximizes the expected cumulative reward by estimating Q-values for each state-action pair.

What made DQN revolutionary was that it was the first deep learning model to successfully learn control policies end-to-end directly from high-dimensional sensory input (raw pixels). Previous successes usually relied on hand-crafted features or low-dimensional state spaces. DQN proved that a Convolutional Neural Network (CNN) could act as the function approximator for raw video frames, and that Experience Replay could stabilize the normally chaotic process of training a neural network on correlated, non-stationary RL data

The paper also popularized the use of Experience Replay. That instead of learning from consecutive samples (which are correlated), we store past experiences in a buffer and sample randomly from it. This breaks the correlation between samples and improves learning stability. It also helps to reduce non-stationarity. It can also be argued to improve sample efficiency, though this is not always the case, and depends on how experience replay is used.And although experience replay is widely used, it does come with some trade-offs, your algorithm must work well off policy (somewhat well, as DQN does), and this prevents experience replay from being used in on-policy algorithms like PPO or A2C, etc (many policy gradient methods). It also leads to a much larger memory foot print, and uses more memory per real interaction with the environment.

Additionally, the paper introduced the concept of a separate target network to stabilize training. Instead of using the same network to calculate the target Q-values, a copy of the network is kept and updated only periodically. This prevents the network from chasing a moving target, which can lead to instability and divergence.

End to end learning is now essentially standard practice, and tabular methods are rarely used in deep RL. Additionally Experience Replay is used in MANY deep RL algorithms, and target networks are very commonly used (although not always, e.g. in actor-critic methods).

NOTE: DQN is fundamentally on off-policy algorithm and in theory works well with offline data. However, in practice, DQN can only be slightly off-policy without performance degradation. There are papers that improve DQNs ability with offline data.

"""

from functional.initialization import layer_init, set_seed
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
    init_buffer,
    circular_write_strategy,
    uniform_sample,
)
from functional.losses import mse_loss
from functional.td import compute_q_td_target
from functional.action_selection import (
    argmax_selector,
    gather_q_values,
    with_epsilon_greedy,
)
from functional.schedules import get_linear_schedule
from functional.optimizer import apply_gradients
from functional.network import hard_update_target_network
from functional.utils import (
    to_tensor,
    to_numpy_action,
)

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
set_seed(SEED)


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
env = gym.wrappers.RecordEpisodeStatistics(env)
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
        "action": (1,),
        "reward": (),
        "terminated": (),
        "truncated": (),
        "next_obs": obs_shape,
        "gamma": (),
    },
    device=device,
)

obs, info = env.reset(seed=SEED)
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
        obs_tensor = to_tensor(obs[None, ...], device=device)

        predictions = model(obs_tensor)
        action, info = action_selector(
            predictions=predictions,
            epsilon=current_epsilon,
            num_actions=num_actions,
            generator=rng_key,
        )
        rng_key = info["generator"]
        action_np = to_numpy_action(action)

    # 2. Step Env
    # Extract the scalar for a non-vectorized Gymnasium environment
    action_int = int(action_np.item())
    next_obs, reward, terminated, truncated, info = env.step(action_int)

    # 3. Add to Buffer
    # TODO: URGENT. Creating a new tensor every step in the hotloop. should do in a more efficient way, maybe some thing like the rollout buffer in PPO (possible reuse?) ie store in a rollout buffer before sending to main replay buffer. Idea being its pre allocated basically. Must consider the N-Step case.
    transition = {
        "obs": to_tensor(obs),
        "action": action.squeeze(0).detach().to(torch.long),
        "reward": to_tensor(reward),
        "terminated": to_tensor(terminated),
        "truncated": to_tensor(truncated),
        "next_obs": to_tensor(next_obs),
        "gamma": torch.tensor(GAMMA, dtype=torch.float32),
    }
    buffer_state, _ = circular_write_strategy(
        buffer_state, TensorDict(transition, batch_size=[]).unsqueeze(0)
    )

    # Update state for next tick
    obs = next_obs

    if terminated or truncated:
        if "episode" in info:
            wandb.log(
                {
                    "episode_return": info["episode"]["r"][0],
                    "episode_length": info["episode"]["l"][0],
                },
                step=step,
            )
        obs, info = env.reset()

    # --- 3. The Update Loop ---
    if step > MIN_BUFFER_SIZE and step % UPDATE_FREQ == 0:
        # Sample
        batch = uniform_sample(buffer_state, rng_key, BATCH_SIZE)

        # 1. Forward Passes (Online and Target)
        q_values = model(batch["obs"])
        with torch.no_grad():
            next_q_values = target_model(batch["next_obs"])

            # 2. Next Action Selection (Pure Primitive)
            # Standard DQN uses greedy selection on the target network
            next_actions, _ = argmax_selector(next_q_values)

            # 3. Target Calculation (Pure Primitive)
            td_target = compute_q_td_target(
                next_q_values,
                next_actions.squeeze(-1),
                batch["reward"],
                batch["terminated"],
                batch["gamma"],
            )

        # 4. Prediction Extraction (Current actions)
        pred_sa = gather_q_values(q_values, batch["action"])

        # 5. Loss Calculation (Pure Primitive)
        loss, info_dict = mse_loss(pred_sa, td_target)
        loss = loss.mean()

        # Apply Updates
        optimizer = apply_gradients(optimizer, loss)

        if step % 100 == 0:
            # Augment info with orchestration-level metrics
            info_dict.update(
                {
                    "loss": loss.item(),
                    "epsilon": current_epsilon,
                    "q_values/mean": pred_sa.mean().detach(),
                    "q_values/min": pred_sa.min().detach(),
                    "q_values/max": pred_sa.max().detach(),
                    "td_targets/mean": td_target.mean().detach(),
                    "rewards/mean": batch["reward"].mean().detach(),
                }
            )
            wandb.log(info_dict, step=step)

    # 4. Target Network Update
    if step % TARGET_NET_UPDATE_FREQ == 0:
        hard_update_target_network(model, target_model)
