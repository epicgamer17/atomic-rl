r"""
Notes on Double DQN:
The idea is to decouple value prediction from action selection. To prevent the estimation errors caused by the maximization step, where the same network is used to select the action and evaluate the value of that action, we use an online network and a delayed target network. The online network is used to select the action, and the target network is used to evaluate the value of that action. This prevents the network from overestimating the value of the action that it selects, which can lead to unstable training and poor performance. This is in contrast to DQN, where the same network is used to select the action and evaluate the value of that action.

Standard DQN Target: $Y_{t}^{DQN} \equiv R_{t+1} + \gamma \max_{a} Q(S_{t+1}, a; \theta_{t}^{-})$.  Double DQN Target: $Y_{t}^{DoubleDQN} \equiv R_{t+1} + \gamma Q(S_{t+1}, \text{argmax}_{a} Q(S_{t+1}, a; \theta_{t}); \theta_{t}^{-})$.

NOTE: this is implemented inline with common Rainbow Implementations and may not be in line with the original paper.
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
from tensordict import TensorDict
from functools import partial
from atomic_rl.utils import to_numpy_action

from atomic_rl.replay_buffer import (
    init_buffer,
    circular_write_strategy,
    uniform_sample,
)
from atomic_rl.losses import mse_loss
from atomic_rl.td import compute_q_td_target
from atomic_rl.action_selection import (
    argmax_selector,
    gather_q_values,
    with_epsilon_greedy,
)
from atomic_rl.schedules import get_linear_schedule
from atomic_rl.optimizer import apply_gradients
from atomic_rl.network import hard_update_target_network_

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
    project="ddqn-cartpole",
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
        obs_tensor = torch.as_tensor(obs[None, ...], dtype=torch.float32, device=device)

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
        "obs": torch.as_tensor(obs, dtype=torch.float32),
        "action": action.squeeze(0).detach().to(torch.long),
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
            next_q_values_online = model(batch["next_obs"])
            next_q_values_target = target_model(batch["next_obs"])

            # 2. Next Action Selection (Double DQN: Online model selects)
            next_actions, _ = argmax_selector(next_q_values_online)

            # 3. Target Calculation (Double DQN: Target model evaluates)
            td_target = compute_q_td_target(
                next_q_values_target,
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
            # W&B handles scalars and histograms of tensors (like priorities) automatically.
            log_dict = info_dict.copy()
            log_dict.update({"loss": loss.item(), "epsilon": current_epsilon})
            wandb.log(log_dict, step=step)

    # 4. Target Network Update
    if step % TARGET_NET_UPDATE_FREQ == 0:
        hard_update_target_network_(model, target_model)
