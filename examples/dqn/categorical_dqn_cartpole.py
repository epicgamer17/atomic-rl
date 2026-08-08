"""
Notes on C51 (Distributional/Categorical) DQN:
Unlike standard DQN which estimates the expected value of the return, distributional RL estimates the
full probability distribution of the return. This allows us to capture the riskiness of different actions and states leading to potentially better performance. The key idea is to discretize the return into a set of bins called atoms.

Although it learns the riskiness of different actions and states C51 does not use it. It still selects actions via the expected value. However learning the distribution instead of the scalar expected values allows for better representation learning and richer gradients, leading to better performance.

This changes the shape of the loss treating the problem as a classification problem, where we are trying to predict the probability of each atom. With MSE or Huber large differences were penalized more, here the scale of the difference between values is not as important as the classification error. This can lead to better stability and performance, especially in environments with stochastic rewards.
It also has an effect on PER priorities.

A specific detail is that for Q = r + gamma * max_a Q(s',a), the distribution needs to be shifted to have the correct expected value. For C51 this is done in a way that maintains the rich distribution predicted by the network, unlike MuZero and other models which compute a scalar target, and then compute a totally fresh distribution from that target (instead of using the prediction).

In short:
- C51 treats the environment and the returns as a distribution. It explicitly learns how uncertainty unfolds over time.
- Other algorithms like MuZero treat the neural network output as a distribution strictly to make the gradient descent smoother. It completely destroys the distributional information of the future by averaging it into a scalar before making its target.

This paradigm/approach of treating q value estimation, value estimation, or reward estimation as classification problems has been applied to other algorithms as well like MuZero and Dreamer, though with different supports and support transformations or targets (e.g., two-hot with a projection or exponentially spaced bins). This is a powerful idea that has led to significant improvements in performance.

NOTE: the below DQN implementation is not the same as used in the paper (it does not use dueling DQN, double DQN, etc).
"""

from atomic_rl.initialization import layer_init_
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

from atomic_rl.buffers.replay import (
    init_buffer,
    circular_write_strategy_,
    uniform_sample,
)
from atomic_rl.losses import cross_entropy_loss
from atomic_rl.action_selection import (
    argmax_selector,
    expected_value,
    gather_q_values,
    with_epsilon_greedy,
)
from atomic_rl.schedules import get_linear_schedule
from atomic_rl.optimizer import apply_gradients_
from atomic_rl.update_target_net import hard_update_target_network_
from atomic_rl.metrics import log_distributional_metrics

from atomic_rl.td import compute_categorical_q_td_target

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
V_MIN = 0
V_MAX = 500
ATOM_SIZE = 51
SUPPORT = torch.linspace(V_MIN, V_MAX, ATOM_SIZE)

SEED = 42

# Seeding for reproducibility
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)


class CategoricalDQN(nn.Module):
    def __init__(self, input_shape: Tuple, num_actions: int, atom_size: int = 51):
        super().__init__()
        self.l1 = layer_init_(nn.Linear(input_shape[0], 512))
        self.l2 = layer_init_(nn.Linear(512, 512))
        self.l3 = layer_init_(nn.Linear(512, num_actions * atom_size), std=1.0)
        self.num_actions = num_actions
        self.atom_size = atom_size

    def forward(self, x):
        x = F.relu(self.l1(x))
        x = F.relu(self.l2(x))
        x = self.l3(x)
        return x.view(x.size(0), self.num_actions, self.atom_size)


# --- 1. Initialization (Defining the State) ---
env = gym.make("CartPole-v1")
env = gym.wrappers.RecordEpisodeStatistics(env)
obs_shape = env.observation_space.shape
num_actions = env.action_space.n
device = torch.device("cpu")

model = CategoricalDQN(obs_shape, num_actions).to(device)
target_model = CategoricalDQN(obs_shape, num_actions).to(device)
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

# 1. Initialize the Distributional DQN rollout selector
action_selector = with_epsilon_greedy(argmax_selector)

# Initialize W&B
wandb.init(
    project="categorical-dqn-cartpole",
    config={
        "batch_size": BATCH_SIZE,
        "gamma": GAMMA,
        "learning_rate": LEARNING_RATE,
        "atom_size": ATOM_SIZE,
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
        expected_qs = expected_value(predictions, support=SUPPORT.to(device))
        action, info = action_selector(
            predictions=expected_qs,
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
    buffer_state, _ = circular_write_strategy_(
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
        logits = model(batch["obs"])
        with torch.no_grad():
            next_logits = target_model(batch["next_obs"])

            # 2. Next Action Selection (Composed inline natively)
            # C51 uses the expected value of the distribution for action selection
            next_expected_qs = expected_value(next_logits, support=SUPPORT.to(device))
            next_actions, _ = argmax_selector(next_expected_qs)

            # 3. Target Calculation (Pure Primitive)
            td_target = compute_categorical_q_td_target(
                next_logits,
                next_actions.squeeze(-1),
                batch["reward"],
                batch["terminated"],
                batch["gamma"],
                support=SUPPORT.to(device),
                v_min=V_MIN,
                v_max=V_MAX,
                atom_size=ATOM_SIZE,
            )

        # 4. Prediction Extraction (Current actions)
        pred_sa_logits = gather_q_values(logits, batch["action"])

        # 5. Loss Calculation (Pure Primitive)
        loss, info_dict = cross_entropy_loss(pred_sa_logits, td_target)
        loss = loss.mean()

        # Apply Updates
        optimizer = apply_gradients_(optimizer, loss)

        if step % 100 == 0:
            # Log all metrics from info_dict. W&B handles scalars and histograms of tensors.
            # We exclude 'predictions' and 'priorities' from the direct log to handle them specially.
            log_dict = {
                k: v
                for k, v in info_dict.items()
                if k not in ["predictions", "priorities"]
            }
            log_dict.update({"epsilon": current_epsilon})

            # Add distributional metrics (Expected Q every 100 steps, Chart every 1000 steps)
            log_dict.update(
                log_distributional_metrics(
                    info_dict, SUPPORT, step, log_chart=(step % 1000 == 0)
                )
            )

            wandb.log(log_dict, step=step)

    # 4. Target Network Update
    if step % TARGET_NET_UPDATE_FREQ == 0:
        hard_update_target_network_(model, target_model)
