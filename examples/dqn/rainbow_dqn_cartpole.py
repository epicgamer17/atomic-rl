"""
Notes on RainbowDQN:
The main idea of the paper was that there had been several incremental improvements to DQN that had been developed, but not combined together. This paper simply combined all of them to achieve state-of-the-art performance on Atari. With a few changes to make certain improvements compatible. The components are:
1. Prioritized Experience Replay
2. Double DQN
3. Dueling Networks
4. Multi-step Learning
5. Distributional RL
6. Noisy Nets
"""

from atomic_rl.initialization import set_seed
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import gymnasium as gym
from typing import Tuple, Dict, Any, Optional
import numpy as np
import random
import wandb
from functools import partial

from atomic_rl.buffers.replay import (
    init_per_buffer,
    sample_per,
    update_priorities,
    circular_write_strategy_,
    with_per_tracking,
    make_n_step_accumulator,
)
from atomic_rl.schedules import get_linear_schedule
from atomic_rl.losses import with_per_weights, cross_entropy_loss
from atomic_rl.td import compute_categorical_q_td_target
from atomic_rl.action_selection import (
    argmax_selector,
    expected_value,
    gather_q_values,
)
from atomic_rl.optimizer import apply_gradients_
from atomic_rl.update_target_net import hard_update_target_network_
from atomic_rl.metrics import log_distributional_metrics
from atomic_rl.utils import (
    to_tensor,
    to_numpy_action,
)
from atomic_rl.networks.noisy_linear import NoisyLinear

# --- Constants ---
BATCH_SIZE = 128
GAMMA = 0.99
LEARNING_RATE = 1e-3  # Rainbow often uses smaller LR, but CartPole is forgiving
MAX_STEPS = 120_000
UPDATE_FREQ = 4
BUFFER_CAPACITY = 50000
MIN_BUFFER_SIZE = 500
TARGET_NET_UPDATE_FREQ = 100  # Less frequent updates for stability with Rainbow
SEED = 42

# PER Constants
ALPHA = 0.6
BETA_START = 0.4
BETA_FRAMES = 100000

# Distributional (C51) Constants
V_MIN = 0.0
V_MAX = 500.0  # CartPole-v1 max episode length
ATOM_SIZE = 51
SUPPORT = torch.linspace(V_MIN, V_MAX, ATOM_SIZE)

# Multi-step
N_STEPS = 3


# Seeding for reproducibility
# TODO/NOTE: figure out if we want to make a network component for these in networks/
class RainbowNetwork(nn.Module):
    """
    Rainbow DQN Network: Combines Dueling architecture, Noisy Nets, and Distributional RL.
    """

    def __init__(self, input_shape: Tuple, num_actions: int, atom_size: int = 51):
        """
        Initialize the Rainbow network.

        Args:
            input_shape (Tuple): The shape of the input observations.
            num_actions (int): The number of possible actions.
            atom_size (int): The number of atoms for the distributional output.
        """
        super().__init__()
        self.input_shape = input_shape
        self.num_actions = num_actions
        self.atom_size = atom_size

        # Shared feature extractor
        self.feature_layer = NoisyLinear(input_shape[0], 512)

        # Dueling Heads: Value and Advantage
        # Both output distributions over atoms
        self.advantage_head = nn.Sequential(
            NoisyLinear(512, 512),
            nn.ReLU(),
            NoisyLinear(512, num_actions * atom_size),
        )
        self.value_head = nn.Sequential(
            NoisyLinear(512, 512),
            nn.ReLU(),
            NoisyLinear(512, atom_size),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the network.

        Args:
            x (torch.Tensor): The input observation tensor.

        Returns:
            torch.Tensor: The output logits of shape (batch, num_actions, atom_size).
        """
        x = F.relu(self.feature_layer(x))

        advantage = self.advantage_head(x).view(-1, self.num_actions, self.atom_size)
        value = self.value_head(x).view(-1, 1, self.atom_size)

        # Dueling combination for distributional RL
        # Q(s,a) = V(s) + (A(s,a) - mean(A(s,a)))
        q_atoms = value + advantage - advantage.mean(dim=1, keepdim=True)

        return q_atoms

    def reset_noise(self):
        """
        Resets noise for all NoisyLinear layers in the network.
        """
        for module in self.modules():
            if isinstance(module, NoisyLinear):
                module.reset_noise()


# --- 1. Initialization (Defining the State) ---
env = gym.make("CartPole-v1")
env = gym.wrappers.RecordEpisodeStatistics(env)
obs_shape = env.observation_space.shape
num_actions = env.action_space.n
device = torch.device("cpu")

model = RainbowNetwork(obs_shape, num_actions, ATOM_SIZE).to(device)
target_model = RainbowNetwork(obs_shape, num_actions, ATOM_SIZE).to(device)
target_model.load_state_dict(model.state_dict())

optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

# Initialize Prioritized Replay Buffer
buffer_state = init_per_buffer(
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
per_add_transition = with_per_tracking(circular_write_strategy_)

# Initialize N-Step Accumulator
accumulate_n_step, reset_accumulator = make_n_step_accumulator(
    n_steps=N_STEPS, gamma=GAMMA
)

obs, info = env.reset(seed=SEED)
rng_key = torch.Generator(device=device)
rng_key.manual_seed(SEED)


# Initialize W&B
wandb.init(
    project="rainbow-dqn-cartpole",
    config={
        "batch_size": BATCH_SIZE,
        "gamma": GAMMA,
        "learning_rate": LEARNING_RATE,
        "buffer_capacity": BUFFER_CAPACITY,
        "n_steps": N_STEPS,
        "atom_size": ATOM_SIZE,
    },
)

# --- 2. The Monolithic Loop ---
for step in range(MAX_STEPS):

    # 1. Act (Noisy Nets handle exploration, no epsilon needed)
    with torch.inference_mode():
        obs_tensor = to_tensor(obs[None, ...], device=device)

        # Resample noise for the actor
        model.reset_noise()

        predictions = model(obs_tensor)
        expected_qs = expected_value(predictions, support=SUPPORT.to(device))
        action, _ = argmax_selector(expected_qs)
        action_np = to_numpy_action(action)

    # 2. Step Env
    # Extract the scalar for a non-vectorized Gymnasium environment
    action_int = int(action_np.item())
    next_obs, reward, terminated, truncated, info = env.step(action_int)

    # 3. N-Step Accumulation and Buffer Addition
    # TODO: URGENT. Creating a new tensor every step in the hotloop. should do in a more efficient way, maybe some thing like the rollout buffer in PPO (possible reuse?) ie store in a rollout buffer before sending to main replay buffer. Idea being its pre allocated basically. Must consider the N-Step case.
    # The accumulator now returns a list of 0, 1, or N transitions.
    n_step_transitions = accumulate_n_step(
        to_tensor(obs).unsqueeze(0),
        action,
        to_tensor([reward]),
        to_tensor(next_obs).unsqueeze(0),
        to_tensor([terminated]),
        to_tensor([truncated]),
    )

    # If the accumulator yielded any transitions, write them as a single batch
    # TODO: this is a bit yuckier than the list but also more efficient. maybe update.
    if n_step_transitions.batch_size[0] > 0:
        buffer_state = per_add_transition(buffer_state, n_step_transitions)

    # Update state
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
        reset_accumulator()

    # --- 3. Update Loop ---
    if step > MIN_BUFFER_SIZE and step % UPDATE_FREQ == 0:
        # Anneal PER Beta
        beta = get_linear_schedule(step, BETA_START, 1.0, BETA_FRAMES)
        beta_tensor = torch.tensor(beta, dtype=torch.float32, device=device)

        # 2. Sample with Prioritization
        batch, tree_indices, is_weights = sample_per(
            buffer_state, BATCH_SIZE, beta=beta_tensor
        )

        # Resample noise for training step
        model.reset_noise()
        target_model.reset_noise()

        # 1. Forward Passes (Online and Target)
        logits = model(batch["obs"])
        with torch.no_grad():
            # Double DQN: Online model selects action
            next_logits_online = model(batch["next_obs"])
            # Target model evaluates the chosen action
            next_logits_target = target_model(batch["next_obs"])

            # 2. Next Action Selection (Composed inline natively)
            next_expected_qs = expected_value(
                next_logits_online, support=SUPPORT.to(device)
            )
            next_actions, _ = argmax_selector(next_expected_qs)

            # 3. Target Calculation (Categorical TD target)
            td_target = compute_categorical_q_td_target(
                next_logits_target,
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

        # 5. Loss Calculation (PER weighted Cross Entropy)
        per_loss_fn = with_per_weights(cross_entropy_loss, is_weights)
        loss, info_dict = per_loss_fn(pred_sa_logits, td_target)

        # Apply Gradients
        optimizer = apply_gradients_(optimizer, loss)

        # Update PER Priorities
        buffer_state = update_priorities(
            buffer_state, tree_indices, info_dict["priorities"], alpha=ALPHA
        )

        if step % 100 == 0:
            # We log the info_dict directly. W&B handles scalars and histograms of tensors.
            # We exclude 'predictions' and 'priorities' from the direct log to handle them specially.
            log_dict = {
                k: v
                for k, v in info_dict.items()
                if k not in ["predictions", "priorities"]
            }
            log_dict.update({"beta": beta})

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
