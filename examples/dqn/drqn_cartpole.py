"""
Notes on DRQN + R2D2 Replay:
Goal: Handle partially observable MDPs (or just sequential state dependencies).
Key Ideas:
- Replaces standard MLP with an LSTM to integrate information over time.
- NOTE: Unlike the original DRQN paper we do not store individual transitions, instead we use an approach similar to R2D2's where we store overlapping sequences and perform a burn-in period per sequence. We also store old hidden states, which is a better start for the burn in that an fresh LSTM state, and also acts as a regularizer for the learning.
"""

# TODO: results seem slightly noisier/worse than normal DQN. in some ways like my PPO+lstm results. should add a comparison on the flickering env of DQN and DRQN.

from functional.initialization import layer_init, set_seed
import random
from typing import Tuple, Optional

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import wandb
from tensordict import TensorDict

from functional.action_selection import (
    argmax_selector,
    gather_q_values,
    with_epsilon_greedy,
)
from functional.losses import mse_loss, with_sequence_mask
from functional.network import hard_update_target_network_
from functional.replay_buffer import (
    circular_write_strategy,
    init_buffer,
    make_padded_chunk_accumulator,
    uniform_sample,
)
from functional.schedules import get_linear_schedule
from functional.td import compute_q_td_target
from functional.utils import to_tensor

# Constants
BATCH_SIZE = 32
GAMMA = 0.99
EPS_START = 1.0
EPS_END = 0.05
EPS_DECAY_FRAMES = 20000
LEARNING_RATE = 1e-4
MAX_STEPS = 100_000
UPDATE_FREQ = 4
TARGET_NET_UPDATE_FREQ = 1000
BUFFER_CAPACITY = 5000  # Number of sequences
MIN_BUFFER_SIZE = 500  # TODO: is this in sequences or in transitions?
SEED = 42

# R2D2 / DRQN Specific Constants
SEQ_LENGTH = 20
BURN_IN_STEPS = 10
HIDDEN_SIZE = 128

# Seeding for reproducibility
set_seed(SEED)


class RecurrentDQN(nn.Module):
    """
    Recurrent Q-Network (DRQN) using an LSTM backbone.
    """

    def __init__(self, obs_dim: int, action_dim: int, hidden_size: int = HIDDEN_SIZE):
        super().__init__()
        self.hidden_size = hidden_size
        self.feature_extractor = nn.Sequential(
            layer_init(nn.Linear(obs_dim, hidden_size)),
            nn.ReLU(),
        )

        # Using batch_first=True makes shape tracking (B, T, Features) much easier
        self.lstm = nn.LSTM(hidden_size, hidden_size, batch_first=True)
        self.q_head = layer_init(nn.Linear(hidden_size, action_dim), std=1.0)

    def forward(
        self,
        obs_sequence: torch.Tensor,
        hidden_state: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Forward pass for a sequence of observations.

        Args:
            obs_sequence: Tensor of shape (Batch, Time, ObsDim).
            hidden_state: Tuple of (hx, cx) tensors.

        Returns:
            q_values: Tensor of shape (Batch, Time, ActionDim)
            new_hidden_state: Tuple of updated (hx, cx)
        """
        batch_size, seq_len, _ = obs_sequence.shape

        # (B, T, ObsDim) -> (B*T, HiddenSize)
        flat_obs = obs_sequence.flatten(0, 1)
        features = self.feature_extractor(flat_obs)
        features = features.view(batch_size, seq_len, -1)

        # lstm_out: (B, T, HiddenSize)
        lstm_out, new_hidden_state = self.lstm(features, hidden_state)

        # (B, T, HiddenSize) -> (B, T, ActionDim)
        q_values = self.q_head(lstm_out)

        return q_values, new_hidden_state


def make_env(env_id: str, seed: int):
    """Utility to create and seed the environment."""
    env = gym.make(env_id)
    env = gym.wrappers.RecordEpisodeStatistics(env)
    env.action_space.seed(seed)
    env.observation_space.seed(seed)
    return env


def train():
    """Main training loop for DRQN on CartPole."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    env = make_env("CartPole-v1", SEED)
    obs_dim = env.observation_space.shape[0]
    num_actions = env.action_space.n

    # Initialize Networks
    model = RecurrentDQN(obs_dim, num_actions).to(device)
    target_model = RecurrentDQN(obs_dim, num_actions).to(device)
    target_model.load_state_dict(model.state_dict())

    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # Initialize Replay Buffer (Stores Sequences)
    buffer_state = init_buffer(
        capacity=BUFFER_CAPACITY,
        shapes={
            "observations": (SEQ_LENGTH, obs_dim),
            "actions": (SEQ_LENGTH, 1),
            "rewards": (SEQ_LENGTH,),
            "terminated": (SEQ_LENGTH,),
            "truncated": (SEQ_LENGTH,),
            "valid": (SEQ_LENGTH,),
            "hx": (SEQ_LENGTH, 1, HIDDEN_SIZE),
            "cx": (SEQ_LENGTH, 1, HIDDEN_SIZE),
        },
        device=device,
    )

    # Accumulator to gather transitions into chunks
    accumulator = make_padded_chunk_accumulator(
        chunk_size=SEQ_LENGTH,
        num_envs=1,
        shapes={
            "observations": (obs_dim,),
            "actions": (1,),
            "rewards": (),
            "terminated": (),
            "truncated": (),
            "hx": (1, HIDDEN_SIZE),
            "cx": (1, HIDDEN_SIZE),
        },
        device=device,
        overlap=BURN_IN_STEPS,
    )

    # Tracking State
    obs, info = env.reset(seed=SEED)
    hx = torch.zeros(1, 1, HIDDEN_SIZE, device=device)
    cx = torch.zeros(1, 1, HIDDEN_SIZE, device=device)
    hidden_state = (hx, cx)

    rng_key = torch.Generator(device="cpu")
    rng_key.manual_seed(SEED)

    action_selector = with_epsilon_greedy(argmax_selector)

    # Initialize W&B
    wandb.init(
        project="drqn-cartpole",
        config={
            "batch_size": BATCH_SIZE,
            "gamma": GAMMA,
            "learning_rate": LEARNING_RATE,
            "seq_length": SEQ_LENGTH,
            "burn_in_steps": BURN_IN_STEPS,
            "buffer_capacity": BUFFER_CAPACITY,
        },
    )

    for step in range(MAX_STEPS):
        current_epsilon = get_linear_schedule(
            step, EPS_START, EPS_END, EPS_DECAY_FRAMES
        )

        # 1. Act
        with torch.inference_mode():
            # (Batch=1, Time=1, ObsDim)
            obs_tensor = to_tensor(obs[None, None, ...], device=device)
            q_values, next_hidden_state = model(obs_tensor, hidden_state)

            action, select_info = action_selector(
                predictions=q_values.squeeze(1).cpu(),  # Move to CPU for selection
                epsilon=current_epsilon,
                num_actions=num_actions,
                generator=rng_key,
            )
            rng_key = select_info["generator"]

        # 2. Step Env
        # Convert the action tensor to a numpy array at the environment boundary for standard Gym.
        # We index the batch dimension instead of using .item() to support vectorized environments.
        action_int = int(action.cpu().numpy()[0])
        next_obs, reward, terminated, truncated, info = env.step(action_int)

        # 3. Accumulate Sequences
        # TODO: are there helpers i should use here?
        # TODO: should i make allocating a tensor/transition into a helper? it seems like we do it very often and its a bit of a pain. also could allow for that preallocation trick to not make a new tensor dict every step.
        transition = TensorDict(
            {
                "observations": to_tensor(obs, device=device),
                "actions": action.squeeze(0),
                "rewards": torch.tensor(reward, dtype=torch.float32, device=device),
                "terminated": torch.tensor(
                    terminated, dtype=torch.float32, device=device
                ),
                "truncated": torch.tensor(
                    truncated, dtype=torch.float32, device=device
                ),
                "hx": hidden_state[0].squeeze(0),
                "cx": hidden_state[1].squeeze(0),
            },
            batch_size=[],
        )

        ready_chunks = accumulator(transition.unsqueeze(0))

        if ready_chunks.batch_size[0] > 0:
            # Write full sequences to buffer
            buffer_state, _ = circular_write_strategy(buffer_state, ready_chunks)

        # Update Recurrent State
        obs = next_obs
        hidden_state = next_hidden_state

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
            # Reset hidden state for new episode
            hidden_state = (
                torch.zeros(1, 1, HIDDEN_SIZE, device=device),
                torch.zeros(1, 1, HIDDEN_SIZE, device=device),
            )

        # 4. Training Phase
        if step > MIN_BUFFER_SIZE and step % UPDATE_FREQ == 0:
            batch = uniform_sample(buffer_state, rng_key, BATCH_SIZE)

            # Sequence Q-learning update
            # We use the stored hidden states from the START of the sequence
            # batch["hx"] shape: [BATCH_SIZE, SEQ_LENGTH, 1, HIDDEN_SIZE]
            b_hx = batch["hx"][:, 0].transpose(0, 1).contiguous().detach()
            b_cx = batch["cx"][:, 0].transpose(0, 1).contiguous().detach()

            # Online pass
            # We unroll through the entire sequence to get all Q-values
            q_all, _ = model(batch["observations"], (b_hx, b_cx))
            # Current Q-values for steps 0 to SEQ_LENGTH-2
            q_pred_flat = gather_q_values(
                q_all[:, :-1].flatten(0, 1),
                batch["actions"][:, :-1].flatten(),
            )

            with torch.no_grad():
                # Target pass (use the same initial hidden state for target net)
                target_q_all, _ = target_model(batch["observations"], (b_hx, b_cx))
                # Target values for steps 1 to SEQ_LENGTH-1 (next states)
                next_q_values_flat = target_q_all[:, 1:].flatten(0, 1)

                # Greedy action selection from target network
                next_actions_flat, _ = argmax_selector(next_q_values_flat)

                # Compute TD Targets [B * (T-1)]
                td_targets_flat = compute_q_td_target(
                    next_q_values=next_q_values_flat,
                    next_actions=next_actions_flat.squeeze(-1),
                    rewards=batch["rewards"][:, :-1].flatten(),
                    terminated=batch["terminated"][:, :-1].flatten(),
                    gamma=torch.tensor(GAMMA, device=device),
                )

            # Masking: Valid transitions AND skip burn-in steps
            # A transition at step t is valid if both s_t and s_{t+1} were real data.
            # However, if s_t was terminal, we still use the transition (next_value is masked).
            # So we just need to ensure s_t was valid.
            valid_mask = batch["valid"][:, :-1].clone()
            if BURN_IN_STEPS > 0:
                valid_mask[:, :BURN_IN_STEPS] = False
            flat_mask = valid_mask.flatten()

            # Use sequence-aware loss
            loss_fn = with_sequence_mask(mse_loss, flat_mask)
            loss, info_dict = loss_fn(q_pred_flat, td_targets_flat)
            loss = loss.mean()

            # Optimize
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
            optimizer.step()

            if step % 100 == 0:
                info_dict.update(
                    {
                        "loss/total": loss.item(),
                        "epsilon": current_epsilon,
                        "q_values/mean": q_pred_flat[flat_mask.to(torch.bool)]
                        .mean()
                        .item(),
                        "td_targets/mean": td_targets_flat[flat_mask.to(torch.bool)]
                        .mean()
                        .item(),
                    }
                )
                wandb.log(info_dict, step=step)

        # 5. Target Network Update
        if step % TARGET_NET_UPDATE_FREQ == 0:
            hard_update_target_network_(model, target_model)

    wandb.finish()


if __name__ == "__main__":
    train()
