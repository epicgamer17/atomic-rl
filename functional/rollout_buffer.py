import torch
from tensordict import TensorDict
from typing import Tuple, Callable, List, Optional, Union
from dataclasses import dataclass, field
import numpy as np
from einops import rearrange
from .utils import _allocate_tensordict


@dataclass
class RolloutBufferState:
    """
    State of a rollout buffer used for on-policy algorithms (PPO, A2C).

    Attributes:
        data (TensorDict): The stored transitions with geometry [Batch, Time].
        truncation_records (List[Tuple[int, int, torch.Tensor]]): Records of
            truncations and their final observations for patching next_values.
    """

    data: TensorDict
    truncation_records: List[Tuple[int, int, torch.Tensor]] = field(
        default_factory=list
    )


def init_rollout_buffer(
    steps_per_env: int, num_envs: int, shapes: dict, device: str = "cpu"
) -> RolloutBufferState:
    """Initializes a rollout buffer [Batch, Time]."""
    # FIX: Swap the order of the batch size allocation
    data = _allocate_tensordict(shapes, [num_envs, steps_per_env], device)
    return RolloutBufferState(data=data, truncation_records=[])


def store_rollout_step(
    buffer: RolloutBufferState, step: int, transition: TensorDict
) -> None:
    """Stores a transition batch into the rollout buffer."""
    # FIX: Write across all batches [:, step] instead of [step]
    buffer.data[:, step] = transition


def record_truncations(
    buffer: RolloutBufferState,
    step: int,
    truncated_envs: torch.Tensor,
    final_observations: torch.Tensor,
) -> None:
    """
    Records final observations for truncated environments.
    Decoupled from Gymnasium's info dictionary structure.

    Args:
        buffer (RolloutBufferState): The rollout buffer state.
        step (int): The current rollout step.
        truncated_envs (torch.Tensor): 1D tensor of environment indices that were truncated.
        final_observations (torch.Tensor): 2D tensor of final observations for those environments.
    """
    for i, env_idx in enumerate(truncated_envs):
        buffer.truncation_records.append((step, env_idx.item(), final_observations[i]))


def get_rollout_next_values(
    buffer: RolloutBufferState,
    last_values: torch.Tensor,
    get_value_fn: Callable[[torch.Tensor], torch.Tensor],
    device: torch.device,
) -> torch.Tensor:
    """Assembles patched next_values [B, T]."""
    B, T = buffer.data.batch_size
    values = buffer.data["values"]

    if "truncated" in buffer.data.keys():
        truncated_steps = buffer.data["truncated"].bool().nonzero(as_tuple=False)
        if truncated_steps.numel() > 0:
            expected_records = {
                (int(step.item()), int(env_idx.item()))
                for env_idx, step in truncated_steps
            }
            actual_records = {
                (int(step), int(env_idx))
                for step, env_idx, _ in buffer.truncation_records
            }
            missing_records = expected_records - actual_records
            if missing_records:
                missing_preview = sorted(missing_records)[:5]
                raise RuntimeError(
                    "Missing final observations for truncated rollout steps. "
                    "Cannot safely bootstrap truncated states because vector env "
                    "autoreset may expose reset observations instead. "
                    f"Missing (step, env_idx) records: {missing_preview}"
                )

    if last_values.ndim == 1:
        last_values = last_values.unsqueeze(1)

    # next_values[b, t] should be values[b, t+1] for t < T-1, and last_values[b] for t = T-1
    next_values = torch.cat([values[:, 1:], last_values], dim=1)

    if buffer.truncation_records:
        obs_batch = torch.stack([r[2] for r in buffer.truncation_records]).to(device)
        with torch.inference_mode():
            v_patch = get_value_fn(obs_batch).squeeze(-1)

        for i, (step, env_idx, _) in enumerate(buffer.truncation_records):
            next_values[env_idx, step] = v_patch[i]

    return next_values


def flatten_rollout_buffer(buffer: RolloutBufferState) -> TensorDict:
    """Flattens [B, T] into [B*T]."""
    return buffer.data.flatten(0, 1)


# TODO: this is both an offline (replay buffer) and online (rollout buffer) sampling method, can we make it more general? can we unify the files? what is happening in one that really isnt happening in the other file?
def yield_shuffled_minibatches(
    batch: TensorDict, minibatch_size: int, generator: torch.Generator = None
):
    """
    Yields shuffled minibatches from a flattened TensorDict.
    Used in PPO, SAC, Behaviour Cloning, and Offline RL (Efficient Zero, MuZero Unplugged, etc).
    """
    assert (
        len(batch.batch_size) == 1
    ), f"Expected 1D batched data, got {batch.batch_size}. Call flatten() first."

    total_size = batch.batch_size[0]
    indices = torch.randperm(total_size, generator=generator, device=batch.device)

    for start_idx in range(0, total_size, minibatch_size):
        end_idx = min(start_idx + minibatch_size, total_size)
        mb_indices = indices[start_idx:end_idx]
        yield batch[mb_indices]


# TODO: is it possible to merge the two yield mini batch functions? is that a good idea?
def yield_sequential_minibatches(
    buffer_data: TensorDict,
    num_envs: int,
    num_minibatches: int,
    initial_lstm_states: Tuple[torch.Tensor, torch.Tensor],
    generator: torch.Generator = None,
):
    """
    Yields mini-batches preserving the sequential order of time steps.
    Instead of shuffling individual steps, it shuffles entire environments
    and yields their full unrolled sequences.

    Args:
        buffer_data: TensorDict of shape [envs, steps, ...]
        num_envs: Total number of environments
        num_minibatches: How many mini-batches to split the environments into
        initial_lstm_states: Tuple of (hidden, cell) states from the start of the rollout.
            Expected shape: [layers, envs, hidden_dim]
        generator: Optional torch generator for reproducibility.

    Yields:
        mb_flat: Flattened TensorDict of shape [steps * envs_per_batch, ...]
        (mb_initial_h, mb_initial_c): Initial LSTM states for the selected environments.
    """
    assert (
        num_envs % num_minibatches == 0
    ), f"num_envs ({num_envs}) must be divisible by num_minibatches ({num_minibatches}) for LSTM PPO"
    assert (
        len(buffer_data.batch_size) == 2
    ), f"Expected [envs, steps] data, got {buffer_data.batch_size}."

    envs_per_batch = num_envs // num_minibatches

    # Shuffle environment indices, NOT time indices
    env_indices = torch.randperm(
        num_envs, generator=generator, device=buffer_data.device
    )

    for start in range(0, num_envs, envs_per_batch):
        end = start + envs_per_batch
        mb_env_inds = env_indices[start:end]

        # 1. Select the initial LSTM states for these specific environments
        # All states are typically [layers, envs, hidden_dim]
        mb_initial_states = tuple(s[:, mb_env_inds].detach() for s in initial_lstm_states)

        # 2. Slice the buffer data for these environments across ALL steps
        # buffer_data is [envs, steps, ...] -> slice to [envs_per_batch, steps, ...]
        mb_data = buffer_data[mb_env_inds, :]
        steps = mb_data.batch_size[1]

        # 3. Flatten the sequence and environment dimensions to [steps * envs_per_batch, ...]
        # We use "b t ... -> (t b) ..." so that when reshaped to (T, B), it stays sequential
        # Output shape: [T * B_per_batch, ...]
        # NOTE: einops doesn't support TensorDict directly, so we use .apply()
        mb_flat = mb_data.apply(
            lambda x: rearrange(x, "b t ... -> (t b) ..."),
            batch_size=[steps * envs_per_batch],
        )

        yield mb_flat, mb_initial_states
