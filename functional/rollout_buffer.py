import torch
from tensordict import TensorDict
from typing import Tuple, Callable, List, Optional, Union
from dataclasses import dataclass, field
import numpy as np
from .replay_buffer import _allocate_tensordict


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
    info: dict,
    truncated: Union[np.ndarray, torch.Tensor],
) -> None:
    """Checks the info dict for final observations (from Gymnasium auto-resets)
    and records them in the buffer's truncation_records list ONLY if they were truncated.
    """
    from .utils import extract_vector_env_final_obs

    env_indices, final_obs = extract_vector_env_final_obs(info)
    for i, env_idx in enumerate(env_indices):
        if truncated[env_idx]:
            obs_tensor = torch.as_tensor(final_obs[i], dtype=torch.float32)
            buffer.truncation_records.append((step, int(env_idx), obs_tensor))


def get_rollout_next_values(
    buffer: RolloutBufferState,
    last_values: torch.Tensor,
    get_value_fn: Callable[[torch.Tensor], torch.Tensor],
    device: torch.device,
) -> torch.Tensor:
    """Assembles patched next_values [B, T]."""
    B, T = buffer.data.batch_size
    values = buffer.data["values"]

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
    total_size = batch.batch_size[0]
    indices = torch.randperm(total_size, generator=generator, device=batch.device)

    for start_idx in range(0, total_size, minibatch_size):
        end_idx = min(start_idx + minibatch_size, total_size)
        mb_indices = indices[start_idx:end_idx]
        yield batch[mb_indices]
