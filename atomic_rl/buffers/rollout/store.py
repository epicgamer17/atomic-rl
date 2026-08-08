from ...initialization import _allocate_tensordict
from .state import RolloutBufferState
import torch
from tensordict import TensorDict
from typing import Tuple, Callable, List, Optional, Union
from dataclasses import dataclass, field
import numpy as np


# TODO: should store.py be renamed to write.py (and the function names too) to be more consistent with our replay buffer organization?


def store_rollout_step_(
    buffer: RolloutBufferState, step: int, transition: TensorDict
) -> None:
    """Stores a transition batch into the rollout buffer."""
    # FIX: Write across all batches [:, step] instead of [step]
    buffer.data[:, step] = transition


def record_truncations_(
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
        buffer.truncation_records.append((step, int(env_idx), final_observations[i]))
