from ...initialization import _allocate_tensordict
import torch
from tensordict import TensorDict
from typing import Tuple, Callable, List, Optional, Union
from dataclasses import dataclass, field
import numpy as np


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
