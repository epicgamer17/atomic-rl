from ...initialization import _allocate_tensordict
import torch
from tensordict import TensorDict
from typing import Tuple, Callable, List, Optional, Union
import random
from collections import deque
from dataclasses import dataclass


@dataclass(kw_only=True)
class BufferState:
    """
    State of a standard replay buffer.

    Attributes:
        data (TensorDict): The stored transitions.
        pointer (int): Current write position.
        size (int): Number of items currently in the buffer.
        capacity (int): Maximum number of items the buffer can hold.
        steps_seen (int): Total number of transitions processed by this buffer.
            Useful for Reservoir Sampling logic.
    """

    data: TensorDict
    pointer: int
    size: int
    capacity: int
    steps_seen: int = 0


def init_buffer(
    capacity: int, shapes: dict, device: Optional[torch.device] = None
) -> BufferState:
    """Initializes a standard buffer."""
    data = _allocate_tensordict(shapes, [capacity], device)
    return BufferState(data=data, pointer=0, size=0, capacity=capacity, steps_seen=0)


@dataclass
class PERBufferState(BufferState):
    """
    State of a Prioritized Experience Replay (PER) buffer.
    """

    sum_tree: torch.Tensor
    min_tree: torch.Tensor
    max_priority: float
    tree_capacity: int


def init_per_buffer(
    capacity: int, shapes: dict, device: Optional[torch.device] = None
) -> PERBufferState:
    """Initializes a PER buffer."""
    tree_capacity = 1 << (capacity - 1).bit_length() if capacity > 0 else 1
    empty_data = _allocate_tensordict(shapes, [capacity], device)
    return PERBufferState(
        data=empty_data,
        sum_tree=torch.zeros(2 * tree_capacity - 1, device=device),
        min_tree=torch.full((2 * tree_capacity - 1,), float("inf"), device=device),
        max_priority=torch.tensor(1.0, device=device),
        pointer=0,
        size=0,
        capacity=capacity,
        tree_capacity=tree_capacity,
        steps_seen=0,
    )
