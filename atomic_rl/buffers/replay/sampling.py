from ...initialization import _allocate_tensordict
from .state import BufferState, PERBufferState
from .write import compute_is_weights
import torch
from tensordict import TensorDict
from typing import Tuple, Callable, List, Optional, Union
import random
from collections import deque
from dataclasses import dataclass


def uniform_sample(
    buffer_state: BufferState, rng_key: torch.Generator, batch_size: int
) -> TensorDict:
    """Uniform sampling."""
    if buffer_state.size < batch_size:
        raise ValueError(
            f"Buffer size ({buffer_state.size}) is smaller than batch size ({batch_size})."
        )
    indices = torch.randint(0, buffer_state.size, (batch_size,), generator=rng_key)
    return buffer_state.data[indices]


# TODO: unify sample API and make sample_per use and RNG key
def sample_per(
    buffer_state: PERBufferState, batch_size: int, beta: torch.Tensor
) -> Tuple[TensorDict, torch.Tensor, torch.Tensor]:
    """
    Prioritized Experience Replay sampling.

    Args:
        buffer_state (PERBufferState): The PER buffer state.
        batch_size (int): The size of the batch to sample.
        beta (torch.Tensor): The beta parameter for PER.
            Using a tensor for beta avoids torch.compile recompilation.

    Returns:
        Tuple[TensorDict, torch.Tensor, torch.Tensor]: The sampled batch, tree indices, and importance weights.
    """
    total_priority = buffer_state.sum_tree[0]

    # Generate batch_size random values between 0 and total_priority
    segment_length = total_priority / batch_size
    targets = (
        torch.rand(batch_size, device=buffer_state.sum_tree.device)
        + torch.arange(batch_size, device=buffer_state.sum_tree.device)
    ) * segment_length

    # Vectorized Tree Traversal
    indices = buffer_state.sum_tree.new_zeros(batch_size, dtype=torch.long)

    # Depth of tree is log2(capacity). We loop exactly this many times.
    import math

    depth = int(math.log2(buffer_state.tree_capacity))

    for _ in range(depth):
        left_children = indices * 2 + 1
        right_children = indices * 2 + 2

        left_priorities = buffer_state.sum_tree[left_children]

        # If target > left_priority, go right and subtract left_priority
        go_right = targets > left_priorities

        targets = torch.where(go_right, targets - left_priorities, targets)
        indices = torch.where(go_right, right_children, left_children)

    # 'indices' are now the tree indices (leaves). We need the data indices.
    data_indices = indices - (buffer_state.tree_capacity - 1)

    # Clamp to handle potential precision issues leading to out-of-bounds indices
    # TODO/NOTE: If an out-of-bounds index is sampled, mathematically it implies a precision error; you should resample or fall back to the last valid tree traversal step rather than clamping.
    data_indices = torch.clamp(data_indices, 0, buffer_state.capacity - 1)

    # Importance Sampling (IS) Weights
    leaf_priorities = buffer_state.sum_tree[indices]
    min_prob = buffer_state.min_tree[0] / total_priority

    is_weights = compute_is_weights(leaf_priorities, min_prob, total_priority, beta)

    batch = buffer_state.data[data_indices]

    return batch, indices, is_weights
