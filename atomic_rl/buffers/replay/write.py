from ...initialization import _allocate_tensordict
from .state import BufferState, PERBufferState
import torch
from tensordict import TensorDict
from typing import Tuple, Callable, List, Optional, Union
import random
from collections import deque
from dataclasses import dataclass


def circular_write_strategy_(
    buffer_state: BufferState, batch: TensorDict
) -> Tuple[BufferState, torch.Tensor]:
    """Standard circular writing."""
    batch_size = batch.batch_size[0]
    start_idx = buffer_state.pointer
    end_idx = start_idx + batch_size

    if end_idx <= buffer_state.capacity:
        indices = torch.arange(start_idx, end_idx, dtype=torch.long)
        buffer_state.data[start_idx:end_idx] = batch
    else:
        overflow = end_idx - buffer_state.capacity
        first_chunk_size = buffer_state.capacity - start_idx
        indices = torch.cat(
            [
                torch.arange(start_idx, buffer_state.capacity, dtype=torch.long),
                torch.arange(0, overflow, dtype=torch.long),
            ]
        )
        buffer_state.data[start_idx : buffer_state.capacity] = batch[:first_chunk_size]
        buffer_state.data[0:overflow] = batch[first_chunk_size:]

    buffer_state.pointer = end_idx % buffer_state.capacity
    buffer_state.size = min(buffer_state.size + batch_size, buffer_state.capacity)
    buffer_state.steps_seen += batch_size
    return buffer_state, indices


def reservoir_write_strategy_(
    buffer_state: BufferState, batch: TensorDict
) -> Tuple[BufferState, torch.Tensor]:
    """
    Writes data using Reservoir Sampling (uniform probability over infinite stream).
    Always expects batched inputs.

    Args:
        buffer_state (BufferState): The buffer state.
        batch (TensorDict): A TensorDict of batched tensors to write.

    Returns:
        Tuple[BufferState, torch.Tensor]: The updated buffer state and indices.
    """
    batch_size = batch.batch_size[0]

    written_indices = []
    for i in range(batch_size):
        # 1. Decide if and where to write
        if buffer_state.steps_seen < buffer_state.capacity:
            idx = buffer_state.steps_seen
        else:
            # Standard reservoir math: keep item with probability (capacity / steps_seen)
            j = random.randint(0, buffer_state.steps_seen)
            if j < buffer_state.capacity:
                idx = j
            else:
                idx = None

        if idx is not None:
            # 2. Write to TensorDict
            buffer_state.data[idx] = batch[i]
            written_indices.append(idx)

        buffer_state.steps_seen += 1
        buffer_state.size = min(buffer_state.steps_seen, buffer_state.capacity)

    indices_tensor = (
        torch.tensor(written_indices, dtype=torch.long)
        if written_indices
        else torch.empty(0, dtype=torch.long)
    )
    return buffer_state, indices_tensor


def with_per_tracking(write_strategy_fn: Callable) -> Callable:
    """
    Higher-order function composing a base writing strategy with PER logic.
    """

    def per_add(buffer_state: PERBufferState, batch: TensorDict) -> PERBufferState:
        # 1. Execute the base writing strategy
        new_state, written_indices = write_strategy_fn(buffer_state, batch)

        # 2. If data was actually written, update the PER sum/min trees
        if written_indices is not None and written_indices.numel() > 0:
            tree_indices = written_indices + new_state.tree_capacity - 1

            # Check for explicit priorities in the batch
            if "priority" in batch.keys():
                priorities = batch["priority"]
                if priorities.ndim == 2:
                    priorities = priorities.squeeze(-1)
            else:
                # Use max priority for all new additions
                priorities = torch.full(
                    (written_indices.shape[0],),
                    new_state.max_priority,
                    dtype=torch.float32,
                    device=written_indices.device,
                )

            new_sum_tree, new_min_tree = _update_tree(
                new_state.sum_tree, new_state.min_tree, tree_indices, priorities
            )

            new_state.sum_tree = new_sum_tree
            new_state.min_tree = new_min_tree
            new_state.max_priority = torch.maximum(
                new_state.max_priority, torch.max(priorities)
            )

        return new_state

    return per_add


def compute_is_weights(
    leaf_priorities: torch.Tensor,
    min_prob: Union[float, torch.Tensor],
    total_priority: Union[float, torch.Tensor],
    beta: torch.Tensor,
) -> torch.Tensor:
    """
    Computes Importance Sampling (IS) weights for Prioritized Experience Replay (PER).

    IS weights correct for the bias introduced by non-uniform sampling in PER.
    Mathematically: $w_i = ( (1/N) * (1/P_i) )^\beta / \max_j w_j$, where $P_i$ is the
    sampling probability of transition $i$. This simplifies to:
    $w_i = ( P_i / \min_j P_j )^{-\beta}$.

    Args:
        leaf_priorities (torch.Tensor): The priority values of the sampled transitions.
        min_prob (Union[float, torch.Tensor]): The minimum sampling probability in the tree.
        total_priority (Union[float, torch.Tensor]): The sum of all priorities in the tree.
        beta (torch.Tensor): The correction coefficient (usually scheduled from 0.4 to 1.0).

    Returns:
        torch.Tensor: The normalized IS weights for the batch.
    """
    # Sampling probability: P_i = priority_i / sum(priorities)
    probs = leaf_priorities / total_priority
    # IS weight: (P_i / min_P)^-beta
    # This simplifies the (1/N * 1/P_i)^beta / max_w normalization
    is_weights = torch.pow(probs / min_prob, -beta)
    return is_weights


@torch.inference_mode()
def _update_tree(
    sum_tree: torch.Tensor,
    min_tree: torch.Tensor,
    tree_indices: torch.Tensor,
    priorities: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Updates the tree nodes and propagates up to the root.
    """
    # This modifies the tensors in-place for speed, but maintains a functional signature
    sum_tree[tree_indices] = priorities
    min_tree[tree_indices] = priorities

    # Propagate up
    parent_indices = (tree_indices - 1) // 2

    while parent_indices.numel() > 0 and (parent_indices >= 0).all():
        left_children = parent_indices * 2 + 1
        right_children = parent_indices * 2 + 2

        sum_tree[parent_indices] = sum_tree[left_children] + sum_tree[right_children]
        min_tree[parent_indices] = torch.minimum(
            min_tree[left_children], min_tree[right_children]
        )

        parent_indices = (parent_indices - 1) // 2

    return sum_tree, min_tree


def update_priorities(
    buffer_state: PERBufferState,
    tree_indices: torch.Tensor,
    td_errors: torch.Tensor,
    alpha: float = 0.6,
) -> PERBufferState:
    """
    Update the PER sum/min trees with the TD errors.
    """
    # Add epsilon to prevent zero priority
    priorities = torch.pow(torch.abs(td_errors) + 1e-6, alpha)

    new_sum_tree, new_min_tree = _update_tree(
        buffer_state.sum_tree, buffer_state.min_tree, tree_indices, priorities
    )

    new_max_priority = torch.maximum(buffer_state.max_priority, torch.max(priorities))

    return PERBufferState(
        data=buffer_state.data,
        sum_tree=new_sum_tree,
        min_tree=new_min_tree,
        max_priority=new_max_priority,
        pointer=buffer_state.pointer,
        size=buffer_state.size,
        capacity=buffer_state.capacity,
        tree_capacity=buffer_state.tree_capacity,
        steps_seen=buffer_state.steps_seen,
    )
