import torch
from dataclasses import dataclass
from tensordict import TensorDict
import random
import numpy as np
from einops import rearrange
from typing import Tuple, Callable, List, Dict, Any, Optional, Union



@dataclass(kw_only=True)
class BufferState:
    data: TensorDict
    pointer: int
    size: int
    capacity: int


@dataclass
class PERBufferState(BufferState):
    sum_tree: torch.Tensor  # 1D Tensor of shape [2 * tree_capacity - 1]
    min_tree: torch.Tensor  # For calculating max IS weights
    max_priority: float  # Track max priority for new additions
    tree_capacity: int  # Smallest power of 2 >= capacity


@dataclass
class ReservoirBufferState(BufferState):
    total_steps_seen: int


def init_buffer(capacity: int, shapes: dict, device: str = "cpu") -> BufferState:
    """
    Initializes a buffer for storing transitions.

    Args:
        capacity (int): The maximum number of transitions the buffer can store.
        shapes (dict): A dictionary where keys are the names of the transition components (e.g., 'obs', 'action', 'reward') and values are their shapes (e.g., (4,), (1,), (1,)).
        device (str, optional): The device on which to store the buffer ('cpu' or 'cuda'). Defaults to 'cpu'.

    Returns:
        BufferState: An instance of BufferState containing the initialized buffer.
    """
    empty_data = TensorDict({}, batch_size=[capacity], device=device)
    for key, shape in shapes.items():
        empty_data.set(key, torch.zeros((capacity, *shape), dtype=torch.float32))
    return BufferState(data=empty_data, pointer=0, size=0, capacity=capacity)


def init_per_buffer(capacity: int, shapes: dict, device="cpu") -> PERBufferState:
    """
    Initializes a Prioritized Experience Replay (PER) buffer.

    Args:
        capacity (int): The maximum number of transitions the buffer can store.
        shapes (dict): A dictionary where keys are the names of the transition components (e.g., 'obs', 'action', 'reward') and values are their shapes (e.g., (4,), (1,), (1,)).
        device (str, optional): The device on which to store the buffer ('cpu' or 'cuda'). Defaults to 'cpu'.

    Returns:
        PERBufferState: An instance of PERBufferState containing the initialized buffer.
    """
    # tree_capacity MUST be a power of 2 for a perfectly balanced tree
    # We find the smallest power of 2 >= capacity
    tree_capacity = 1 << (capacity - 1).bit_length() if capacity > 0 else 1

    empty_data = TensorDict({}, batch_size=[capacity], device=device)
    for key, shape in shapes.items():
        empty_data.set(key, torch.zeros((capacity, *shape), dtype=torch.float32))

    return PERBufferState(
        data=empty_data,
        sum_tree=torch.zeros(2 * tree_capacity - 1, device=device),
        min_tree=torch.full((2 * tree_capacity - 1,), float("inf"), device=device),
        max_priority=1.0,
        pointer=0,
        size=0,
        capacity=capacity,
        tree_capacity=tree_capacity,
    )


def circular_write_strategy(
    buffer_state: BufferState, batch: TensorDict
) -> Tuple[BufferState, torch.Tensor]:
    """
    Unified write strategy. Always expects batched inputs.
    For a single transition, batch_dict tensors should have shape [1, ...]

    Args:
        buffer_state (BufferState): The current buffer state.
        batch_dict (dict): A dictionary of batched tensors to write.

    Returns:
        Tuple[BufferState, torch.Tensor]: The updated buffer state and the indices where data was written.
    """
    # Assuming all tensors in batch_dict have the same batch size
    batch_size = batch.batch_size[0]

    start_idx = buffer_state.pointer
    end_idx = start_idx + batch_size

    if end_idx <= buffer_state.capacity:
        indices = torch.arange(start_idx, end_idx, dtype=torch.long)
        buffer_state.data[start_idx:end_idx] = batch
    else:
        # Wrap-around logic
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

    return buffer_state, indices


def reservoir_write_strategy(
    buffer_state: ReservoirBufferState, batch: TensorDict
) -> Tuple[ReservoirBufferState, torch.Tensor]:
    """
    Writes data using Reservoir Sampling (uniform probability over infinite stream).
    Always expects batched inputs.

    Args:
        buffer_state (ReservoirBufferState): The reservoir buffer state.
        batch (TensorDict): A TensorDict of batched tensors to write.

    Returns:
        Tuple[ReservoirBufferState, torch.Tensor]: The updated reservoir buffer state and indices.
    """
    batch_size = batch.batch_size[0]

    written_indices = []
    for i in range(batch_size):
        # 1. Decide if and where to write
        if buffer_state.total_steps_seen < buffer_state.capacity:
            idx = buffer_state.total_steps_seen
        else:
            # Standard reservoir math: keep item with probability (capacity / steps_seen)
            j = random.randint(0, buffer_state.total_steps_seen)
            if j < buffer_state.capacity:
                idx = j
            else:
                idx = None

        if idx is not None:
            # 2. Write to TensorDict
            buffer_state.data[idx] = batch[i]
            written_indices.append(idx)

        buffer_state.total_steps_seen += 1
        buffer_state.size = min(buffer_state.total_steps_seen, buffer_state.capacity)

    indices_tensor = (
        torch.tensor(written_indices, dtype=torch.long)
        if written_indices
        else torch.empty(0, dtype=torch.long)
    )
    return buffer_state, indices_tensor


def uniform_sample(
    buffer_state: BufferState, rng_key: torch.Generator, batch_size: int
) -> TensorDict:
    """
    Uniformly samples from the buffer.

    Args:
        buffer_state (BufferState): The buffer state.
        rng_key (torch.Generator): The random number generator key.
        batch_size (int): The size of the batch to sample.

    Returns:
        TensorDict: The sampled batch.
    """
    if buffer_state.size < batch_size:
        raise ValueError("Buffer size is smaller than batch size.")
    indices = torch.randint(0, buffer_state.size, (batch_size,), generator=rng_key)
    return buffer_state.data[indices]


# @torch.compile  # Compile this for massive GPU speedups
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
    targets = (torch.rand(batch_size) + torch.arange(batch_size)) * segment_length

    # Vectorized Tree Traversal
    indices = torch.zeros(batch_size, dtype=torch.long)

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
    # TODO: why does this only seem to be needed with ape-x?
    data_indices = torch.clamp(data_indices, 0, buffer_state.capacity - 1)

    # Importance Sampling (IS) Weights
    leaf_priorities = buffer_state.sum_tree[indices]
    # TODOL is there a NaN risk here?
    min_prob = buffer_state.min_tree[0] / total_priority

    probs = leaf_priorities / total_priority
    is_weights = torch.pow(probs / min_prob, -beta)

    batch = buffer_state.data[data_indices]

    return batch, indices, is_weights


def _update_tree(
    sum_tree: torch.Tensor,
    min_tree: torch.Tensor,
    tree_indices: torch.Tensor,
    priorities: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Updates the tree nodes and propagates up to the root.

    Args:
        sum_tree (torch.Tensor): The sum tree to update.
        min_tree (torch.Tensor): The min tree to update.
        tree_indices (torch.Tensor): The indices of the tree to update.
        priorities (torch.Tensor): The priorities to update the tree with.

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: The updated sum tree and min tree.
    """
    # This modifies the tensors in-place for speed, but maintains a functional signature
    sum_tree[tree_indices] = priorities
    min_tree[tree_indices] = priorities

    # Propagate up
    parent_indices = (tree_indices - 1) // 2

    # In pure PyTorch, we can use a loop here because the depth is log2(N) (e.g., ~16 steps for 100k capacity)
    # torch.compile will unroll this beautifully.
    while parent_indices.numel() > 0 and parent_indices[0] >= 0:
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

    Args:
        buffer_state (PERBufferState): The PER buffer state.
        tree_indices (torch.Tensor): The indices of the tree to update.
        td_errors (torch.Tensor): The TD errors to update the tree with.
        alpha (float): The alpha parameter for PER.

    Returns:
        PERBufferState: The updated PER buffer state.
    """
    # Add epsilon to prevent zero priority
    priorities = torch.pow(torch.abs(td_errors) + 1e-6, alpha)

    new_sum_tree, new_min_tree = _update_tree(
        buffer_state.sum_tree, buffer_state.min_tree, tree_indices, priorities
    )

    new_max_priority = max(buffer_state.max_priority, torch.max(priorities).item())

    return PERBufferState(
        data=buffer_state.data,
        sum_tree=new_sum_tree,
        min_tree=new_min_tree,
        max_priority=new_max_priority,
        pointer=buffer_state.pointer,
        size=buffer_state.size,
        capacity=buffer_state.capacity,
        tree_capacity=buffer_state.tree_capacity,
    )


def with_per_tracking(write_strategy_fn: Callable) -> Callable:
    """
    Higher-order function composing a base writing strategy with PER logic.

    Args:
        write_strategy_fn: The base writing strategy function.

    Returns:
        per_add: The PER tracking function.
    """

    def per_add(
        buffer_state: PERBufferState, batch: TensorDict
    ) -> PERBufferState:
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
            new_state.max_priority = max(
                new_state.max_priority, torch.max(priorities).item()
            )

        return new_state

    return per_add


from collections import deque




def make_n_step_accumulator(n_steps: int, gamma: float) -> Callable:
    """
    Creates a stateful function that accumulates transitions.
    Returns a list of N-step transitions ready to be written to the buffer.

    Args:
        n_steps (int): The number of steps to accumulate.
        gamma (float): The discount factor.

    Returns:
        process_transition: The function that processes transitions.

    Note: Computes N-step TD targets at write time. This allows you to use 1-step transitions for N-step target. Because of this it does not work for sequence based algorithms.
    This is also a stateful function. You need to call reset() when you reset the environment.
    It is also possible to use a sequence based approach to get the N-step TD target. As done in R2D2 and MuZero.
    """
    history = deque(maxlen=n_steps)

    def process_transition(
        obs: torch.Tensor,
        action: torch.Tensor,
        reward: torch.Tensor,
        next_obs: torch.Tensor,
        terminated: torch.Tensor,
        truncated: torch.Tensor,
    ) -> List[TensorDict]:
        history.append((obs, action, reward, next_obs, terminated, truncated))
        transitions_to_yield = []

        # Case 1: Normal stepping. The window is full, slide it forward by 1.
        if len(history) == n_steps and not (terminated or truncated):
            n_step_reward = sum(t[2] * (gamma**i) for i, t in enumerate(history))

            first_obs, first_action, _, _, _, _ = history[0]
            _, _, _, final_next_obs, final_terminated, final_truncated = history[-1]

            transitions_to_yield.append(
                TensorDict(
                    {
                        "obs": rearrange(torch.as_tensor(first_obs), "... -> 1 ..."),
                        "action": torch.tensor([first_action], dtype=torch.long),
                        "reward": torch.tensor([n_step_reward], dtype=torch.float32),
                        "terminated": torch.tensor([final_terminated], dtype=torch.float32),
                        "truncated": torch.tensor([final_truncated], dtype=torch.float32),
                        "next_obs": rearrange(
                            torch.as_tensor(final_next_obs), "... -> 1 ..."
                        ),
                        "gamma": torch.tensor([gamma**n_steps], dtype=torch.float32),
                    },
                    batch_size=[1],
                )
            )
            history.popleft()

        # Case 2: Episode ended. Flush the remaining tail!
        elif terminated or truncated:
            while len(history) > 0:
                # Calculate the return for the remaining items in the shrinking window
                n_step_reward = sum(t[2] * (gamma**i) for i, t in enumerate(history))

                first_obs, first_action, _, _, _, _ = history[0]
                _, _, _, final_next_obs, final_terminated, final_truncated = history[-1]

                transitions_to_yield.append(
                    TensorDict(
                        {
                            "obs": rearrange(torch.as_tensor(first_obs), "... -> 1 ..."),
                            "action": torch.tensor([first_action], dtype=torch.long),
                            "reward": torch.tensor([n_step_reward], dtype=torch.float32),
                            "terminated": torch.tensor(
                                [final_terminated], dtype=torch.float32
                            ),
                            "truncated": torch.tensor(
                                [final_truncated], dtype=torch.float32
                            ),
                            "next_obs": rearrange(
                                torch.as_tensor(final_next_obs), "... -> 1 ..."
                            ),
                            "gamma": torch.tensor(
                                [gamma ** len(history)], dtype=torch.float32
                            ),
                        },
                        batch_size=[1],
                    )
                )
                history.popleft()  # Shrink the window until empty

        return transitions_to_yield

    def reset():
        """
        Clears the history.
        Note: Because the termination case flushes the queue, this is mostly
        a safety net for hard resets (e.g., if you manually interrupt an episode).
        """
        history.clear()

    return process_transition, reset
