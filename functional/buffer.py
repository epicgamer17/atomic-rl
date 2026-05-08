import torch
import torch.nn as nn

from dataclasses import dataclass, field
from tensordict import TensorDict
import random
import numpy as np
from einops import rearrange
from typing import Tuple, Callable, List, Dict, Any, Optional, Union

from collections import deque


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


@dataclass
class RolloutBufferState:
    data: TensorDict
    truncation_records: List[Tuple[int, int, torch.Tensor]] = field(
        default_factory=list
    )


def _allocate_tensordict(
    shapes: dict, batch_size: List[int], device: str = "cpu"
) -> TensorDict:
    """Allocates a zeroed TensorDict of any arbitrary geometry."""
    data = TensorDict({}, batch_size=batch_size, device=device)
    for key, shape in shapes.items():
        # Use long for actions, float32 for everything else by default
        dtype = torch.long if "action" in key else torch.float32
        data.set(key, torch.zeros((*batch_size, *shape), dtype=dtype))
    return data


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
    # Geometry: [Capacity]
    data = _allocate_tensordict(shapes, [capacity], device)
    return BufferState(data=data, pointer=0, size=0, capacity=capacity)


# TODO: should i merge this with init_buffer somehow?
def init_rollout_buffer(
    steps_per_env: int, num_envs: int, shapes: dict, device: str = "cpu"
) -> RolloutBufferState:
    """
    Initializes a buffer for storing on-policy rollouts.
    Uses [Time, Batch, ...] dimensionality.

    Args:
        steps_per_env (int): Number of steps to store per environment.
        num_envs (int): Number of parallel environments.
        shapes (dict): Dictionary of names and shapes for each component.
        device (str): Device to store the buffer on.

    Returns:
        RolloutBufferState: State containing the pre-allocated TensorDict.
    """
    # Geometry: [Time, Batch]
    data = _allocate_tensordict(shapes, [steps_per_env, num_envs], device)
    return RolloutBufferState(data=data, truncation_records=[])


def store_rollout_step(
    buffer: RolloutBufferState, step: int, transition: TensorDict
) -> None:
    """
    Stores a transition batch into the rollout buffer at the given time step.

    Args:
        buffer (RolloutBufferState): The rollout buffer state.
        step (int): The time step index to store at.
        transition (TensorDict): A TensorDict containing the transition data for this step.
            Must have batch_size [num_envs].
    """
    buffer.data[step] = transition


def record_truncations(
    buffer: RolloutBufferState,
    step: int,
    info: dict,
    truncated: Union[np.ndarray, torch.Tensor],
) -> None:
    """
    Checks the info dict for final observations (from Gymnasium auto-resets)
    and records them in the buffer's truncation_records list ONLY if they were truncated.

    Args:
        buffer (RolloutBufferState): The buffer to record into.
        step (int): The current step index in the rollout.
        info (dict): The info dictionary from env.step().
        truncated (Union[np.ndarray, torch.Tensor]): The truncated boolean mask from env.step().
    """
    from .utils import extract_vector_env_final_obs

    env_indices, final_obs = extract_vector_env_final_obs(info)

    for i, env_idx in enumerate(env_indices):
        # Only record it if the environment actually truncated!
        if truncated[env_idx]:
            # We store the observation as a tensor to avoid repeated conversions later
            obs_tensor = torch.as_tensor(final_obs[i], dtype=torch.float32)
            buffer.truncation_records.append((step, int(env_idx), obs_tensor))


def get_rollout_next_values(
    buffer: RolloutBufferState,
    last_values: torch.Tensor,
    get_value_fn: Callable[[torch.Tensor], torch.Tensor],
    device: torch.device,
) -> torch.Tensor:
    """
    Assembles the next_values tensor [T, B] and patches any truncated states
    using the recorded truncation_records.

    Args:
        buffer (RolloutBufferState): The rollout buffer.
        last_values (torch.Tensor): The value of the final states in the rollout [B, 1] or [B].
        get_value_fn (Callable[[torch.Tensor], torch.Tensor]): A function that takes an observation
            tensor and returns the corresponding state value tensor.
        device (torch.device): Device for computation.

    Returns:
        torch.Tensor: The patched next_values tensor of shape [T, B].
    """
    T, B = buffer.data.batch_size
    values = buffer.data["values"]  # [T, B]

    # 1. Standard assembly
    if last_values.ndim == 1:
        last_values = last_values.unsqueeze(1)  # [B, 1]

    # values is [T, B], so values[1:] is [T-1, B].
    # last_values is [B, 1]. We need it to be [1, B].
    last_v_reshaped = rearrange(last_values, "b 1 -> 1 b")

    next_values = torch.cat([values[1:], last_v_reshaped], dim=0)  # [T, B]

    # 2. Patch truncations
    if buffer.truncation_records:
        obs_batch = torch.stack([r[2] for r in buffer.truncation_records]).to(device)
        with torch.inference_mode():
            v_patch = get_value_fn(obs_batch)
            v_patch = v_patch.squeeze(-1)  # [N]

        for i, (step, env_idx, _) in enumerate(buffer.truncation_records):
            next_values[step, env_idx] = v_patch[i]

    return next_values


def flatten_rollout_buffer(buffer: RolloutBufferState) -> TensorDict:
    """
    Flattens the [Time, Batch] dimensions into a single [Time * Batch] dimension.

    Args:
        buffer (RolloutBufferState): The rollout buffer state.

    Returns:
        TensorDict: The flattened data.
    """
    # TODO: should we/can we use einops here (on a tensordict)?
    return buffer.data.flatten(0, 1)


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
            new_state.max_priority = max(
                new_state.max_priority, torch.max(priorities).item()
            )

        return new_state

    return per_add


def make_n_step_accumulator(n_steps: int, gamma: float, num_envs: int = 1) -> Callable:
    """
    Creates a stateful function that accumulates transitions for both single
    and vectorized environments. Maintains a separate history for each environment.

    Args:
        n_steps (int): The number of steps to lookahead.
        gamma (float): The discount factor.
        num_envs (int): The number of parallel environments (default: 1).
    """
    histories = [deque(maxlen=n_steps) for _ in range(num_envs)]

    def process_transition(
        obs: torch.Tensor,
        action: torch.Tensor,
        reward: torch.Tensor,
        next_obs: torch.Tensor,
        terminated: torch.Tensor,
        truncated: torch.Tensor,
    ) -> TensorDict:

        # 1. THE BOUNCER: Safely format unbatched inputs to [1, ...]
        # If the reward is a 0D tensor (scalar), we know it lacks a batch dimension.
        # TODO: should bouncer handle these or should user pass in batched inputs always? maybe user always batching is cleaner?
        if getattr(reward, "ndim", 0) == 0:
            obs = obs.unsqueeze(0)
            action = action.unsqueeze(0)
            reward = reward.view(1)
            next_obs = next_obs.unsqueeze(0)
            terminated = terminated.view(1)
            truncated = truncated.view(1)

        ready_transitions = []

        # 2. Process each environment's timeline individually
        for i in range(num_envs):
            t_obs = obs[i]
            t_action = action[i]
            t_reward = reward[i].item()
            t_next_obs = next_obs[i]
            t_term = terminated[i].item()
            t_trunc = truncated[i].item()

            history = histories[i]
            history.append((t_obs, t_action, t_reward, t_next_obs, t_term, t_trunc))

            is_done = t_term or t_trunc

            # Case 1: Window is full, env is still running. Slide forward by 1.
            if len(history) == n_steps and not is_done:
                n_step_reward = sum(t[2] * (gamma**j) for j, t in enumerate(history))

                first_obs, first_action, _, _, _, _ = history[0]
                _, _, _, final_next_obs, final_term, final_trunc = history[-1]

                ready_transitions.append(
                    {
                        "obs": first_obs,
                        "action": first_action,
                        "reward": torch.tensor(n_step_reward, dtype=torch.float32),
                        "next_obs": final_next_obs,
                        "terminated": torch.tensor(final_term, dtype=torch.float32),
                        "truncated": torch.tensor(final_trunc, dtype=torch.float32),
                        "gamma": torch.tensor(gamma**n_steps, dtype=torch.float32),
                    }
                )
                history.popleft()

            # Case 2: Episode ended. Flush the remaining tail for this specific env!
            elif is_done:
                while len(history) > 0:
                    n_step_reward = sum(
                        t[2] * (gamma**j) for j, t in enumerate(history)
                    )

                    first_obs, first_action, _, _, _, _ = history[0]
                    _, _, _, final_next_obs, final_term, final_trunc = history[-1]

                    ready_transitions.append(
                        {
                            "obs": first_obs,
                            "action": first_action,
                            "reward": torch.tensor(n_step_reward, dtype=torch.float32),
                            "next_obs": final_next_obs,
                            "terminated": torch.tensor(final_term, dtype=torch.float32),
                            "truncated": torch.tensor(final_trunc, dtype=torch.float32),
                            "gamma": torch.tensor(
                                gamma ** len(history), dtype=torch.float32
                            ),
                        }
                    )
                    history.popleft()

        # 3. Batch the results for unified buffer writing
        if ready_transitions:
            stacked_dict = {
                key: torch.stack([t[key] for t in ready_transitions])
                for key in ready_transitions[0].keys()
            }
            # Returns a TensorDict of shape [N] where N is ready transitions
            return TensorDict(stacked_dict, batch_size=[len(ready_transitions)])
        else:
            # Returns an empty TensorDict if nothing is ready yet
            return TensorDict({}, batch_size=[0])

    def reset():
        for h in histories:
            h.clear()

    return process_transition, reset
