from .initialization import _allocate_tensordict
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


@dataclass
class PERBufferState(BufferState):
    """
    State of a Prioritized Experience Replay (PER) buffer.
    """

    sum_tree: torch.Tensor
    min_tree: torch.Tensor
    max_priority: float
    tree_capacity: int


def init_buffer(
    capacity: int, shapes: dict, device: Optional[torch.device] = None
) -> BufferState:
    """Initializes a standard buffer."""
    data = _allocate_tensordict(shapes, [capacity], device)
    return BufferState(data=data, pointer=0, size=0, capacity=capacity, steps_seen=0)


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


def circular_write_strategy(
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


def reservoir_write_strategy(
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


# TODO: this doesnt work with sequence accumulator (should it?). i think generally for sequences and sequence accumulators we use returns.py
def make_n_step_accumulator(n_steps: int, gamma: float, num_envs: int = 1) -> Callable:
    """
    Creates a stateful function for N-step transition accumulation.

    # TODO: Performance Violation
    # This implementation iterates over the batch dimension (`num_envs`) in Python
    # using a for-loop and uses Python `deque`s. This directly violates the rule
    # "STRICTLY AVOID iterating over tensor dimensions in Python" and creates a massive
    # CPU overhead for highly vectorized environments (e.g., thousands of envs).
    # Future enhancement: Rewrite this to use a fully vectorized PyTorch circular
    # buffer [num_envs, n_steps, ...] using cumprod and sum on GPU/CPU natively.
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
        ready_transitions = []
        # Fix for bitwise_or on Float tensors
        is_done = torch.logical_or(terminated.bool(), truncated.bool())

        for i in range(num_envs):
            h = histories[i]
            h.append(
                (
                    obs[i].detach(),
                    action[i].detach(),
                    reward[i].detach(),
                    next_obs[i].detach(),
                    terminated[i].detach(),
                    truncated[i].detach(),
                )
            )

            done_i = is_done[i]

            if len(h) == n_steps and not done_i:
                n_step_reward = 0.0
                curr_gamma = 1.0
                for transition in h:
                    n_step_reward += transition[2] * curr_gamma
                    curr_gamma *= gamma

                first_obs, first_action, _, _, _, _ = h[0]
                _, _, _, final_next_obs, final_term, final_trunc = h[-1]

                # Coerce to tensor safely without introspection
                ready_transitions.append(
                    {
                        "obs": first_obs,
                        "action": first_action,
                        "reward": torch.as_tensor(n_step_reward).detach(),
                        "next_obs": final_next_obs,
                        "terminated": torch.as_tensor(final_term).detach(),
                        "truncated": torch.as_tensor(final_trunc).detach(),
                        "gamma": torch.as_tensor(gamma**n_steps),
                    }
                )
                h.popleft()

            elif done_i:
                while h:
                    n_step_reward = 0.0
                    curr_gamma = 1.0
                    for transition in h:
                        n_step_reward += transition[2] * curr_gamma
                        curr_gamma *= gamma

                    first_obs, first_action, _, _, _, _ = h[0]
                    _, _, _, final_next_obs, final_term, final_trunc = h[-1]

                    ready_transitions.append(
                        {
                            "obs": first_obs,
                            "action": first_action,
                            "reward": torch.as_tensor(n_step_reward).detach(),
                            "next_obs": final_next_obs,
                            "terminated": torch.as_tensor(final_term).detach(),
                            "truncated": torch.as_tensor(final_trunc).detach(),
                            "gamma": torch.as_tensor(gamma ** len(h)),
                        }
                    )
                    h.popleft()

        if ready_transitions:
            stacked = {
                k: torch.stack([t[k] for t in ready_transitions])
                for k in ready_transitions[0].keys()
            }
            return TensorDict(stacked, batch_size=[len(ready_transitions)])
        return TensorDict({}, batch_size=[0])

    def reset():
        for h in histories:
            h.clear()

    return process_transition, reset


# TODO: unrolling on sampling (ie not using a full chunk every time for muzero). must add padding. Value Target: 0, Reward Target: 0, Policy Target: Uniform distribution (or just mask the policy loss to 0 for absorbing steps).
# TODO: sampling for DRQN no clipped chunks. (easiest when storing 1 by 1 transitions). There will be the problem of chunks that end early that will either have to be thrown out or not be usable. How to handle. How does R2D2 handle this?
# TODO: make PER for R2D2 (method of calculating PER for a sequence of transitions), uses mean and max of per in sequence or something.
# TODO: make PER for MuZero which differs from R2D2 (it is closer to DQN's PER, ie individual transitions). May be difficult when storing full chunks of data. Likely best approach is to store step priorities and use standard sum tree for seqeunces (ie do similar to R2D2 and then instead of uniform in the sequence its also prioritized)
# TODO: R2D2 stores sequences of (s, a, r) (so will DRQN) (should we also store gamma with our system?)
# TODO: MuZero stores sequences of (s, a, r, ?player, search_value, search_policy_target) (should we also store gamma with our system?). AlphaZero stores the same but does not need to unroll.
def make_padded_chunk_accumulator(
    chunk_size: int,
    num_envs: int,
    shapes: dict,
    device: Optional[torch.device] = None,
    overlap: int = 0,  # Used for R2D2 burn-in
) -> Callable[[TensorDict], TensorDict]:
    """
    Stateful accumulator that yields padded chunks.
    If an env terminates early, the rest of the chunk is zero-padded.
    Returns batches of variable size [num_ready_envs, chunk_size, ...].
    Used for storing chunks of data in off-policy RL. For example storing episodes of board games for AlphaZero, overlapping chunks for R2D2, or discrete chunks for MuZero.
    """
    # Note the shape swap: [num_envs, chunk_size].
    # This makes slicing individual ready environments much faster.
    history = _allocate_tensordict(shapes, [num_envs, chunk_size], device)

    # Track the current write index for each environment independently
    current_steps = torch.zeros(num_envs, dtype=torch.long, device=device)
    env_indices = torch.arange(num_envs, device=device)

    def process_transition(transition: TensorDict) -> TensorDict:
        nonlocal current_steps

        # 1. Write transition into the history for each env at its specific step
        history[env_indices, current_steps] = transition
        current_steps += 1

        # 2. Check which environments are ready to be flushed
        dones = transition["terminated"].bool() | transition["truncated"].bool()
        is_full = current_steps == chunk_size
        ready_mask = dones | is_full

        # If any environment finished an episode OR filled a chunk
        if ready_mask.any():
            ready_indices = ready_mask.nonzero(as_tuple=True)[0]

            # Extract only the chunks that are ready.
            # Shape: [num_ready, chunk_size, ...]
            ready_chunks = history[ready_indices].clone()

            # 3. Apply Zero-Padding
            # If an env finished at step 5, steps 5-39 contain garbage from old episodes.
            # We generate a mask to zero them out.
            steps_taken = current_steps[ready_indices]
            seq_indices = torch.arange(chunk_size, device=device).unsqueeze(0)

            # valid_mask is True for real data, False for padding
            valid_mask = seq_indices < steps_taken.unsqueeze(
                1
            )  # [num_ready, chunk_size]
            valid_mask_float = valid_mask.float()

            for key, tensor in ready_chunks.items():
                # Dynamically broadcast the mask to match the tensor's shape (e.g. images)
                mask = valid_mask_float
                for _ in range(tensor.ndim - 2):
                    mask = mask.unsqueeze(-1)
                # Zero out the padded steps
                ready_chunks[key] = tensor * mask

            # Embed the valid mask so your loss function knows which steps to ignore
            ready_chunks["valid"] = valid_mask

            # 4. Handle State Reset & Overlap
            current_steps[ready_mask] = 0

            if overlap > 0:
                # Overlap is ONLY applied to envs that reached chunk_size but DID NOT terminate
                full_not_done = is_full & ~dones
                full_indices = full_not_done.nonzero(as_tuple=True)[0]

                if full_indices.numel() > 0:
                    # Copy the last `overlap` steps to the beginning of the next chunk
                    history[full_indices, :overlap] = history[
                        full_indices, chunk_size - overlap :
                    ].clone()
                    current_steps[full_not_done] = overlap

            return ready_chunks

        # If no envs are ready, return an empty TensorDict
        return TensorDict({}, batch_size=[0])

    return process_transition
