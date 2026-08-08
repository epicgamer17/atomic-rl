from ...initialization import _allocate_tensordict
import torch
from tensordict import TensorDict
from typing import Tuple, Callable, List, Optional, Union
import random
from collections import deque
from dataclasses import dataclass


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
