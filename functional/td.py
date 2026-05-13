import torch
import torch.nn.functional as F
from einops import rearrange

# TODO: I think truncation right now works for single env trajectories without auto resetting, but would fail for vectorized envs with auto resetting. as the next_q_values would be on the resetted state instead of the one from the info. verify this, and if it is an issue, unify the vectorization logic in our buffer to work with offline buffers as well somehow.

def compute_v_td_target(
    next_values: torch.Tensor,  # [B]
    rewards: torch.Tensor,      # [B]
    terminated: torch.Tensor,   # [B]
    gamma: torch.Tensor,        # [B] or scalar
) -> torch.Tensor:
    """
    Calculates the 1-step Temporal Difference target for state values V(s).
    Formula: y = R_{t} + gamma * (1 - terminated) * V(s_{t+1})

    Args:
        next_values: Value estimates of the next states.
        rewards: Rewards for the transitions.
        terminated: Booleans indicating whether the episodes terminated.
        gamma: Discount factors.

    Returns:
        The TD target of shape [B].
    """
    # Fail Fast: Ensure shape alignment
    assert (
        next_values.ndim == 1
    ), f"Expected 1D next_values [B], got {next_values.shape}"
    assert (
        rewards.shape == terminated.shape == next_values.shape
    ), f"Shape mismatch: rewards {rewards.shape}, terminated {terminated.shape}, next_values {next_values.shape}"

    return rewards + gamma * next_values.detach() * (1 - terminated.float())


def compute_td_error(
    values: torch.Tensor,  # [B]
    next_values: torch.Tensor,  # [B]
    rewards: torch.Tensor,  # [B]
    terminated: torch.Tensor,  # [B]
    gamma: torch.Tensor,  # [B] or scalar
) -> torch.Tensor:
    """
    Computes the standard 1-step TD error (delta).
    Formula: delta_t = r_t + gamma * (1 - term) * V(s_{t+1}) - V(s_t)
    """
    target = compute_v_td_target(next_values, rewards, terminated, gamma)
    return target - values


def compute_q_td_target(
    next_q_values: torch.Tensor, # [B, A]
    next_actions: torch.Tensor,  # [B]
    rewards: torch.Tensor,       # [B]
    terminated: torch.Tensor,    # [B]
    gamma: torch.Tensor,         # [B]
) -> torch.Tensor:
    """
    Calculates the TD target for scalar Q-values.
    Composes the V-target function by extracting the value of the next state.

    Args:
        next_q_values: Q-values of the next states.
        next_actions: Indices of the actions taken in the next states (greedy for Q-learning, sampled for SARSA).
        rewards: Rewards for the transitions.
        terminated: Booleans indicating whether the episodes terminated.
        gamma: Discount factors.

    Returns:
        The TD target of shape [B].
    """
    assert (
        next_q_values.ndim == 2
    ), f"Expected 2D next_q_values [B, A], got {next_q_values.shape}"
    assert (
        next_actions.ndim == 1
    ), f"Expected [B] next_actions, got {next_actions.shape}"
    
    # 1. Extract the Q-value of the selected next action -> This IS V(s')
    next_values = torch.gather(next_q_values, 1, next_actions.unsqueeze(-1)).squeeze(-1)
    
    # 2. Compute standard V-target
    return compute_v_td_target(next_values, rewards, terminated, gamma)


def compute_categorical_q_td_target(
    next_logits: torch.Tensor,  # [B, A, Atoms]
    next_actions: torch.Tensor,  # [B]
    rewards: torch.Tensor,       # [B]
    terminated: torch.Tensor,    # [B]
    gamma: torch.Tensor,         # [B]
    support: torch.Tensor,       # [Atoms]
    v_min: float,
    v_max: float,
    atom_size: int,
) -> torch.Tensor:
    """
    Calculates the projected Categorical TD target distribution (C51 style).

    This function handles both 1-step and N-step TD targets. For N-step,
    the `rewards` should be the pre-computed discounted sum of rewards,
    and `gamma` should be the pre-computed effective discount factor (gamma^n).

    Args:
        next_logits: Logits of the next states.
        next_actions: Indices of the actions taken in the next states.
        rewards: Rewards for the transitions.
        terminated: Booleans indicating whether the episodes terminated.
        gamma: Discount factors.
        support: Support values for the distribution.
        v_min: The minimum value of the support.
        v_max: The maximum value of the support.
        atom_size: The number of atoms in the support.

    Returns:
        The projected Categorical TD target distribution [B, Atoms].
    """
    assert (
        next_logits.ndim == 3
    ), f"Expected 3D next_logits [B, A, Atoms], got {next_logits.shape}"
    assert (
        next_actions.ndim == 1
    ), f"Expected [B] next_actions, got {next_actions.shape}"

    # 1. Get probabilities of the next states
    next_probs = F.softmax(next_logits, dim=-1)

    # 2. Gather the probabilities for the chosen next actions
    next_actions_expanded = next_actions.view(-1, 1, 1).expand(-1, -1, atom_size)
    next_probs_a = next_probs.gather(1, next_actions_expanded).squeeze(1)  # [B, Atoms]

    # 3. Compute the target support (Tz) [B, Atoms]
    support_b = rearrange(support, "a -> 1 a")
    rewards_b = rearrange(rewards, "b -> b 1")
    gamma_b = rearrange(gamma, "b -> b 1")
    term_b = rearrange(terminated, "b -> b 1")

    Tz = rewards_b + gamma_b * support_b * (1 - term_b.float())
    Tz = Tz.clamp(min=v_min, max=v_max)

    # 4. Compute projection bins
    dz = (v_max - v_min) / (atom_size - 1)
    b = (Tz - v_min) / dz
    l = b.floor().long()
    u = b.ceil().long()

    # Handle boundary conditions where the target falls exactly on a bin
    l[(u > 0) & (l == u)] -= 1
    u[(l < (atom_size - 1)) & (l == u)] += 1

    # 5. Distribute probabilities onto the fixed support (Projection)
    batch_size = rewards.size(0)
    m = torch.zeros(batch_size, atom_size, device=rewards.device)
    offset = (
        torch.linspace(
            0,
            (batch_size - 1) * atom_size,
            batch_size,
            dtype=torch.long,
            device=rewards.device,
        )
        .unsqueeze(1)
        .expand(batch_size, atom_size)
    )

    # Flatten views for categorical projection
    m_flat = rearrange(m, "b a -> (b a)")
    offset_l = rearrange(l + offset, "b a -> (b a)")
    offset_u = rearrange(u + offset, "b a -> (b a)")

    prob_lower = rearrange(next_probs_a * (u.float() - b), "b a -> (b a)")
    prob_upper = rearrange(next_probs_a * (b - l.float()), "b a -> (b a)")

    # Index Add becomes clean and fast:
    m_flat.index_add_(0, offset_l, prob_lower)
    m_flat.index_add_(0, offset_u, prob_upper)

    return m
