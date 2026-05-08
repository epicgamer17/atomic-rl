import torch
import torch.nn.functional as F
from einops import rearrange


def scalar_td_target(
    next_q_values: torch.Tensor,  # [batch, num_actions]
    next_actions: torch.Tensor,  # [batch, 1]
    rewards: torch.Tensor,  # [batch, 1]
    terminated: torch.Tensor,  # [batch, 1]
    truncated: torch.Tensor,  # [batch, 1]
    gamma: torch.Tensor,  # [batch, 1]
) -> torch.Tensor:
    """
    Calculates the TD target for scalar Q-values.

    This function handles both 1-step and N-step TD targets. For N-step,
    the `rewards` should be the pre-computed discounted sum of rewards,
    and `gamma` should be the pre-computed effective discount factor (gamma^n).

    Formula: y = R_{t:t+n} + gamma^n * (1 - terminated) * max_a Q(s_{t+n}, a)
    Note: Truncated states still bootstrap as they are not true environment terminations.

    Args:
        next_q_values: Q-values of the next states.
        next_actions: Indices of the actions taken in the next states.
        rewards: Rewards for the transitions (or n-step discounted sum).
        terminated: Booleans indicating whether the episodes terminated.
        truncated: Booleans indicating whether the episodes were truncated.
        gamma: Discount factors (or effective gamma^n).

    Returns:
        The TD target of shape [batch, 1].
    """
    assert (
        next_q_values.ndim == 2
    ), f"Expected 2D next_q_values [B, A], got {next_q_values.shape}"
    assert (
        next_actions.shape
        == rewards.shape
        == terminated.shape
        == truncated.shape
        == gamma.shape
    ), "Shape mismatch in TD target inputs"
    assert (
        next_actions.ndim == 2 and next_actions.shape[1] == 1
    ), f"Expected [B, 1] next_actions, got {next_actions.shape}"

    # Pure Math: Guaranteed to broadcast perfectly if shapes are correct
    max_q_next = torch.gather(next_q_values, 1, next_actions)

    # Bootstrap if not terminated.
    return rewards + gamma * max_q_next * (1 - terminated.float())


def categorical_td_target(
    next_logits: torch.Tensor,  # [batch, num_actions, atom_size]
    next_actions: torch.Tensor,  # [batch, 1]
    rewards: torch.Tensor,  # [batch, 1]
    terminated: torch.Tensor,  # [batch, 1]
    truncated: torch.Tensor,  # [batch, 1]
    gamma: torch.Tensor,  # [batch, 1]
    support: torch.Tensor,  # [atom_size]
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
        rewards: Rewards for the transitions (or n-step discounted sum).
        terminated: Booleans indicating whether the episodes terminated.
        truncated: Booleans indicating whether the episodes were truncated.
        gamma: Discount factors (or effective gamma^n).
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
        next_actions.shape
        == rewards.shape
        == terminated.shape
        == truncated.shape
        == gamma.shape
    ), "Shape mismatch in Categorical target inputs"
    assert (
        next_actions.ndim == 2 and next_actions.shape[1] == 1
    ), f"Expected [B, 1] next_actions, got {next_actions.shape}"

    # 1. Get probabilities of the next states
    next_probs = F.softmax(next_logits, dim=-1)

    # 2. Gather the probabilities for the chosen next actions
    # next_actions is [B, 1], expand to [B, 1, Atoms] to match next_probs
    next_actions_expanded = next_actions.unsqueeze(-1).expand(-1, -1, atom_size)
    next_probs_a = next_probs.gather(1, next_actions_expanded).squeeze(1)  # [B, Atoms]

    # 3. Compute the target support (Tz) [B, Atoms]
    # Formula: Tz = R + gamma * support * (1 - terminated)
    support_b = rearrange(support, "a -> 1 a")
    Tz = rewards + gamma * support_b * (1 - terminated.float())
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

    # Flatten views for categorical projection (Projection Efficiency)
    m_flat = rearrange(m, "b a -> (b a)")
    offset_l = rearrange(l + offset, "b a -> (b a)")
    offset_u = rearrange(u + offset, "b a -> (b a)")

    prob_lower = rearrange(next_probs_a * (u.float() - b), "b a -> (b a)")
    prob_upper = rearrange(next_probs_a * (b - l.float()), "b a -> (b a)")

    # Index Add becomes clean and fast:
    m_flat.index_add_(0, offset_l, prob_lower)
    m_flat.index_add_(0, offset_u, prob_upper)

    return m
