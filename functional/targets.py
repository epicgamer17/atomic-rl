import torch
import torch.nn.functional as F
from einops import rearrange


def standard_td_target(
    next_q_values: torch.Tensor,
    next_actions: torch.Tensor,
    rewards: torch.Tensor,
    terminated: torch.Tensor,
    truncated: torch.Tensor,
    gamma: torch.Tensor,
):
    """
    Calculates the standard 1-step TD target.
    Handles truncated states by bootstrapping from the next state's Q-value.

    Args:
        next_q_values (torch.Tensor): Tensor of shape (batch_size, num_actions) containing the Q-values of the next states.
        next_actions (torch.Tensor): Tensor of shape (batch_size, 1) containing the indices of the actions taken in the next states.
        rewards (torch.Tensor): Tensor of shape (batch_size, 1) containing the rewards.
        terminated (torch.Tensor): Tensor of shape (batch_size, 1) containing booleans indicating whether the episodes terminated (MDP end).
        truncated (torch.Tensor): Tensor of shape (batch_size, 1) containing booleans indicating whether the episodes were truncated (e.g. time limit).
        gamma (torch.Tensor): Discount factor.
    Returns:
        torch.Tensor: The standard 1-step TD target.
    """
    # 1. THE BOUNCER: Safely handle [B] OR [B, 1], and format exactly to [B, 1]
    rewards = rearrange(rewards.squeeze(-1), "b -> b 1")
    gamma = rearrange(gamma.squeeze(-1), "b -> b 1")
    terminated = rearrange(terminated.squeeze(-1), "b -> b 1").float()
    truncated = rearrange(truncated.squeeze(-1), "b -> b 1").float()
    next_actions = rearrange(next_actions.squeeze(-1), "b -> b 1")

    # 2. PURE MATH: Now guaranteed to broadcast perfectly
    max_q_next = torch.gather(next_q_values, 1, next_actions)
    
    # Bootstrap if not terminated. Truncated states bootstrap from max_q_next.
    return rewards + gamma * max_q_next * (1 - terminated)


def n_step_td_target(
    next_q_values: torch.Tensor,
    next_actions: torch.Tensor,
    rewards: torch.Tensor,
    terminated: torch.Tensor,
    truncated: torch.Tensor,
    gamma: torch.Tensor,
):
    """
    Calculates the N-step TD target using an effective gamma.
    Assumes rewards is already the n-step discounted sum and gamma is gamma^n.
    Handles truncated states by bootstrapping from the next state's Q-value.

    Args:
        next_q_values (torch.Tensor): Tensor of shape (batch_size, num_actions) containing the Q-values of the next states.
        next_actions (torch.Tensor): Tensor of shape (batch_size, 1) containing the indices of the actions taken in the next states.
        rewards (torch.Tensor): Tensor of shape (batch_size, 1) containing the rewards.
        terminated (torch.Tensor): Tensor of shape (batch_size, 1) containing booleans indicating whether the episodes terminated.
        truncated (torch.Tensor): Tensor of shape (batch_size, 1) containing booleans indicating whether the episodes were truncated.
        gamma (torch.Tensor): Effective discount factor (gamma^n).
    Returns:
        torch.Tensor: The N-step TD target.
    """
    # 1. THE BOUNCER: Safely handle [B] OR [B, 1], and format exactly to [B, 1]
    rewards = rearrange(rewards.squeeze(-1), "b -> b 1")
    gamma = rearrange(gamma.squeeze(-1), "b -> b 1")
    terminated = rearrange(terminated.squeeze(-1), "b -> b 1").float()
    truncated = rearrange(truncated.squeeze(-1), "b -> b 1").float()
    next_actions = rearrange(next_actions.squeeze(-1), "b -> b 1")

    # 2. PURE MATH: Now guaranteed to broadcast perfectly
    max_q_next = torch.gather(next_q_values, 1, next_actions)
    
    # Bootstrap if not terminated.
    return rewards + gamma * max_q_next * (1 - terminated)


def categorical_td_target(
    next_logits: torch.Tensor,
    next_actions: torch.Tensor,
    rewards: torch.Tensor,
    terminated: torch.Tensor,
    truncated: torch.Tensor,
    gamma: torch.Tensor,
    support: torch.Tensor,
    v_min: float,
    v_max: float,
    atom_size: int,
):
    """
    Calculates the projected Categorical TD target distribution.
    Handles truncated states by bootstrapping from the support values.

    Args:
        next_logits (torch.Tensor): Tensor of shape (batch_size, num_actions, atom_size) containing the logits of the next states.
        next_actions (torch.Tensor): Tensor of shape (batch_size, 1) containing the indices of the actions taken in the next states.
        rewards (torch.Tensor): Tensor of shape (batch_size, 1) containing the rewards.
        terminated (torch.Tensor): Tensor of shape (batch_size, 1) containing booleans indicating whether the episodes terminated.
        truncated (torch.Tensor): Tensor of shape (batch_size, 1) containing booleans indicating whether the episodes were truncated.
        gamma (torch.Tensor): Tensor of shape (batch_size, 1) containing the discount factors.
        support (torch.Tensor): Tensor of shape (atom_size,) containing the support values.
        v_min (float): The minimum value of the support.
        v_max (float): The maximum value of the support.
        atom_size (int): The number of atoms in the support.
    Returns:
        torch.Tensor: The projected Categorical TD target distribution.
    """
    # 1. THE BOUNCER: Safely handle [B] OR [B, 1], and format exactly to [B, 1]
    rewards_b = rearrange(rewards.squeeze(-1), "b -> b 1")
    gamma_b = rearrange(gamma.squeeze(-1), "b -> b 1")
    terminated_b = rearrange(terminated.squeeze(-1), "b -> b 1").float()
    truncated_b = rearrange(truncated.squeeze(-1), "b -> b 1").float()
    next_actions_b = rearrange(next_actions.squeeze(-1), "b -> b 1")
    support_b = rearrange(support, "a -> 1 a")

    batch_size = rewards_b.size(0)

    # 2. Get probabilities of the next states
    next_probs = F.softmax(next_logits, dim=-1)

    # 3. Gather the probabilities for the chosen next actions
    # next_actions_b is [B, 1], expand to [B, 1, Atoms] to match next_probs
    next_actions_expanded = next_actions_b.unsqueeze(-1).expand(-1, -1, atom_size)
    next_probs_a = next_probs.gather(1, next_actions_expanded).squeeze(1)  # [B, Atoms]

    # 4. Compute the target support (Tz) [B, Atoms]
    # Pure Math (Readable)
    # Bootstrap if not terminated. Truncated states bootstrap from support_b.
    Tz = rewards_b + gamma_b * support_b * (1 - terminated_b)
    Tz = Tz.clamp(min=v_min, max=v_max)

    # 4. Compute projection bins
    dz = (v_max - v_min) / (atom_size - 1)
    b = (Tz - v_min) / dz
    l = b.floor().long()
    u = b.ceil().long()

    # Handle boundary conditions where the target falls exactly on a bin
    l[(u > 0) & (l == u)] -= 1
    u[(l < (atom_size - 1)) & (l == u)] += 1

    # 5. Distribute probabilities onto the fixed support
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

    # Flatten views for categorical projection (Flatten Once)
    m_flat = rearrange(m, "b a -> (b a)")
    offset_l = rearrange(l + offset, "b a -> (b a)")
    offset_u = rearrange(u + offset, "b a -> (b a)")

    prob_lower = rearrange(next_probs_a * (u.float() - b), "b a -> (b a)")
    prob_upper = rearrange(next_probs_a * (b - l.float()), "b a -> (b a)")

    # Index Add becomes clean:
    m_flat.index_add_(0, offset_l, prob_lower)
    m_flat.index_add_(0, offset_u, prob_upper)

    return m  # This is the target probability distribution
