import torch
from tensordict import TensorDict
from typing import Tuple, Callable, List, Optional
from ..utils import add_dirichlet_noise


def get_mcts_visit_policy(
    visit_counts: torch.Tensor,
    temperature: float = 1.0,
) -> torch.Tensor:
    """
    Computes a target policy distribution from MCTS visit counts.

    Formula:
        for tau > 0: pi(a|s) = N(s, a)^(1/tau) / sum_b N(s, b)^(1/tau)
        for tau = 0: pi(a|s) = one_hot(argmax_a N(s, a))

    Args:
        visit_counts: Tensor of visit counts [B, A] or [A].
        temperature: Temperature parameter tau >= 0.

    Returns:
        torch.Tensor: Target policy probability distribution with same shape as visit_counts.
    """
    assert temperature >= 0.0, f"Temperature must be non-negative, got {temperature}"

    if temperature == 0.0:
        is_max = (
            visit_counts == torch.max(visit_counts, dim=-1, keepdim=True).values
        ).float()
        return is_max / is_max.sum(dim=-1, keepdim=True)

    if temperature == 1.0:
        total_visits = visit_counts.sum(dim=-1, keepdim=True)
        total_visits = torch.where(
            total_visits > 0, total_visits, torch.ones_like(total_visits)
        )
        return visit_counts / total_visits

    exponent = 1.0 / temperature
    scaled_visits = torch.pow(visit_counts.float(), exponent)
    total_scaled = scaled_visits.sum(dim=-1, keepdim=True)
    total_scaled = torch.where(
        total_scaled > 0, total_scaled, torch.ones_like(total_scaled)
    )
    return scaled_visits / total_scaled
