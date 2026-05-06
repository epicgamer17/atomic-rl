import torch
from typing import List, Tuple


# TODO: Support batched [B, T] inputs and optional 'terminated' flags to handle multiple episodes within a single trajectory/batch.
def compute_mc_returns(rewards: torch.Tensor, gamma: float) -> torch.Tensor:
    """
    Computes standard Monte Carlo discounted returns.

    NOTE: This implementation currently only supports 1D [T] reward tensors.

    """
    returns = torch.zeros_like(rewards)
    R = 0.0
    for t in reversed(range(len(rewards))):
        R = rewards[t] + gamma * R
        returns[t] = R
    return returns
