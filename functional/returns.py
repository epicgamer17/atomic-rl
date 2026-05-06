import torch
from typing import List, Tuple


# TODO: should this live in targets.py?
def compute_mc_returns(rewards: torch.Tensor, gamma: float) -> torch.Tensor:
    """
    Computes standard Monte Carlo discounted returns.
    """
    returns = torch.zeros_like(rewards)
    R = 0.0
    for t in reversed(range(len(rewards))):
        R = rewards[t] + gamma * R
        returns[t] = R
    return returns
