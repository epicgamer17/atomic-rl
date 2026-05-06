import torch
import warnings


def compute_mc_returns(
    rewards: torch.Tensor,
    terminals: torch.Tensor,
    gamma: float,
) -> torch.Tensor:
    """
    Computes batched Monte Carlo discounted returns.

    Args:
        rewards (torch.Tensor): The reward tensor of shape [batch, time].
        terminals (torch.Tensor): Boolean/float mask indicating episode termination.
            If 1.0/True, the return is not propagated from the next step.
            Shape [batch, time].
        gamma (float): The discount factor.

    Returns:
        torch.Tensor: The discounted returns of shape [batch, time].
    """
    assert rewards.ndim == 2, f"Expected 2D rewards tensor [B, T], got {rewards.shape}"
    assert (
        terminals.shape == rewards.shape
    ), f"Terminals shape {terminals.shape} must match rewards shape {rewards.shape}"

    # Validation: Warn if any trajectory in the batch is never terminal (always False)
    # This usually indicates that the trajectory is incomplete or not an full episode.
    is_never_terminal = ~terminals.any(dim=1)
    if is_never_terminal.any():
        warnings.warn(
            "Found trajectory in batch where terminals is always False (never terminal). "
            "MC returns for these trajectories may be biased if they were intended to be full episodes."
        )

    returns = torch.zeros_like(rewards)
    R = torch.zeros_like(rewards[:, 0])  # [B]

    # Iterate backwards through time T
    for t in reversed(range(rewards.size(1))):
        # If terminal is True (1.0), the next R gets multiplied by 0
        mask = 1.0 - terminals[:, t].float()

        R = rewards[:, t] + gamma * R * mask
        returns[:, t] = R

    return returns
