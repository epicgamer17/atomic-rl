import torch
import warnings
from einops import rearrange


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
    # The Bouncer: Ensure [B, T] shapes
    rewards = rearrange(rewards, "b t -> b t")
    terminals = rearrange(terminals, "b t -> b t")

    # Validation: Warn if any trajectory in the batch is never terminal (always False)
    # This usually indicates that the trajectory is incomplete or not an full episode.
    is_never_terminal = ~(terminals.bool().any(dim=1))
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


# NOTE: Could change the approach so user must properly slice (remove V_0) and append the final values to `next_values` to ensure the returns are computed correctly.
def compute_n_step_returns(
    rewards: torch.Tensor,
    terminals: torch.Tensor,
    values: torch.Tensor,
    last_values: torch.Tensor,
    gamma: float,
    n: int,
) -> torch.Tensor:
    """
    Computes batched n-step bootstrapped discounted returns.

    Args:
        rewards (torch.Tensor): The reward tensor of shape [batch, time].
        terminals (torch.Tensor): Boolean/float mask indicating episode termination.
            If 1.0/True, the return is not propagated from the next step.
            Shape [batch, time].
        values (torch.Tensor): The values tensor of shape [batch, time].
            Note: This should be V(s_0), ..., V(s_{T-1}).
        last_values (torch.Tensor): The value for the final state in the batch.
            Note: This should be V(s_T). Shape [batch] or [batch, 1].
        gamma (float): The discount factor.
        n (int): The number of steps to lookahead.

    Returns:
        torch.Tensor: The discounted returns of shape [batch, time].
    """
    # The Bouncer: Ensure [B, T] shapes
    # 1. Safely strip trailing 1s, assert it's 2D, and lock in the exact (B, T) sizes
    rewards = rearrange(rewards.squeeze(-1), "b t -> b t")
    b, t = rewards.shape  # Capture the exact batch and time dimensions

    # 2. Force terminals and values to match 'b' and 't' exactly, or crash!
    terminals = rearrange(terminals.squeeze(-1), "b t -> b t", b=b, t=t)
    values = rearrange(values.squeeze(-1), "b t -> b t", b=b, t=t)
    # 3. Last Value Bouncer: Safely handle [B] or [B, 1] without destroying batch size 1
    # NOTE: not sure how much i love this block
    if last_values.ndim == 1:
        last_values = rearrange(last_values, "b -> b 1", b=b)
    else:
        last_values = rearrange(last_values, "b 1 -> b 1", b=b)

    assert n >= 1, f"n-step must be at least 1, got {n}"

    # Construct next_values [V(s_1), ..., V(s_T)] by shifting values [V(s_0), ..., V(s_{T-1})]
    # and appending last_values [V(s_T)].
    next_values = torch.cat([values[:, 1:], last_values], dim=1)
    returns = next_values.clone()

    for i in range(n):
        # On iterations after the first, we shift the previously computed returns
        # to the left to become the bootstrap for the next step.
        if i > 0:
            # Shift left: G_{t+1}^{(i)}
            # [B, T] -> [B, T]
            next_returns = torch.zeros_like(returns)
            next_returns[:, :-1] = returns[:, 1:]

            # We only update the returns for time steps where we have a valid future step
            # for this iteration. This ensures that at the end of the trajectory, the
            # return gracefully falls back to the best available n-step return.
            mask = torch.zeros_like(returns)
            mask[:, :-i] = 1.0

            # Compute the (i+1)-step return
            updated_returns = rewards + gamma * (1.0 - terminals.float()) * next_returns
            returns = updated_returns * mask + returns * (1.0 - mask)
        else:
            # First iteration: Compute 1-step returns for all t
            # [B, T] = [B, T] + scalar * [B, T] * [B, T]
            returns = rewards + gamma * (1.0 - terminals.float()) * returns

    return returns


def compute_td_lambda_returns():
    raise NotImplementedError("TD Lambda returns not yet implemented")
