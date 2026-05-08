import torch
import warnings
from einops import rearrange


# TODO: should also reset on truncated not just terminated
def compute_mc_returns(
    rewards: torch.Tensor,
    terminated: torch.Tensor,
    truncated: torch.Tensor,
    gamma: float,
) -> torch.Tensor:
    """
    Computes batched Monte Carlo discounted returns.

    Args:
        rewards (torch.Tensor): The reward tensor of shape [batch, time].
        terminated (torch.Tensor): Boolean/float mask indicating episode termination (MDP end).
            If 1.0/True, the return is not propagated from the next step.
            Shape [batch, time].
        truncated (torch.Tensor): Boolean/float mask indicating episode truncation (e.g. time limit).
            If 1.0/True, the return is not propagated from the next step.
            Shape [batch, time].
        gamma (float): The discount factor.

    Returns:
        torch.Tensor: The discounted returns of shape [batch, time].
    """
    # The Bouncer: Ensure [B, T] shapes
    rewards = rearrange(rewards, "b t -> b t")
    terminated = rearrange(terminated, "b t -> b t")
    truncated = rearrange(truncated, "b t -> b t")

    # Combine masks: MC returns stop at either terminated or truncated
    done = (terminated.bool() | truncated.bool()).float()

    # Validation: Warn if any trajectory in the batch is never done
    is_never_terminal = ~(terminated.bool().any(dim=1))
    if is_never_terminal.any():
        warnings.warn(
            "Found trajectory in batch where terminated is always False (never terminal). "
            "MC returns for these trajectories may be biased if they were intended to be full episodes."
        )

    returns = torch.zeros_like(rewards)
    R = torch.zeros_like(rewards[:, 0])  # [B]

    # Iterate backwards through time T
    for t in reversed(range(rewards.size(1))):
        # If done is True (1.0), the next R gets multiplied by 0
        mask = 1.0 - done[:, t]

        R = rewards[:, t] + gamma * R * mask
        returns[:, t] = R

    return returns


# NOTE: Could change the approach so user must properly slice (remove V_0) and append the final values to `next_values` to ensure the returns are computed correctly.
# TODO: should also reset on truncated not just terminated
def compute_n_step_returns(
    rewards: torch.Tensor,
    terminated: torch.Tensor,
    truncated: torch.Tensor,
    values: torch.Tensor,
    next_values: torch.Tensor,
    gamma: float,
    n: int,
) -> torch.Tensor:
    """
    Computes batched n-step bootstrapped discounted returns.
    Correctly handles truncated states by bootstrapping from the value function,
    while terminated states do not bootstrap.

    Args:
        rewards (torch.Tensor): The reward tensor of shape [batch, time].
        terminated (torch.Tensor): Boolean/float mask indicating episode termination.
            If 1.0/True, the return is not propagated from the next step and NO bootstrap occurs.
            Shape [batch, time].
        truncated (torch.Tensor): Boolean/float mask indicating episode truncation.
            If 1.0/True, the return is not propagated from the next step but bootstrapping DOES occur.
            Shape [batch, time].
        values (torch.Tensor): The values tensor of shape [batch, time].
            Note: This should be V(s_0), ..., V(s_{T-1}).
        next_values (torch.Tensor): The values for the next state in the batch.
            Note: This should be V(s_1), ..., V(s_T). Shape [batch, time].
        gamma (float): The discount factor.
        n (int): The number of steps to lookahead.

    Returns:
        torch.Tensor: The discounted returns of shape [batch, time].
    """
    rewards = rearrange(rewards.squeeze(-1), "b t -> b t")
    b, t = rewards.shape

    terminated = rearrange(terminated.squeeze(-1), "b t -> b t", b=b, t=t)
    truncated = rearrange(truncated.squeeze(-1), "b t -> b t", b=b, t=t)
    values = rearrange(values.squeeze(-1), "b t -> b t", b=b, t=t)
    next_values = rearrange(next_values.squeeze(-1), "b t -> b t", b=b, t=t).detach()

    assert n >= 1, f"n-step must be at least 1, got {n}"

    done = (terminated.bool() | truncated.bool()).float()

    # Base case: 1-step returns (Pure Bellman math)
    returns = rewards + gamma * (1.0 - terminated.float()) * next_values

    # Iteratively compute n-step returns
    for i in range(1, n):
        # Shift left: G_{t+1}^{(i)}
        next_returns = torch.zeros_like(returns)
        next_returns[:, :-1] = returns[:, 1:]

        # Valid update mask: steps that have a future step available within the batch
        valid_mask = torch.zeros_like(returns)
        valid_mask[:, :-i] = 1.0

        # Compute (i+1)-step return: G_t^{(i+1)} = R_t + gamma * G_{t+1}^{(i)}
        # We only use G_{t+1} if the trajectory hasn't ended (not done)
        # If it has ended, we keep the previous k-step return (which eventually is the 1-step return)
        updated_returns = rewards + gamma * next_returns

        # Update logic:
        # 1. Must be a valid step in the batch (valid_mask)
        # 2. Must not be the end of a trajectory (not done)
        should_update = valid_mask.bool() & ~done.bool()
        returns = torch.where(should_update, updated_returns, returns)

    return returns


def compute_td_lambda_returns():
    raise NotImplementedError("TD Lambda returns not yet implemented")
