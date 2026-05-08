import torch
from einops import rearrange


EPS = 1e-8


def compute_mean_advantages(returns: torch.Tensor) -> torch.Tensor:
    """
    Compute advantage function as return minus the mean return.
    This is often used in actor critic methods as a simple baseline.

    Args:
        returns (torch.Tensor): The returns for the episode.

    Returns:
        torch.Tensor: The advantages for the episode.
    """
    advantages = returns - returns.mean(dim=-1, keepdim=True)
    if advantages.numel() > 1:
        std = advantages.std()
        if std > EPS:
            # Mean center and scale
            advantages = (advantages - advantages.mean()) / std
        else:
            advantages = torch.zeros_like(advantages)
    else:
        # NOTE if only 1 step was taken, advantage is 0 (just skip that update not enough info to learn from)
        advantages = torch.zeros_like(advantages)
    return advantages


def compute_ema_advantages(returns: torch.Tensor, ema_baseline: float) -> torch.Tensor:
    """
    Computes advantages using an Exponential Moving Average baseline.
    Does NOT mean-center, because the EMA already centers the data.

    Args:
        returns (torch.Tensor): The returns for the episode.
        ema_baseline (float): The exponential moving average of the returns.

    Returns:
        torch.Tensor: The advantages for the episode.
    """
    advantages = returns - ema_baseline
    if advantages.numel() > 1:
        std = advantages.std()
        if std > EPS:
            # Only scale by std, do not subtract the mean!
            advantages = advantages / std
        else:
            advantages = torch.zeros_like(advantages)
    else:
        # NOTE if only 1 step was taken, advantage is 0 (just skip that update not enough info to learn from)
        advantages = torch.zeros_like(advantages)
    return advantages


# TODO: good name for this?
def compute_critic_advantages(
    returns: torch.Tensor, values: torch.Tensor
) -> torch.Tensor:
    """
    Computes simple advantages for Vanilla Policy Gradient using a Critic network.
    Mean-centering is standard practice here.

    Args:
        returns (torch.Tensor): The returns for the episode.
        values (torch.Tensor): The values for the episode.

    Returns:
        torch.Tensor: The advantages for the episode.
    """
    # TODO: should values be detached here or in main loop? what about returns?
    # Values are detached to treat the baseline as a constant for the policy gradient.
    advantages = returns - values.detach()
    if advantages.numel() > 1:
        std = advantages.std()
        if std > EPS:
            # Mean center and scale
            advantages = (advantages - advantages.mean()) / std
        else:
            advantages = torch.zeros_like(advantages)
    else:
        # NOTE if only 1 step was taken, advantage is 0 (just skip that update not enough info to learn from)
        advantages = torch.zeros_like(advantages)
    return advantages


def compute_gae(
    rewards: torch.Tensor,
    terminated: torch.Tensor,
    truncated: torch.Tensor,
    values: torch.Tensor,
    next_values: torch.Tensor,
    gamma: float,
    gae_lambda: float,
) -> torch.Tensor:
    """
    Computes the Generalized Advantage Estimate (GAE) for a batch of trajectories.
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
        gae_lambda (float): The GAE lambda parameter (typically 0.95 or 0.99).

    Returns:
        torch.Tensor: The GAE values of shape [batch, time].
    """
    # 1. Safely strip trailing 1s, assert it's 2D, lock in (B, T) sizes
    rewards = rearrange(rewards.squeeze(-1), "b t -> b t")
    b, t = rewards.shape

    terminated = rearrange(terminated.squeeze(-1), "b t -> b t", b=b, t=t)
    truncated = rearrange(truncated.squeeze(-1), "b t -> b t", b=b, t=t)
    values = rearrange(values.squeeze(-1), "b t -> b t", b=b, t=t)
    next_values = rearrange(next_values.squeeze(-1), "b t -> b t", b=b, t=t)

    # Combined done mask to prevent bleedthrough between trajectories
    done = (terminated.bool() | truncated.bool()).float()

    # Pure Bellman Math (No more torch.cat assembly here!)
    deltas = rewards + gamma * (1.0 - terminated.float()) * next_values - values

    gae = torch.zeros_like(rewards)
    last_gae = torch.zeros_like(rewards[:, 0])  # [B]

    for t in reversed(range(rewards.shape[1])):
        mask = 1.0 - done[:, t].float()
        last_gae = deltas[:, t] + gamma * gae_lambda * mask * last_gae
        gae[:, t] = last_gae

    return gae
