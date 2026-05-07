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
    terminals: torch.Tensor,
    values: torch.Tensor,
    last_values: torch.Tensor,
    gamma: float,
    gae_lambda: float,
):
    """
    Computes the Generalized Advantage Estimate (GAE) for a batch of trajectories.

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
        gae_lambda (float): The GAE lambda parameter (typically 0.95 or 0.99).

    Returns:
        torch.Tensor: The GAE values of shape [batch, time].
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

    next_values = torch.cat([values[:, 1:], last_values], dim=1)

    deltas = rewards + gamma * (1.0 - terminals.float()) * next_values - values
    gae = torch.zeros_like(rewards)
    last_gae = torch.zeros_like(rewards[:, 0])  # [B]
    for t in reversed(range(rewards.shape[1])):
        mask = 1.0 - terminals[:, t].float()

        # A_t = delta_t + gamma * lambda * mask * A_{t+1}
        last_gae = deltas[:, t] + gamma * gae_lambda * mask * last_gae
        gae[:, t] = last_gae
    return gae
