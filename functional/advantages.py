import torch


# TODO once we implement it replace the inlined version in reinforce_cartpole
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
    if len(advantages) > 1:
        # Mean center and scale
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
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
    if len(advantages) > 1:
        # Only scale by std, do not subtract the mean!
        advantages = advantages / (advantages.std() + 1e-8)
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
    if len(advantages) > 1:
        # Mean center and scale
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
    else:
        # NOTE if only 1 step was taken, advantage is 0 (just skip that update not enough info to learn from)
        advantages = torch.zeros_like(advantages)
    return advantages
