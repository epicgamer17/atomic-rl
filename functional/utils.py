import torch
import math


def exponential_moving_average(
    old_ema: torch.Tensor, new_value: torch.Tensor, alpha: float
):
    """
    Calculates the exponential moving average (EMA) of a value.

    Args:
        old_ema (torch.Tensor): The previous EMA value.
        new_value (torch.Tensor): The new value to incorporate into the EMA.
        alpha (float): The EMA smoothing factor (between 0 and 1).

    Returns:
        torch.Tensor: The updated EMA value.
    """
    return (1 - alpha) * old_ema + alpha * new_value


# TODO: make this not only for epsilon from epsilon greedy
def get_linear_epsilon(
    step: int, start_eps: float, end_eps: float, decay_steps: int
) -> float:
    """
    Linearly decays epsilon from start_eps to end_eps over decay_steps.

    Args:
        step (int): The current step.
        start_eps (float): The starting epsilon.
        end_eps (float): The ending epsilon.
        decay_steps (int): The number of steps over which to decay epsilon.
    """
    # Calculate the fraction of the way through the decay period (capped at 1.0)
    fraction = min(1.0, float(step) / decay_steps)
    return start_eps - fraction * (start_eps - end_eps)


# TODO: make this not only for epsilon from epsilon greedy
def get_exponential_epsilon(
    step: int, start_eps: float, end_eps: float, decay_rate: float
) -> float:
    """
    Exponentially decays epsilon, decay rate controls how fast it drops.

    Args:
        step (int): The current step.
        start_eps (float): The starting epsilon.
        end_eps (float): The ending epsilon.
        decay_rate (float): The decay rate.
    """
    return end_eps + (start_eps - end_eps) * math.exp(-1.0 * step / decay_rate)


def get_linear_beta(
    step: int, start_beta: float, end_beta: float, anneal_steps: int
) -> float:
    """
    Linearly anneals beta from start_beta to end_beta over anneal_steps.
    Beta is used for Importance Sampling correction in PER.
    """
    fraction = min(1.0, float(step) / anneal_steps)
    return start_beta + fraction * (end_beta - start_beta)
