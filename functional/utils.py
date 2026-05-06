import torch


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
