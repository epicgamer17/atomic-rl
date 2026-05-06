import math


def get_linear_schedule(
    step: int, start_val: float, end_val: float, decay_steps: int
) -> float:
    """
    Linearly decays a value from start_val to end_val over decay_steps.

    Args:
        step (int): The current step.
        start_val (float): The starting value.
        end_val (float): The ending value.
        decay_steps (int): The number of steps over which to decay the value.

    Returns:
        float: The scheduled value at the current step.
    """
    # Calculate the fraction of the way through the decay period (capped at 1.0)
    fraction = min(1.0, float(step) / decay_steps)
    return start_val + fraction * (end_val - start_val)


def get_exponential_schedule(
    step: int, start_val: float, end_val: float, decay_rate: float
) -> float:
    """
    Exponentially decays a value, decay rate controls how fast it drops.

    Args:
        step (int): The current step.
        start_val (float): The starting value.
        end_val (float): The ending value.
        decay_rate (float): The decay rate.

    Returns:
        float: The scheduled value at the current step.
    """
    return end_val + (start_val - end_val) * math.exp(-1.0 * step / decay_rate)
