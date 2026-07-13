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


def get_ape_x_epsilon(
    actor_id: int, num_actors: int, base_eps: float = 0.4, alpha: float = 7.0
) -> float:
    """
    Calculates the fixed epsilon for a specific actor in APE-X.

    Args:
        actor_id (int): The ID of the actor.
        num_actors (int): The total number of actors.
        base_eps (float): The base epsilon value.
        alpha (float): The alpha parameter for the distribution.
    """
    if num_actors <= 1:
        return base_eps
    return base_eps ** (1 + (actor_id / (num_actors - 1)) * alpha)
