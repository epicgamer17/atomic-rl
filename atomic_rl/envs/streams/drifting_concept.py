import torch
import math
from typing import Iterator, Tuple


def make_drifting_concept_task(
    num_features: int = 20,
    num_relevant: int = 5,
    flip_interval: int = 20,
    obs_noise_var: float = 0.0,
) -> Iterator[Tuple[torch.Tensor, torch.Tensor]]:
    """
    Creates a non-stationary tracking task as defined in Sutton (1992a) IDBD.
    The true weights are +1 or -1. Every `flip_interval` steps, one of the
    relevant weights randomly flips its sign.

    Args:
        num_features: Total number of features.
        num_relevant: Number of features that affect the target.
        flip_interval: Steps between a weight sign flip.
        obs_noise_var: Variance of the observation noise.

    Yields:
        Tuple of (inputs, target)
    """
    true_weights = torch.zeros(num_features)
    # Initialize relevant weights to randomly +1 or -1
    true_weights[:num_relevant] = torch.randint(0, 2, (num_relevant,)).float() * 2 - 1

    step_count = 0
    while True:
        # Flip sign of one random relevant feature occasionally
        if step_count > 0 and step_count % flip_interval == 0:
            idx = torch.randint(0, num_relevant, (1,)).item()
            true_weights[idx] *= -1

        inputs = torch.randn(num_features)
        noise = torch.randn(1).squeeze() * math.sqrt(obs_noise_var)
        target = torch.dot(true_weights, inputs) + noise

        step_count += 1
        yield inputs, target
