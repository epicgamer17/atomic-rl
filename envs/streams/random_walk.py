import torch
import math
from typing import Iterator, Tuple


def make_random_walk_tracking_task(
    num_features: int = 20,
    num_relevant: int = 5,
    obs_noise_var: float = 1.0,
    drift_var: float = 1.0,
) -> Iterator[Tuple[torch.Tensor, torch.Tensor]]:
    """
    Creates a non-stationary tracking task as defined in Sutton (1992b).
    The true weights undergo a random walk every step.

    Args:
        num_features: Total number of features.
        num_relevant: Number of features that actually drift and affect the target.
        obs_noise_var: Variance of the observation noise (R).
        drift_var: Variance of the random walk drift (Q).

    Yields:
        Tuple of (inputs, target)
    """
    true_weights = torch.zeros(num_features)

    while True:
        # 1. Update true weights (Drift)
        drift = torch.zeros(num_features)
        drift[:num_relevant] = torch.randn(num_relevant) * math.sqrt(drift_var)
        true_weights += drift

        # 2. Generate inputs and noisy target
        inputs = torch.randn(num_features)
        noise = torch.randn(1).squeeze() * math.sqrt(obs_noise_var)
        target = torch.dot(true_weights, inputs) + noise

        yield inputs, target
