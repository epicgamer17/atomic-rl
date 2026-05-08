import torch
import numpy as np
from typing import Tuple


# NOTE: DONT LET THIS FILE BUILD UP AND HAVE A LOT FUNCTIONS, THAT IS A SIGN OF BAD ORGANIZATION.


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


def extract_vector_env_final_obs(info: dict) -> Tuple[np.ndarray, np.ndarray]:
    """
    Extracts the true final observations from a Gymnasium Vector Environment's info dict.
    Safely handles the auto-reset hidden states.

    Args:
        info (dict): The info dictionary returned by `envs.step()`.

    Returns:
        Tuple[np.ndarray, np.ndarray]:
            - env_indices: 1D array of environment indices that terminated or truncated.
            - final_obs: Stacked array of the true final observations for those environments.
              Returns (empty_array, empty_array) if no environments ended.
    """
    # Vector envs only add this key if at least one environment ended
    if "final_observation" not in info:
        return np.array([]), np.array([])

    # Gymnasium uses "_final_observation" as the boolean mask for which envs actually ended
    mask = info.get("_final_observation")
    if mask is None:
        return np.array([]), np.array([])

    env_indices = np.where(mask)[0]

    if len(env_indices) == 0:
        return np.array([]), np.array([])

    # Extract only the valid observations (the rest are usually None)
    true_final_obs = np.stack(info["final_observation"][env_indices])

    return env_indices, true_final_obs

