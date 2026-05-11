import torch
import numpy as np
from typing import Tuple, List
from tensordict import TensorDict


def _allocate_tensordict(
    shapes: dict, batch_size: List[int], device: str = "cpu"
) -> TensorDict:
    """Allocates a zeroed TensorDict of any arbitrary geometry."""
    data = TensorDict({}, batch_size=batch_size, device=device)
    for key, shape in shapes.items():
        dtype = torch.long if "action" in key else torch.float32
        data.set(key, torch.zeros((*batch_size, *shape), dtype=dtype))
    return data


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
    assert (
        old_ema.shape == new_value.shape
    ), f"EMA shape mismatch: {old_ema.shape} vs {new_value.shape}"
    return (1 - alpha) * old_ema + alpha * new_value


# TODO: messy, hard to read, and doesnt work reliably or at least not tested reliably on both gym vector envs and pufferlib's vector envs.
def extract_vector_env_final_obs(info) -> Tuple[np.ndarray, np.ndarray]:
    """
    Extracts the true final observations from a Gymnasium Vector Environment's info dict.
    Safely handles the auto-reset hidden states.

    Some vector-env wrappers (notably PufferLib's `pufferlib.vector`) deliver
    `info` as a list rather than a Gymnasium-style dict, and do not preserve
    the truncation-state observation at all. In those cases this returns
    empty arrays — the caller should not bootstrap through the truncated
    next-state since `V(s_truncated)` is not recoverable.

    Args:
        info: The info object returned by `envs.step()`. Expected to be a dict
            with `final_observation` / `_final_observation` keys (Gymnasium
            vector envs). Any other type (list, None, etc.) is treated as
            "no final observations available" and yields empty arrays.

    Returns:
        Tuple[np.ndarray, np.ndarray]:
            - env_indices: 1D array of environment indices that terminated or truncated.
            - final_obs: Stacked array of the true final observations for those environments.
              Returns (empty_array, empty_array) if no environments ended or if the
              info format does not expose final observations.
    """
    # Non-dict info formats (e.g. PufferLib emits a list) cannot expose
    # final observations; bail out cleanly.
    # TODO: implement correct behaviour for pufferlib
    if not isinstance(info, dict):
        return np.array([]), np.array([])

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

    # Use explicit list comprehension to extract items using the NumPy array indices.
    # This avoids the TypeError from "fancy indexing" a Python list and avoids
    # allocating a slow intermediate np.array(..., dtype=object).
    valid_observations = [info["final_observation"][i] for i in env_indices]
    true_final_obs = np.stack(valid_observations)

    return env_indices, true_final_obs


def standardize_tensor(tensor: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """
    Mean-centers and scales a tensor by its standard deviation.
    Used when the baseline does not perfectly center the current batch (e.g., PPO Critic).
    """
    if tensor.numel() <= 1:
        return torch.zeros_like(tensor)

    return (tensor - tensor.mean()) / (tensor.std() + eps)


def scale_tensor_by_std(tensor: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """
    Scales a tensor by its standard deviation WITHOUT mean-centering.
    Used when the data is already centered (e.g., EMA Advantages).
    """
    if tensor.numel() <= 1:
        return torch.zeros_like(tensor)

    return tensor / (tensor.std() + eps)


def add_dirichlet_noise(
    probs: torch.Tensor,
    epsilon: float,
    alpha: float,
) -> torch.Tensor:
    """
    Adds Dirichlet noise to the given probabilities for exploration.
    Formula: P(s, a) = (1 - epsilon) * P(s, a) + epsilon * noise
    where noise ~ Dirichlet(alpha).

    Args:
        probs: The probabilities to add noise to [..., Num_Actions].
        epsilon: The weight of the noise.
        alpha: The concentration parameter of the Dirichlet distribution.

    Returns:
        The probabilities with added noise.
    """
    # 1. Sample noise from Dirichlet distribution
    num_actions = probs.shape[-1]
    alphas = torch.full((num_actions,), alpha, device=probs.device, dtype=probs.dtype)

    # torch.distributions.Dirichlet handles batch shapes correctly via .sample()
    dist = torch.distributions.Dirichlet(alphas)
    noise = dist.sample(probs.shape[:-1])  # [..., Num_Actions]

    # 2. Combine with original probabilities
    return (1.0 - epsilon) * probs + epsilon * noise


def add_gumbel_noise():
    # TODO: implement this
    pass
