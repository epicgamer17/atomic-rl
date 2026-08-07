import torch
import torch.nn as nn
import numpy as np
import random
from typing import Tuple, List, Callable, Optional, Union
from tensordict import TensorDict


# NOTE: DONT LET THIS FILE BUILD UP AND HAVE A LOT FUNCTIONS, THAT IS A SIGN OF BAD ORGANIZATION.


# TODO: do we need the inplace and not inplace operations?
def ema_update(
    old_ema: torch.Tensor, new_value: torch.Tensor, alpha: float
) -> torch.Tensor:
    """
    Calculates the exponential moving average (EMA).
    Formula: (1 - alpha) * old_ema + alpha * new_value
    """
    assert (
        old_ema.shape == new_value.shape
    ), f"EMA shape mismatch: {old_ema.shape} vs {new_value.shape}"

    return (1.0 - alpha) * old_ema + alpha * new_value


def ema_update_(
    old_ema: torch.Tensor, new_value: torch.Tensor, alpha: float
) -> torch.Tensor:
    """
    In-place exponential moving average (EMA).
    Formula: (1 - alpha) * old_ema + alpha * new_value
    """
    assert (
        old_ema.shape == new_value.shape
    ), f"EMA shape mismatch: {old_ema.shape} vs {new_value.shape}"

    # Use optimized in-place kernels: old = old * (1-a) + new * a
    # This is rearranged for .add_ usage: old = old - a*old + a*new => old += a*(new - old)
    return old_ema.lerp_(new_value, alpha)


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
    # Note: If `info` is not a dict (e.g. PufferLib lists), this will explicitly crash,
    # which is intended under the fail-fast philosophy unless handled explicitly.
    # TODO: implement correct behaviour for pufferlib

    # Handle None or non-dict inputs gracefully
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
    mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Adds Dirichlet noise to the given probabilities for exploration.
    Formula: P(s, a) = (1 - epsilon) * P(s, a) + epsilon * noise
    where noise ~ Dirichlet(alpha).

    Args:
        probs: The probabilities to add noise to [..., Num_Actions].
        epsilon: The weight of the noise.
        alpha: The concentration parameter of the Dirichlet distribution.
        mask: Optional boolean mask [..., Num_Actions] of valid actions.

    Returns:
        The probabilities with added noise.
    """
    # 1. Sample noise from Dirichlet distribution
    num_actions = probs.shape[-1]
    alphas = torch.full((num_actions,), alpha, device=probs.device, dtype=probs.dtype)

    # torch.distributions.Dirichlet handles batch shapes correctly via .sample()
    dist = torch.distributions.Dirichlet(alphas)
    noise = dist.sample(probs.shape[:-1])  # [..., Num_Actions]

    if mask is not None:
        # Mask out noise for illegal actions
        noise = noise * mask.to(probs.dtype)

        # Re-normalize the noise so it still sums to 1.0 across valid actions
        # Add 1e-8 to prevent division by zero in case of severe precision issues
        noise_sum = noise.sum(dim=-1, keepdim=True)
        noise = noise / (noise_sum + 1e-8)

    return (1.0 - epsilon) * probs + epsilon * noise


def to_tensor(
    numpy_array: Union[np.ndarray, int, float, bool],
    device: torch.device = torch.device("cpu"),
    dtype: Optional[torch.dtype] = torch.float32,
) -> torch.Tensor:
    """
    Converts a numpy array (or primitive) to a PyTorch tensor on the specified device.

    Args:
        numpy_array: The numpy array or primitive to convert.
        device: The target device.
        dtype: The target data type. Defaults to float32.

    Returns:
        A PyTorch tensor.
    """
    return torch.as_tensor(numpy_array, dtype=dtype, device=device)


# TODO: does this handle pendulum?
def to_numpy_action(action_tensor: torch.Tensor) -> np.ndarray:
    """
    Converts a PyTorch action tensor to a numpy array for Gymnasium consumption.
    Handles detaching, moving to CPU, and flattening for discrete spaces.

    Args:
        action_tensor: The action tensor from the model.

    Returns:
        A numpy array of actions.
    """
    res = action_tensor.detach().cpu().numpy()
    # For discrete actions (LongTensor/IntTensor), we cast to int32
    if action_tensor.dtype in [torch.long, torch.int]:
        # Flatten if it's a single action per batch entry (e.g., [B, 1] -> [B])
        # This is the standard for Gymnasium VectorEnv consumption of discrete actions.
        if res.ndim > 1 and res.shape[-1] == 1:
            return res.flatten().astype(np.int32)
        return res.astype(np.int32)
    # For continuous actions, we keep the original shape and dtype (float32)
    return res


# TODO: should this be inplace?
# TODO: there is a tension between Gym which uses numpy and training loops which use torch. If this is to be used in a wrapper for the environment, then it should probably be numpy, but if we want to use it in simple training loops, torch is nicer.
# TODO: Rename to sample mean var?
def update_welford_stats(
    mean: torch.Tensor, sq_diff: torch.Tensor, count: torch.Tensor, batch: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Pure mathematical update for running mean and variance using Welford's online algorithm.

    Args:
        mean: current running mean of the features, initially 0.0. [*F]
        sq_diff: current running sum of squared differences from the mean, initially 1.0. [*F]
        count: current number of samples seen. (1)
        batch: the new batch of features to update with. (B, *F)

    Returns:
        new_mean, new_sq_diff, new_var, new_count
    """

    # Fail Fast: Strict layout validation to catch dimension errors immediately
    assert batch.ndim > mean.ndim, (
        f"Shape Mismatch: Incoming batch must contain an explicit leading batch dimension. "
        f"Batch shape: {batch.shape}, Mean shape: {mean.shape}"
    )
    assert batch.shape[1:] == mean.shape == sq_diff.shape, (
        f"Geometry mismatch. Batch features: {batch.shape[1:]}, "
        f"Mean features: {mean.shape}, Sq_diff features: {sq_diff.shape}"
    )

    batch_size = batch.size(0)

    new_count = count + batch_size
    delta = batch - mean.unsqueeze(0)
    new_mean = mean + delta.sum(dim=0) / new_count

    delta2 = batch - new_mean.unsqueeze(0)
    new_sq_diff = sq_diff + (delta * delta2).sum(dim=0)
    new_var = new_sq_diff / (new_count - 1)
    if new_count < 2:
        new_var = torch.ones_like(new_var)

    return new_mean, new_sq_diff, new_var, new_count


# TODO: where to put this?
# TODO: kind of messy, untested, and very mountaincar specific (in respect to defaults and dimensionality)
def compute_tile_coding_features(
    state: np.ndarray,
    action: int,
    num_actions: int,
    num_tilings: int = 10,
    tiles_per_tiling: int = 10,
    state_low: np.ndarray = np.array([-1.2, -0.07]),
    state_high: np.ndarray = np.array([0.6, 0.07]),
) -> torch.Tensor:
    """
    Computes a sparse binary feature vector phi(s, a) using tile coding.
    Follows 'Documentation by Signature' and 'Fail Fast' by strictly enforcing shapes.

    According to the Stream RL paper, tile coding has been shown to reduce forgetting, as a form of "sparse initialization" but really sparse representation.
    """
    # 1. Clip the state to the bounds strictly
    clipped_state = np.clip(state, state_low, state_high)

    # 2. Normalize state to [0, tiles_per_tiling]
    state_normalized = (
        (clipped_state - state_low) / (state_high - state_low) * tiles_per_tiling
    )

    # 2. Compute asymmetrical offsets for each tiling to prevent diagonal artifacts
    # Using 1 and 3 as displacement multipliers (Sutton's recommendation for 2D)
    offsets = np.zeros((num_tilings, len(state)))
    for i in range(num_tilings):
        offsets[i, 0] = ((i * 1) % num_tilings) / num_tilings
        if len(state) > 1:
            offsets[i, 1] = ((i * 3) % num_tilings) / num_tilings

    # 3. Find the active tile in each tiling
    active_tiles = []
    for i in range(num_tilings):
        tile_coords = np.floor(state_normalized + offsets[i]).astype(int)
        # Map 2D coordinates to a 1D index for this specific tiling
        # Handling up to 2D for Mountain Car.
        if len(state) == 2:
            tile_idx = (
                i * ((tiles_per_tiling + 1) ** 2)
                + tile_coords[0] * (tiles_per_tiling + 1)
                + tile_coords[1]
            )
        else:
            tile_idx = i * (tiles_per_tiling + 1) + tile_coords[0]

        active_tiles.append(tile_idx)

    # 4. Create the state-action flat vector
    features_per_action = num_tilings * ((tiles_per_tiling + 1) ** 2)
    total_features = features_per_action * num_actions

    phi = torch.zeros(total_features, dtype=torch.float64)

    # Offset by the action to make it phi(s, a)
    action_offset = action * features_per_action
    for tile_idx in active_tiles:
        phi[action_offset + tile_idx] = 1.0

    return phi
