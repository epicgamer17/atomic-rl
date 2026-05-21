import torch
import torch.nn as nn
import numpy as np
import random
from typing import Tuple, List, Callable, Optional
from tensordict import TensorDict


def set_seed(seed: int) -> None:
    """
    Sets random seeds for reproducibility across random, numpy, and torch.

    Args:
        seed: The integer seed to use.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        # Ensure deterministic behavior for some ops if needed,
        # though this can slow down performance.
        # torch.backends.cudnn.deterministic = True
        # torch.backends.cudnn.benchmark = False

    if torch.backends.mps.is_available():
        torch.mps.manual_seed(seed)


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


def ema_update(
    old_ema: torch.Tensor, new_value: torch.Tensor, alpha: float, inplace: bool = False
) -> torch.Tensor:
    """
    Calculates the exponential moving average (EMA).
    Formula: (1 - alpha) * old_ema + alpha * new_value
    """
    assert (
        old_ema.shape == new_value.shape
    ), f"EMA shape mismatch: {old_ema.shape} vs {new_value.shape}"

    if inplace:
        # Use optimized in-place kernels: old = old * (1-a) + new * a
        # This is rearranged for .add_ usage: old = old - a*old + a*new => old += a*(new - old)
        return old_ema.lerp_(new_value, alpha)

    return (1.0 - alpha) * old_ema + alpha * new_value


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


def gnt_init_wrapper(
    init_fn: Callable[[torch.Tensor], None],
) -> Callable[[torch.Tensor], None]:
    """
    Standardizes initialization for Generate and Test methods (CBP/SWR).

    Wraps/Returns a new init fn to be used for GnT. If it's a weight matrix (2D+),
    it uses the provided init_fn. If it's a bias vector (1D), it uses zeros.

    Args:
        init_fn: The base initialization function to use for weight matrices.

    Returns:
        Callable[[torch.Tensor], None]: A wrapped initialization function that
            dispatches to init_fn for weights and zeros_ for biases.
    """

    def wrapped_init(tensor: torch.Tensor) -> None:
        if tensor.dim() >= 2:
            init_fn(tensor)
        elif tensor.dim() == 1:
            nn.init.zeros_(tensor)

    return wrapped_init


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


def to_tensor(
    numpy_array: np.ndarray, device: torch.device = torch.device("cpu")
) -> torch.Tensor:
    """
    Converts a numpy array to a float32 PyTorch tensor on the specified device.

    Args:
        numpy_array: The numpy array to convert.
        device: The target device.

    Returns:
        A PyTorch tensor.
    """
    return torch.as_tensor(numpy_array, dtype=torch.float32, device=device)


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


# TODO: is this the same as ema?
def update_welford_stats(
    mean: torch.Tensor, var: torch.Tensor, count: torch.Tensor, batch: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Pure mathematical update for running mean and variance using Welford's online algorithm.
    """
    assert batch.dim() == 2, f"Expected [Batch, Features], got {batch.shape}"
    batch_size = batch.size(0)

    new_count = count + batch_size
    delta = batch - mean.unsqueeze(0)
    new_mean = mean + delta.sum(dim=0) / new_count

    delta2 = batch - new_mean.unsqueeze(0)
    new_var = var + (delta * delta2).sum(dim=0)

    return new_mean, new_var, new_count


def normalize_features(
    features: torch.Tensor,
    mean: torch.Tensor,
    var: torch.Tensor,
    count: torch.Tensor,
    eps: float = 1e-8,
) -> torch.Tensor:
    """
    Applies running normalization to the features.
    """
    # Use unbiased variance if we have more than 1 sample
    unbiased_var = var / torch.clamp(count - 1, min=1.0)
    std = torch.sqrt(unbiased_var + eps)
    return (features - mean) / std
