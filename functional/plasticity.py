"""
Notes on Selective Weight Reinitialization:

It is a much simpler and in many cases more effective method that Continual Backprop. In fact I tried to implement continual backprop but decided to remove it from the repository due to its unecessary complexity. Unlike continual backprop, you do not need to design different methods generating and testing or a new CBP class for each layer type, instead you can use a generic implementation for all layer types. Not only does this have the benefit of being MUCH easier to implement and use, it is also often better for large networks with layer normalization (where CBP performed poorly) which is now standard practice to use in most state of the art models. However unlike SWR, CBP doesn't just reset weights; it identifies useless features (whole neurons or filters), replaces them with newly generated ones, and wires them into the network. In some ways this may be more correct or at least an important alternative. TODO: in a clean way implement CBP

CBP:
- Harder to generalize architecturally,
- fewer replacement decisions,
- more semantically meaningful changes.
- preserves feature-level meaning

SWR:
- Easy to apply everywhere,
- potentially noisier,
- less coherent,
- more local/random.
"""

import torch
import torch.nn as nn
from typing import Callable, Iterable


def compute_gradient_utility(weight: torch.Tensor, grad: torch.Tensor) -> torch.Tensor:
    """
    Computes the Gradient Utility of weights as defined in the SWR paper.
    Utility = |w * g_w| (First-order Taylor approximation of loss change if weight is zeroed).

    Args:
        weight (torch.Tensor): The parameter tensor.
        grad (torch.Tensor): The gradient tensor of the parameter.

    Returns:
        torch.Tensor: Utility tensor of the same shape as weight.
    """
    assert weight.shape == grad.shape, "Weight and gradient shapes must match."
    return torch.abs(weight * grad)


def compute_magnitude_utility(weight: torch.Tensor) -> torch.Tensor:
    """
    Computes the Magnitude Utility of weights.
    Utility = |w|

    Args:
        weight (torch.Tensor): The parameter tensor.

    Returns:
        torch.Tensor: Utility tensor of the same shape as weight.
    """
    return torch.abs(weight)


def get_threshold_pruning_mask(
    utilities: torch.Tensor, threshold_factor: float
) -> torch.BoolTensor:
    """
    Computes a boolean mask for Threshold Pruning.
    Prunes weights where utility <= (threshold_factor * mean(utilities)).

    Args:
        utilities (torch.Tensor): The utility values.
        threshold_factor (float): The multiplier 'k' for the threshold.

    Returns:
        torch.BoolTensor: A boolean mask where True indicates the weight should be pruned.
    """
    threshold = threshold_factor * utilities.mean()
    return utilities <= threshold


def get_proportional_pruning_mask(
    utilities: torch.Tensor, proportional_factor: float
) -> torch.BoolTensor:
    """
    Computes a boolean mask for Proportional Pruning.
    Prunes a fixed proportion 'k' of the weights. Handles decimals via Bernoulli sampling
    as explicitly specified in the paper: Integer Part(k*d) + Bernoulli(Decimal Part(k*d)).

    Args:
        utilities (torch.Tensor): The utility values.
        proportional_factor (float): The target proportion 'k' of weights to prune.

    Returns:
        torch.BoolTensor: A boolean mask where True indicates the weight should be pruned.
    """
    d = utilities.numel()
    kd = proportional_factor * d

    num_pruned = int(kd)
    dec_part = kd - num_pruned

    if dec_part > 0:
        # Bernoulli sample the fractional remainder to get exact expected pruning rate
        num_pruned += torch.bernoulli(torch.tensor(dec_part)).int().item()

    num_pruned = min(num_pruned, d)
    mask = torch.zeros_like(utilities, dtype=torch.bool)

    if num_pruned > 0:
        # Find the indices of the smallest 'num_pruned' utilities
        flat_utils = utilities.view(-1)
        _, indices = torch.topk(flat_utils, num_pruned, largest=False)

        # Apply to flat mask
        flat_mask = mask.view(-1)
        flat_mask[indices] = True

    return mask


def reset_optimizer_states_elementwise(
    optimizer: torch.optim.Optimizer, param: nn.Parameter, mask: torch.BoolTensor
) -> None:
    """
    Zeroes out the optimizer states (e.g., Adam momentum and variance) for specific
    elements of a parameter defined by a boolean mask. This prevents the optimizer from
    immediately pulling freshly reinitialized weights back to their old trajectories.

    Args:
        optimizer (torch.optim.Optimizer): The optimizer maintaining state.
        param (nn.Parameter): The parameter whose state needs resetting.
        mask (torch.BoolTensor): Boolean mask where True means reset the state.
    """
    if param not in optimizer.state:
        return

    state = optimizer.state[param]
    keys_to_reset = ["exp_avg", "exp_avg_sq", "momentum_buffer", "max_exp_avg_sq"]

    with torch.no_grad():
        for key in keys_to_reset:
            if key in state:
                state[key][mask] = 0.0


# --- High-Level Orchestration ---


def apply_selective_weight_reinitialization(
    parameters: Iterable[nn.Parameter],
    optimizer: torch.optim.Optimizer,
    init_fn: Callable[[torch.Tensor], None],
    k: float = 1e-4,
    utility_type: str = "gradient",
    prune_type: str = "threshold",
) -> None:
    """
    Orchestrates Selective Weight Reinitialization (SWR) for a given set of parameters.
    This modifies the parameters and the optimizer in-place.

    Fail Fast Note: If using 'gradient' utility, this MUST be called after loss.backward()
    so param.grad is populated, and before optimizer.zero_grad().

    Args:
        parameters: Iterable of parameters (e.g., model.parameters()).
        optimizer: The optimizer being used to train the parameters.
        init_fn: A function that applies the desired reinitialization (e.g., orthogonal init) to a tensor.
        k: The reinitialization factor/threshold multiplier.
        utility_type: "gradient" or "magnitude".
        prune_type: "threshold" or "proportional".
    """
    for param in parameters:
        if not param.requires_grad:
            continue

        with torch.no_grad():
            # 1. Utility Calculation
            if utility_type == "gradient":
                if param.grad is None:
                    raise RuntimeError(
                        "Gradient utility requested, but param.grad is None. "
                        "Ensure apply_selective_weight_reinitialization is called AFTER loss.backward()."
                    )
                utilities = compute_gradient_utility(param, param.grad)
            elif utility_type == "magnitude":
                utilities = compute_magnitude_utility(param)
            else:
                raise ValueError(
                    f"Unknown utility_type: '{utility_type}'. Use 'gradient' or 'magnitude'."
                )

            # 2. Pruning Mask Calculation
            if prune_type == "threshold":
                mask = get_threshold_pruning_mask(utilities, k)
            elif prune_type == "proportional":
                mask = get_proportional_pruning_mask(utilities, k)
            else:
                raise ValueError(
                    f"Unknown prune_type: '{prune_type}'. Use 'threshold' or 'proportional'."
                )

            if not mask.any():
                continue

            # 3. Resample Reinitialization
            # We create a temporary tensor, apply the init logic, and copy only the masked elements
            temp_tensor = torch.empty_like(param)
            init_fn(temp_tensor)
            param.copy_(torch.where(mask, temp_tensor, param))

        # 4. Reset Optimizer Momentum (Must be out of the main no_grad context but acts in-place)
        reset_optimizer_states_elementwise(optimizer, param, mask)


# METRICS TODO: find a good file to put these in
def compute_dead_units_proportion(activations: torch.Tensor) -> float:
    """
    Computes the proportion of dead units in a layer.
    A unit is considered 'dead' if it outputs exactly 0 for an entire batch of data.

    Args:
        activations (torch.Tensor): The output tensor of a ReLU layer of shape [Batch, Features].

    Returns:
        float: The percentage of dead units (0.0 to 1.0).
    """
    # Check if a unit is 0 across the entire batch dimension (dim=0)
    dead_mask = (activations <= 0).all(dim=0)
    return dead_mask.sum().item() / dead_mask.numel()


def compute_average_weight_magnitude(parameters: Iterable[nn.Parameter]) -> float:
    """Computes the mean absolute value of all weights in the given parameters."""
    total_mag = 0.0
    total_elements = 0
    with torch.no_grad():
        for param in parameters:
            total_mag += torch.abs(param).sum().item()
            total_elements += param.numel()
    return total_mag / total_elements if total_elements > 0 else 0.0


def compute_average_gradient_magnitude(parameters: Iterable[nn.Parameter]) -> float:
    """Computes the mean absolute value of all gradients in the given parameters."""
    total_mag = 0.0
    total_elements = 0
    with torch.no_grad():
        for param in parameters:
            if param.grad is not None:
                total_mag += torch.abs(param.grad).sum().item()
                total_elements += param.grad.numel()
    return total_mag / total_elements if total_elements > 0 else 0.0


def compute_stable_rank(activations: torch.Tensor) -> float:
    """
    Computes the stable rank of the representation matrix.
    Stable Rank = (Frobenius Norm)^2 / (Spectral Norm)^2
    It measures the effective dimensionality/expressivity of the features.

    Args:
        activations (torch.Tensor): Tensor of shape [Batch, Features].

    Returns:
        float: The stable rank.
    """
    # Center the activations
    centered = activations - activations.mean(dim=0, keepdim=True)

    # Compute singular values
    # We use svdvals for numerical stability instead of full SVD
    singular_values = torch.linalg.svdvals(centered)

    squared_sv = singular_values**2

    # Frobenius norm squared is the sum of squared singular values
    # Spectral norm squared is the max squared singular value
    max_sq_sv = squared_sv.max().item()
    if max_sq_sv < 1e-8:
        return 0.0

    return squared_sv.sum().item() / max_sq_sv
