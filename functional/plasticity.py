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

TODO: add CBP Notes.
"""

import torch
import torch.nn as nn
from typing import Callable, Iterable
from functional.utils import ema_update

# TODO: some messy is instance checks. these are necessary for IDBD and linear value heads and stuff, but maybe there is a cleaner way to do this.

# TODO: The functions now return masks, but examples have not been updated to use this interface yet, and at the moment pretend there are no masks. NOTE: it may be that for nn.Modules (ie standard deep learning) the function has an implicit effect, whereas for linear and other methods (alberta plan) the mask is used. im not sure if i love this and ideally it would be unified across both.


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
    optimizer: torch.optim.Optimizer,
    param: nn.Parameter | torch.Tensor,
    mask: torch.BoolTensor,
) -> None:
    """
    Zeroes out the optimizer states (e.g., Adam momentum and variance) for specific
    elements of a parameter defined by a boolean mask. This prevents the optimizer from
    immediately pulling freshly reinitialized weights back to their old trajectories.

    Args:
        optimizer (torch.optim.Optimizer): The optimizer maintaining state.
        param (nn.Parameter | torch.Tensor): The parameter whose state needs resetting.
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
    init_fn: Callable[
        [torch.Tensor], None
    ],  # TODO: does this need to be from a distribution like CBP?
    k: float = 1e-4,
    utility_type: str = "gradient",
    prune_type: str = "threshold",
) -> dict[nn.Parameter, torch.BoolTensor]:
    """
    Orchestrates Selective Weight Reinitialization (SWR) for a given set of parameters.
    This modifies the parameters and the optimizer in-place.

    Fail Fast Note: If using 'gradient' utility, this MUST be called after loss.backward()
    so param.grad is populated, and before optimizer.zero_grad().

    Args:
        parameters: Iterable of parameters (e.g., model.parameters()).
        optimizer: The optimizer being used to train the parameters.
        init_fn: A function that applies the desired reinitialization (e.g., orthogonal init) to a tensor.
            TIP: Use `functional.utils.gnt_init_wrapper` to ensure biases are correctly zeroed.
        k: The reinitialization factor/threshold multiplier.
        utility_type: "gradient" or "magnitude".
        prune_type: "threshold" or "proportional".

    Example:
        >>> from functional.utils import gnt_init_wrapper
        >>> init_fn = gnt_init_wrapper(nn.init.orthogonal_)
        >>> masks_applied = apply_selective_weight_reinitialization(model.parameters(), optimizer, init_fn)
    """
    masks_applied = {}
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
                masks_applied[param] = mask
                continue

            # 3. Resample Reinitialization
            # We create a temporary tensor, apply the init logic, and copy only the masked elements
            temp_tensor = torch.empty_like(param)
            init_fn(temp_tensor)
            param.copy_(torch.where(mask, temp_tensor, param))

        # 4. Reset Optimizer Momentum (Must be out of the main no_grad context but acts in-place)
        reset_optimizer_states_elementwise(optimizer, param, mask)
        masks_applied[param] = mask

    return masks_applied


def init_cbp_state(layer: nn.Linear) -> dict[str, torch.Tensor]:
    """
    Initializes the state tracking tensors required for Continual Backpropagation (CBP)
    for a given linear layer. This state should be preserved across training steps.

    Args:
        layer (nn.Linear): The layer whose features/neurons will be tracked.

    Returns:
        dict[str, torch.Tensor]: A dictionary containing initialized state tensors.
    """
    device = layer.weight.device
    num_features = layer.out_features
    return {
        "ages": torch.zeros(num_features, device=device),
        "utilities": torch.zeros(num_features, device=device),
        "avg_activations": torch.zeros(num_features, device=device),
    }


def get_cbp_replacement_mask(
    utilities: torch.Tensor,
    eligible_mask: torch.BoolTensor,
    replacement_rate: float,
) -> torch.BoolTensor:
    """
    Computes the boolean mask for CBP replacement.
    Selects a fraction of features (defined by replacement_rate) that have the
    lowest utility, but only among those marked as eligible (past maturity).

    Args:
        utilities (torch.Tensor): Bias-corrected utilities for the features.
        eligible_mask (torch.BoolTensor): True for features past the maturity threshold.
        replacement_rate (float): The target proportion 'rho' of total weights to replace.

    Returns:
        torch.BoolTensor: A boolean mask where True indicates the feature should be replaced.
    """
    num_features = utilities.numel()
    kd = replacement_rate * num_features

    num_replace = int(kd)
    dec_part = kd - num_replace

    if dec_part > 0:
        # Bernoulli sample the fractional remainder to get exact expected replacement rate
        num_replace += torch.bernoulli(torch.tensor(dec_part)).int().item()

    num_eligible = eligible_mask.sum().item()
    num_replace = min(num_replace, num_eligible)

    mask = torch.zeros_like(utilities, dtype=torch.bool)

    if num_replace > 0:
        # We want to find the lowest utility among *eligible* units.
        # Give ineligible units a utility of infinity so they aren't chosen.
        masked_utils = torch.where(eligible_mask, utilities, torch.inf)
        _, indices = torch.topk(masked_utils, num_replace, largest=False)

        flat_mask = mask.view(-1)
        flat_mask[indices] = True

    return mask


def apply_continual_backprop(
    layer_pairs: Iterable[tuple[nn.Linear, nn.Linear | torch.Tensor]],
    activations: Iterable[torch.Tensor],
    cbp_states: dict[nn.Linear, dict[str, torch.Tensor]],
    optimizer: torch.optim.Optimizer,  # Contains step size
    init_fn: Callable[
        [torch.Tensor], None
    ],  # TODO: does this need to be from a distribution (does orthogonal work?)
    eta: float = 0.99,  # Decay rate
    maturity_threshold: int = 100,
    replacement_rate: float = 1e-4,
) -> dict[nn.Linear, torch.BoolTensor]:
    """
    Orchestrates Continual Backpropagation (CBP) for a set of feedforward linear layer pairs.
    This calculates contribution/adaptation utilities, updates running statistics, and selectively
    reinitializes the lowest utility neurons alongside their optimizer momentum.

    Fail Fast Note: Call this function AFTER `loss.backward()` and `optimizer.step()`, but ensure
    you pass the post-activation tensors captured during the forward pass.

    TODO/NOTE: PyTorch's default Adam tracks the `step` count as a single scalar per parameter tensor, not element-wise. Therefore, the element-wise timestep reset mentioned in the CBP paper is omitted. Element-wise momentum (`exp_avg`, `exp_avg_sq`) resetting is fully supported but in future perhaps make a GnT version of Adam to fully support the CBP paper?

    Args:
        layer_pairs: Iterable of (layer, next_layer) tuples (e.g., [(layer1, layer2)]).
        activations: Iterable of activation tensors corresponding to the output of the first
                     layer in each pair (post-activation, shape: [Batch, Features]).
        cbp_states: Dictionary mapping `layer` to its state dictionary (from `init_cbp_state`).
        optimizer: The optimizer being used to train the parameters.
        init_fn: A function that applies the desired reinitialization to a tensor.
            NOTE: Use `functional.utils.gnt_init_wrapper` to ensure biases are correctly zeroed. TODO can we do this automatically for the user or somehow enforce Fail Fast on this.
        eta: Decay rate for running averages.
        maturity_threshold: Minimum age before a unit is eligible for replacement.
        replacement_rate: The fraction of units to replace per step (rho).

    Example:
        >>> from functional.utils import gnt_init_wrapper
        >>> init_fn = gnt_init_wrapper(nn.init.orthogonal_)
        >>> apply_continual_backprop(layer_pairs, activations, cbp_states, optimizer, init_fn)

        NOTE: Corresponds to lines 8 onwards in the CBP paper pseudocode (the for each layer loop)
    """

    # Calculate alpha for use in EMA updates
    alpha = 1.0 - eta

    replacement_masks = {}

    for (layer, next_layer), act in zip(layer_pairs, activations):
        state = cbp_states[layer]
        ages = state["ages"]
        utilities = state["utilities"]
        avg_activations = state["avg_activations"]

        with torch.no_grad():
            # 1. Update Age
            ages += 1

            # 2. Update feature utility: Using Equations 4, 5, and 6 TODO where is equation 4 in this block as in the pseudocode?
            # a. Eq 3
            bias_correction = 1.0 - eta ** ages.clamp(min=1)
            f_hat = avg_activations / bias_correction

            # Instantaneous overall utility (Eq 5)
            # Mean absolute difference of activations from running average over the batch
            act_diff = torch.abs(act - f_hat.unsqueeze(0)).mean(dim=0)

            # Sum of absolute outgoing and incoming weights
            if isinstance(next_layer, torch.Tensor):
                out_weight_sum = next_layer.abs().sum(dim=0)
            else:
                out_weight_sum = next_layer.weight.abs().sum(dim=0)
            in_weight_sum = layer.weight.abs().sum(dim=1).clamp(min=1e-8)

            # NOTE: We skip explicitly tracking Eq 4 (Contribution EMA) to save memory.
            # Instead, we calculate instantaneous contribution/adaptation and apply
            # the EMA (Eq 6) to the overall result. This is mathematically equivalent.
            instant_utility = (act_diff * out_weight_sum) / in_weight_sum

            # c. Update running averages (Eq 2, Eq 6 and Eq 4?)
            ema_update(avg_activations, act.mean(dim=0), alpha=alpha, inplace=True)
            ema_update(utilities, instant_utility, alpha=alpha, inplace=True)

            u_hat = utilities / bias_correction
            # 3. Find eligible features: Features with age more than m
            eligible_mask = ages > maturity_threshold

            # 4. Features to replace: n_l * rho of eligible features with smallest utility
            mask = get_cbp_replacement_mask(u_hat, eligible_mask, replacement_rate)

            if not mask.any():
                replacement_masks[layer] = mask
                continue

            # Apply Replacement, Lines 5, 6, 7 of the pseudo code loop over layers

            # Transfer contribution to the bias of consumer (next_layer) (NOTE: not in pseudocode but mentioned in paper)
            if not isinstance(next_layer, torch.Tensor) and next_layer.bias is not None:
                # Multiply outgoing weights by the bias-corrected average activation of the removed unit
                contribution = (
                    next_layer.weight[:, mask] * f_hat[mask].unsqueeze(0)
                ).sum(dim=1)
                next_layer.bias.add_(contribution)

            # 7. Reset input weights and bias (NOTE bias part not in pseudocode but mentioned in paper)
            # TODO: if using standard init_fns can this be simpler?
            temp_weight = torch.empty_like(layer.weight)
            init_fn(temp_weight)
            # Using mask.unsqueeze(1) expands the 1D feature mask across the in_features dimension (Rows)
            layer.weight.copy_(
                torch.where(mask.unsqueeze(1), temp_weight, layer.weight)
            )

            # TODO: how should this be handled? Should it be initialized like it was at the beginning of training or something similar to how we do SWR (ie our custom init func)? If like SWR
            if layer.bias is not None:
                layer.bias.masked_fill_(mask, 0.0)

            # 6. Reset output weights
            # Using mask.unsqueeze(0) expands the 1D feature mask across the next_out_features dimension (Columns)
            if isinstance(next_layer, torch.Tensor):
                next_layer.masked_fill_(mask.unsqueeze(0), 0.0)
            else:
                next_layer.weight.masked_fill_(mask.unsqueeze(0), 0.0)

            # 7. Reset CBP state for replaced features
            ages.masked_fill_(mask, 0.0)
            utilities.masked_fill_(mask, 0.0)
            avg_activations.masked_fill_(mask, 0.0)

        # 8. Reset Optimizer States (must act in-place outside of no_grad context) - See Appendix of CBP paper for Adam
        # Initialize moment estimates: Set ml−1[:, r], ml[r, :], vl−1[:, r], and vl[r, :] to 0
        # TODO: Initialize timestep: Set tl−1[:, r], and tl[r, :] to 0

        # Input weights (Resetting rows)
        reset_optimizer_states_elementwise(
            optimizer, layer.weight, mask.unsqueeze(1).expand_as(layer.weight)
        )
        if layer.bias is not None:
            reset_optimizer_states_elementwise(optimizer, layer.bias, mask)

        # Output weights (Resetting columns)
        if isinstance(next_layer, torch.Tensor):
            reset_optimizer_states_elementwise(
                optimizer, next_layer, mask.unsqueeze(0).expand_as(next_layer)
            )
        else:
            reset_optimizer_states_elementwise(
                optimizer,
                next_layer.weight,
                mask.unsqueeze(0).expand_as(next_layer.weight),
            )

        replacement_masks[layer] = mask

    return replacement_masks
