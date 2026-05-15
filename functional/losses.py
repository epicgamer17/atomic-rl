import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as D
from einops import rearrange
from tensordict import TensorDict
from typing import Callable, Tuple, Optional, Union
from functional.action_selection import argmax_selector
from functional.td import compute_v_td_target


def mse_loss(
    predictions: torch.Tensor, targets: torch.Tensor
) -> Tuple[torch.Tensor, dict]:
    """
    Standard MSE Loss. Also returns priorities for PER.

    Args:
        predictions (torch.Tensor): Predicted Q-values.
        targets (torch.Tensor): Target Q-values.

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: A tuple containing the raw losses and the info dictionary.

    Expects flat tensors [B].
    """
    assert (
        predictions.shape == targets.shape
    ), f"Shape mismatch: {predictions.shape} vs {targets.shape}"
    raw_losses = F.mse_loss(predictions, targets, reduction="none")
    # Priorities are usually the absolute TD error
    priorities = torch.abs(predictions - targets).detach()

    info = {
        "priorities": priorities,
        "loss/mse": raw_losses.mean().detach(),
    }
    return raw_losses, info


def cross_entropy_loss(
    predictions: torch.Tensor, targets: torch.Tensor
) -> Tuple[torch.Tensor, dict]:
    """
    Categorical Cross-Entropy Loss. Also returns priorities for PER.

    Args:
        predictions (torch.Tensor): Predicted Q-values.
        targets (torch.Tensor): Target Q-values.

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: A tuple containing the raw losses and the info dictionary.

    Expects [B, Atoms] or [B, Actions].
    """
    assert (
        predictions.shape == targets.shape
    ), f"Shape mismatch: {predictions.shape} vs {targets.shape}"
    log_probs = F.log_softmax(predictions, dim=-1)
    # Cross-entropy: - sum(p_target * log(p_online))
    raw_losses = -(targets * log_probs).sum(dim=-1)

    info = {
        "priorities": raw_losses.detach(),
        "loss/cross_entropy": raw_losses.mean().detach(),
        # Functional tensor for plotting distributions
        "predictions": predictions.detach(),
    }
    return raw_losses, info


def huber_loss(
    predictions: torch.Tensor, targets: torch.Tensor, delta: float = 1.0
) -> Tuple[torch.Tensor, dict]:
    """
    Huber Loss (Smooth L1 Loss). Also returns priorities for PER.
        Args:
        predictions (torch.Tensor): Predicted Q-values.
        targets (torch.Tensor): Target Q-values.
        delta (float, optional): The threshold at which to change between L1 and L2 loss. Defaults to 1.0.

    Returns:
        Tuple[torch.Tensor, dict]: A tuple containing the raw losses and the info dictionary.

    Expects flat tensors [B].
    """
    assert (
        predictions.shape == targets.shape
    ), f"Shape mismatch: {predictions.shape} vs {targets.shape}"
    raw_losses = F.huber_loss(predictions, targets, reduction="none", delta=delta)
    priorities = torch.abs(predictions - targets).detach()

    info = {
        "priorities": priorities,
        "loss/huber": raw_losses.mean().detach(),
    }
    return raw_losses, info


def policy_gradient_loss(
    advantages: torch.Tensor,
    log_probs: torch.Tensor,
) -> Tuple[torch.Tensor, dict]:
    """
    Calculate the policy gradient loss for a batch of transitions.


    Args:
        advantages (torch.Tensor): Tensor of advantages.
        log_probs (torch.Tensor): Tensor of log probabilities of actions.

    Returns:
        Tuple[torch.Tensor, dict]: A tuple containing the raw losses and the info dictionary.

    Expects flat tensors [B * T] or [B].
    """
    assert (
        advantages.shape == log_probs.shape
    ), f"Shape mismatch: {advantages.shape} vs {log_probs.shape}"
    # PG Loss: -log_prob * advantage (Advantage is treated as constant)
    loss = -log_probs * advantages.detach()
    # NOTE: no priorities, PG is on-policy and so PER doesn't really apply.abs
    # TODO: what about A-PPO or A3C?
    info = {"loss/policy_gradient": loss.mean().detach()}

    return loss, info


def entropy_loss(
    dist: D.Distribution,
) -> Tuple[torch.Tensor, dict]:
    """
    Calculate the entropy loss for an action distribution.
    Returns the NEGATIVE mean entropy, so that minimizing this "loss" maximizes entropy.

    Rule Enforcement (Explicit over Implicit):
    This function expects the distribution's `event_shape` to be configured correctly.
    If you have a multi-dimensional action space and want joint entropy, you MUST wrap
    your base distribution in `torch.distributions.Independent` before passing it here.
    We strictly avoid guessing whether to sum over extra dimensions.

    Args:
        dist: The PyTorch distribution object.

    Returns:
        Tuple[torch.Tensor, dict]: The negative mean entropy and logging info.
    """
    # 1. dist.entropy() automatically handles the underlying math based on event_shape
    entropy = dist.entropy()

    # Fail Fast: If the user didn't use `Independent`, entropy will incorrectly retain
    # the action dimension as a batch dimension. We check against the expected batch dim.
    # Note: We assume scalar entropy per transition here.
    assert entropy.ndim == 1, (
        f"Expected 1D entropy [B], got {entropy.shape}. If your action space is "
        "multi-dimensional, ensure you wrap your distribution in `Independent`."
    )

    loss = -entropy.mean()
    return loss, {"loss/entropy": loss.detach()}


def with_per_weights(base_loss_fn: Callable, is_weights: torch.Tensor) -> Callable:
    """
    Higher-order function that wraps a standard loss function to apply
    PER Importance Sampling weights and extract TD errors.

    Args:
        base_loss_fn (Callable): Function to calculate loss. Must return a tuple of
            (raw_losses, info_dict).
        is_weights (torch.Tensor): Importance sampling weights.

    Returns:
        Callable: The loss function with PER weights applied.
    """

    def per_loss_fn(
        predictions: torch.Tensor, targets: torch.Tensor
    ) -> Tuple[torch.Tensor, dict]:
        """
        Calculate the weighted loss for a batch of transitions.

        Args:
            predictions (torch.Tensor): Predicted Q-values.
            targets (torch.Tensor): Target Q-values.

        Returns:
            torch.Tensor: The weighted loss for the batch.
            info_dict (dict): A dictionary containing the weighted loss and priorities.
        """
        # 1. Compute raw loss and priorities
        raw_losses, info_dict = base_loss_fn(predictions, targets)

        # 2. Compute weighted loss for the optimizer
        weighted_loss = (raw_losses * is_weights).mean()

        # Update info with weighted loss and priorities
        info_dict["loss/weighted"] = weighted_loss.detach()

        return weighted_loss, info_dict

    return per_loss_fn


def with_sequence_mask(base_loss_fn: Callable, mask: torch.Tensor) -> Callable:
    """
    Higher-order function that masks out invalid sequence transitions before calculating the mean loss.

    Args:
        base_loss_fn (Callable): Function to calculate loss. Must return a tuple of (raw_losses, info_dict).
        mask (torch.Tensor): Mask indicating valid transitions. In other words masking out the loss for observations or predictions past terminal or truncated states or that are for another episode (an artifact of our how our buffer handles sequences)

    Returns:
        Callable: The loss function with sequence mask applied.
    """

    def masked_loss_fn(
        predictions: torch.Tensor, targets: torch.Tensor
    ) -> Tuple[torch.Tensor, dict]:
        # predictions and targets are [B * T]
        raw_losses, info_dict = base_loss_fn(predictions, targets)

        flat_mask = mask.view(-1).float()
        masked_losses = raw_losses * flat_mask

        # Mean over only valid transitions to avoid diluting the gradient
        valid_count = flat_mask.sum().clamp(min=1.0)
        mean_loss = masked_losses.sum() / valid_count

        info_dict["loss/masked_mean"] = mean_loss.detach()
        return masked_losses, info_dict

    return masked_loss_fn


# TODO: is this good? should this take in torch.Distribution objects instead?
def probability_ratio(
    old_log_probs: torch.Tensor,
    new_log_probs: torch.Tensor,
):
    """
    Calculate the probability ratio between a new and an old log probability.
    Used in algorithms like PPO, TRPO, and V-trace.
    NOTE: The probability ratio $r_t(\theta) = \frac{\pi_\theta(a_t|s_t)}{\pi_{\theta_{old}}(a_t|s_t)}$ is mathematically an importance sampling weight.

    Args:
        old_log_probs (torch.Tensor): Tensor of old log probabilities.
        new_log_probs (torch.Tensor): Tensor of new log probabilities.

    Returns:
        torch.Tensor: The probability ratio.

    Expects flat tensors [B * T] or [B].
    """
    assert (
        old_log_probs.shape == new_log_probs.shape
    ), f"Shape mismatch: {old_log_probs.shape} vs {new_log_probs.shape}"
    return torch.exp(new_log_probs - old_log_probs.detach())


def clipped_surrogate_loss(
    ratio: torch.Tensor, advantages: torch.Tensor, clip_coef: float
) -> Tuple[torch.Tensor, dict]:
    """
    Computes the PPO clipped surrogate loss.

    Mathematically, PPO defines L^CLIP as an objective to be maximized.
    Because PyTorch optimizers minimize by default, this function returns
    the NEGATED objective so it can be directly minimized. L^CLIP is based on L^CLI which is the same loss without the penalty for moving the ratio away from 1 (the min statement with the clip coefficient)

    Args:
        ratio: The probability ratio r_t(θ) [B] or [B * T]
        advantages: The estimated advantages [B] or [B * T]
        clip_coef: The clipping coefficient (epsilon)

    Returns:
        Tuple containing the loss tensor and an info dictionary for logging.
    """
    # FAIL FAST: Prevent catastrophic broadcasting
    assert (
        ratio.shape == advantages.shape
    ), f"Shape mismatch: ratio {ratio.shape} vs adv {advantages.shape}"

    # Calculate the objective
    # TODO: should this be its own function?
    unclipped_objective = ratio * advantages.detach()
    clipped_objective = (
        torch.clamp(ratio, 1.0 - clip_coef, 1.0 + clip_coef) * advantages.detach()
    )

    # The PPO Objective to maximize
    objective = torch.min(unclipped_objective, clipped_objective)

    # Negate to turn the objective into a loss for gradient descent
    loss = -objective

    # Metrics for debugging and evaluating the clip_coef
    with torch.no_grad():
        log_ratio = torch.log(ratio)
        # Standard PPO approximate KL divergence: ((ratio - 1.0) - log_ratio).mean()
        approx_kl = ((ratio - 1.0) - log_ratio).mean()
        # Fraction of the batch that was clipped
        clip_fraction = (torch.abs(ratio - 1.0) > clip_coef).float().mean()

    info = {
        "loss/policy": loss.mean().detach(),
        "policy/approx_kl": approx_kl.detach(),
        "policy/clip_fraction": clip_fraction.detach(),
        "objective/unclipped": unclipped_objective.mean().detach(),
        "objective/clipped": clipped_objective.mean().detach(),
    }

    return loss, info


def clipped_mse_loss(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    old_predictions: torch.Tensor,
    clip_coef: float,
) -> Tuple[torch.Tensor, dict]:
    """
    Computes a Clipped MSE loss, commonly used in PPO for value function updates.

    This loss clips the predicted value to be within a certain range of the old value,
    and then takes the maximum of the unclipped and clipped MSE losses. This provides
    a pessimistic estimate of the value, which can help stabilize training.

    Args:
        predictions: The current predictions (e.g. new state values).
        targets: The target values (e.g. returns).
        old_predictions: The predictions from the previous iteration.
        clip_coef: The clipping coefficient.

    Returns:
        Tuple[torch.Tensor, dict]: The clipped MSE loss (raw) and logging info.
    """
    assert (
        predictions.shape == old_predictions.shape == targets.shape
    ), f"Shape mismatch: pred {predictions.shape}, old {old_predictions.shape}, targets {targets.shape}"

    # Explicitly detach old_predictions to prevent gradient leak
    old_predictions = old_predictions.detach()

    # Unclipped MSE
    v_loss_unclipped = F.mse_loss(predictions, targets, reduction="none")

    # Clipped MSE: clip the predicted value to be within [old - clip, old + clip]
    v_clipped = old_predictions + torch.clamp(
        predictions - old_predictions, -clip_coef, clip_coef
    )
    v_loss_clipped = F.mse_loss(v_clipped, targets, reduction="none")

    # The value loss is the max of clipped and unclipped losses (pessimistic estimate)
    v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
    loss = 0.5 * v_loss_max

    info = {
        "loss/value": loss.mean().detach(),
        "value/unclipped_loss": v_loss_unclipped.mean().detach(),
        "value/clipped_loss": v_loss_clipped.mean().detach(),
    }

    return loss, info
