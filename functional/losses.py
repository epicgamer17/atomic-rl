import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as D
from einops import rearrange
from tensordict import TensorDict
from typing import Callable, Tuple, Optional, Union
from functional.action_selection import argmax_selector


def bellman_error(
    model: torch.nn.Module,
    batch: TensorDict,
    selector_model: torch.nn.Module,
    target_calculator_fn: Callable,
    eval_model: Optional[torch.nn.Module] = None,
    loss_fn: Optional[Callable] = None,
    # TODO: should this just be part of a partial on the selector or be a "selector_fn" instead of argmax selector always?
    extractor_fn: Optional[Callable] = None,
) -> Tuple[torch.Tensor, dict]:
    """
    Calculate the Bellman error for a batch of transitions.
    The 'Imperative Shell' for DQN-style updates.

    This function orchestrates the evaluation of next-states and target calculation,
    ensuring that the pure math functions receive correctly formatted tensors.

    Args:
        model (nn.Module): The online Q-network.
        batch (dict): A dictionary containing the batch of transitions.
        selector_model (nn.Module): Model used to select the best action for bootstrapping.
        target_calculator_fn (Callable): Function to calculate targets.
        eval_model (Optional[nn.Module]): Model used to evaluate the selected action's value.
            Defaults to selector_model (standard DQN).
        loss_fn (Callable): Function to calculate loss (e.g. mse_loss).
        extractor_fn (Optional[Callable]): Function to extract scalar values for action selection.

    Returns:
        torch.Tensor: The loss for the batch.
        dict: Information for logging and debugging.

    Note:
        - Assumes model.forward directly returns q-values for all actions for all observations.
    """

    # 1. Current Q-values (Prediction)
    predictions = model(batch["obs"])

    batch_size = predictions.shape[0]
    batch_idx = torch.arange(batch_size, device=predictions.device)
    actions = batch["action"].long().squeeze(-1)
    pred_sa = predictions[batch_idx, actions]

    # 2. Next State Evaluation
    with torch.no_grad():
        # NOTE: Noisy DQN with Double DQN/Dueling samples a 3rd epsilon here but we do not,
        # and neither do most implementations online.
        next_obs = batch["next_obs"]
        selector_predictions = selector_model(next_obs)
        if eval_model is None:
            # Standard DQN
            next_preds = selector_predictions
        else:
            # Double DQN
            next_preds = eval_model(next_obs)

        # Select the best next action using the selector model
        next_actions, _ = argmax_selector(selector_predictions, extractor_fn)

        # 3. Target Calculation
        # Ensure rewards and terminated are [B, 1] for target calculator
        rewards = rearrange(batch["reward"], "b -> b 1")
        terminated = rearrange(batch["terminated"], "b -> b 1")
        truncated = rearrange(batch["truncated"], "b -> b 1")

        # Calculate TD target (standard, n-step, or categorical)
        td_target = target_calculator_fn(
            next_preds,
            next_actions,
            rewards,
            terminated,
            truncated,
        )

    # 4. Compute Loss (Force shape alignment to prevent broadcasting bugs)
    if loss_fn is None:
        loss_fn = mse_loss

    # Align shapes for loss: [B] or [B, Atoms]
    # If standard DQN, pred_sa is [B], td_target is [B, 1] -> squeeze td_target
    if td_target.dim() == 2 and td_target.shape[1] == 1:
        td_target = td_target.squeeze(-1)

    loss, info = loss_fn(pred_sa, td_target)

    # 5. Augment info with orchestration-level metrics for W&B
    info.update(
        {
            "q_values/mean": pred_sa.mean().detach(),
            "q_values/min": pred_sa.min().detach(),
            "q_values/max": pred_sa.max().detach(),
            "td_targets/mean": td_target.mean().detach(),
            "rewards/mean": batch["reward"].mean().detach(),
        }
    )

    return loss, info


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
    Calculate the entropy loss for ANY action distribution (Discrete or Continuous).

    Args:
        dist (torch.distributions.Distribution): The PyTorch distribution object.

    Returns:
        Tuple[torch.Tensor, dict]: The mean entropy loss and logging info.
    """
    # 1. dist.entropy() automatically handles the underlying math,
    # whether it's Shannon Entropy (Categorical) or Differential Entropy (Normal)
    entropy = dist.entropy()

    # 2. Take the mean across the batch
    # NOTE: for multivariate distributions, entropy might be [B, A] or [B].
    # Normal returns [B, A], so we sum across actions first if needed,
    # but dist.entropy() for Normal usually returns [B, A].
    # Wait, torch.distributions.Normal(mu, std).entropy() returns [B, A].
    # We usually want the total entropy of the joint distribution, which is the sum.
    if entropy.dim() > 1:
        entropy = entropy.sum(dim=-1)

    loss = entropy.mean()

    info = {"loss/entropy": loss.detach()}

    return loss, info
