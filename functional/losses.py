import torch
import torch.nn as nn
import torch.nn.functional as F
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

    Args:
        model (nn.Module): The online Q-network.
        batch (dict): A dictionary containing the batch of transitions.
        selector_model (nn.Module): Model used to select the best action for bootstrapping.
        target_calculator_fn (Callable): Function to calculate targets.
        eval_model (Optional[nn.Module]): Model used to evaluate the selected action's value.
            Defaults to selector_model.
        loss_fn (Callable): Function to calculate loss.
        extractor_fn (Optional[Callable]): Function to extract scalar values for action selection.

    Returns:
        torch.Tensor: The loss for the batch.

    Note:
        - Assumes model.forward directly returns q-values for all actions for all observations.
    """

    # 1. Current Q-values
    predictions = model(batch["obs"])

    batch_size = predictions.shape[0]
    batch_idx = torch.arange(batch_size, device=predictions.device)
    actions = batch["action"].long().squeeze(-1)
    pred_sa = predictions[batch_idx, actions]

    # 2. Next State Evaluation (Inlined q_value_bootstrapping_evaluator)
    with torch.no_grad():
        # NOTE: Noisy DQN with Double DQN/Dueling samples a 3rd epsilon here but we do not, and neither do most implementations online.
        next_obs = batch["next_obs"]
        selector_predictions = selector_model(next_obs)
        if eval_model is None:
            next_preds = selector_predictions
        else:
            next_preds = eval_model(next_obs)

        next_actions = argmax_selector(selector_predictions, extractor_fn)

        # 3. Target Calculation
        td_target = target_calculator_fn(
            next_preds,
            next_actions,
            batch["reward"],
            batch["terminated"],
            batch["truncated"],
        )

    # 4. Compute Loss (Force shape alignment to prevent broadcasting)
    if loss_fn is None:
        loss_fn = mse_loss

    # The Polymorphic Bouncer:
    # If standard DQN [B, 1] -> safely becomes [B]
    # If C51 [B, 51] -> safely ignores the squeeze, stays [B, 51]
    td_target = td_target.squeeze(-1)
    pred_sa = pred_sa.squeeze(-1)  # Do this to pred_sa too just to be perfectly safe!

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
        Callable: The loss function with PER weights.
    """

    def per_loss_fn(
        predictions: torch.Tensor, targets: torch.Tensor
    ) -> Tuple[torch.Tensor, dict]:
        """
        Calculate the loss for a batch of transitions.

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
    """
    targets = rearrange(targets, "b -> b")
    raw_losses = F.mse_loss(predictions, targets, reduction="none")
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
    """
    targets = rearrange(targets, "b a -> b a")
    log_probs = F.log_softmax(predictions, dim=-1)
    # Cross-entropy: - sum(p_target * log(p_online))
    raw_losses = -(targets * log_probs).sum(dim=-1)

    info = {
        "priorities": raw_losses.detach(),
        "loss/cross_entropy": raw_losses.mean().detach(),
        # Functional tensor for plotting
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
    """
    targets = rearrange(targets, "b -> b")
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
        torch.Tensor: The loss for the batch.

    Note: log_probs must be shape [T, ] (advantages must match it)
    """
    # NOTE: doesnt follow the exact policy gradient of returns - baseline but we calculate the baseline outside. Optionally could move into here and do: -log_probs * (returns - baseline) instead
    # Ensure advantages matches the shape of log_probs for element-wise multiplication
    advantages = rearrange(advantages, "t -> t")
    log_probs = rearrange(log_probs, "t -> t")

    loss = -log_probs * advantages.detach()
    # NOTE: no priorities, PG is on-policy and so PER doesn't really apply.abs
    # TODO: what about A-PPO or A3C?
    info = {"loss/policy_gradient": loss.mean().detach()}

    return loss, info


def entropy_loss(
    logits: torch.Tensor,
) -> Tuple[torch.Tensor, dict]:
    """
    Calculate the entropy loss for a batch of transitions.

    Args:
        logits (torch.Tensor): Tensor of logits.

    Returns:
        torch.Tensor: The loss for the batch.

    Note: logits should be of shape [Batch, num_actions].
    """
    logits = rearrange(logits, "b a -> b a")

    # Option 1
    # dist = torch.distributions.Categorical(logits=logits)
    # entropy = dist.entropy()

    # Option 2
    # # 1. Get probabilities and log probabilities
    # probs = F.softmax(logits, dim=-1)
    # log_probs = F.log_softmax(logits, dim=-1)

    # # 2. Compute Shannon entropy: -sum(p * log(p))
    # entropy = -(probs * log_probs).sum(dim=-1)

    # # 3. Take the mean across the batch
    # loss = entropy.mean()

    # Option 3 (Fastest?)
    # Compute probabilities once
    probs = F.softmax(logits, dim=-1)

    # Cross-entropy of probabilities with their own logits = Entropy
    # PyTorch's backend handles the log-sum-exp fusion automatically here
    loss = F.cross_entropy(logits, probs)
    info = {"loss/entropy": loss.detach()}

    return loss, info
