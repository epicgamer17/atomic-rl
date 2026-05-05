import torch
import torch.nn as nn
import torch.nn.functional as F
from tensordict import TensorDict
from typing import Callable, Tuple


def bellman_error(
    model: torch.nn.Module,
    target_model: torch.nn.Module,
    batch: dict,
    action_selector_fn,  # Inject the behavior
    target_calculator_fn,  # Inject the behavior
    loss_fn=None,  # Inject the behavior
) -> Tuple[torch.Tensor, dict]:
    """
    Calculate the Bellman error for a batch of transitions.

    Args:
        model (nn.Module): The online Q-network.
        target_model (nn.Module): The target Q-network.
        batch (dict): A dictionary containing the batch of transitions.
        action_selector_fn (Callable): Function to select actions.
        target_calculator_fn (Callable): Function to calculate targets.
        loss_fn (Callable): Function to calculate loss.

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

    # 2. Next State Evaluation (Delegated to injected function)
    with torch.no_grad():
        # NOTE: Noisy DQN with Double DQN/Dueling samples a 3rd epsilon here but we do not, and niether do most implementations online.
        next_preds, next_actions = action_selector_fn(
            model, target_model, batch["next_obs"]
        )

        # 3. Target Calculation (Delegated to injected function)
        td_target = target_calculator_fn(
            next_preds, next_actions, batch["reward"], batch["terminated"]
        )

    # 4. Compute Loss (Force shape alignment to prevent broadcasting)
    if loss_fn is None:
        from functional.losses import mse_loss

        loss_fn = mse_loss

    loss, info = loss_fn(pred_sa, td_target.view_as(pred_sa))

    # 5. Augment info with orchestration-level metrics for W&B
    info.update(
        {
            "q_values/mean": pred_sa.mean().item(),
            "q_values/min": pred_sa.min().item(),
            "q_values/max": pred_sa.max().item(),
            "td_targets/mean": td_target.mean().item(),
            "rewards/mean": batch["reward"].mean().item(),
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
        info_dict["loss/weighted"] = weighted_loss.item()

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
    targets = targets.view_as(predictions)
    raw_losses = F.mse_loss(predictions, targets, reduction="none")
    priorities = torch.abs(predictions - targets).detach()

    info = {
        "priorities": priorities,
        "loss/mse": raw_losses.mean().item(),
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
    targets = targets.view_as(predictions)
    log_probs = F.log_softmax(predictions, dim=-1)
    # Cross-entropy: - sum(p_target * log(p_online))
    raw_losses = -(targets * log_probs).sum(dim=-1)

    info = {
        "priorities": raw_losses.detach(),
        "loss/cross_entropy": raw_losses.mean().item(),
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
    targets = targets.view_as(predictions)
    raw_losses = F.huber_loss(predictions, targets, reduction="none", delta=delta)
    priorities = torch.abs(predictions - targets).detach()

    info = {
        "priorities": priorities,
        "loss/huber": raw_losses.mean().item(),
    }
    return raw_losses, info
