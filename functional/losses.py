import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as D
from einops import rearrange
from tensordict import TensorDict
from typing import Callable, Tuple, Optional, Union
from functional.action_selection import argmax_selector


# TODO: possibly make this take in a selector_fn in future when i implement sarsa or soft q learning etc
def compute_q_td_loss(
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

    NOTE:
        - Assumes model.forward directly returns q-values for all actions for all observations.
    """

    # 1. Current Q-values (Prediction)
    predictions = model(batch["obs"])

    batch_size = predictions.shape[0]
    batch_idx = torch.arange(batch_size, device=predictions.device)
    # Handle both [B] and [B, 1] action shapes gracefully
    actions = batch["action"].long()
    if actions.dim() == 2:
        actions = actions.squeeze(-1)
    
    pred_sa = predictions[batch_idx, actions]

    # 2. Next State Evaluation
    with torch.no_grad():
        # NOTE: Noisy DQN with Double DQN/Dueling samples a 3rd epsilon here but we do not,
        # and neither do most implementations online.
        next_obs = batch["next_obs"]
        selector_predictions = selector_model(next_obs)
        # TODO: possibly always require an eval_model and just do a check if eval_model = selector_model etc?
        if eval_model is None:
            # Standard DQN
            next_preds = selector_predictions
        else:
            # Double DQN
            next_preds = eval_model(next_obs)

        # Select the best next action using the selector model
        next_actions, _ = argmax_selector(selector_predictions, extractor_fn)

        # 3. Target Calculation
        # Pass through rewards and termination flags as they are (expected to be [B])
        rewards = batch["reward"]
        terminated = batch["terminated"]
        truncated = batch["truncated"]
        gamma = batch["gamma"]

        # Calculate TD target (standard, n-step, or categorical)
        td_target = target_calculator_fn(
            next_preds,
            next_actions,
            rewards,
            terminated,
            truncated,
            gamma,
        )

    # 4. Compute Loss (Force shape alignment to prevent broadcasting bugs)
    if loss_fn is None:
        loss_fn = mse_loss

    # FAIL FAST: Ensure shapes match exactly for standard DQN
    if pred_sa.dim() == 1:
        assert pred_sa.shape == td_target.shape, f"Shape mismatch: pred {pred_sa.shape} vs target {td_target.shape}"

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


# TODO: wondering if this is necessary or can simply be inlined. For now its okay.
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


# TODO: this is the objective but do we want to make it a loss (ie something to minimize) instead of making the user put a - sign in front of it when they use it?
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


def compute_is_weights(
    leaf_priorities: torch.Tensor,
    min_prob: Union[float, torch.Tensor],
    total_priority: Union[float, torch.Tensor],
    beta: torch.Tensor,
) -> torch.Tensor:
    """
    Computes Importance Sampling (IS) weights for Prioritized Experience Replay (PER).

    IS weights correct for the bias introduced by non-uniform sampling in PER.
    Mathematically: $w_i = ( (1/N) * (1/P_i) )^\beta / \max_j w_j$, where $P_i$ is the
    sampling probability of transition $i$. This simplifies to:
    $w_i = ( P_i / \min_j P_j )^{-\beta}$.

    Args:
        leaf_priorities (torch.Tensor): The priority values of the sampled transitions.
        min_prob (Union[float, torch.Tensor]): The minimum sampling probability in the tree.
        total_priority (Union[float, torch.Tensor]): The sum of all priorities in the tree.
        beta (torch.Tensor): The correction coefficient (usually scheduled from 0.4 to 1.0).

    Returns:
        torch.Tensor: The normalized IS weights for the batch.
    """
    # Sampling probability: P_i = priority_i / sum(priorities)
    probs = leaf_priorities / total_priority
    # IS weight: (P_i / min_P)^-beta
    # This simplifies the (1/N * 1/P_i)^beta / max_w normalization
    is_weights = torch.pow(probs / min_prob, -beta)
    return is_weights


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

    info = {
        "loss/policy": loss.mean().detach(),
        "objective/unclipped": unclipped_objective.mean().detach(),
        "objective/clipped": clipped_objective.mean().detach(),
    }

    return loss, info


# TODO: this is a value loss only at the moment. should this instead be considered more like a clipped MSE loss? right now switching this with mse is weird because all the arg names change despite being the same underlying loss. Thoughts?
def clipped_value_loss(
    new_values: torch.Tensor,
    old_values: torch.Tensor,
    returns: torch.Tensor,
    clip_coef: float,
) -> Tuple[torch.Tensor, dict]:
    """
    Computes the PPO clipped value loss.

    This prevents the value network from updating too aggressively in a single epoch
    by clipping the value function estimate similar to how the policy ratio is clipped.

    Args:
        new_values (torch.Tensor): Predicted values from the current policy.
        old_values (torch.Tensor): Predicted values from the old policy (at collection time).
        returns (torch.Tensor): Target values (e.g., TD-lambda returns).
        clip_coef (float): Clipping coefficient (epsilon).

    Returns:
        Tuple[torch.Tensor, dict]: The clipped value loss (raw) and logging info.
    """
    assert (
        new_values.shape == old_values.shape == returns.shape
    ), f"Shape mismatch: new {new_values.shape}, old {old_values.shape}, returns {returns.shape}"

    # Unclipped MSE
    v_loss_unclipped = F.mse_loss(new_values, returns, reduction="none")

    # Clipped MSE: clip the predicted value to be within [old - clip, old + clip]
    v_clipped = old_values + torch.clamp(new_values - old_values, -clip_coef, clip_coef)
    v_loss_clipped = F.mse_loss(v_clipped, returns, reduction="none")

    # PPO value loss is the max of clipped and unclipped losses
    # (pessemistic estimate)
    v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
    loss = 0.5 * v_loss_max

    info = {
        "loss/value": loss.mean().detach(),
        "value/unclipped_loss": v_loss_unclipped.mean().detach(),
        "value/clipped_loss": v_loss_clipped.mean().detach(),
    }

    return loss, info
