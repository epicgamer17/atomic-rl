import math
from typing import Tuple, Callable, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


def expected_value(predictions: torch.Tensor, support: torch.Tensor) -> torch.Tensor:
    """
    Calculate expected values from a distribution.
    Useful for distributional value prediction methods like Categorical DQN, Dreamer, MuZero, etc.

    Args:
        predictions: The logits output from the model for the distribution [B, A, N].
        support: The support for the distribution [N] or [B, A, N].

    Returns:
        The expected values of the distributions [B, A].
    """
    # TODO: also accept B, N? for value prediction instead of action value (q value) prediction for algos like muzero?
    assert (
        predictions.ndim == 3
    ), f"Expected 3D predictions [B, A, N], got {predictions.shape}"
    assert support.ndim in [
        1,
        3,
    ], f"Expected 1D [N] or 3D [B, A, N] support, got {support.shape}"

    # B x A x N
    probs = F.softmax(predictions, dim=-1)

    # Handle both [N] and [B, A, N] support
    # NOTE: support must be same shape as predictions (B x A x N) to compute correctly
    if support.dim() == 1:
        # TODO: will this shape work for muzero categorical value prediction?
        support = rearrange(support, "n -> 1 1 n")

    values = (probs * support).sum(dim=-1)
    return values


def argmax_selector(
    predictions: torch.Tensor,
    extractor_fn: Optional[Callable] = None,
) -> Tuple[torch.Tensor, dict]:
    """
    Selects the action with the maximum value.
    Expects predictions to have a batch dimension [B, ...].

    Args:
        predictions: The model predictions (e.g. Q-values or logits).
        extractor_fn: Function to extract scalar values from predictions.

    Returns:
        A tuple containing:
            - The selected actions of shape [B, 1].
            - An empty info dictionary.
    """
    if extractor_fn is not None:
        vals = extractor_fn(predictions)
    else:
        vals = predictions

    # Force [B, 1] output for consistency
    action = torch.argmax(vals, dim=1, keepdim=True)
    return action, {}


# TODO: right now this only handles logits, muzero will need to be able to handle probs.
# TODO: should we make categorical sampling selector and gaussian sampling selector take in a distributions.Distribution object instead of taking in predictions/action_mean and action_std? (could we then merge these functions?)
# TODO: is it better to have a universal dist selector that takes in a distribution object and samples/computes log_prob from it?
def categorical_sampling_selector(
    predictions: torch.Tensor,
    extractor_fn: Optional[Callable] = None,
    temperature: float = 1.0,
) -> Tuple[torch.Tensor, dict]:
    """
    Samples an action from a categorical distribution.
    Expects predictions [B, A] or similar.

    Args:
        predictions: The model predictions (logits).
        extractor_fn: Optional extractor.
        temperature: Sampling temperature for exploration.

    Returns:
        A tuple containing:
            - The selected actions of shape [B, 1].
            - An info dictionary containing "log_prob".
    """
    # TODO: does this assert work for multi-discrete action spaces? does this code work for multi-discrete action spaces?
    assert (
        predictions.ndim >= 2
    ), f"Expected predictions with at least batch and action dims, got {predictions.shape}"

    if temperature == 0.0:
        # For temp=0, we just take the argmax
        # TODO: how should log_prob be computed here?
        action, info = argmax_selector(predictions, extractor_fn)
        info["log_prob"] = torch.zeros_like(action, dtype=torch.float32)
        return action, info

    if extractor_fn is not None:
        vals = extractor_fn(predictions)
    else:
        vals = predictions

    temperature_logits = vals / temperature
    dist = torch.distributions.Categorical(logits=temperature_logits)

    # action will be [Batch] for standard, or [Batch, Num_Vars] for multi-discrete
    action = dist.sample()
    log_prob = dist.log_prob(action)

    # TODO: CONSISTENT LOG PROB SHAPE ACROSS FUNCTIONAL LIBRARY.
    # IN POLICY GRADIENT LOSS WE USE [T, ] for log_prob, NOT [T, 1]. WHY???.
    # For now we ensure [B, 1] for consistency in off-policy selectors.
    if log_prob.ndim > 1 and log_prob.shape[-1] > 1:
        # NOTE: sum assumes independent for each action in the vector.
        log_prob = log_prob.sum(dim=-1, keepdim=True)

    if action.ndim == 1:
        action = action.unsqueeze(-1)
    if log_prob.ndim == 1:
        log_prob = log_prob.unsqueeze(-1)

    return action, {"log_prob": log_prob}


def gaussian_sampling_selector(
    action_mean: torch.Tensor, action_std: torch.Tensor, explore: bool = True
) -> Tuple[torch.Tensor, dict]:
    """
    Samples from a Gaussian policy for continuous actions.

    Args:
        action_mean (torch.Tensor): The mean of the Gaussian distribution.
        action_std (torch.Tensor): The standard deviation of the Gaussian distribution.
        explore (bool): Whether to explore or not.

    Returns:
        A tuple containing:
            - The selected actions.
            - An info dictionary containing "log_prob".
    """
    # TODO: does this assert work for multi-discrete action spaces? does this code work for multi-discrete action spaces?
    assert (
        action_mean.shape == action_std.shape
    ), f"Mean {action_mean.shape} and Std {action_std.shape} must match"
    assert (
        action_mean.ndim >= 1
    ), f"Expected at least 1D tensors, got {action_mean.ndim}D"

    if not explore:
        # log prob is 0 for deterministic. Ensure [B, 1] if input is [B, 1]
        log_prob = torch.zeros_like(action_mean)
        if log_prob.ndim == 1:
            log_prob = log_prob.unsqueeze(-1)
        return action_mean, {"log_prob": log_prob}

    dist = torch.distributions.Normal(action_mean, action_std)
    action = dist.sample()
    log_prob = dist.log_prob(action)

    # NOTE: Many continuous envs have an action dimension (multiple values per step).
    # If the action space has multiple dimensions (e.g., [Batch, 6]),
    # we sum the log probs of each independent joint to get the total joint probability.
    if log_prob.ndim > 1 and log_prob.shape[-1] > 1:
        # NOTE: sum assumes independent for each action in the vector.
        log_prob = log_prob.sum(dim=-1, keepdim=True)
    elif log_prob.ndim == 1:
        log_prob = log_prob.unsqueeze(-1)

    return action, {"log_prob": log_prob}


def with_epsilon_greedy(selector_fn: Callable) -> Callable:
    """
    Higher-order function that augments a selector with epsilon-greedy logic.

    Args:
        selector_fn (Callable): The function to use for action selection.
            Must return (actions [B, 1], info_dict).
    """

    def epsilon_greedy_selector(
        predictions: torch.Tensor,
        epsilon: float,
        num_actions: int,
        generator: torch.Generator = None,
    ) -> Tuple[torch.Tensor, dict]:
        greedy_actions, info = selector_fn(predictions)  # Expected [B, 1], dict

        if epsilon <= 0.0:
            return greedy_actions, {**info, "generator": generator}

        batch_size = predictions.shape[0]

        # Sample random actions
        random_actions = torch.randint(
            0,
            num_actions,
            (batch_size, 1),
            generator=generator,
            device=predictions.device,
        )

        # Decide which actions are random
        random_mask = (
            torch.rand((batch_size, 1), generator=generator, device=predictions.device)
            < epsilon
        )

        final_actions = torch.where(random_mask, random_actions, greedy_actions)
        return final_actions, {**info, "generator": generator}

    return epsilon_greedy_selector


def get_ape_x_epsilon(
    actor_id: int, num_actors: int, base_eps: float = 0.4, alpha: float = 7.0
) -> float:
    """
    Calculates the fixed epsilon for a specific actor in APE-X.

    Args:
        actor_id (int): The ID of the actor.
        num_actors (int): The total number of actors.
        base_eps (float): The base epsilon value.
        alpha (float): The alpha parameter for the distribution.
    """
    if num_actors <= 1:
        return base_eps
    return base_eps ** (1 + (actor_id / (num_actors - 1)) * alpha)
