import math
from typing import Tuple, Callable, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F


def expected_value(predictions: torch.Tensor, support: torch.Tensor) -> torch.Tensor:
    """
    Calculate expected values from a distribution, then argmax.
    Useful for distributional value prediction methods like Categorical DQN, Dreamer, MuZero, etc.

    Args:
        predictions (torch.Tensor): The logits output from the model for the distribution.
        support (torch.Tensor): The support for the distribution.

    Returns:
        torch.Tensor: the expected values of the distributions
    """
    # B x A x N
    probs = F.softmax(predictions, dim=-1)
    # B x A x N @ B x A x N -> B x A x N
    # NOTE: support must be same shape as predictions (B x A x N) to compute correctly
    values = (probs * support).sum(dim=-1)
    return values


def argmax_selector(
    predictions: torch.Tensor,
    extractor_fn: Optional[Callable] = None,
) -> torch.Tensor:
    """
    Selects the action with the maximum value.

    Args:
        predictions (torch.Tensor): The model predictions (e.g. Q-values or logits).
        extractor_fn (Optional[Callable]): Function to extract scalar values from predictions.

    Returns:
        torch.Tensor: The selected actions.
    """
    if extractor_fn is not None:
        vals = extractor_fn(predictions)
    else:
        vals = predictions
    return torch.argmax(vals, dim=1, keepdim=True)


# TODO: write now this only handles logits, muzero with need to be able to handle probs.
def categorical_sampling_selector(
    predictions: torch.Tensor,
    extractor_fn: Optional[Callable] = None,
    temperature: float = 1.0,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Samples an action from a categorical distribution.

    Args:
        predictions (torch.Tensor): The model predictions (logits).
        extractor_fn (Optional[Callable]): Not often used, but could be used to extract Q-values from a Categorical DQN for Boltzman exploration (for example).
        temperature (float): The temperature for the Boltzman exploration.

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: A tuple containing the sampled actions and the log probabilities.
    """
    if temperature == 0.0:
        return argmax_selector(predictions, extractor_fn)

    if extractor_fn is not None:
        vals = extractor_fn(predictions)
    else:
        vals = predictions

    temperature_logits = vals / temperature
    dist = torch.distributions.Categorical(logits=temperature_logits)

    action = dist.sample().unsqueeze(-1)
    log_prob = dist.log_prob(action).unsqueeze(-1)
    # If it's a standard discrete env, log_prob shape is [Batch] or [Batch, 1]
    # If multi-discrete, it might be [Batch, Num_Categorical_Variables]
    if log_prob.dim() > 1 and log_prob.shape[-1] > 1:
        log_prob = log_prob.sum(dim=-1, keepdim=True)
    else:
        log_prob = log_prob.view(-1, 1)  # Ensure [Batch, 1] for consistency
        action = action.view(-1, 1)

    return action, log_prob


def gaussian_sampling_selector(
    action_mean: torch.Tensor, action_std: torch.Tensor, explore: bool = True
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Samples from a Gaussian policy for continuous actions.

    Args:
        action_mean (torch.Tensor): The mean of the Gaussian distribution.
        action_std (torch.Tensor): The standard deviation of the Gaussian distribution.
        explore (bool): Whether to explore or not.

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: A tuple containing the sampled actions and the log probabilities.
    """
    if not explore:
        return action_mean, torch.zeros_like(
            action_mean
        )  # Log prob is 0 for deterministic

    dist = torch.distributions.Normal(action_mean, action_std)
    action = dist.sample()
    log_prob = dist.log_prob(action)
    # NOTE: Many continuous envs have an action dimension (multiple values per step). In other words the action is a vector.
    # If the action space has multiple dimensions (e.g., [Batch, 6]),
    # we sum the log probs of each independent joint to get the total joint probability.
    # keepdim=True ensures the output is [Batch, 1] rather than [Batch]
    if log_prob.dim() > 1 and log_prob.shape[-1] > 1:
        log_prob = log_prob.sum(dim=-1, keepdim=True)
    else:
        # If it's a 1D action space, just ensure it's explicitly [Batch, 1]
        log_prob = log_prob.view(-1, 1)
    return action, log_prob


def with_epsilon_greedy(selector_fn: Callable) -> Callable:
    """
    Higher-order function that augments a selector with epsilon-greedy logic.

    Args:
        selector_fn (Callable): The function to use for action selection.
            Must have the signature (predictions).

    Returns:
        Callable: The action selection function with epsilon-greedy logic.
            (predictions, epsilon, num_actions, generator) -> (actions, generator)
    """

    def epsilon_greedy_selector(
        predictions: torch.Tensor,
        epsilon: float,
        num_actions: int,
        generator: torch.Generator = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Generator]:
        greedy_actions = selector_fn(predictions)

        if epsilon <= 0.0:
            return greedy_actions, generator

        batch_size = predictions.shape[0]

        random_actions = torch.randint(
            0,
            num_actions,
            (batch_size, 1),
            generator=generator,
            device=predictions.device,
        )
        random_mask = (
            torch.rand((batch_size, 1), generator=generator, device=predictions.device)
            < epsilon
        )

        final_actions = torch.where(random_mask, random_actions, greedy_actions)
        return final_actions, generator

    return epsilon_greedy_selector


# TODO: should we generalize this in utils?
def get_linear_epsilon(
    step: int, start_eps: float, end_eps: float, decay_steps: int
) -> float:
    """
    Linearly decays epsilon from start_eps to end_eps over decay_steps.

    Args:
        step (int): The current step.
        start_eps (float): The starting epsilon.
        end_eps (float): The ending epsilon.
        decay_steps (int): The number of steps over which to decay epsilon.
    """
    # Calculate the fraction of the way through the decay period (capped at 1.0)
    fraction = min(1.0, float(step) / decay_steps)
    return start_eps - fraction * (start_eps - end_eps)


# TODO: should we generalize this in utils?
def get_exponential_epsilon(
    step: int, start_eps: float, end_eps: float, decay_rate: float
) -> float:
    """
    Exponentially decays epsilon, decay rate controls how fast it drops.

    Args:
        step (int): The current step.
        start_eps (float): The starting epsilon.
        end_eps (float): The ending epsilon.
        decay_rate (float): The decay rate.
    """
    return end_eps + (start_eps - end_eps) * math.exp(-1.0 * step / decay_rate)


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
