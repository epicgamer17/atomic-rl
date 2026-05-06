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


# TODO: what is the point of this do we even need/want this? its just an argmax wrapper?
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


# TODO: does this need an extractor_fn?
# TODO: write now this only handles logits, muzero with need to be able to handle probs.
# TODO: do we even need this or should we just inline it into the loops/code? its basically a torch dists wrapper.
def categorical_sampling_selector(
    predictions: torch.Tensor,
    extractor_fn: Optional[Callable] = None,
    temperature: float = 1.0,
) -> torch.Tensor:
    """
    Samples an action from a categorical distribution.

    Args:
        predictions (torch.Tensor): The model predictions (logits).
        extractor_fn (Optional[Callable]): Unused for now.

    Returns:
        torch.Tensor: The sampled actions.
    """
    assert extractor_fn is None  # For now no extractors
    temperature_logits = predictions / temperature
    dist = torch.distributions.Categorical(logits=temperature_logits)
    return dist.sample()


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
