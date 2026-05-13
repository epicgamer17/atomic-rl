import math
from typing import Tuple, Callable, Optional, Dict
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


def expected_value(predictions: torch.Tensor, support: torch.Tensor) -> torch.Tensor:
    """
    Calculate expected values from a categorical distribution over a support.

    Works for both Q-value prediction [B, A, N] and state-value prediction [B, N].

    Args:
        predictions: Logits output from the model [B, A, N] or [B, N].
        support: The support for the distribution [N], [B, N], or [B, A, N].

    Returns:
        The expected values of shape [B, A] or [B].
    """
    # Fail Fast: Enforce exact dimensionality explicitly
    assert predictions.ndim in [
        2,
        3,
    ], f"Expected 2D [B, N] or 3D [B, A, N] predictions, got {predictions.shape}"

    probs = F.softmax(predictions, dim=-1)

    # Broadcast support safely to match probs
    if support.ndim == 1:
        # Support is [N], broadcast to match [..., N]
        # TODO: using expand_as instead of einops. make a decision to keep or remove einops
        support = support.expand_as(probs)
    else:
        assert (
            support.shape == probs.shape
        ), f"If support is not 1D, it must match predictions shape exactly. Got {support.shape}"

    # Dot product over the atoms dimension
    values = (probs * support).sum(dim=-1)

    # Returns [B] if input was [B, N], or [B, A] if input was [B, A, N]
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


# TODO: is multidiscrete sampling handled and is it handled well? should we make a helper function for making the multi discrete distribution?
def sample_distribution(
    dist: torch.distributions.Distribution, explore: bool = True
) -> Tuple[torch.Tensor, dict]:
    """
    Samples an action from a generic PyTorch distribution.

    Rule Enforcement (Explicit over Implicit):
    The caller is responsible for constructing the appropriate distribution
    (e.g., Categorical, Normal) and wrapping it in `Independent` if joint
    probabilities are required.

    Args:
        dist: The PyTorch distribution object.
        explore: Whether to sample from the distribution or take the mode/mean.

    Returns:
        A tuple containing:
            - The selected actions.
            - An info dictionary containing "log_prob".
    """
    if not explore:
        # Deterministic selection
        if isinstance(dist, torch.distributions.Categorical):
            action = torch.argmax(dist.probs, dim=-1)
        elif hasattr(dist, "mean"):
            action = dist.mean
        else:
            raise NotImplementedError(
                f"Deterministic selection for {type(dist)} is not implemented."
            )
    else:
        action = dist.sample()

    log_prob = dist.log_prob(action)

    if action.ndim == 1:
        action = action.unsqueeze(-1)
    if log_prob.ndim == 1:
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


# TODO: should we make this work for q value selection? or should we at least make it more clear with the args?
def apply_action_mask(logits: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """
    Explicitly replaces logits of invalid actions with a large negative number (-1e8).
    This forces the probability of these actions to approach 0 after softmax.
    """
    mask_bool = mask.to(torch.bool)
    return torch.where(
        mask_bool,
        logits,
        torch.tensor(-1e8, device=logits.device, dtype=logits.dtype),
    )


def compute_masked_entropy(
    logits: torch.Tensor, probs: torch.Tensor, mask: torch.Tensor
) -> torch.Tensor:
    """
    Computes categorical entropy manually to prevent NaNs.
    PyTorch's Categorical.entropy() can yield NaNs if logits are extremely negative.

    Math: entropy = -sum(p * log(p))
    """
    mask_bool = mask.to(torch.bool)
    p_log_p = logits * probs
    # Force masked out p_log_p elements to 0.0 before summing
    p_log_p = torch.where(
        mask_bool,
        p_log_p,
        torch.tensor(0.0, device=logits.device, dtype=logits.dtype),
    )
    return -p_log_p.sum(dim=-1)
