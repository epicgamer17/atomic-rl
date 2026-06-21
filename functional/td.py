"""
Temporal Difference (TD) Learning Utilities.

This module is divided into two distinct paradigms:

1. Target Generators (`compute_*_target`):
   - Agnostic to function approximator (Works with Deep NNs).
   - Returns the target tensor to be used with standard PyTorch Loss functions and Optimizers.
   - Typically used in batched/replay settings (DQN, Actor-Critic).

2. Explicit Weight Updaters (`*_update`):
   - STRICTLY for Linear Function Approximation (V(s) = theta^T phi).
   - Manually applies the mathematical gradient step and returns the new weight vector.
   - Typically used in pure online, streaming RL settings without standard PyTorch Optimizers.
"""

import torch
import torch.nn.functional as F
from typing import Tuple


# TODO: I think truncation right now works for single env trajectories without auto resetting, but would fail for vectorized envs with auto resetting. as the next_q_values would be on the resetted state instead of the one from the info. verify this, and if it is an issue, unify the vectorization logic in our buffer to work with offline buffers as well somehow.
# TODO: should we make something like torch.Optim classes for TD optimization. I feel like we have these update rules similar to things like our IDBD or ObGD update rules and we could make an optimizer class for these or something? Or is that a bad idea?


def compute_v_td_target(
    next_values: torch.Tensor,  # [B]
    rewards: torch.Tensor,  # [B]
    terminated: torch.Tensor,  # [B]
    gamma: torch.Tensor,  # [B] or scalar
) -> torch.Tensor:
    """
    Calculates the 1-step Temporal Difference target for state values V(s).
    Formula: y = R_{t} + gamma * (1 - terminated) * V(s_{t+1})

    Args:
        next_values: Value estimates of the next states.
        rewards: Rewards for the transitions.
        terminated: Booleans indicating whether the episodes terminated.
        gamma: Discount factors.

    Returns:
        The TD target of shape [B].
        Note: The returned tensor is not explicitly detached.
    """
    # Fail Fast: Ensure shape alignment
    assert (
        next_values.ndim == 1
    ), f"Expected 1D next_values [B], got {next_values.shape}"
    assert (
        rewards.shape == terminated.shape == next_values.shape
    ), f"Shape mismatch: rewards {rewards.shape}, terminated {terminated.shape}, next_values {next_values.shape}"

    return rewards + gamma * next_values.detach() * (1 - terminated.float())


def compute_q_td_target(
    next_q_values: torch.Tensor,  # [B, A]
    next_actions: torch.Tensor,  # [B]
    rewards: torch.Tensor,  # [B]
    terminated: torch.Tensor,  # [B]
    gamma: torch.Tensor,  # [B]
) -> torch.Tensor:
    """
    Calculates the TD target for scalar Q-values.
    Composes the V-target function by extracting the value of the next state.

    Args:
        next_q_values: Q-values of the next states.
        next_actions: Indices of the actions taken in the next states (greedy for Q-learning, sampled for SARSA).
        rewards: Rewards for the transitions.
        terminated: Booleans indicating whether the episodes terminated.
        gamma: Discount factors.

    Returns:
        The TD target of shape [B].
        Note: The returned tensor is not explicitly detached.
    """
    assert (
        next_q_values.ndim == 2
    ), f"Expected 2D next_q_values [B, A], got {next_q_values.shape}"
    assert (
        next_actions.ndim == 1
    ), f"Expected [B] next_actions, got {next_actions.shape}"

    # 1. Extract the Q-value of the selected next action -> This IS V(s')
    next_values = torch.gather(next_q_values, 1, next_actions.unsqueeze(-1)).squeeze(-1)

    # 2. Compute standard V-target
    return compute_v_td_target(next_values, rewards, terminated, gamma)


def compute_categorical_q_td_target(
    next_logits: torch.Tensor,  # [B, A, Atoms]
    next_actions: torch.Tensor,  # [B]
    rewards: torch.Tensor,  # [B]
    terminated: torch.Tensor,  # [B]
    gamma: torch.Tensor,  # [B]
    support: torch.Tensor,  # [Atoms]
    v_min: float,
    v_max: float,
    atom_size: int,
) -> torch.Tensor:
    """
    Calculates the projected Categorical TD target distribution (C51 style).

    This function handles both 1-step and N-step TD targets. For N-step,
    the `rewards` should be the pre-computed discounted sum of rewards,
    and `gamma` should be the pre-computed effective discount factor (gamma^n).

    Args:
        next_logits: Logits of the next states.
        next_actions: Indices of the actions taken in the next states.
        rewards: Rewards for the transitions.
        terminated: Booleans indicating whether the episodes terminated.
        gamma: Discount factors.
        support: Support values for the distribution.
        v_min: The minimum value of the support.
        v_max: The maximum value of the support.
        atom_size: The number of atoms in the support.

    Returns:
        The projected Categorical TD target distribution [B, Atoms].
        Note: The returned tensor is not explicitly detached.
    """
    assert (
        next_logits.ndim == 3
    ), f"Expected 3D next_logits [B, A, Atoms], got {next_logits.shape}"
    assert (
        next_actions.ndim == 1
    ), f"Expected [B] next_actions, got {next_actions.shape}"

    # 1. Get probabilities of the next states
    next_probs = F.softmax(next_logits, dim=-1)

    # 2. Gather the probabilities for the chosen next actions
    next_actions_expanded = next_actions.view(-1, 1, 1).expand(-1, -1, atom_size)
    next_probs_a = next_probs.gather(1, next_actions_expanded).squeeze(1)  # [B, Atoms]

    # 3. Compute the target support (Tz) [B, Atoms]
    support_b = support.unsqueeze(0)
    rewards_b = rewards.unsqueeze(1)
    gamma_b = gamma.unsqueeze(1)
    term_b = terminated.unsqueeze(1)

    Tz = rewards_b + gamma_b * support_b * (1 - term_b.float())
    Tz = Tz.clamp(min=v_min, max=v_max)

    # 4. Compute projection bins
    dz = (v_max - v_min) / (atom_size - 1)
    b = (Tz - v_min) / dz
    l = b.floor().long()
    u = b.ceil().long()

    # Handle boundary conditions where the target falls exactly on a bin
    l[(u > 0) & (l == u)] -= 1
    u[(l < (atom_size - 1)) & (l == u)] += 1

    # 5. Distribute probabilities onto the fixed support (Projection)
    batch_size = rewards.size(0)
    m = rewards.new_zeros(batch_size, atom_size)
    offset = (
        torch.linspace(
            0,
            (batch_size - 1) * atom_size,
            batch_size,
            dtype=torch.long,
            device=rewards.device,
        )
        .unsqueeze(1)
        .expand(batch_size, atom_size)
    )

    # Flatten views for categorical projection
    m_flat = m.view(-1)
    offset_l = (l + offset).view(-1)
    offset_u = (u + offset).view(-1)

    prob_lower = (next_probs_a * (u.float() - b)).view(-1)
    prob_upper = (next_probs_a * (b - l.float())).view(-1)

    # Index Add becomes clean and fast:
    m_flat.index_add_(0, offset_l, prob_lower)
    m_flat.index_add_(0, offset_u, prob_upper)

    return m


# GRADIENT TD METHODS
# TODO: more semantic naming instead of phi and theta etc?
# TODO: actually use/test importance sampling
# TODO: batch updates where phi comes from vectorized envs. "batched online learning"
# TODO: remove value specific logic to make these work for Policy updates URGENT FOR STREAM RL.
# TODO: allow for entropy regularization with TD policy method
# TODO: is it possible to unify these?
# TODO: there is an orginization and semantic issue arising here. not all interfaces are the same. and there are some stream TD methods that use gradients to update weights (as in stream RL works) some that work only on linear methods, some that get expanded to work on linear and non linear methods with backprop. So there is a like a mix of things going on here.
# TODO: does this work with non linear weights/networks?
# TODO: should this be inplace?
# TODO: should this be a function since now that we passed error instead of computing it in the function its one line.
def semi_gradient_td_update(
    error: float | torch.Tensor,
    weights: torch.Tensor,
    alpha: float | torch.Tensor,
    update_vector: torch.Tensor,
    rho: float | torch.Tensor = 1.0,
) -> torch.Tensor:
    """
    Performs a generic semi-gradient update for linear function approximation. Allows for eligibility traces.
    Can be used for both value functions (where error is TD error) and policies (where error is advantage).

    Args:
        error: The scalar error term (e.g., TD error or Advantage).
        weights: Current weight vector [features].
        alpha: Learning rate.
        update_vector: The vector used to step the weights.
            - For TD(0) value update, pass `features`.
            - For TD(lambda) value update, pass the accumulated `eligibility_trace`.
            - For Policy update, pass the policy gradient or its trace.
        rho: Importance sampling ratio (default: 1.0 for on-policy).

    Returns:
        The updated weight vector weights [features].

    NOTE: Strictly linear function approximation.
    """
    return weights + alpha * rho * error * update_vector


def gtd0_update(
    error: float | torch.Tensor,
    features: torch.Tensor,
    next_features: torch.Tensor,
    gamma: float | torch.Tensor,
    weights: torch.Tensor,
    u: torch.Tensor,
    alpha: float | torch.Tensor,
    beta: float | torch.Tensor,
    terminated: bool | torch.Tensor,
    rho: float | torch.Tensor = 1.0,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    GTD(0) update from Sutton et al. (2009). NOTE: Faithful to the original 2009 paper, not modern GTD2/TDC.

    Args:
        error: The scalar error term (e.g., TD error or Advantage).
        features: Feature vector of the current state [features].
        next_features: Feature vector of the next state [features].
        gamma: Discount factor.
        weights: Current weight vector [features].
        u: Auxiliary weight vector for GTD(0) [features].
        alpha: Learning rate.
        beta: Step size for auxiliary weight updates.
        terminated: Whether the next state is a terminal state.
        rho: Importance sampling ratio (default: 1.0 for on-policy).

    Returns:
        The updated weight vector weights [features] and auxiliary weight vector u [features].

    NOTE: This implementation is strictly TD(0). It does not yet support eligibility traces.
    NOTE: Strictly linear function approximation.
    """
    # Update auxiliary weights (u)
    u_new = u + beta * rho * (error * features - u)

    # Update primary weights (weights)
    weights_new = weights + alpha * rho * (
        features - gamma * next_features * (1.0 - float(terminated))
    ) * torch.dot(features, u)

    return weights_new, u_new


def tdc_update(
    error: float | torch.Tensor,
    features: torch.Tensor,
    next_features: torch.Tensor,
    gamma: float | torch.Tensor,
    weights: torch.Tensor,
    w: torch.Tensor,
    alpha: float | torch.Tensor,
    beta: float | torch.Tensor,
    terminated: bool | torch.Tensor,
    rho: float | torch.Tensor = 1.0,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Fast-GTD / TDC update from Sutton et al. (2009).

    Args:
        error: The scalar error term (e.g., TD error or Advantage).
        features: Feature vector of the current state [features].
        next_features: Feature vector of the next state [features].
        gamma: Discount factor.
        weights: Current weight vector [features].
        w: Auxiliary weight vector for TDC [features].
        alpha: Learning rate.
        beta: Step size for auxiliary weight updates.
        terminated: Whether the next state is a terminal state.
        rho: Importance sampling ratio (default: 1.0 for on-policy).

    Returns:
        The updated weight vector weights [features] and auxiliary weight vector w [features].

    NOTE: This implementation is strictly TD(0). It does not yet support eligibility traces.
    NOTE: Strictly linear function approximation.
    """
    # Update auxiliary weights (w)
    w_new = w + beta * rho * (error - torch.dot(w, features)) * features

    # Update primary weights (weights) with gradient correction
    weights_new = (
        weights
        + alpha * rho * error * features
        - alpha
        * rho
        * gamma
        * next_features
        * (1.0 - float(terminated))
        * torch.dot(w, features)
    )

    return weights_new, w_new


# TODO: should v_next be handled here?
def true_online_td_update(
    error: float | torch.Tensor,
    v_current: float | torch.Tensor,
    v_old: float | torch.Tensor,
    features: torch.Tensor,
    weights: torch.Tensor,
    alpha: float | torch.Tensor,
    trace: torch.Tensor,
) -> torch.Tensor:
    """
    Performs a True Online Temporal Difference (TD) update for linear function approximation.

    Args:
        error: The scalar error term (e.g., TD error or Advantage).
        v_current: The value of the current state computed using the current weight vector.
        v_old: The value of the current state computed using the previous weight vector.
        features: Feature vector of the current state [features].
        weights: Current weight vector [features].
        alpha: Learning rate.
        trace: The updated True Online eligibility trace for the current step (e_t) [features].

    Returns:
        weights_new: The updated weight vector [features].

    NOTE: Strictly linear function approximation.
    NOTE: We implement True Online TD(lambda) weight update from Suttons Textbook (2nd Ed.) not from the True Online TD(lambda) paper.
    """
    # Fail Fast: Ensure shape alignment
    assert features.ndim == 1, f"Expected 1D features [features], got {features.shape}"
    assert (
        weights.shape == features.shape
    ), f"Shape mismatch: weights {weights.shape}, features {features.shape}"

    # w <- w + \alpha * (\delta + V - V_old) * z - \alpha * (V - V_old) * x
    v_diff = v_current - v_old
    weights_new = weights + alpha * (error + v_diff) * trace - alpha * v_diff * features

    return weights_new
