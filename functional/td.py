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
from einops import rearrange
from typing import Tuple


# TODO: I think truncation right now works for single env trajectories without auto resetting, but would fail for vectorized envs with auto resetting. as the next_q_values would be on the resetted state instead of the one from the info. verify this, and if it is an issue, unify the vectorization logic in our buffer to work with offline buffers as well somehow.
# TODO: should PPO and A2C etc use this for their value head? what do they currently do/use?
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
    support_b = rearrange(support, "a -> 1 a")
    rewards_b = rearrange(rewards, "b -> b 1")
    gamma_b = rearrange(gamma, "b -> b 1")
    term_b = rearrange(terminated, "b -> b 1")

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
    m = torch.zeros(batch_size, atom_size, device=rewards.device)
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
    m_flat = rearrange(m, "b a -> (b a)")
    offset_l = rearrange(l + offset, "b a -> (b a)")
    offset_u = rearrange(u + offset, "b a -> (b a)")

    prob_lower = rearrange(next_probs_a * (u.float() - b), "b a -> (b a)")
    prob_upper = rearrange(next_probs_a * (b - l.float()), "b a -> (b a)")

    # Index Add becomes clean and fast:
    m_flat.index_add_(0, offset_l, prob_lower)
    m_flat.index_add_(0, offset_u, prob_upper)

    return m


# GRADIENT TD METHODS
# TODO: more semantic naming instead of phi and theta etc?
# TODO: actually use/test importance sampling
# TODO: document that terminated defaults to false for the continual learning case by default.
# TODO: should we remove in favor of TDC? or True Online TD?
# TODO: batch updates where phi comes from vectorized envs. "batched online learning"
def semi_gradient_td_update(
    features: torch.Tensor,
    reward: float | torch.Tensor,
    next_features: torch.Tensor,
    gamma: float | torch.Tensor,
    weights: torch.Tensor,
    alpha: float | torch.Tensor,
    update_vector: torch.Tensor,
    rho: float | torch.Tensor = 1.0,
    terminated: bool | torch.Tensor = False,
) -> torch.Tensor:
    """
    Performs a semi-gradient Temporal Difference (TD) update for linear function approximation. Allows for eligibility traces.
    Args:
        features: Feature vector of the current state [features].
        reward: Reward received after transitioning from the current state.
        next_features: Feature vector of the next state [features].
        gamma: Discount factor.
        weights: Current weight vector [features].
        alpha: Learning rate.
        update_vector: The vector used to step the weights.
            - For TD(0), pass `features`.
            - For TD(lambda), pass the accumulated `eligibility_trace`.
        rho: Importance sampling ratio (default: 1.0 for on-policy).
        terminated: Whether the next state is a terminal state (default: False).

    Returns:
        The updated weight vector weights [features].

    NOTE: Strictly linear function approximation.
    """
    # 1. Compute state values using linear function approximation: V(s) = weights^T * features
    v_t = torch.dot(weights, features)
    # Mask v_next if terminated
    v_next = torch.dot(weights, next_features) * (1.0 - float(terminated))

    # 2. Compute TD error: delta = r + gamma * v_next - v_t
    td_error = reward + gamma * v_next - v_t

    # 3. Apply semi-gradient update
    # weights = weights + alpha * rho * delta * update_vector
    weights_new = weights + alpha * rho * td_error * update_vector

    return weights_new


# TODO: should we remove in favor of TDC? or True Online TD?
def gtd0_update(
    features: torch.Tensor,
    reward: float | torch.Tensor,
    next_features: torch.Tensor,
    gamma: float | torch.Tensor,
    weights: torch.Tensor,
    u: torch.Tensor,
    alpha: float | torch.Tensor,
    beta: float | torch.Tensor,
    rho: float | torch.Tensor = 1.0,
    terminated: bool | torch.Tensor = False,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    GTD(0) update from Sutton et al. (2009). NOTE: Faithful to the original 2009 paper, not modern GTD2/TDC.

    Args:
        features: Feature vector of the current state [features].
        reward: Reward received after transitioning from the current state.
        next_features: Feature vector of the next state [features].
        gamma: Discount factor.
        weights: Current weight vector [features].
        u: Auxiliary weight vector for GTD(0) [features].
        alpha: Learning rate.
        beta: Step size for auxiliary weight updates.
        rho: Importance sampling ratio (default: 1.0 for on-policy).
        terminated: Whether the next state is a terminal state (default: False).

    Returns:
        The updated weight vector weights [features] and auxiliary weight vector u [features].

    NOTE: This implementation is strictly TD(0). It does not yet support eligibility traces.
    NOTE: Strictly linear function approximation.
    """
    v_t = torch.dot(weights, features)
    v_next = torch.dot(weights, next_features) * (1.0 - float(terminated))
    td_error = reward + gamma * v_next - v_t

    # Update auxiliary weights (u)
    u_new = u + beta * rho * (td_error * features - u)

    # Update primary weights (weights)
    weights_new = weights + alpha * rho * (
        features - gamma * next_features * (1.0 - float(terminated))
    ) * torch.dot(features, u)

    return weights_new, u_new


def tdc_update(
    features: torch.Tensor,
    reward: float | torch.Tensor,
    next_features: torch.Tensor,
    gamma: float | torch.Tensor,
    weights: torch.Tensor,
    w: torch.Tensor,
    alpha: float | torch.Tensor,
    beta: float | torch.Tensor,
    rho: float | torch.Tensor = 1.0,
    terminated: bool | torch.Tensor = False,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Fast-GTD / TDC update from Sutton et al. (2009).

    Args:
        features: Feature vector of the current state [features].
        reward: Reward received after transitioning from the current state.
        next_features: Feature vector of the next state [features].
        gamma: Discount factor.
        weights: Current weight vector [features].
        w: Auxiliary weight vector for TDC [features].
        alpha: Learning rate.
        beta: Step size for auxiliary weight updates.
        rho: Importance sampling ratio (default: 1.0 for on-policy).
        terminated: Whether the next state is a terminal state (default: False).

    Returns:
        The updated weight vector weights [features] and auxiliary weight vector w [features].

    NOTE: This implementation is strictly TD(0). It does not yet support eligibility traces.
    NOTE: Strictly linear function approximation.
    """
    v_t = torch.dot(weights, features)
    v_next = torch.dot(weights, next_features) * (1.0 - float(terminated))
    td_error = reward + gamma * v_next - v_t

    # Update auxiliary weights (w)
    w_new = w + beta * rho * (td_error - torch.dot(w, features)) * features

    # Update primary weights (weights) with gradient correction
    weights_new = (
        weights
        + alpha * rho * td_error * features
        - alpha * rho * gamma * next_features * (1.0 - float(terminated)) * torch.dot(w, features)
    )

    return weights_new, w_new


def true_online_td_update(
    features: torch.Tensor,
    reward: float | torch.Tensor,
    next_features: torch.Tensor,
    v_old: float | torch.Tensor,
    gamma: float | torch.Tensor,
    weights: torch.Tensor,
    alpha: float | torch.Tensor,
    trace: torch.Tensor,
    terminated: bool | torch.Tensor = False,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Performs a True Online Temporal Difference (TD) update for linear function approximation.

    Args:
        features: Feature vector of the current state [features].
        reward: Reward received after transitioning from the current state.
        next_features: Feature vector of the next state [features].
        v_old: The value of the current state computed using the previous weight vector (weights_{t-1}^T features_t).
        gamma: Discount factor.
        weights: Current weight vector [features].
        alpha: Learning rate.
        trace: The updated True Online eligibility trace for the current step (e_t) [features].
        terminated: Whether the next state is a terminal state (default: False).

    Returns:
        A tuple of (weights_new, v_next):
            - weights_new: The updated weight vector [features].
            - v_next: The value of the next state computed with the current weights (weights_t^T features_{t+1}).
                      This must be passed as `v_old` in the next environment step.

    NOTE: Strictly linear function approximation.
    """
    # Fail Fast: Ensure shape alignment
    assert features.ndim == 1, f"Expected 1D features [features], got {features.shape}"
    assert (
        weights.shape == features.shape
    ), f"Shape mismatch: weights {weights.shape}, features {features.shape}"

    # 1. Compute v_next using the CURRENT weights: v_{S'} = weights^T * next_features
    v_next = torch.dot(weights, next_features) * (1.0 - float(terminated))

    # 2. Compute TD error: delta = R + gamma * v_{S'} - v_old
    td_error = reward + gamma * v_next - v_old

    # 3. Compute v_current using the CURRENT weights: v_S = weights^T * features
    v_current = torch.dot(weights, features)

    # 4. Apply True Online TD weight update
    # weights_{t+1} = weights_t + delta_t * e_t + alpha * (v_old - v_current) * features_t
    weights_new = weights + td_error * trace + alpha * (v_old - v_current) * features

    return weights_new, v_next
