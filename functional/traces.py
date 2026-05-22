import torch


# TODO: should these be in my returns.py or something? what about td.py?
# TODO: should we add alphas here?
def update_accumulating_traces(
    traces: torch.Tensor,  # [batch, num_features]
    gradients: torch.Tensor,  # [batch, num_features]
    gamma: float,  # TODO: should these be tensors like gamma in my other functions?
    lam: float,  # TODO: should these be tensors like gamma in my other functions?
    terminated: torch.Tensor,  # [batch]
) -> torch.Tensor:
    """
    Updates eligibility traces using the accumulating trace method.
    Formula: e_t = gamma * lambda * e_{t-1} + grad_V(s_t)

    If the episode terminates, the trace is reset to zero for that batch element.

    Args:
        traces: The eligibility traces from the previous step.
        gradients: The gradient of the value function with respect to weights (phi_t).
        gamma: Discount factor.
        lam: Trace decay rate (lambda).
        terminated: Mask [B] indicating episode termination to clear traces.

    Returns:
        The updated traces of shape [batch, num_features].
    """
    assert traces.shape == gradients.shape, "Trace and gradient shapes must match"

    # Expand terminated to match feature dimensions [B, 1] for broadcasting
    term_mask = terminated.unsqueeze(-1).float()

    # Reset trace if terminated, otherwise decay and accumulate
    new_traces = (gamma * lam * traces * (1.0 - term_mask)) + gradients
    return new_traces


def update_replacing_traces(
    traces: torch.Tensor,
    features: torch.Tensor,  # Can be continuous/non-binary
    gamma: float | torch.Tensor,
    lam: float | torch.Tensor,
    terminated: torch.Tensor,
) -> torch.Tensor:
    """
    Updates eligibility traces using the replacing trace method (Sutton & Barto).
    Formula: e_t = max(gamma * lambda * e_{t-1}, phi_t)

    NOTE: Replacing traces are usually defined only for discrete states or linear function approximation with binary features (that are either 1 or 0, present or not present)

    Extended in True Online TD(lambda) to handle non-binary features as follows:
    e_{i,t} = γλe_{i,t−1} if φ_{i,t} = 0
            = αφ_{i,t} if φ_{i,t} != 0

    TODO: possible future work True Online TD Lambda for offline case (is there a paper for this?)
    """
    # Expand terminated to match feature dimensions [B, 1]
    term_mask = terminated.unsqueeze(-1).float()

    # 1. Calculate the standard decayed trace (resetting if terminated)
    decayed_traces = gamma * lam * traces * (1.0 - term_mask)

    # 2. Apply the conditional replacement
    # Using a small epsilon or exact 0 check depending on your feature precision
    feature_is_zero = features == 0.0

    # If the feature is 0, keep the decayed trace. Otherwise, REPLACE it.
    new_traces = torch.where(feature_is_zero, decayed_traces, features)

    return new_traces


def update_true_online_traces(
    traces: torch.Tensor,  # [B, features]
    features: torch.Tensor,  # [B, features]
    alpha: float | torch.Tensor,
    gamma: float | torch.Tensor,
    lam: float | torch.Tensor,
    terminated: torch.Tensor,  # [B]
) -> torch.Tensor:
    """
    Updates eligibility traces using the True Online TD(lambda) method (Sutton 2014).
    Formula: e_t = gamma * lambda * e_{t-1} + alpha * (1 - gamma * lambda * e_{t-1}^T features_t) * features_t

    Args:
        traces: The eligibility traces from the previous step.
        features: The feature vector of the current state.
        alpha: Learning rate.
        gamma: Discount factor.
        lam: Trace decay rate (lambda).
        terminated: Mask [B] indicating episode termination to clear traces.

    Returns:
        The updated traces of shape [batch, num_features].

    NOTE: We implement True Online TD(lambda) trace update from Suttons Textbook (2nd Ed.) not from the True Online TD(lambda) paper.
    """
    # Fail Fast: Ensure shape alignment
    assert (
        traces.shape == features.shape
    ), f"Trace {traces.shape} and feature {features.shape} shapes must match"
    assert (
        terminated.ndim == 1
    ), f"Expected 1D terminated tensor [B], got {terminated.shape}"

    term_mask = terminated.unsqueeze(-1).float()

    # \gamma * \lambda * z
    z_decay = gamma * lam * traces * (1.0 - term_mask)

    # z^T * x
    inner_dot = torch.sum(traces * features, dim=-1, keepdim=True)

    # z_t = gamma * lambda * z_{t-1} + (1 - alpha * gamma * lambda * z_{t-1}^T features_t) * features_t
    new_traces = z_decay + (1.0 - alpha * gamma * lam * inner_dot) * features

    return new_traces
