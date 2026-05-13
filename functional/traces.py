import torch


# TODO: should these be in my returns.py or something? what about td.py?
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
    features: torch.Tensor,  # Binary features (phi_t) for tabular/linear cases
    gamma: float,
    lam: float,
    terminated: torch.Tensor,
) -> torch.Tensor:
    """
    Updates eligibility traces using the replacing trace method (Sutton & Barto).
    Formula: e_t = max(gamma * lambda * e_{t-1}, phi_t)
    """
    term_mask = terminated.unsqueeze(-1).float()
    decayed_traces = gamma * lam * traces * (1.0 - term_mask)
    return torch.max(decayed_traces, features)
