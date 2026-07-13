import torch
import warnings

# TODO: eventually will have to figure out how to get these working with MuZero (alternating players and all). it works fine for single agent or if we treat the opponent as part of the environment (ie not an agent), but for things like muzero where the agent learns as both players and value depends on the agents perspective this needs to be updated. (ie for chess, players alternate so if p1 won at the end of the game they get a reward 1, but the mc returns for all the states is not 1, it is 1, -1, -1, 1, -1, ...). There may be several solutions, for one i know mctx does there tree search without worrying about player turn, is there a way to do that? maybe some kind of higher order function to make it player turn aware? or another function to flip the sign of the returns depending on the player the user uses before passing to these or something?


def compute_mc_returns(
    rewards: torch.Tensor,
    terminated: torch.Tensor,
    truncated: torch.Tensor,
    gamma: float,
) -> torch.Tensor:
    """
    Computes batched Monte Carlo discounted returns. The same as a compute discounted rewards function, with the soft expectation that all trajectories have been terminated at some point.

    Formula: G_t = r_t + gamma * r_{t+1} + gamma^2 * r_{t+2} + ...

    MC returns are computed by backward-iterating through the rewards, stopping at
    either termination or truncation.

    Args:
        rewards: Reward tensor [B, T].
        terminated: Termination mask [B, T].
        truncated: Truncation mask [B, T].
        gamma: Discount factor.

    Returns:
        Discounted returns [B, T].
        Note: The returned tensor is not explicitly detached.
    """
    assert rewards.ndim == 2, f"Expected 2D rewards [B, T], got {rewards.shape}"
    assert (
        rewards.shape == terminated.shape == truncated.shape
    ), "Shape mismatch in returns inputs"

    # Combine masks: MC returns stop at either terminated or truncated
    # Both mean the trajectory ended for the purpose of the current return calculation.
    done = (terminated.bool() | truncated.bool()).float()

    # Validation: Warn if any trajectory in the batch is never done.
    # Infinite-horizon MC returns are biased if not properly handled.
    is_never_terminal = ~(terminated.bool().any(dim=1))
    if is_never_terminal.any():
        warnings.warn(
            "Found trajectory in batch where terminated is always False (never terminal). "
            "MC returns for these trajectories may be biased."
        )
    is_truncated_but_not_terminated = (truncated.bool() & ~terminated.bool()).any(dim=1)
    if is_truncated_but_not_terminated.any():
        warnings.warn(
            "Found trajectory in batch where truncated is True but not terminated."
            "MC returns for these trajectories may be biased."
        )

    returns = torch.zeros_like(rewards)
    R = torch.zeros_like(rewards[:, 0])  # [B]

    # Backward iteration for efficient O(T) calculation
    for t in reversed(range(rewards.size(1))):
        # If done is True (1.0), the next R gets multiplied by 0 (bootstrapping stops)
        mask = 1.0 - done[:, t]
        R = rewards[:, t] + gamma * R * mask
        returns[:, t] = R

    return returns


def compute_n_step_returns(
    rewards: torch.Tensor,
    terminated: torch.Tensor,
    truncated: torch.Tensor,
    values: torch.Tensor,
    next_values: torch.Tensor,
    gamma: float,
    n: int,
) -> torch.Tensor:
    """
    Computes batched n-step bootstrapped discounted returns.

    Formula: G_t^{(n)} = r_t + gamma r_{t+1} + ... + gamma^{n-1} r_{t+n-1} + gamma^n V(s_{t+n})

    Correctly handles both terminated (no bootstrap) and truncated (bootstrap) states.

    Args:
        rewards: Reward tensor [B, T].
        terminated: Termination mask [B, T].
        truncated: Truncation mask [B, T].
        values: State values V(s_t) [B, T].
        next_values: Next state values V(s_{t+1}) [B, T].
        gamma: Discount factor.
        n: Number of steps to look ahead.

    Returns:
        N-step returns [B, T].
        Note: The returned tensor is not explicitly detached.
    """
    assert rewards.ndim == 2, f"Expected 2D rewards [B, T], got {rewards.shape}"
    assert (
        rewards.shape == terminated.shape == truncated.shape == next_values.shape
    ), "Shape mismatch in n-step returns inputs"
    assert n >= 1, f"n-step must be at least 1, got {n}"

    T = rewards.size(1)
    done = (terminated.bool() | truncated.bool()).float()

    # Base case: 1-step returns (Pure Bellman math)
    # G_t^{(1)} = r_t + gamma * (1 - terminated_t) * V(s_{t+1})
    returns = rewards + gamma * (1.0 - terminated.float()) * next_values.detach()

    # Recursive calculation: G_t^{(i)} = r_t + gamma * (1 - done_t) * G_{t+1}^{(i-1)}
    for i in range(1, n):
        if T > i:
            # Only update steps where the episode hasn't ended.
            # If done[:, t] is True, G_t already contains the terminal/truncated return.
            mask = (1.0 - done[:, :-i]).bool()
            returns[:, :-i] = torch.where(
                mask,
                rewards[:, :-i] + gamma * returns[:, 1 : T - i + 1],
                returns[:, :-i],
            )

    return returns


def compute_gae(
    rewards: torch.Tensor,
    terminated: torch.Tensor,
    truncated: torch.Tensor,
    values: torch.Tensor,
    next_values: torch.Tensor,
    gamma: float,
    gae_lambda: float,
) -> torch.Tensor:
    """
    Computes the Generalized Advantage Estimate (GAE) for a batch of trajectories.
    (Moved from advantages.py to consolidate temporal credit assignment math).

    Args:
        rewards (torch.Tensor): The reward tensor of shape [batch, time].
        terminated (torch.Tensor): Boolean/float mask indicating episode termination.
        truncated (torch.Tensor): Boolean/float mask indicating episode truncation.
        values (torch.Tensor): The values tensor of shape [batch, time].
        next_values (torch.Tensor): The values for the next state in the batch.
        gamma (float): The discount factor.
        gae_lambda (float): The GAE lambda parameter (typically 0.95 or 0.99).
    Expects exact [B, T] shapes.

    Returns:
        The Generalized Advantage Estimate [B, T].
        Note: The returned tensor is not explicitly detached.
    """
    assert rewards.ndim == 2, f"Expected 2D rewards [B, T], got {rewards.shape}"
    assert (
        rewards.shape
        == terminated.shape
        == truncated.shape
        == values.shape
        == next_values.shape
    ), "Shape mismatch in GAE inputs"

    # Combined done mask
    done = (terminated.bool() | truncated.bool()).float()

    # Pure Bellman Math
    deltas = rewards + gamma * (1.0 - terminated.float()) * next_values - values

    gae = torch.zeros_like(rewards)
    last_gae = torch.zeros_like(rewards[:, 0])  # [B]

    for t in reversed(range(rewards.shape[1])):
        mask = 1.0 - done[:, t].float()
        last_gae = deltas[:, t] + gamma * gae_lambda * mask * last_gae
        gae[:, t] = last_gae

    return gae


def compute_td_lambda_returns(
    rewards: torch.Tensor,
    terminated: torch.Tensor,
    truncated: torch.Tensor,
    values: torch.Tensor,
    next_values: torch.Tensor,
    gamma: float,
    lam: float,
) -> torch.Tensor:
    """
    Computes TD(lambda) returns.
    Mathematically, TD(λ) returns are equivalent to GAE + State Values.
    We preserve compute_gae as a separate function for semantic readability,
    but compose it here to unify the returns API.

    Returns:
        The TD(lambda) returns [B, T].
        Note: The returned tensor is not explicitly detached.
    """
    advantages = compute_gae(
        rewards, terminated, truncated, values, next_values, gamma, lam
    )
    return advantages + values.detach()
