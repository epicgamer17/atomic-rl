import pytest
import torch
from atomic_rl.returns import (
    compute_mc_returns,
    compute_n_step_returns,
    compute_gae,
    compute_td_lambda_returns,
)

pytestmark = pytest.mark.unit


def test_compute_mc_returns_1d_via_2d():
    """
    Test 1D return calculation (passed as 2D).
    Analytical Oracle:
    rewards = [1, 1, 1], gamma = 0.9
    G_2 = 1.0
    G_1 = 1.0 + 0.9*1.0 = 1.9
    G_0 = 1.0 + 0.9*1.9 = 2.71
    """
    rewards = torch.tensor([[1.0, 1.0, 1.0]])
    terminated = torch.tensor([[0.0, 0.0, 1.0]])
    truncated = torch.zeros_like(terminated)
    gamma = 0.9
    expected = torch.tensor([[2.71, 1.9, 1.0]])

    returns = compute_mc_returns(rewards, terminated, truncated, gamma)
    torch.testing.assert_close(returns, expected)


def test_compute_mc_returns_batched():
    """Test batched return calculation."""
    rewards = torch.tensor([[1.0, 1.0, 1.0], [2.0, 2.0, 2.0]])
    terminated = torch.tensor([[0.0, 0.0, 1.0], [0.0, 0.0, 1.0]])
    truncated = torch.zeros_like(terminated)
    gamma = 0.9
    # Batch 0: same as above
    # Batch 1:
    # G_2 = 2.0
    # G_1 = 2.0 + 0.9*2.0 = 2.0 + 1.8 = 3.8
    # G_0 = 2.0 + 0.9*3.8 = 2.0 + 3.42 = 5.42
    expected = torch.tensor([[2.71, 1.9, 1.0], [5.42, 3.8, 2.0]])

    returns = compute_mc_returns(rewards, terminated, truncated, gamma)
    torch.testing.assert_close(returns, expected)


def test_compute_mc_returns_with_terminated():
    """Test return calculation with episode boundaries."""
    rewards = torch.tensor([[1.0, 1.0, 1.0, 1.0]])
    terminated = torch.tensor([[0.0, 1.0, 0.0, 1.0]])
    truncated = torch.zeros_like(terminated)
    gamma = 0.9

    # Episode 1 ends at t=1
    # G_1 = reward[1] = 1.0
    # G_0 = reward[0] + gamma * G_1 = 1.0 + 0.9 * 1.0 = 1.9
    # Episode 2 starts at t=2
    # G_3 = reward[3] = 1.0
    # G_2 = reward[2] + gamma * G_3 = 1.0 + 0.9 * 1.0 = 1.9

    expected = torch.tensor([[1.9, 1.0, 1.9, 1.0]])

    returns = compute_mc_returns(rewards, terminated, truncated, gamma)
    torch.testing.assert_close(returns, expected)


def test_compute_mc_returns_never_done_warning():
    """Test that a warning is issued if neither terminated nor truncated is True."""
    rewards = torch.tensor([[1.0, 1.0]])
    terminated = torch.tensor([[0.0, 0.0]])
    truncated = torch.tensor([[0.0, 0.0]])
    gamma = 0.9

    with pytest.warns(
        UserWarning,
        match="Found trajectory in batch where terminated is always False \(never terminal\).",
    ):
        compute_mc_returns(rewards, terminated, truncated, gamma)


def test_compute_mc_returns_truncation():
    """Test that MC returns stop at truncation point (no bootstrap)."""
    rewards = torch.tensor([[1.0, 1.0]])
    terminated = torch.tensor([[0.0, 0.0]])
    truncated = torch.tensor([[0.0, 1.0]])
    gamma = 0.9

    # G_1 = 1.0
    # G_0 = 1.0 + 0.9 * 0 * G_next = 1.0 (Wait, no, it stops at G_1)
    # Actually, if t=1 is truncated, G_1 is 1.0.
    # G_0 = 1.0 + 0.9 * G_1 = 1.9.
    # BUT if t=0 is truncated, G_0 = 1.0.

    rewards = torch.tensor([[1.0, 2.0]])
    truncated = torch.tensor([[1.0, 0.0]])
    # G_1 = 2.0
    # G_0 = 1.0 (stopped because t=0 is truncated)
    expected = torch.tensor([[1.0, 2.0]])
    returns = compute_mc_returns(rewards, terminated, truncated, gamma)
    torch.testing.assert_close(returns, expected)


def test_compute_n_step_returns_basic():
    """Test n-step returns for various n values."""
    rewards = torch.tensor([[1.0, 1.0, 1.0]])
    terminated = torch.tensor([[0.0, 0.0, 0.0]])
    truncated = torch.zeros_like(terminated)
    # values are V(s0), V(s1), V(s2)
    values = torch.tensor([[0.5, 0.5, 0.5]])
    # last_values is V(s3)
    last_values = torch.tensor([[0.5]])
    gamma = 0.9

    # n=1: G0 = 1 + 0.9*0.5 = 1.45
    expected_n1 = torch.tensor([[1.45, 1.45, 1.45]])
    # Construct next_values [V(s_1), ..., V(s_T)]
    next_values = torch.cat([values[:, 1:], last_values], dim=1)

    returns_n1 = compute_n_step_returns(
        rewards, terminated, truncated, values, next_values, gamma, n=1
    )
    torch.testing.assert_close(returns_n1, expected_n1)

    # n=2: G0 = 1 + 0.9*1 + 0.81*0.5 = 2.305; G2 stays 1-step = 1.45
    expected_n2 = torch.tensor([[2.305, 2.305, 1.45]])
    returns_n2 = compute_n_step_returns(
        rewards, terminated, truncated, values, next_values, gamma, n=2
    )
    torch.testing.assert_close(returns_n2, expected_n2)

    # n=3: G0 = 1 + 0.9*1 + 0.81*1 + 0.729*0.5 = 3.0745
    expected_n3 = torch.tensor([[3.0745, 2.305, 1.45]])
    returns_n3 = compute_n_step_returns(
        rewards, terminated, truncated, values, next_values, gamma, n=3
    )
    torch.testing.assert_close(returns_n3, expected_n3)


def test_compute_n_step_returns_with_terminals():
    """Test n-step returns with episode termination."""
    rewards = torch.tensor([[1.0, 1.0, 1.0, 1.0]])
    terminated = torch.tensor([[0.0, 1.0, 0.0, 0.0]])  # Episode ends at t=1
    truncated = torch.zeros_like(terminated)
    values = torch.tensor([[0.5, 0.5, 0.5, 0.5]])
    last_values = torch.tensor([[0.5]])
    gamma = 0.9

    # n=2:
    # G0 = r0 + gamma * (1-d0) * r1 = 1 + 0.9 * 1 = 1.9 (no bootstrap because d1=1)
    # G1 = r1 = 1.0 (terminal)
    # G2 = r2 + gamma * r3 + gamma^2 * V4 = 1 + 0.9 + 0.405 = 2.305
    # G3 = r3 + gamma * V4 = 1.45
    expected = torch.tensor([[1.9, 1.0, 2.305, 1.45]])
    # Construct next_values [V(s_1), ..., V(s_T)]
    next_values = torch.cat([values[:, 1:], last_values], dim=1)

    returns = compute_n_step_returns(
        rewards, terminated, truncated, values, next_values, gamma, n=2
    )
    torch.testing.assert_close(returns, expected)


def test_compute_n_step_returns_truncation():
    """Test that n-step returns correctly bootstrap on truncation."""
    rewards = torch.tensor([[1.0, 1.0]])
    terminated = torch.tensor([[0.0, 1.0]])  # Terminated at t=1
    truncated = torch.tensor([[1.0, 0.0]])  # Truncated at t=0
    values = torch.tensor([[10.0, 20.0]])
    last_values = torch.tensor([[5.0]])
    gamma = 0.9
    n = 2

    # G_1 = R_1 + gamma * (1-d_1) * V_next = 1.0 + 0.9 * 0 * 5.0 = 1.0
    # G_0 = R_0 + gamma * (1-d_0) * V_1 = 1.0 + 0.9 * 1 * 20.0 = 19.0
    # Since truncated_0 is 1, n-step propagation stops at t=0.

    expected = torch.tensor([[19.0, 1.0]])
    # Construct next_values [V(s_1), ..., V(s_T)]
    next_values = torch.cat([values[:, 1:], last_values], dim=1)

    returns = compute_n_step_returns(
        rewards, terminated, truncated, values, next_values, gamma, n=n
    )
    torch.testing.assert_close(returns, expected)


def test_gae_analytical_oracle():
    """
    Test GAE with a hand-calculated example.

    Setup:
    B=1, T=2
    gamma = 0.9
    gae_lambda = 0.95

    rewards = [[1.0, 2.0]]
    terminals = [[0.0, 1.0]]
    values = [[10.0, 20.0]]
    last_values = [5.0]

    Calculations:
    t=1:
        delta_1 = r_1 + gamma * (1-d_1) * V_next - V_1
                = 2.0 + 0.9 * (1-1.0) * 5.0 - 20.0
                = 2.0 + 0.0 - 20.0 = -18.0
        A_1 = delta_1 + gamma * lambda * (1-d_1) * A_next
            = -18.0 + 0.9 * 0.95 * 0.0 * 0.0 = -18.0

    t=0:
        delta_0 = r_0 + gamma * (1-d_0) * V_1 - V_0
                = 1.0 + 0.9 * (1-0.0) * 20.0 - 10.0
                = 1.0 + 18.0 - 10.0 = 9.0
        A_0 = delta_0 + gamma * lambda * (1-d_0) * A_1
            = 9.0 + 0.9 * 0.95 * 1.0 * (-18.0)
            = 9.0 + 0.855 * (-18.0)
            = 9.0 - 15.39 = -6.39

    Expected GAE: [[-6.39, -18.0]]
    """
    rewards = torch.tensor([[1.0, 2.0]])
    terminals = torch.tensor([[0.0, 1.0]])
    values = torch.tensor([[10.0, 20.0]])
    last_values = torch.tensor([5.0])
    gamma = 0.9
    gae_lambda = 0.95

    # Construct next_values [V(s_1), ..., V(s_T)]
    next_values = torch.cat([values[:, 1:], last_values.unsqueeze(1)], dim=1)

    gae = compute_gae(
        rewards=rewards,
        terminated=terminals,
        truncated=torch.zeros_like(terminals),
        values=values,
        next_values=next_values,
        gamma=gamma,
        gae_lambda=gae_lambda,
    )

    expected = torch.tensor([[-6.39, -18.0]])

    assert gae.shape == (1, 2), f"Expected shape (1, 2), got {gae.shape}"
    torch.testing.assert_close(gae, expected)


def test_gae_batch_independence():
    """
    Test that batches are processed independently.
    We create two identical trajectories in a batch and ensure they have identical GAEs.
    """
    B, T = 2, 5
    rewards = torch.randn(B, T)
    terminals = torch.zeros(B, T)
    values = torch.randn(B, T)
    last_values = torch.randn(B)

    # Make batch 1 identical to batch 0
    rewards[1] = rewards[0]
    terminals[1] = terminals[0]
    values[1] = values[0]
    last_values[1] = last_values[0]

    next_values = torch.cat([values[:, 1:], last_values.unsqueeze(1)], dim=1)

    gae = compute_gae(
        rewards=rewards,
        terminated=terminals,
        truncated=torch.zeros_like(terminals),
        values=values,
        next_values=next_values,
        gamma=0.99,
        gae_lambda=0.95,
    )

    assert gae.shape == (B, T)
    torch.testing.assert_close(gae[0], gae[1])


def test_gae_terminal_reset():
    """
    Test that GAE calculation resets at terminal states.
    If a terminal occurs at t=1, A_0 should not depend on A_2.
    """
    # T=3
    # r0, r1, r2
    # d0, d1, d2
    # If d1 = 1.0, then A_1 depends on delta_1 but NOT on A_2.
    # And A_0 depends on delta_0 and A_1.

    rewards = torch.tensor([[1.0, 1.0, 1.0]])
    terminals = torch.tensor([[0.0, 1.0, 0.0]])  # Terminal at t=1
    values = torch.zeros((1, 3))
    last_values = torch.zeros(1)
    gamma = 1.0
    gae_lambda = 1.0

    # delta_t = r_t + 1.0 * (1-d_t) * V_{t+1} - V_t = r_t
    # t=2: A_2 = delta_2 + 1.0 * 1.0 * (1-0.0) * 0 = r_2 = 1.0
    # t=1: A_1 = delta_1 + 1.0 * 1.0 * (1-1.0) * A_2 = delta_1 = r_1 = 1.0
    # t=0: A_0 = delta_0 + 1.0 * 1.0 * (1-0.0) * A_1 = r_0 + A_1 = 1.0 + 1.0 = 2.0

    next_values = torch.cat([values[:, 1:], last_values.unsqueeze(1)], dim=1)

    gae = compute_gae(
        rewards=rewards,
        terminated=terminals,
        truncated=torch.zeros_like(terminals),
        values=values,
        next_values=next_values,
        gamma=gamma,
        gae_lambda=gae_lambda,
    )

    expected = torch.tensor([[2.0, 1.0, 1.0]])
    torch.testing.assert_close(gae, expected)


def test_gae_lambda_zero():
    """
    When lambda=0, GAE should be equal to the 1-step TD residuals (deltas).
    """
    B, T = 1, 3
    rewards = torch.randn(B, T)
    terminals = torch.zeros(B, T)
    values = torch.randn(B, T)
    last_values = torch.randn(B)
    gamma = 0.99

    next_values = torch.cat([values[:, 1:], last_values.unsqueeze(1)], dim=1)

    gae = compute_gae(
        rewards=rewards,
        terminated=terminals,
        truncated=torch.zeros_like(terminals),
        values=values,
        next_values=next_values,
        gamma=gamma,
        gae_lambda=0.0,
    )

    # Calculate deltas manually
    next_values = torch.cat([values[:, 1:], last_values.unsqueeze(1)], dim=1)
    expected_deltas = rewards + gamma * (1.0 - terminals) * next_values - values

    torch.testing.assert_close(gae, expected_deltas)


def test_gae_zeros():
    """
    Test with all zeros to ensure no NaNs or crashes.
    """
    B, T = 2, 5
    rewards = torch.zeros(B, T)
    terminals = torch.zeros(B, T)
    values = torch.zeros(B, T)
    last_values = torch.zeros(B)

    next_values = torch.cat([values[:, 1:], last_values.unsqueeze(1)], dim=1)

    gae = compute_gae(
        rewards=rewards,
        terminated=terminals,
        truncated=torch.zeros_like(terminals),
        values=values,
        next_values=next_values,
        gamma=0.99,
        gae_lambda=0.95,
    )

    assert torch.all(gae == 0.0)
    assert not torch.any(torch.isnan(gae))


def test_gae_truncation_bootstrapping():
    """
    Test that GAE correctly bootstraps on truncated states but not on terminated states.
    """
    rewards = torch.tensor([[1.0, 1.0]])
    terminated = torch.tensor([[0.0, 1.0]])  # Terminated at t=1
    truncated = torch.tensor([[1.0, 0.0]])  # Truncated at t=0
    values = torch.tensor([[10.0, 20.0]])
    last_values = torch.tensor([5.0])
    gamma = 0.9
    gae_lambda = 1.0

    # t=1 (Terminated):
    # delta_1 = r_1 + gamma * (1-d_1) * V_next - V_1 = 1.0 + 0.9 * 0 * 5.0 - 20.0 = -19.0
    # A_1 = delta_1 = -19.0

    # t=0 (Truncated):
    # delta_0 = r_0 + gamma * (1-d_0) * V_1 - V_0 = 1.0 + 0.9 * 1 * 20.0 - 10.0 = 1.0 + 18.0 - 10.0 = 9.0
    # A_0 = delta_0 + gamma * lambda * (1 - done_0) * A_1
    # Since truncated_0 is 1, done_0 is 1.
    # A_0 = delta_0 = 9.0

    next_values = torch.cat([values[:, 1:], last_values.unsqueeze(1)], dim=1)

    gae = compute_gae(
        rewards=rewards,
        terminated=terminated,
        truncated=truncated,
        values=values,
        next_values=next_values,
        gamma=gamma,
        gae_lambda=gae_lambda,
    )

    expected = torch.tensor([[9.0, -19.0]])
    torch.testing.assert_close(gae, expected)


def test_gae_assertions():
    """Test that GAE raises assertions on invalid input shapes."""
    rewards = torch.randn(4, 10)
    # Shape mismatch
    with pytest.raises(AssertionError, match="Shape mismatch in GAE inputs"):
        compute_gae(
            rewards=rewards,
            terminated=torch.randn(4, 11),
            truncated=rewards,
            values=rewards,
            next_values=rewards,
            gamma=0.99,
            gae_lambda=0.95,
        )

    # Dimensionality mismatch
    with pytest.raises(AssertionError, match="Expected 2D rewards"):
        compute_gae(
            rewards=torch.randn(4, 10, 1),
            terminated=torch.randn(4, 10, 1),
            truncated=torch.randn(4, 10, 1),
            values=torch.randn(4, 10, 1),
            next_values=torch.randn(4, 10, 1),
            gamma=0.99,
            gae_lambda=0.95,
        )


def test_td_lambda_returns():
    """
    Test TD(lambda) returns.
    Analytical Oracle (composed from GAE oracle):
    GAE = [[-6.39, -18.0]]
    Values = [[10.0, 20.0]]
    Returns = GAE + Values = [[3.61, 2.0]]
    """
    rewards = torch.tensor([[1.0, 2.0]])
    terminals = torch.tensor([[0.0, 1.0]])
    values = torch.tensor([[10.0, 20.0]])
    last_values = torch.tensor([5.0])
    gamma = 0.9
    lam = 0.95

    next_values = torch.cat([values[:, 1:], last_values.unsqueeze(1)], dim=1)

    returns = compute_td_lambda_returns(
        rewards=rewards,
        terminated=terminals,
        truncated=torch.zeros_like(terminals),
        values=values,
        next_values=next_values,
        gamma=gamma,
        lam=lam,
    )

    expected = torch.tensor([[3.61, 2.0]])
    torch.testing.assert_close(returns, expected)
