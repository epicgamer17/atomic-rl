import pytest
import torch
from functional.returns import compute_mc_returns, compute_n_step_returns

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
