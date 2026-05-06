import pytest
import torch
from functional.returns import compute_mc_returns

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
    terminals = torch.tensor([[0.0, 0.0, 1.0]])  # Last step is terminal
    gamma = 0.9
    expected = torch.tensor([[2.71, 1.9, 1.0]])
    
    returns = compute_mc_returns(rewards, terminals, gamma)
    torch.testing.assert_close(returns, expected)


def test_compute_mc_returns_batched():
    """Test batched return calculation."""
    rewards = torch.tensor([
        [1.0, 1.0, 1.0],
        [2.0, 2.0, 2.0]
    ])
    terminals = torch.tensor([
        [0.0, 0.0, 1.0],
        [0.0, 0.0, 1.0]
    ])
    gamma = 0.9
    # Batch 0: same as above
    # Batch 1:
    # G_2 = 2.0
    # G_1 = 2.0 + 0.9*2.0 = 2.0 + 1.8 = 3.8
    # G_0 = 2.0 + 0.9*3.8 = 2.0 + 3.42 = 5.42
    expected = torch.tensor([
        [2.71, 1.9, 1.0],
        [5.42, 3.8, 2.0]
    ])
    
    returns = compute_mc_returns(rewards, terminals, gamma)
    torch.testing.assert_close(returns, expected)


def test_compute_mc_returns_with_terminated():
    """Test return calculation with episode boundaries."""
    rewards = torch.tensor([[1.0, 1.0, 1.0, 1.0]])
    terminals = torch.tensor([[0.0, 1.0, 0.0, 1.0]])
    gamma = 0.9
    
    # Episode 1 ends at t=1
    # G_1 = reward[1] = 1.0
    # G_0 = reward[0] + gamma * G_1 = 1.0 + 0.9 * 1.0 = 1.9
    # Episode 2 starts at t=2
    # G_3 = reward[3] = 1.0
    # G_2 = reward[2] + gamma * G_3 = 1.0 + 0.9 * 1.0 = 1.9
    
    expected = torch.tensor([[1.9, 1.0, 1.9, 1.0]])
    
    returns = compute_mc_returns(rewards, terminals, gamma)
    torch.testing.assert_close(returns, expected)


def test_compute_mc_returns_never_terminal_warning():
    """Test that a warning is issued if terminals is always False."""
    rewards = torch.tensor([[1.0, 1.0]])
    terminals = torch.tensor([[0.0, 0.0]])
    gamma = 0.9
    
    with pytest.warns(UserWarning, match="Found trajectory in batch where terminals is always False"):
        compute_mc_returns(rewards, terminals, gamma)
