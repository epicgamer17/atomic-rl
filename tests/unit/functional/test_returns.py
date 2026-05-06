import pytest
import torch
from functional.returns import compute_mc_returns

pytestmark = pytest.mark.unit


def test_compute_mc_returns_1d():
    """
    Test 1D return calculation.
    Analytical Oracle:
    rewards = [1, 1, 1], gamma = 0.9
    G_2 = 1.0
    G_1 = 1.0 + 0.9*1.0 = 1.9
    G_0 = 1.0 + 0.9*1.9 = 2.71
    """
    rewards = torch.tensor([1.0, 1.0, 1.0])
    gamma = 0.9
    expected = torch.tensor([2.71, 1.9, 1.0])
    
    returns = compute_mc_returns(rewards, gamma)
    torch.testing.assert_close(returns, expected)
