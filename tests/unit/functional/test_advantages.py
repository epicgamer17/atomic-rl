import pytest
import torch
from functional.advantages import (
    compute_mean_advantages,
    compute_ema_advantages,
    compute_critic_advantages,
)

pytestmark = pytest.mark.unit

def test_compute_mean_advantages():
    """
    Test advantage computation with mean baseline.
    Analytical Oracle:
    returns = [1.0, 2.0, 3.0]
    mean = 2.0
    centered = [-1.0, 0.0, 1.0]
    std = 1.0
    normalized = [-1.0, 0.0, 1.0] (approx)
    """
    returns = torch.tensor([[1.0, 2.0, 3.0]]) # [1, 3]
    advantages = compute_mean_advantages(returns)
    
    assert advantages.shape == (1, 3)
    # centered: [-1, 0, 1]
    # std: sqrt(((-1)^2 + 0^2 + 1^2) / (3-1)) = sqrt(2/2) = 1.0 
    # (Note: torch.std uses Bessel's correction by default, unbiased=True)
    expected = torch.tensor([[-1.0, 0.0, 1.0]])
    torch.testing.assert_close(advantages, expected, atol=1e-6, rtol=1e-6)

    # Edge case: single step
    single_return = torch.tensor([[5.0]])
    single_advantage = compute_mean_advantages(single_return)
    assert torch.all(single_advantage == 0.0)

def test_compute_ema_advantages():
    """
    Test advantage computation with EMA baseline.
    Analytical Oracle:
    returns = [2.0, 3.0]
    ema_baseline = 1.0
    centered = [1.0, 2.0] (only subtracted ema_baseline)
    std = 0.7071
    normalized = [1.4142, 2.8284]
    """
    returns = torch.tensor([[2.0, 3.0]])
    ema_baseline = 1.0
    advantages = compute_ema_advantages(returns, ema_baseline)
    
    assert advantages.shape == (1, 2)
    diff = returns - ema_baseline
    expected = diff / (diff.std() + 1e-8)
    torch.testing.assert_close(advantages, expected)

    # Edge case: single step
    single_return = torch.tensor([[5.0]])
    single_advantage = compute_ema_advantages(single_return, 1.0)
    assert torch.all(single_advantage == 0.0)

def test_compute_critic_advantages():
    """
    Test advantage computation with Critic baseline.
    Analytical Oracle:
    returns = [1.0, 2.0]
    values = [0.5, 1.5]
    raw_adv = [0.5, 0.5]
    mean = 0.5
    centered = [0.0, 0.0]
    """
    returns = torch.tensor([[1.0, 2.0]])
    values = torch.tensor([[0.5, 1.5]])
    advantages = compute_critic_advantages(returns, values)
    
    assert advantages.shape == (1, 2)
    # raw advantages are both 0.5, so mean centering makes them 0.0
    assert torch.all(advantages == 0.0)
    
    # Another case with variance
    returns_v = torch.tensor([[1.0, 3.0]])
    values_v = torch.tensor([[0.0, 0.0]]) # zero baseline
    advantages_v = compute_critic_advantages(returns_v, values_v)
    # centered: [1-2, 3-2] = [-1, 1]
    # std = sqrt((-1^2 + 1^2)/1) = 1.4142
    # norm = [-0.7071, 0.7071]
    expected_v = torch.tensor([[-0.70710678, 0.70710678]])
    torch.testing.assert_close(advantages_v, expected_v)

    # Verify detach
    v_with_grad = torch.tensor([[0.5, 1.5]], requires_grad=True)
    advantages_grad = compute_critic_advantages(returns, v_with_grad)
    assert advantages_grad.requires_grad == False # values.detach() should break graph
