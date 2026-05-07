import pytest
import torch
from einops import rearrange
from functional.advantages import compute_gae

pytestmark = pytest.mark.unit

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
    
    gae = compute_gae(
        rewards=rewards,
        terminals=terminals,
        values=values,
        last_values=last_values,
        gamma=gamma,
        gae_lambda=gae_lambda
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
    
    gae = compute_gae(
        rewards=rewards,
        terminals=terminals,
        values=values,
        last_values=last_values,
        gamma=0.99,
        gae_lambda=0.95
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
    terminals = torch.tensor([[0.0, 1.0, 0.0]]) # Terminal at t=1
    values = torch.zeros((1, 3))
    last_values = torch.zeros(1)
    gamma = 1.0
    gae_lambda = 1.0
    
    # delta_t = r_t + 1.0 * (1-d_t) * V_{t+1} - V_t = r_t
    # t=2: A_2 = delta_2 + 1.0 * 1.0 * (1-0.0) * 0 = r_2 = 1.0
    # t=1: A_1 = delta_1 + 1.0 * 1.0 * (1-1.0) * A_2 = delta_1 = r_1 = 1.0
    # t=0: A_0 = delta_0 + 1.0 * 1.0 * (1-0.0) * A_1 = r_0 + A_1 = 1.0 + 1.0 = 2.0
    
    gae = compute_gae(
        rewards=rewards,
        terminals=terminals,
        values=values,
        last_values=last_values,
        gamma=gamma,
        gae_lambda=gae_lambda
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
    
    gae = compute_gae(
        rewards=rewards,
        terminals=terminals,
        values=values,
        last_values=last_values,
        gamma=gamma,
        gae_lambda=0.0
    )
    
    # Calculate deltas manually
    next_values = torch.cat([values[:, 1:], last_values.unsqueeze(1)], dim=1)
    expected_deltas = rewards + gamma * (1.0 - terminals) * next_values - values
    
    torch.testing.assert_close(gae, expected_deltas)

def test_gae_input_shapes():
    """
    Test that the function handles [B, T, 1] inputs gracefully as per the 'Bouncer' logic.
    """
    B, T = 4, 10
    rewards = torch.randn(B, T, 1)
    terminals = (torch.rand(B, T, 1) > 0.8).float()
    values = torch.randn(B, T, 1)
    last_values = torch.randn(B, 1)
    
    gae = compute_gae(
        rewards=rewards,
        terminals=terminals,
        values=values,
        last_values=last_values,
        gamma=0.99,
        gae_lambda=0.95
    )
    
    assert gae.shape == (B, T)

def test_gae_zeros():
    """
    Test with all zeros to ensure no NaNs or crashes.
    """
    B, T = 2, 5
    rewards = torch.zeros(B, T)
    terminals = torch.zeros(B, T)
    values = torch.zeros(B, T)
    last_values = torch.zeros(B)
    
    gae = compute_gae(
        rewards=rewards,
        terminals=terminals,
        values=values,
        last_values=last_values,
        gamma=0.99,
        gae_lambda=0.95
    )
    
    assert torch.all(gae == 0.0)
    assert not torch.any(torch.isnan(gae))
