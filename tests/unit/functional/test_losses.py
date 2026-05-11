import pytest
import torch
import torch.nn as nn
from functional.losses import (
    mse_loss,
    huber_loss,
    cross_entropy_loss,
    with_per_weights,
    policy_gradient_loss,
    entropy_loss,
    compute_q_td_loss,
    probability_ratio,
    clipped_surrogate_loss,
    clipped_value_loss,
)
import math

pytestmark = pytest.mark.unit


def test_mse_loss():
    predictions = torch.tensor([1.0, 2.0, 3.0])
    targets = torch.tensor([1.5, 2.0, 2.5])

    raw_losses, info = mse_loss(predictions, targets)

    expected_losses = torch.tensor([0.25, 0.0, 0.25])
    torch.testing.assert_close(raw_losses, expected_losses)
    torch.testing.assert_close(info["priorities"], torch.tensor([0.5, 0.0, 0.5]))
    assert math.isclose(info["loss/mse"], 0.5 / 3.0, rel_tol=1e-5)


def test_huber_loss():
    predictions = torch.tensor([1.0, 5.0])
    targets = torch.tensor([1.1, 1.0])
    delta = 1.0

    raw_losses, info = huber_loss(predictions, targets, delta=delta)

    expected_losses = torch.tensor([0.005, 3.5])
    torch.testing.assert_close(raw_losses, expected_losses)
    torch.testing.assert_close(info["priorities"], torch.tensor([0.1, 4.0]))


def test_cross_entropy_loss():
    predictions = torch.tensor([[10.0, 0.0, 0.0]])
    targets = torch.tensor([[1.0, 0.0, 0.0]])

    raw_losses, info = cross_entropy_loss(predictions, targets)

    assert raw_losses.item() < 0.01
    assert "priorities" in info
    assert "predictions" in info


def test_with_per_weights():
    def base_loss(p, t):
        return (p - t) ** 2, {"base": 1.0}

    is_weights = torch.tensor([0.5, 2.0])
    weighted_loss_fn = with_per_weights(base_loss, is_weights)

    predictions = torch.tensor([1.0, 2.0])
    targets = torch.tensor([2.0, 3.0])

    loss, info = weighted_loss_fn(predictions, targets)

    assert loss.item() == 1.25
    assert info["loss/weighted"] == 1.25


def test_policy_gradient_loss():
    advantages = torch.tensor([1.0, -1.0])
    log_probs = torch.tensor([-0.5, -2.0])

    raw_loss, info = policy_gradient_loss(advantages, log_probs)

    torch.testing.assert_close(raw_loss, torch.tensor([0.5, -2.0]))
    assert info["loss/policy_gradient"] == (0.5 - 2.0) / 2.0


def test_entropy_loss():
    # Test Discrete (Categorical)
    logits = torch.tensor([[0.0, 0.0]])
    dist = torch.distributions.Categorical(logits=logits)
    loss, info = entropy_loss(dist)
    expected_entropy = math.log(2.0)
    assert math.isclose(loss.item(), expected_entropy, rel_tol=1e-5)
    assert math.isclose(info["loss/entropy"].item(), expected_entropy, rel_tol=1e-5)

    logits = torch.tensor([[100.0, 0.0]])
    dist = torch.distributions.Categorical(logits=logits)
    loss, info = entropy_loss(dist)
    assert loss.item() < 1e-5

    # Test Continuous (Normal)
    mu = torch.tensor([[0.0, 0.0]])
    std = torch.tensor([[1.0, 1.0]])
    dist = torch.distributions.Normal(mu, std)
    loss, info = entropy_loss(dist)
    # Entropy of Normal(0, 1) is 0.5 * log(2 * pi * e) approx 1.4189
    expected_entropy = (
        0.5 * math.log(2 * math.pi * math.e) * 2
    )  # sum across 2 dimensions
    assert math.isclose(loss.item(), expected_entropy, rel_tol=1e-5)


def test_compute_q_td_loss():
    class MockModel(nn.Module):
        def forward(self, x):
            return torch.tensor([[1.0, 2.0]])

    model = MockModel()
    batch = {
        "obs": torch.zeros(1, 4),
        "action": torch.tensor([1]),  # [B]
        "next_obs": torch.zeros(1, 4),
        "reward": torch.tensor([0.5]),  # [B]
        "terminated": torch.tensor([0.0]),  # [B]
        "truncated": torch.tensor([0.0]),
    }

    def target_calculator(next_preds, next_actions, reward, terminated, truncated):
        # target = reward + gamma * next_q
        # reward is [B, 1], terminated is [B, 1], next_actions is [B, 1]
        return reward + 0.9 * next_preds[0, next_actions.squeeze(-1)]

    loss, info = compute_q_td_loss(
        model=model,
        batch=batch,
        selector_model=model,
        target_calculator_fn=target_calculator,
    )

    # pred_sa = 2.0
    # target = 0.5 + 0.9 * 2.0 = 2.3
    # MSE loss = (2.0 - 2.3)^2 = 0.09
    assert math.isclose(loss.mean().item(), 0.09, rel_tol=1e-5)
    assert info["q_values/mean"] == 2.0
    assert math.isclose(info["td_targets/mean"], 2.3, rel_tol=1e-5)


def test_compute_q_td_loss_with_eval_model():
    class MockModel(nn.Module):
        def __init__(self, val):
            super().__init__()
            self.val = val

        def forward(self, x):
            return torch.tensor([[self.val, self.val]])

    selector_model = MockModel(1.0)
    eval_model = MockModel(2.0)
    batch = {
        "obs": torch.zeros(1, 4),
        "action": torch.tensor([0]),
        "next_obs": torch.zeros(1, 4),
        "reward": torch.tensor([0.5]),
        "terminated": torch.tensor([0.0]),
        "truncated": torch.tensor([0.0]),
    }

    def target_calculator(next_preds, next_actions, reward, terminated, truncated):
        return reward + 0.9 * next_preds[0, next_actions.squeeze(-1)]

    loss, info = compute_q_td_loss(
        model=selector_model,
        batch=batch,
        selector_model=selector_model,
        eval_model=eval_model,
        target_calculator_fn=target_calculator,
    )

    assert math.isclose(info["td_targets/mean"], 2.3, rel_tol=1e-5)


def test_probability_ratio():
    """Test the probability ratio calculation."""
    old_log_probs = torch.tensor([-0.5, -1.0, -2.0])
    new_log_probs = torch.tensor([-0.4, -1.0, -2.5])

    ratio = probability_ratio(old_log_probs, new_log_probs)

    # ratio = exp(new - old)
    expected_ratio = torch.exp(new_log_probs - old_log_probs)
    torch.testing.assert_close(ratio, expected_ratio)
    
    # Test shape mismatch
    with pytest.raises(AssertionError, match="Shape mismatch"):
        probability_ratio(old_log_probs, new_log_probs[:2])


def test_clipped_surrogate_loss():
    """Test the PPO clipped surrogate loss."""
    ratio = torch.tensor([0.9, 1.1, 1.3, 0.7])
    advantages = torch.tensor([1.0, 1.0, -1.0, -1.0])
    clip_coef = 0.2

    # Case 1: ratio=0.9, adv=1.0 (positive adv, ratio < 1) -> unclipped 0.9, clipped 0.9 -> objective 0.9
    # Case 2: ratio=1.1, adv=1.0 (positive adv, 1 < ratio < 1.2) -> unclipped 1.1, clipped 1.1 -> objective 1.1
    # Case 3: ratio=1.3, adv=-1.0 (negative adv, ratio > 1.2) -> unclipped -1.3, clipped 1.2*-1 = -1.2 -> objective min(-1.3, -1.2) = -1.3
    # Case 4: ratio=0.7, adv=-1.0 (negative adv, ratio < 0.8) -> unclipped -0.7, clipped 0.8*-1 = -0.8 -> objective min(-0.7, -0.8) = -0.8

    loss, info = clipped_surrogate_loss(ratio, advantages, clip_coef)

    expected_loss = torch.tensor([-0.9, -1.1, 1.3, 0.8])
    torch.testing.assert_close(loss, expected_loss)
    
    assert "loss/policy" in info
    assert "objective/unclipped" in info
    assert "objective/clipped" in info
    
    # Test shape mismatch
    with pytest.raises(AssertionError, match="Shape mismatch"):
        clipped_surrogate_loss(ratio, advantages[:2], clip_coef)


def test_clipped_value_loss():
    """Test the PPO clipped value loss."""
    new_values = torch.tensor([1.0, 2.0, 3.0, 4.0])
    old_values = torch.tensor([1.1, 1.9, 2.5, 4.5])
    returns = torch.tensor([1.5, 1.5, 1.5, 1.5])
    clip_coef = 0.2

    # Case 1: new=1.0, old=1.1, return=1.5
    # unclipped_loss = (1.0 - 1.5)^2 = 0.25
    # clipped_v = 1.1 + clamp(1.0-1.1, -0.2, 0.2) = 1.1 - 0.1 = 1.0
    # clipped_loss = (1.0 - 1.5)^2 = 0.25
    # max(0.25, 0.25) * 0.5 = 0.125

    # Case 2: new=2.0, old=1.9, return=1.5
    # unclipped_loss = (2.0 - 1.5)^2 = 0.25
    # clipped_v = 1.9 + clamp(2.0-1.9, -0.2, 0.2) = 1.9 + 0.1 = 2.0
    # clipped_loss = (2.0 - 1.5)^2 = 0.25
    # max(0.25, 0.25) * 0.5 = 0.125

    # Case 3: new=3.0, old=2.5, return=1.5
    # unclipped_loss = (3.0 - 1.5)^2 = 2.25
    # clipped_v = 2.5 + clamp(3.0-2.5, -0.2, 0.2) = 2.5 + 0.2 = 2.7
    # clipped_loss = (2.7 - 1.5)^2 = 1.44
    # max(2.25, 1.44) * 0.5 = 1.125

    # Case 4: new=4.0, old=4.5, return=1.5
    # unclipped_loss = (4.0 - 1.5)^2 = 6.25
    # clipped_v = 4.5 + clamp(4.0-4.5, -0.2, 0.2) = 4.5 - 0.2 = 4.3
    # clipped_loss = (4.3 - 1.5)^2 = 7.84
    # max(6.25, 7.84) * 0.5 = 3.92

    loss, info = clipped_value_loss(new_values, old_values, returns, clip_coef)

    expected_loss = torch.tensor([0.125, 0.125, 1.125, 3.92])
    torch.testing.assert_close(loss, expected_loss)

    assert "loss/value" in info
    assert "value/unclipped_loss" in info
    assert "value/clipped_loss" in info

    # Test shape mismatch
    with pytest.raises(AssertionError, match="Shape mismatch"):
        clipped_value_loss(new_values, old_values[:2], returns, clip_coef)
