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
    clipped_mse_loss,
    compute_v_td_loss,
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
    assert math.isclose(loss.item(), -expected_entropy, rel_tol=1e-5)
    assert math.isclose(info["loss/entropy"].item(), -expected_entropy, rel_tol=1e-5)

    logits = torch.tensor([[100.0, 0.0]])
    dist = torch.distributions.Categorical(logits=logits)
    loss, info = entropy_loss(dist)
    assert abs(loss.item()) < 1e-5

    # Test Continuous (Normal)
    mu = torch.tensor([[0.0, 0.0]])
    std = torch.tensor([[1.0, 1.0]])
    dist = torch.distributions.Normal(mu, std)
    dist = torch.distributions.Independent(dist, 1)
    loss, info = entropy_loss(dist)
    # Entropy of Normal(0, 1) is 0.5 * log(2 * pi * e) approx 1.4189
    expected_entropy = (
        0.5 * math.log(2 * math.pi * math.e) * 2
    )  # sum across 2 dimensions
    assert math.isclose(loss.item(), -expected_entropy, rel_tol=1e-5)


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
        "gamma": torch.tensor([0.9]),
    }

    def target_calculator(next_preds, next_actions, reward, terminated, gamma):
        # target = reward + gamma * next_q
        return reward + gamma * next_preds[0, next_actions.squeeze(-1)]

    loss, info = compute_q_td_loss(
        model=model,
        batch=batch,
        target_model=model,
        next_action_selector_fn=lambda obs, preds: torch.tensor([1]),
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

    online_model = MockModel(1.0)
    target_model = MockModel(2.0)
    batch = {
        "obs": torch.zeros(1, 4),
        "action": torch.tensor([0]),
        "next_obs": torch.zeros(1, 4),
        "reward": torch.tensor([0.5]),
        "terminated": torch.tensor([0.0]),
        "gamma": torch.tensor([0.9]),
    }

    def target_calculator(next_preds, next_actions, reward, terminated, gamma):
        return reward + gamma * next_preds[0, next_actions.squeeze(-1)]

    loss, info = compute_q_td_loss(
        model=online_model,
        batch=batch,
        target_model=target_model,
        next_action_selector_fn=lambda obs, preds: torch.tensor([0]),
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
    assert "policy/approx_kl" in info
    assert "policy/clip_fraction" in info
    assert "objective/unclipped" in info
    assert "objective/clipped" in info
    
    # Test shape mismatch
    with pytest.raises(AssertionError, match="Shape mismatch"):
        clipped_surrogate_loss(ratio, advantages[:2], clip_coef)


def test_clipped_mse_loss():
    """Test the PPO clipped MSE loss."""
    predictions = torch.tensor([1.0, 2.0, 3.0, 4.0])
    old_predictions = torch.tensor([1.1, 1.9, 2.5, 4.5])
    targets = torch.tensor([1.5, 1.5, 1.5, 1.5])
    clip_coef = 0.2

    # Case 1: pred=1.0, old=1.1, target=1.5
    # unclipped_loss = (1.0 - 1.5)^2 = 0.25
    # clipped_v = 1.1 + clamp(1.0-1.1, -0.2, 0.2) = 1.1 - 0.1 = 1.0
    # clipped_loss = (1.0 - 1.5)^2 = 0.25
    # max(0.25, 0.25) * 0.5 = 0.125

    # Case 2: pred=2.0, old=1.9, target=1.5
    # unclipped_loss = (2.0 - 1.5)^2 = 0.25
    # clipped_v = 1.9 + clamp(2.0-1.9, -0.2, 0.2) = 1.9 + 0.1 = 2.0
    # clipped_loss = (2.0 - 1.5)^2 = 0.25
    # max(0.25, 0.25) * 0.5 = 0.125

    # Case 3: pred=3.0, old=2.5, target=1.5
    # unclipped_loss = (3.0 - 1.5)^2 = 2.25
    # clipped_v = 2.5 + clamp(3.0-2.5, -0.2, 0.2) = 2.5 + 0.2 = 2.7
    # clipped_loss = (2.7 - 1.5)^2 = 1.44
    # max(2.25, 1.44) * 0.5 = 1.125

    # Case 4: pred=4.0, old=4.5, target=1.5
    # unclipped_loss = (4.0 - 1.5)^2 = 6.25
    # clipped_v = 4.5 + clamp(4.0-4.5, -0.2, 0.2) = 4.5 - 0.2 = 4.3
    # clipped_loss = (4.3 - 1.5)^2 = 7.84
    # max(6.25, 7.84) * 0.5 = 3.92

    loss, info = clipped_mse_loss(predictions, targets, old_predictions, clip_coef)

    expected_loss = torch.tensor([0.125, 0.125, 1.125, 3.92])
    torch.testing.assert_close(loss, expected_loss)

    assert "loss/value" in info
    assert "value/unclipped_loss" in info
    assert "value/clipped_loss" in info

    # Test shape mismatch
    with pytest.raises(AssertionError, match="Shape mismatch"):
        clipped_mse_loss(predictions, targets[:2], old_predictions, clip_coef)


def test_compute_v_td_loss():
    from functional.losses import compute_v_td_loss
    from tensordict import TensorDict
    
    class MockModel(nn.Module):
        def forward(self, x):
            return x.sum(dim=-1, keepdim=True) # V(s) = sum(obs)
            
    model = MockModel()
    batch = TensorDict({
        "obs": torch.tensor([[1.0, 2.0], [1.0, 1.0]]),
        "next_obs": torch.tensor([[2.0, 2.0], [0.0, 0.0]]),
        "reward": torch.tensor([1.0, 0.5]),
        "terminated": torch.tensor([0.0, 1.0]),
        "gamma": torch.tensor([0.9, 0.9])
    }, batch_size=[2])
    
    # V(s) = [3.0, 2.0]
    # V(s') = [4.0, 0.0]
    # targets: 
    #   0: 1.0 + 0.9 * 4.0 * 1 = 4.6
    #   1: 0.5 + 0.9 * 0.0 * 0 = 0.5
    # MSE loss:
    #   0: (3.0 - 4.6)^2 = 1.6^2 = 2.56
    #   1: (2.0 - 0.5)^2 = 1.5^2 = 2.25
    # avg loss = (2.56 + 2.25)/2 = 2.405
    
    loss, info = compute_v_td_loss(model, batch)
    assert math.isclose(loss.mean().item(), 2.405, rel_tol=1e-4)
    assert "v_values/mean" in info
    assert "v_targets/mean" in info


def test_losses_assertions():
    with pytest.raises(AssertionError, match="Shape mismatch"):
        mse_loss(torch.randn(2), torch.randn(3))
        
    with pytest.raises(AssertionError, match="Shape mismatch"):
        cross_entropy_loss(torch.randn(2), torch.randn(3))
        
    with pytest.raises(AssertionError, match="Shape mismatch"):
        huber_loss(torch.randn(2), torch.randn(3))
        
    with pytest.raises(AssertionError, match="Shape mismatch"):
        policy_gradient_loss(torch.randn(2), torch.randn(3))
        
    dist = torch.distributions.Normal(torch.zeros(1, 2), torch.ones(1, 2))
    with pytest.raises(AssertionError, match="Expected 1D entropy"):
        entropy_loss(dist)
        
    class MockModel(nn.Module):
        def forward(self, x): return torch.tensor([[1.0, 2.0]])
    batch = {"obs": torch.zeros(1, 4), "action": torch.tensor([1]), "next_obs": torch.ones(1, 4), "reward": torch.tensor([0.5]), "terminated": torch.tensor([0.0]), "gamma": torch.tensor([0.9])}
    
    def bad_target_calculator(*args):
        return torch.tensor([1.0, 2.0]) # Returns [2] instead of [1]
        
    with pytest.raises(AssertionError, match="Shape mismatch"):
        compute_q_td_loss(MockModel(), batch, MockModel(), lambda obs, preds: torch.tensor([1]), bad_target_calculator)
        
    class BadModel(nn.Module):
        def forward(self, x):
            # Return 2D for obs, 1D for next_obs to bypass v_next shape checks
            if torch.equal(x, batch["obs"]):
                return torch.tensor([[1.0, 2.0]]) # v_pred squeezed to [1, 2]
            return torch.tensor([[1.0]]) # v_next squeezed to [1]
            
    with pytest.raises(AssertionError, match="Prediction and target shapes must match"):
        compute_v_td_loss(BadModel(), batch)
