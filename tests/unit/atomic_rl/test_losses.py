import pytest
import torch
import torch.nn as nn
from atomic_rl.losses import (
    mse_loss,
    huber_loss,
    cross_entropy_loss,
    with_per_weights,
    policy_gradient_loss,
    entropy_loss,
    probability_ratio,
    clipped_surrogate_loss,
    clipped_mse_loss,
    with_sequence_mask,
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


# ==========================================
# Tests for with_sequence_mask (New Code Coverage)
# ==========================================


def test_with_sequence_mask_standard():
    """Verify that the loss is correctly zeroed out for masked transitions and the mean is un-diluted."""

    def dummy_base_loss(p, t):
        return torch.abs(p - t), {"base_metric": torch.tensor(0.0)}

    # Pass flat 1D sequence tensors [B * T] as the function expects
    predictions = torch.tensor([1.0, 2.0, 3.0, 4.0])
    targets = torch.tensor([2.0, 4.0, 6.0, 8.0])
    mask = torch.tensor([1, 0, 1, 1])  # 3 valid steps, 1 masked step

    masked_loss_fn = with_sequence_mask(dummy_base_loss, mask)
    masked_losses, info = masked_loss_fn(predictions, targets)

    # Raw errors are: [1.0, 2.0, 3.0, 4.0]
    # Masked errors should be: [1.0, 0.0, 3.0, 4.0]
    expected_losses = torch.tensor([1.0, 0.0, 3.0, 4.0])
    torch.testing.assert_close(masked_losses, expected_losses)

    # Valid count is 3. Sum of valid losses = 1.0 + 3.0 + 4.0 = 8.0
    # Expected masked mean = 8.0 / 3.0 = 2.66666...
    expected_mean = 8.0 / 3.0
    torch.testing.assert_close(info["loss/masked_mean"], torch.tensor(expected_mean))


def test_with_sequence_mask_all_masked():
    """Verify that if every transition is masked out, clamp(min=1.0) prevents a division-by-zero NaN."""

    def dummy_base_loss(p, t):
        return torch.abs(p - t), {}

    predictions = torch.tensor([1.0, 2.0])
    targets = torch.tensor([2.0, 4.0])
    mask = torch.tensor([0, 0])  # All elements masked out

    masked_loss_fn = with_sequence_mask(dummy_base_loss, mask)
    masked_losses, info = masked_loss_fn(predictions, targets)

    # All outputs should be perfectly zeroed out
    torch.testing.assert_close(masked_losses, torch.tensor([0.0, 0.0]))

    # Sum is 0.0, Valid count is clamped to 1.0 -> mean must be 0.0, NOT NaN
    torch.testing.assert_close(info["loss/masked_mean"], torch.tensor(0.0))


def test_with_sequence_mask_shape_assertions():
    """Verify that with_sequence_mask raises an AssertionError on unflattened or mismatched shapes."""

    def dummy_base_loss(p, t):
        return torch.abs(p - t), {}

    # Mask has 4 elements total
    mask = torch.tensor([1, 0, 1, 1])

    # Case 1: Passing unflattened 2D tensors [2, 2] instead of [4]
    predictions_2d = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    targets_2d = torch.tensor([[2.0, 4.0], [6.0, 8.0]])

    masked_loss_fn = with_sequence_mask(dummy_base_loss, mask)

    with pytest.raises(AssertionError, match="Ensure inputs are flattened"):
        masked_loss_fn(predictions_2d, targets_2d)

    # Case 2: Passing a flat tensor but with the entirely wrong element count (e.g., 3 instead of 4)
    predictions_short = torch.tensor([1.0, 2.0, 3.0])
    targets_short = torch.tensor([2.0, 4.0, 6.0])

    with pytest.raises(AssertionError, match="must match mask total elements"):
        masked_loss_fn(predictions_short, targets_short)
