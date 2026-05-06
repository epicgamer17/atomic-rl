import pytest
import torch
import torch.nn as nn
from functional.losses import (
    mse_loss,
    huber_loss,
    cross_entropy_loss,
    with_per_weights,
    policy_gradient_loss,
    bellman_error,
)
import math  # Add math for isclose

pytestmark = pytest.mark.unit


def test_mse_loss():
    predictions = torch.tensor([1.0, 2.0, 3.0])
    targets = torch.tensor([1.5, 2.0, 2.5])

    raw_losses, info = mse_loss(predictions, targets)

    # (1-1.5)^2 = 0.25, (2-2)^2 = 0, (3-2.5)^2 = 0.25
    expected_losses = torch.tensor([0.25, 0.0, 0.25])
    torch.testing.assert_close(raw_losses, expected_losses)
    # Priorities = abs(diff) = [0.5, 0.0, 0.5]
    torch.testing.assert_close(info["priorities"], torch.tensor([0.5, 0.0, 0.5]))
    assert math.isclose(info["loss/mse"], 0.5 / 3.0, rel_tol=1e-5)


def test_huber_loss():
    predictions = torch.tensor([1.0, 5.0])
    targets = torch.tensor([1.1, 1.0])  # diffs: 0.1 (small), 4.0 (large)
    delta = 1.0

    raw_losses, info = huber_loss(predictions, targets, delta=delta)

    # Small diff: 0.5 * 0.1^2 = 0.005
    # Large diff: delta * (abs_diff - 0.5 * delta) = 1.0 * (4.0 - 0.5) = 3.5
    expected_losses = torch.tensor([0.005, 3.5])
    torch.testing.assert_close(raw_losses, expected_losses)
    torch.testing.assert_close(info["priorities"], torch.tensor([0.1, 4.0]))


def test_cross_entropy_loss():
    # [Batch=1, Classes=3]
    predictions = torch.tensor([[10.0, 0.0, 0.0]])  # High prob on class 0
    targets = torch.tensor([[1.0, 0.0, 0.0]])  # Target is class 0

    raw_losses, info = cross_entropy_loss(predictions, targets)

    # Loss should be very small
    assert raw_losses.item() < 0.01
    assert "priorities" in info
    assert "predictions" in info


def test_with_per_weights():
    def base_loss(p, t):
        return (p - t) ** 2, {"base": 1.0}

    is_weights = torch.tensor([0.5, 2.0])
    weighted_loss_fn = with_per_weights(base_loss, is_weights)

    predictions = torch.tensor([1.0, 2.0])
    targets = torch.tensor([2.0, 3.0])  # diffs are 1.0, 1.0

    loss, info = weighted_loss_fn(predictions, targets)

    # raw_losses = [1.0, 1.0]
    # weighted = [1.0*0.5, 1.0*2.0] = [0.5, 2.0]
    # mean = (0.5 + 2.0) / 2 = 1.25
    assert loss.item() == 1.25
    assert info["loss/weighted"] == 1.25


def test_policy_gradient_loss():
    advantages = torch.tensor([1.0, -1.0])
    log_probs = torch.tensor([-0.5, -2.0])  # probs: exp(-0.5), exp(-2.0)

    # PG Loss = -log_prob * advantage
    # [-(-0.5)*1.0, -(-2.0)*(-1.0)] = [0.5, -2.0]
    raw_loss, info = policy_gradient_loss(advantages, log_probs)

    torch.testing.assert_close(raw_loss, torch.tensor([0.5, -2.0]))
    assert info["loss/policy_gradient"] == (0.5 - 2.0) / 2.0


def test_bellman_error():
    # Setup mocks
    class MockModel(nn.Module):
        def forward(self, x):
            # return 1.0 for action 0, 2.0 for action 1
            return torch.tensor([[1.0, 2.0]])

    model = MockModel()
    batch = {
        "obs": torch.zeros(1, 4),
        "action": torch.tensor([[1]]),  # predicted Q should be 2.0
        "next_obs": torch.zeros(1, 4),
        "reward": torch.tensor([[0.5]]),
        "terminated": torch.tensor([[False]]),
    }

    def target_calculator(next_preds, next_actions, reward, terminated):
        # target = reward + gamma * next_q
        # next_q is from next_preds[next_actions]
        # next_actions will be 1 (argmax of [1, 2])
        # next_q will be 2.0
        return reward + 0.9 * next_preds[0, next_actions]

    loss, info = bellman_error(
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


def test_bellman_error_with_eval_model():
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
        "action": torch.tensor([[0]]),
        "next_obs": torch.zeros(1, 4),
        "reward": torch.tensor([[0.5]]),
        "terminated": torch.tensor([[False]]),
    }

    def target_calculator(next_preds, next_actions, reward, terminated):
        # next_preds will be from eval_model (2.0)
        return reward + 0.9 * next_preds[0, next_actions]

    loss, info = bellman_error(
        model=selector_model,
        batch=batch,
        selector_model=selector_model,
        eval_model=eval_model,
        target_calculator_fn=target_calculator,
    )

    # target = 0.5 + 0.9 * 2.0 = 2.3
    assert math.isclose(info["td_targets/mean"], 2.3, rel_tol=1e-5)
