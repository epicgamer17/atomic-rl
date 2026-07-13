import pytest
import torch
import torch.nn.functional as F
import math
from functional.action_selection import (
    expected_value,
    argmax_selector,
    sample_distribution,
    with_epsilon_greedy,
)
from functional.schedules import get_linear_schedule, get_exponential_schedule

pytestmark = pytest.mark.unit


def test_expected_value():
    """
    Test the expected_value function.
    Analytical Oracle:
    predictions = [[0, 100]] (logits) -> probs = [0, 1]
    support = [[-1, 1]]
    expected_value = 0*-1 + 1*1 = 1
    """
    # Batch size 1, 2 actions, 2 support atoms
    predictions = torch.tensor([[[0.0, 100.0], [100.0, 0.0]]])  # [1, 2, 2]
    support = torch.tensor([[[-1.0, 1.0], [-1.0, 1.0]]])  # [1, 2, 2]

    # Expected: First action should have value ~1.0, second action ~-1.0
    values = expected_value(predictions, support)

    assert values.shape == (1, 2), f"Expected shape (1, 2), got {values.shape}"
    torch.testing.assert_close(values, torch.tensor([[1.0, -1.0]]))

    # Test with 1D support
    support_1d = torch.tensor([-1.0, 1.0])
    values_1d = expected_value(predictions, support_1d)
    torch.testing.assert_close(values_1d, torch.tensor([[1.0, -1.0]]))


def test_argmax_selector():
    """Test argmax action selection."""
    predictions = torch.tensor([[0.1, 0.9, 0.2], [0.8, 0.1, 0.1]])  # [2, 3]

    # Test without extractor_fn
    actions, info = argmax_selector(predictions)
    assert actions.shape == (2, 1)
    assert actions[0, 0] == 1
    assert actions[1, 0] == 0
    assert isinstance(info, dict)

    # Test with extracted values
    def dummy_extractor(x):
        return x * -1  # flip signs

    actions_flipped, _ = argmax_selector(dummy_extractor(predictions))
    assert actions_flipped[0, 0] == 0
    assert actions_flipped[1, 0] == 1


def test_sample_distribution_categorical():
    """Test sample_distribution with Categorical."""
    torch.manual_seed(42)  # Local seed for determinism in sampling

    # [1, 3] logits
    predictions = torch.tensor([[0.0, 10.0, 0.0]])
    dist = torch.distributions.Categorical(logits=predictions)

    # explore=False
    action, info = sample_distribution(dist, explore=False)
    assert action.item() == 1
    # Note: deterministic mode doesn't zero log_prob by default, it computes the actual log_prob of the argmax
    assert info["log_prob"].shape == (1, 1)

    # explore=True
    action, info = sample_distribution(dist, explore=True)
    log_prob = info["log_prob"]
    assert action.shape == (1, 1)
    assert log_prob.shape == (1, 1)
    # With logits [0, 10, 0], action 1 is extremely likely
    assert action.item() == 1

    expected_log_prob = F.log_softmax(predictions, dim=-1)[0, 1]
    torch.testing.assert_close(log_prob[0], expected_log_prob.view(1))


def test_sample_distribution_gaussian():
    """Test sample_distribution with Gaussian."""
    torch.manual_seed(42)

    mean = torch.tensor([[10.0, -10.0]])
    std = torch.tensor([[0.1, 0.1]])
    dist = torch.distributions.Normal(mean, std)

    # Test explore=False
    action, info = sample_distribution(dist, explore=False)
    log_prob = info["log_prob"]
    torch.testing.assert_close(action, mean)

    # Test explore=True
    action, info = sample_distribution(dist, explore=True)
    log_prob = info["log_prob"]
    assert action.shape == (1, 2)
    assert log_prob.shape == (1, 2)  # Log prob is independent, so [1, 2]

    # Manual check for log_prob
    expected_log_prob = dist.log_prob(action)
    torch.testing.assert_close(log_prob, expected_log_prob)

    # Test multi-dimensional continuous action space
    mean_multi = torch.randn(2, 3)  # [Batch 2, Actions 3]
    std_multi = torch.ones(2, 3) * 0.1
    dist_multi = torch.distributions.Normal(mean_multi, std_multi)
    action_multi, info_multi = sample_distribution(dist_multi)
    log_prob_multi = info_multi["log_prob"]
    assert action_multi.shape == (2, 3)
    assert log_prob_multi.shape == (2, 3)


def test_with_epsilon_greedy():
    """Test epsilon greedy higher-order function."""
    greedy_selector = lambda x: (
        torch.argmax(x, dim=1, keepdim=True),
        {"test_info": True},
    )
    epsilon_selector = with_epsilon_greedy(greedy_selector)

    predictions = torch.tensor([[1.0, 0.0], [1.0, 0.0]])  # Greedy action is 0

    # Epsilon 0.0 -> always greedy
    actions, info = epsilon_selector(predictions, epsilon=0.0, num_actions=2)
    assert torch.all(actions == 0)
    assert info["test_info"] is True

    # Epsilon 1.0 -> always random
    # Use a fixed generator for reproducibility
    gen = torch.Generator()
    gen.manual_seed(42)
    actions, _ = epsilon_selector(
        predictions, epsilon=1.0, num_actions=2, generator=gen
    )
    # With seed 42, we expect some random actions
    assert actions.shape == (2, 1)


def test_linear_schedule():
    """Test linear schedule decay."""
    # start 1.0, end 0.1, decay_steps 10
    assert math.isclose(get_linear_schedule(0, 1.0, 0.1, 10), 1.0)
    assert math.isclose(
        get_linear_schedule(5, 1.0, 0.1, 10), 0.55
    )  # 1.0 + 0.5 * (-0.9)
    assert math.isclose(get_linear_schedule(10, 1.0, 0.1, 10), 0.1)
    assert math.isclose(
        get_linear_schedule(20, 1.0, 0.1, 10), 0.1
    )  # Capped at 1.0 fraction


def test_exponential_schedule():
    """Test exponential schedule decay."""
    # start 1.0, end 0.1, decay_rate 10
    # val = end + (start - end) * exp(-step/rate)
    assert math.isclose(get_exponential_schedule(0, 1.0, 0.1, 10), 1.0)
    expected_middle = 0.1 + 0.9 * math.exp(-5 / 10)
    assert math.isclose(get_exponential_schedule(5, 1.0, 0.1, 10), expected_middle)


def test_action_selection_assertions():
    """Test that action selection functions raise assertions on invalid input shapes."""
    # expected_value
    with pytest.raises(
        AssertionError, match="Expected 2D \[B, N\] or 3D \[B, A, N\] predictions"
    ):
        expected_value(torch.randn(2), torch.randn(2))
    with pytest.raises(
        AssertionError,
        match="If support is not 1D, it must match predictions shape exactly.",
    ):
        expected_value(torch.randn(1, 2, 3), torch.randn(2, 3))


def test_sample_distribution_not_implemented():
    class DummyDist:
        pass

    dist = DummyDist()
    with pytest.raises(NotImplementedError, match="Deterministic selection"):
        sample_distribution(dist, explore=False)


def test_apply_action_mask():
    from functional.action_selection import apply_action_mask

    logits = torch.tensor([[1.0, 2.0, 3.0]])
    mask = torch.tensor([[1, 0, 1]])
    masked = apply_action_mask(logits, mask)
    expected = torch.tensor([[1.0, -1e8, 3.0]])
    torch.testing.assert_close(masked, expected)


def test_compute_masked_entropy():
    from functional.action_selection import compute_masked_entropy

    logits = torch.tensor([[-0.6931, -1e8, -0.6931]])
    probs = torch.tensor([[0.5, 0.0, 0.5]])
    mask = torch.tensor([[1, 0, 1]])
    entropy = compute_masked_entropy(logits, probs, mask)
    # entropy = -(0.5 * -0.6931 + 0 + 0.5 * -0.6931) = 0.6931
    expected = torch.tensor([0.6931])
    torch.testing.assert_close(entropy, expected)
