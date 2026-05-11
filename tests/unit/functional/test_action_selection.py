import pytest
import torch
import torch.nn.functional as F
import math
from functional.action_selection import (
    expected_value,
    argmax_selector,
    categorical_sampling_selector,
    gaussian_sampling_selector,
    with_epsilon_greedy,
    get_ape_x_epsilon,
    multidiscrete_sampling_selector,
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

    # Test with extractor_fn
    def dummy_extractor(x):
        return x * -1  # flip signs

    actions_flipped, _ = argmax_selector(predictions, extractor_fn=dummy_extractor)
    assert actions_flipped[0, 0] == 0
    assert actions_flipped[1, 0] == 1


def test_categorical_sampling_selector():
    """Test categorical sampling selector."""
    torch.manual_seed(42)  # Local seed for determinism in sampling

    # [1, 3] logits
    predictions = torch.tensor([[0.0, 10.0, 0.0]])

    # Temperature 0 should be argmax and return 0 log_prob
    action, info = categorical_sampling_selector(predictions, temperature=0.0)
    assert action.item() == 1
    assert info["log_prob"].item() == 0.0

    # Temperature 0 with extractor_fn
    action_0_ext, _ = categorical_sampling_selector(
        predictions, extractor_fn=lambda x: x * -1.0, temperature=0.0
    )
    # flipped signs: [0, -10, 0] -> argmax is 0 or 2. argmax(0) = 0
    assert action_0_ext.item() == 0

    # Temperature 1.0
    action, info = categorical_sampling_selector(predictions, temperature=1.0)
    log_prob = info["log_prob"]
    assert action.shape == (1, 1)
    assert log_prob.shape == (1, 1)
    # With logits [0, 10, 0], action 1 is extremely likely
    assert action.item() == 1

    # Verify log_prob calculation manually
    # probs = softmax([0, 10, 0]) approx [0, 1, 0]
    # log_prob = log(probs[1]) approx 0
    expected_log_prob = F.log_softmax(predictions, dim=-1)[0, 1]
    torch.testing.assert_close(log_prob[0], expected_log_prob.view(1))

    # Test with extractor_fn
    def dummy_extractor(x):
        return x * 2.0

    action_ext, _ = categorical_sampling_selector(
        predictions, extractor_fn=dummy_extractor
    )
    assert action_ext.item() == 1

    # Test multi-discrete (e.g. 2 categorical variables)
    # Shape [Batch, Num_Vars, Num_Actions] -> [1, 2, 3]
    multi_predictions = torch.tensor([[[10.0, 0.0, 0.0], [0.0, 10.0, 0.0]]])
    action_multi, info_multi = categorical_sampling_selector(multi_predictions)
    log_prob_multi = info_multi["log_prob"]
    # The correct implementation preserves batch dimension: [Batch, Num_Vars]
    assert action_multi.shape == (1, 2)
    assert log_prob_multi.shape == (1, 1)
    assert action_multi[0, 0] == 0
    assert action_multi[0, 1] == 1


def test_gaussian_sampling_selector():
    """Test gaussian sampling selector."""
    torch.manual_seed(42)

    mean = torch.tensor([[10.0, -10.0]])
    std = torch.tensor([[0.1, 0.1]])

    # Test explore=False
    action, info = gaussian_sampling_selector(mean, std, explore=False)
    log_prob = info["log_prob"]
    torch.testing.assert_close(action, mean)
    torch.testing.assert_close(log_prob, torch.zeros_like(log_prob))

    # Test explore=True
    action, info = gaussian_sampling_selector(mean, std, explore=True)
    log_prob = info["log_prob"]
    assert action.shape == (1, 2)
    assert log_prob.shape == (1, 1)  # Summed over action dimension

    # Manual check for log_prob
    # log_prob = log_pdf(action, mean, std).sum()
    dist = torch.distributions.Normal(mean, std)
    expected_log_prob = dist.log_prob(action).sum(dim=-1, keepdim=True)
    torch.testing.assert_close(log_prob, expected_log_prob)

    # Test multi-dimensional continuous action space
    mean_multi = torch.randn(2, 3)  # [Batch 2, Actions 3]
    std_multi = torch.ones(2, 3) * 0.1
    action_multi, info_multi = gaussian_sampling_selector(mean_multi, std_multi)
    log_prob_multi = info_multi["log_prob"]
    assert action_multi.shape == (2, 3)
    assert log_prob_multi.shape == (2, 1)


def test_gaussian_sampling_selector_1d():
    mean = torch.tensor([10.0, -10.0])  # 1D
    std = torch.tensor([0.1, 0.1])

    # Covers line 157 (explore=False)
    _, info_det = gaussian_sampling_selector(mean, std, explore=False)
    assert info_det["log_prob"].shape == (2, 1)

    # Covers lines 170-171 (explore=True)
    _, info_sample = gaussian_sampling_selector(mean, std, explore=True)
    assert info_sample["log_prob"].shape == (2, 1)


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


def test_ape_x_epsilon():
    """Test Ape-X fixed epsilon calculation."""
    # If num_actors <= 1, return base_eps
    assert get_ape_x_epsilon(0, 1, base_eps=0.4) == 0.4

    # Check extremes for multiple actors
    # actor 0 should have base_eps ^ (1 + 0) = base_eps
    assert math.isclose(get_ape_x_epsilon(0, 5, base_eps=0.4), 0.4)
    # actor last should have base_eps ^ (1 + alpha)
    expected_last = 0.4 ** (1 + 7.0)
    assert math.isclose(get_ape_x_epsilon(4, 5, base_eps=0.4, alpha=7.0), expected_last)


def test_action_selection_assertions():
    """Test that action selection functions raise assertions on invalid input shapes."""
    # expected_value
    with pytest.raises(AssertionError, match="Expected 3D predictions"):
        expected_value(torch.randn(2, 2), torch.randn(2))
    with pytest.raises(
        AssertionError, match="Expected 1D \[N\] or 3D \[B, A, N\] support"
    ):
        expected_value(torch.randn(1, 2, 3), torch.randn(2, 3))

    # categorical_sampling_selector
    with pytest.raises(
        AssertionError, match="Expected predictions with at least batch and action dims"
    ):
        categorical_sampling_selector(torch.randn(3))

    # gaussian_sampling_selector
    with pytest.raises(AssertionError, match="Mean .* and Std .* must match"):
        gaussian_sampling_selector(torch.randn(2, 2), torch.randn(2, 3))
    with pytest.raises(AssertionError, match="Expected at least 1D tensors"):
        gaussian_sampling_selector(torch.tensor(0.0), torch.tensor(1.0))


def test_multidiscrete_sampling_selector():
    """Test MultiDiscrete action sampling selector."""
    torch.manual_seed(42)

    # nvec = (3, 2)
    # logits shape [Batch=1, sum(nvec)=5]
    nvec = (3, 2)
    # High logit for index 1 in first component, index 0 in second component
    logits = torch.tensor([[0.0, 10.0, 0.0, 10.0, 0.0]])

    actions, info = multidiscrete_sampling_selector(logits, nvec, temperature=1.0)
    log_prob = info["log_prob"]

    assert actions.shape == (1, 2)
    assert log_prob.shape == (1,)

    # Check selected actions
    assert actions[0, 0] == 1
    assert actions[0, 1] == 0

    # Manual check for log_prob
    # First component: Categorical(logits=[0, 10, 0]) -> log_prob[1]
    # Second component: Categorical(logits=[10, 0]) -> log_prob[0]
    split_logits = torch.split(logits, list(nvec), dim=-1)
    expected_log_prob = torch.distributions.Categorical(
        logits=split_logits[0]
    ).log_prob(actions[0, 0]) + torch.distributions.Categorical(
        logits=split_logits[1]
    ).log_prob(actions[0, 1])
    torch.testing.assert_close(log_prob, expected_log_prob)
