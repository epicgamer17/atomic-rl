import pytest
import torch
import torch.nn.functional as F
import math
from atomic_rl.action_selection import (
    expected_value,
    argmax_selector,
    sample_distribution,
    with_epsilon_greedy,
)

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


def test_expected_value_2d_input():
    """Test the expected_value function with 2D state-value predictions [B, N]."""
    # Batch size 2, 3 atoms
    predictions = torch.tensor(
        [
            [0.0, 100.0, 0.0],  # Softmax concentrates heavily on index 1
            [100.0, 0.0, 0.0],  # Softmax concentrates heavily on index 0
        ]
    )
    support = torch.tensor([-1.0, 0.0, 1.0])  # 1D Support

    values = expected_value(predictions, support)
    assert values.shape == (2,)
    torch.testing.assert_close(values, torch.tensor([0.0, -1.0]), atol=1e-4, rtol=1e-4)


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


# ==========================================
# Comprehensive Tests for with_epsilon_greedy Masking
# ==========================================


def test_with_epsilon_greedy_with_mask():
    """Verify that action masking restricts exploration to allowed actions."""
    greedy_selector = lambda x: (torch.zeros(x.shape[0], 1, dtype=torch.long), {})
    epsilon_selector = with_epsilon_greedy(greedy_selector)

    # 1 batch, 4 actions. Action 0 is the greedy action, but masked out!
    predictions = torch.tensor([[10.0, 0.0, 0.0, 0.0]])
    mask = torch.tensor([[0, 1, 1, 0]])  # Only actions 1 and 2 are valid

    gen = torch.Generator()
    gen.manual_seed(101)

    # Force 100% exploration
    for _ in range(20):
        actions, _ = epsilon_selector(
            predictions, epsilon=1.0, num_actions=4, generator=gen, mask=mask
        )
        # Action must strictly be either 1 or 2, never 0 or 3
        assert actions.item() in [1, 2]


def test_with_epsilon_greedy_mask_assertions():
    """Verify fail-fast protections for invalid shapes or empty masks."""
    greedy_selector = lambda x: (torch.zeros(x.shape[0], 1, dtype=torch.long), {})
    epsilon_selector = with_epsilon_greedy(greedy_selector)

    predictions = torch.tensor([[1.0, 2.0, 3.0]])  # Batch=1, Actions=3

    # Case 1: Mask shape mismatch
    bad_mask = torch.tensor([[1, 1]])  # Expecting length 3
    with pytest.raises(AssertionError, match="Mask shape .* does not match expected"):
        epsilon_selector(predictions, epsilon=0.5, num_actions=3, mask=bad_mask)

    # Case 2: Zero valid actions in the environment
    dead_mask = torch.tensor([[0, 0, 0]])
    with pytest.raises(
        AssertionError,
        match="Encountered a mask where an environment has 0 valid actions",
    ):
        epsilon_selector(predictions, epsilon=0.5, num_actions=3, mask=dead_mask)


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
    from atomic_rl.action_selection import apply_action_mask

    logits = torch.tensor([[1.0, 2.0, 3.0]])
    mask = torch.tensor([[1, 0, 1]])
    masked = apply_action_mask(logits, mask)
    expected = torch.tensor([[1.0, -1e8, 3.0]])
    torch.testing.assert_close(masked, expected)


def test_compute_masked_entropy():
    from atomic_rl.action_selection import compute_masked_entropy

    logits = torch.tensor([[-0.6931, -1e8, -0.6931]])
    probs = torch.tensor([[0.5, 0.0, 0.5]])
    mask = torch.tensor([[1, 0, 1]])
    entropy = compute_masked_entropy(logits, probs, mask)
    # entropy = -(0.5 * -0.6931 + 0 + 0.5 * -0.6931) = 0.6931
    expected = torch.tensor([0.6931])
    torch.testing.assert_close(entropy, expected)


# ==========================================
# Tests for gather_q_values
# ==========================================


def test_gather_q_values_2d():
    from atomic_rl.action_selection import gather_q_values

    # q_values: [B, A] -> [2, 3]
    q_values = torch.tensor([[10.0, 20.0, 30.0], [40.0, 50.0, 60.0]])

    # Test with 1D actions [B]
    actions_1d = torch.tensor([1, 2])  # Selects 20.0 and 60.0
    out_1d = gather_q_values(q_values, actions_1d)
    torch.testing.assert_close(out_1d, torch.tensor([20.0, 60.0]))

    # Test with 2D actions [B, 1]
    actions_2d = torch.tensor([[1], [2]])
    out_2d = gather_q_values(q_values, actions_2d)
    torch.testing.assert_close(out_2d, torch.tensor([20.0, 60.0]))


def test_gather_q_values_3d():
    from atomic_rl.action_selection import gather_q_values

    # q_values: [B, A, Atoms] -> [2, 2, 3]
    q_values = torch.tensor(
        [
            [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]],  # Batch 0: Action 0, Action 1
            [[7.0, 8.0, 9.0], [10.0, 11.0, 12.0]],  # Batch 1: Action 0, Action 1
        ]
    )
    actions = torch.tensor([1, 0])  # Batch 0 -> Action 1, Batch 1 -> Action 0

    expected = torch.tensor([[4.0, 5.0, 6.0], [7.0, 8.0, 9.0]])

    out = gather_q_values(q_values, actions)
    assert out.shape == (2, 3)
    torch.testing.assert_close(out, expected)


def test_gather_q_values_assertions():
    from atomic_rl.action_selection import gather_q_values

    # Invalid q_values dimensions (4D)
    with pytest.raises(AssertionError, match="Expected 2D or 3D q_values"):
        gather_q_values(torch.randn(2, 2, 2, 2), torch.tensor([0, 0]))

    # Invalid action dimensions (2D but wrong shape, or 3D)
    with pytest.raises(AssertionError, match="Expected 1D actions"):
        gather_q_values(torch.randn(2, 3), torch.tensor([[0, 1], [1, 0]]))
