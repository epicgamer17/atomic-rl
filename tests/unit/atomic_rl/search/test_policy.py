import pytest
import torch

from atomic_rl.search import get_mcts_visit_policy

pytestmark = pytest.mark.unit


def test_get_mcts_visit_policy_temperature_1():
    """Verify temperature tau=1.0 yields visit count proportional policy."""
    visit_counts = torch.tensor([[10.0, 30.0, 60.0]])
    policy = get_mcts_visit_policy(visit_counts, temperature=1.0)
    expected = torch.tensor([[0.1, 0.3, 0.6]])
    torch.testing.assert_close(policy, expected)


def test_get_mcts_visit_policy_temperature_greedy():
    """Verify temperature tau=0.0 yields greedy one-hot policy."""
    visit_counts = torch.tensor([[10.0, 30.0, 60.0]])
    policy = get_mcts_visit_policy(visit_counts, temperature=0.0)
    expected = torch.tensor([[0.0, 0.0, 1.0]])
    torch.testing.assert_close(policy, expected)


def test_get_mcts_visit_policy_negative_temperature():
    """Verify fail-fast assertion on negative temperature."""
    visit_counts = torch.tensor([[10.0, 30.0]])
    with pytest.raises(AssertionError, match="Temperature must be non-negative"):
        get_mcts_visit_policy(visit_counts, temperature=-0.5)
