import pytest
import torch
from atomic_rl.td import (
    compute_q_td_target,
    compute_categorical_q_td_target,
)

pytestmark = pytest.mark.unit


def test_compute_q_td_target():
    # Batch size 2, Actions 3
    next_q = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    next_actions = torch.tensor([2, 0])  # Q-values: 3.0, 4.0
    rewards = torch.tensor([0.5, 1.0])
    terminated = torch.tensor([0.0, 1.0])

    gamma = torch.tensor([0.9, 0.9])

    # Target 0: Terminated=0 -> Should bootstrap
    # 0.5 + 0.9 * 3.0 = 0.5 + 2.7 = 3.2
    # Target 1: Terminated=1 -> Should NOT bootstrap
    # 1.0 + 0.9 * 4.0 * (1 - 1) = 1.0
    expected = torch.tensor([3.2, 1.0])

    target = compute_q_td_target(next_q, next_actions, rewards, terminated, gamma)
    torch.testing.assert_close(target, expected)


def test_compute_categorical_q_td_target():
    # Small scale example for C51 projection
    atom_size = 3
    v_min, v_max = 0.0, 2.0
    support = torch.linspace(v_min, v_max, atom_size)  # [0.0, 1.0, 2.0]

    # Batch size 1, Action 1
    next_logits = torch.tensor(
        [[[0.0, 0.0, 0.0], [0.0, 100.0, 0.0]]]
    )  # Next action 1 is certain at atom 1 (val 1.0)
    next_actions = torch.tensor([1])
    rewards = torch.tensor([0.5])
    terminated = torch.tensor([0.0])

    gamma = torch.tensor([1.0])  # For simplicity

    # Tz = 0.5 + 1.0 * [0, 1, 2] = [0.5, 1.5, 2.5]
    # Clamped = [0.5, 1.5, 2.0]
    # atom 0 (0.0): projected from Tz=0.5 -> between bin 0 and 1. 0.5 is exactly half.
    # next_probs_a = [0, 1, 0] (since atom 1 was certain)
    # Tz for that atom was 0.5 + 1.0 * 1.0 = 1.5
    # 1.5 is exactly between bin 1 (1.0) and bin 2 (2.0)
    # so m[1] = 0.5, m[2] = 0.5

    target_dist = compute_categorical_q_td_target(
        next_logits,
        next_actions,
        rewards,
        terminated,
        gamma,
        support,
        v_min,
        v_max,
        atom_size,
    )

    expected_dist = torch.tensor([[0.0, 0.5, 0.5]])
    torch.testing.assert_close(target_dist, expected_dist)


def test_compute_categorical_q_td_target_terminal():
    atom_size = 3
    v_min, v_max = 0.0, 2.0
    support = torch.linspace(v_min, v_max, atom_size)  # [0, 1, 2]

    next_logits = torch.randn(1, 1, atom_size)  # doesn't matter
    next_actions = torch.tensor([0])
    rewards = torch.tensor([1.2])
    terminated = torch.tensor([1.0])

    gamma = torch.tensor([0.9])

    # Terminal means Tz = rewards = 1.2
    # 1.2 is between bin 1 (1.0) and bin 2 (2.0)
    # distance from bin 1 = 0.2. distance to bin 2 = 0.8.
    # prob to bin 1 = 1 - 0.2/1.0 = 0.8
    # prob to bin 2 = 0.2/1.0 = 0.2

    target_dist = compute_categorical_q_td_target(
        next_logits,
        next_actions,
        rewards,
        terminated,
        gamma,
        support,
        v_min,
        v_max,
        atom_size,
    )

    expected_dist = torch.tensor([[0.0, 0.8, 0.2]])
    torch.testing.assert_close(target_dist, expected_dist)


def test_compute_v_td_target():
    from atomic_rl.td import compute_v_td_target

    next_values = torch.tensor([3.0, 4.0])
    rewards = torch.tensor([0.5, 1.0])
    terminated = torch.tensor([0.0, 1.0])
    gamma = torch.tensor([0.9, 0.9])

    # Target 0: 0.5 + 0.9 * 3.0 = 3.2
    # Target 1: 1.0 + 0.9 * 4.0 * 0 = 1.0
    expected = torch.tensor([3.2, 1.0])
    target = compute_v_td_target(next_values, rewards, terminated, gamma)
    torch.testing.assert_close(target, expected)


def test_td_assertions():
    from atomic_rl.td import (
        compute_v_td_target,
        compute_q_td_target,
        compute_categorical_q_td_target,
    )

    with pytest.raises(AssertionError, match="Expected 1D next_values"):
        compute_v_td_target(
            torch.randn(2, 2), torch.randn(2), torch.randn(2), torch.randn(2)
        )

    with pytest.raises(AssertionError, match="Shape mismatch"):
        compute_v_td_target(
            torch.randn(2), torch.randn(3), torch.randn(2), torch.randn(2)
        )

    with pytest.raises(AssertionError, match="Expected 2D next_q_values"):
        compute_q_td_target(
            torch.randn(2),
            torch.randn(2),
            torch.randn(2),
            torch.randn(2),
            torch.randn(2),
        )

    with pytest.raises(AssertionError, match="Expected \[B\] next_actions"):
        compute_q_td_target(
            torch.randn(2, 2),
            torch.randn(2, 2),
            torch.randn(2),
            torch.randn(2),
            torch.randn(2),
        )

    with pytest.raises(AssertionError, match="Expected 3D next_logits"):
        compute_categorical_q_td_target(
            torch.randn(2, 2),
            torch.randn(2),
            torch.randn(2),
            torch.randn(2),
            torch.randn(2),
            torch.randn(2),
            0.0,
            1.0,
            2,
        )

    with pytest.raises(AssertionError, match="Expected \[B\] next_actions"):
        compute_categorical_q_td_target(
            torch.randn(2, 2, 2),
            torch.randn(2, 2),
            torch.randn(2),
            torch.randn(2),
            torch.randn(2),
            torch.randn(2),
            0.0,
            1.0,
            2,
        )
