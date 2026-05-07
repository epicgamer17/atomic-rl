import pytest
import torch
from functional.targets import (
    standard_td_target,
    n_step_td_target,
    categorical_td_target,
)

pytestmark = pytest.mark.unit


def test_standard_td_target():
    # Batch size 2, Actions 3
    next_q = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    next_actions = torch.tensor([[2], [0]])  # Q-values: 3.0, 4.0
    rewards = torch.tensor([[0.5], [1.0]])
    terminated = torch.tensor([[False], [True]])
    gamma = torch.tensor([0.9, 0.9])

    # Target 0: 0.5 + 0.9 * 3.0 = 0.5 + 2.7 = 3.2
    # Target 1: 1.0 + 0.9 * 4.0 * (1 - 1) = 1.0
    expected = torch.tensor([[3.2], [1.0]])

    target = standard_td_target(next_q, next_actions, rewards, terminated, gamma)
    torch.testing.assert_close(target, expected)


def test_n_step_td_target():
    # Effectively same as standard but with n-step rewards/gamma passed in
    next_q = torch.tensor([[10.0]])
    next_actions = torch.tensor([[0]])
    rewards = torch.tensor([[1.0]])
    terminated = torch.tensor([[False]])
    gamma = torch.tensor([[0.9**3]])  # N=3 steps

    # 1.0 + 0.729 * 10.0 = 8.29
    expected = torch.tensor([[8.29]])

    target = n_step_td_target(next_q, next_actions, rewards, terminated, gamma)
    torch.testing.assert_close(target, expected)


def test_categorical_td_target():
    # Small scale example for C51 projection
    atom_size = 3
    v_min, v_max = 0.0, 2.0
    support = torch.linspace(v_min, v_max, atom_size)  # [0.0, 1.0, 2.0]

    # Batch size 1, Action 1
    next_logits = torch.tensor(
        [[[0.0, 0.0, 0.0], [0.0, 100.0, 0.0]]]
    )  # Next action 1 is certain at atom 1 (val 1.0)
    next_actions = torch.tensor([[1]])
    rewards = torch.tensor([[0.5]])
    terminated = torch.tensor([[False]])
    gamma = torch.tensor([[1.0]])  # For simplicity

    # Tz = 0.5 + 1.0 * [0, 1, 2] = [0.5, 1.5, 2.5]
    # Clamped = [0.5, 1.5, 2.0]
    # atom 0 (0.0): projected from Tz=0.5 -> between bin 0 and 1. 0.5 is exactly half.
    # next_probs_a = [0, 1, 0] (since atom 1 was certain)
    # Tz for that atom was 0.5 + 1.0 * 1.0 = 1.5
    # 1.5 is exactly between bin 1 (1.0) and bin 2 (2.0)
    # so m[1] = 0.5, m[2] = 0.5

    target_dist = categorical_td_target(
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


def test_categorical_td_target_terminal():
    atom_size = 3
    v_min, v_max = 0.0, 2.0
    support = torch.linspace(v_min, v_max, atom_size)  # [0, 1, 2]

    next_logits = torch.randn(1, 1, atom_size)  # doesn't matter
    next_actions = torch.tensor([[0]])
    rewards = torch.tensor([[1.2]])
    terminated = torch.tensor([[True]])
    gamma = torch.tensor([[0.9]])

    # Terminal means Tz = rewards = 1.2 (for all atoms)
    # 1.2 is between bin 1 (1.0) and bin 2 (2.0)
    # distance from bin 1 = 0.2. distance to bin 2 = 0.8.
    # prob to bin 1 = 1 - 0.2/1.0 = 0.8
    # prob to bin 2 = 0.2/1.0 = 0.2

    target_dist = categorical_td_target(
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
