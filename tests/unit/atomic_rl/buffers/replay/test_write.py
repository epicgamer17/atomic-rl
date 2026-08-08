import pytest
import torch
import random
import math
from tensordict import TensorDict

from atomic_rl.buffers.replay import (
    circular_write_strategy,
    compute_is_weights,
    init_buffer,
    init_per_buffer,
    reservoir_write_strategy,
    update_priorities,
    with_per_tracking,
)

pytestmark = pytest.mark.unit


def test_circular_write_strategy():
    capacity = 5
    shapes = {"data": (1,)}
    state = init_buffer(capacity, shapes)

    # 1. Write 3 items
    batch = TensorDict({"data": torch.tensor([[1.0], [2.0], [3.0]])}, batch_size=[3])
    state, indices = circular_write_strategy(state, batch)

    assert state.size == 3
    assert state.pointer == 3
    assert state.steps_seen == 3
    torch.testing.assert_close(indices, torch.tensor([0, 1, 2]))
    torch.testing.assert_close(state.data["data"][:3], batch["data"])

    # 2. Write 4 items (should wrap around)
    batch2 = TensorDict(
        {"data": torch.tensor([[4.0], [5.0], [6.0], [7.0]])}, batch_size=[4]
    )
    state, indices2 = circular_write_strategy(state, batch2)

    assert state.size == 5  # capped at capacity
    assert state.pointer == 2  # (3 + 4) % 5 = 2
    assert state.steps_seen == 7
    torch.testing.assert_close(indices2, torch.tensor([3, 4, 0, 1]))
    # Verify values at indices
    assert state.data["data"][3] == 4.0
    assert state.data["data"][4] == 5.0
    assert state.data["data"][0] == 6.0
    assert state.data["data"][1] == 7.0


def test_reservoir_write_strategy():
    random.seed(42)
    capacity = 5
    shapes = {"data": (1,)}

    state = init_buffer(capacity, shapes)

    # Write 10 items
    batch = TensorDict(
        {"data": torch.arange(10, dtype=torch.float32).reshape(-1, 1)}, batch_size=[10]
    )
    state, indices = reservoir_write_strategy(state, batch)

    assert state.steps_seen == 10
    assert state.size == 5
    # Since it's random, we just check that indices are valid
    assert indices.numel() <= 10
    assert torch.all(indices < 5)


def test_per_tree_logic():
    capacity = 4
    state = init_per_buffer(capacity, {"data": (1,)})
    # tree_capacity = 4, tree_nodes = 7
    # leaf indices: 3, 4, 5, 6

    # Update priorities manually
    tree_indices = torch.tensor([3, 4, 5, 6], dtype=torch.long)
    priorities = torch.tensor([1.0, 2.0, 3.0, 4.0])
    state = update_priorities(state, tree_indices, priorities, alpha=1.0)

    # Sum tree should be:
    # root: 10
    # children: 3, 7
    # leaves: 1, 2, 3, 4
    assert math.isclose(state.sum_tree[0].item(), 10.0, rel_tol=1e-5)
    assert math.isclose(state.sum_tree[1].item(), 3.0, rel_tol=1e-5)
    assert math.isclose(state.sum_tree[2].item(), 7.0, rel_tol=1e-5)
    torch.testing.assert_close(state.sum_tree[3:], priorities)

    # Min tree:
    assert math.isclose(state.min_tree[0].item(), 1.0, rel_tol=1e-5)
    assert math.isclose(state.min_tree[1].item(), 1.0, rel_tol=1e-5)
    assert math.isclose(state.min_tree[2].item(), 3.0, rel_tol=1e-5)


def test_with_per_tracking():
    capacity = 4
    state = init_per_buffer(capacity, {"data": (1,)})
    wrapped_write = with_per_tracking(circular_write_strategy)

    # 1. Automatic max priority
    batch = TensorDict({"data": torch.tensor([[10.0]])}, batch_size=[1])
    state = wrapped_write(state, batch)

    assert state.size == 1
    assert state.pointer == 1
    # Check if sum tree was updated at leaf index 3
    assert state.sum_tree[3] == state.max_priority
    assert state.sum_tree[0] == state.max_priority

    # 2. Explicit priority override
    batch_prio = TensorDict(
        {"data": torch.tensor([[20.0]]), "priority": torch.tensor([[5.0]])},
        batch_size=[1],
    )
    state = wrapped_write(state, batch_prio)
    # pointer was 1, so written to index 1, tree index 1 + 4 - 1 = 4
    assert state.sum_tree[4] == 5.0


def test_compute_is_weights():
    """Test Importance Sampling weight computation."""
    leaf_priorities = torch.tensor([1.0, 2.0, 4.0])
    total_priority = 10.0
    min_prob = 0.05
    beta = torch.tensor(0.5)

    is_weights = compute_is_weights(leaf_priorities, min_prob, total_priority, beta)

    # probs = [0.1, 0.2, 0.4]
    # is = [ (0.1/0.05)^-0.5, (0.2/0.05)^-0.5, (0.4/0.05)^-0.5 ]
    # is = [ 2^-0.5, 4^-0.5, 8^-0.5 ]
    # is = [ 0.7071, 0.5, 0.3535 ]
    expected_is = torch.tensor([0.70710678118, 0.5, 0.35355339059])
    torch.testing.assert_close(is_weights, expected_is)
