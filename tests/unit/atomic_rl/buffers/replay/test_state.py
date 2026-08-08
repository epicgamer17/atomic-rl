import pytest
import torch

from atomic_rl.buffers.replay import init_buffer, init_per_buffer

pytestmark = pytest.mark.unit


def test_init_buffer():
    capacity = 10
    shapes = {"obs": (4,), "action": (1,)}
    state = init_buffer(capacity, shapes)

    assert state.capacity == capacity
    assert state.size == 0
    assert state.pointer == 0
    assert state.steps_seen == 0
    assert "obs" in state.data.keys()
    assert state.data["obs"].shape == (capacity, 4)
    assert state.data["action"].shape == (capacity, 1)


def test_init_per_buffer():
    capacity = 10
    shapes = {"obs": (4,)}
    state = init_per_buffer(capacity, shapes)

    # Smallest power of 2 >= 10 is 16
    assert state.tree_capacity == 16
    # tree nodes = 2 * 16 - 1 = 31
    assert state.sum_tree.shape == (31,)
    assert state.min_tree.shape == (31,)
    assert state.max_priority == 1.0
