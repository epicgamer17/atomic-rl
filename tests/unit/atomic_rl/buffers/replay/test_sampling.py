import pytest
import torch
from tensordict import TensorDict

from atomic_rl.buffers.replay import (
    init_buffer,
    init_per_buffer,
    sample_per,
    uniform_sample,
    update_priorities,
)

pytestmark = pytest.mark.unit


def test_uniform_sample():
    capacity = 10
    state = init_buffer(capacity, {"data": (1,)})
    state.size = 5
    state.data["data"][:5] = torch.arange(5, dtype=torch.float32).reshape(-1, 1)

    gen = torch.Generator()
    gen.manual_seed(42)

    sample = uniform_sample(state, gen, batch_size=3)
    assert sample.batch_size == torch.Size([3])
    assert torch.all(sample["data"] < 5)

    with pytest.raises(ValueError):
        uniform_sample(state, gen, batch_size=10)


def test_sample_per():
    capacity = 4
    state = init_per_buffer(capacity, {"data": (1,)})
    state.size = 4
    state.data["data"][:] = torch.arange(4, dtype=torch.float32).reshape(-1, 1)

    # Set priorities such that the last one is very likely
    tree_indices = torch.tensor([3, 4, 5, 6], dtype=torch.long)
    priorities = torch.tensor([0.1, 0.1, 0.1, 10.0])
    state = update_priorities(state, tree_indices, priorities, alpha=1.0)

    beta = torch.tensor(0.4)
    batch, indices, is_weights = sample_per(state, batch_size=100, beta=beta)

    # Most sampled should be index 6 (data index 3)
    data_indices = indices - (state.tree_capacity - 1)
    counts = torch.bincount(data_indices)
    assert torch.argmax(counts) == 3

    # Check IS weights shape
    assert is_weights.shape == (100,)
    # max weight should be for the lowest priority item
    assert is_weights[data_indices == 0].max() > is_weights[data_indices == 3].max()
