import pytest
import torch
import random
import math
from tensordict import TensorDict
from functional.buffer import (
    init_buffer,
    init_per_buffer,
    circular_write_strategy,
    reservoir_write_strategy,
    uniform_sample,
    sample_per,
    update_priorities,
    with_per_tracking,
    make_n_step_accumulator,
    init_rollout_buffer,
    store_rollout_step,
    flatten_rollout_buffer,
)

pytestmark = pytest.mark.unit


def test_init_buffer():
    capacity = 10
    shapes = {"obs": (4,), "action": (1,)}
    state = init_buffer(capacity, shapes)

    assert state.capacity == capacity
    assert state.size == 0
    assert state.pointer == 0
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


def test_circular_write_strategy():
    capacity = 5
    shapes = {"data": (1,)}
    state = init_buffer(capacity, shapes)

    # 1. Write 3 items
    batch = TensorDict({"data": torch.tensor([[1.0], [2.0], [3.0]])}, batch_size=[3])
    state, indices = circular_write_strategy(state, batch)

    assert state.size == 3
    assert state.pointer == 3
    torch.testing.assert_close(indices, torch.tensor([0, 1, 2]))
    torch.testing.assert_close(state.data["data"][:3], batch["data"])

    # 2. Write 4 items (should wrap around)
    batch2 = TensorDict({"data": torch.tensor([[4.0], [5.0], [6.0], [7.0]])}, batch_size=[4])
    state, indices2 = circular_write_strategy(state, batch2)

    assert state.size == 5  # capped at capacity
    assert state.pointer == 2  # (3 + 4) % 5 = 2
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
    from functional.buffer import ReservoirBufferState

    state = ReservoirBufferState(
        data=TensorDict(
            {"data": torch.zeros((capacity, 1))}, batch_size=[capacity], device="cpu"
        ),
        pointer=0,
        size=0,
        capacity=capacity,
        total_steps_seen=0,
    )

    # Write 10 items
    batch = TensorDict({"data": torch.arange(10, dtype=torch.float32).reshape(-1, 1)}, batch_size=[10])
    state, indices = reservoir_write_strategy(state, batch)

    assert state.total_steps_seen == 10
    assert state.size == 5
    # Since it's random, we just check that indices are valid
    assert indices.numel() <= 10
    assert torch.all(indices < 5)


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
    batch_prio = TensorDict({"data": torch.tensor([[20.0]]), "priority": torch.tensor([[5.0]])}, batch_size=[1])
    state = wrapped_write(state, batch_prio)
    # pointer was 1, so written to index 1, tree index 1 + 4 - 1 = 4
    assert state.sum_tree[4] == 5.0


def test_n_step_accumulator():
    n_steps = 3
    gamma = 0.9
    process, reset = make_n_step_accumulator(n_steps, gamma)

    obs = torch.zeros(4)
    action = torch.tensor(1, dtype=torch.long)
    reward = torch.tensor(1.0, dtype=torch.float32)
    next_obs = torch.ones(4)
    terminated = torch.tensor(0.0, dtype=torch.float32)
    truncated = torch.tensor(0.0, dtype=torch.float32)

    # 1. Step once
    trans = process(obs, action, reward, next_obs, terminated, truncated)
    assert len(trans) == 0

    # 2. Step until window full
    process(obs, action, reward, next_obs, terminated, truncated)
    trans = process(obs, action, reward, next_obs, terminated, truncated)
    assert len(trans) == 1
    # reward = 1 + 0.9*1 + 0.9^2 * 1 = 1 + 0.9 + 0.81 = 2.71
    torch.testing.assert_close(trans[0]["reward"], torch.tensor(2.71))
    torch.testing.assert_close(trans[0]["gamma"], torch.tensor(gamma**3))

    # 3. Terminate
    terminated_true = torch.tensor(1.0, dtype=torch.float32)
    trans_term = process(obs, action, reward, next_obs, terminated_true, truncated)
    # history had 2 items left before termination, plus the terminal one = 3
    # it should flush all 3
    assert len(trans_term) == 3
    # last one should have terminated=1.0
    assert trans_term[-1]["terminated"] == 1.0
    # last one's reward should be just the last step's reward
    torch.testing.assert_close(trans_term[-1]["gamma"], torch.tensor(gamma**1))

    # 4. Reset
    process(obs, action, reward, next_obs, terminated, truncated)
    reset()
    trans_after_reset = process(obs, action, reward, next_obs, terminated, truncated)
    assert len(trans_after_reset) == 0  # history was cleared


def test_init_rollout_buffer():
    steps = 10
    num_envs = 4
    shapes = {"obs": (8,), "action": ()}
    device = "cpu"

    buffer = init_rollout_buffer(steps, num_envs, shapes, device=device)

    assert buffer.data["obs"].shape == (steps, num_envs, 8)
    assert buffer.data["action"].shape == (steps, num_envs)
    assert buffer.data["action"].dtype == torch.long


def test_store_rollout_step():
    steps = 5
    num_envs = 2
    shapes = {"obs": (4,), "action": ()}
    buffer = init_rollout_buffer(steps, num_envs, shapes)

    step = 0
    transition = TensorDict(
        {"obs": torch.randn(num_envs, 4), "action": torch.randint(0, 5, (num_envs,))},
        batch_size=[num_envs],
    )

    store_rollout_step(buffer, step, transition)

    torch.testing.assert_close(buffer.data[step], transition)


def test_flatten_rollout_buffer():
    steps = 3
    num_envs = 2
    shapes = {"obs": (4,), "action": ()}
    buffer = init_rollout_buffer(steps, num_envs, shapes)

    # Fill buffer
    for i in range(steps):
        transition = TensorDict(
            {
                "obs": torch.ones(num_envs, 4) * i,
                "action": torch.ones(num_envs, dtype=torch.long) * i,
            },
            batch_size=[num_envs],
        )
        store_rollout_step(buffer, i, transition)

    flat_data = flatten_rollout_buffer(buffer)

    assert flat_data.batch_size == torch.Size([steps * num_envs])
    assert flat_data["obs"].shape == (steps * num_envs, 4)
    assert torch.all(flat_data["obs"][0:2] == 0)
    assert torch.all(flat_data["obs"][2:4] == 1)
    assert torch.all(flat_data["obs"][4:6] == 2)
