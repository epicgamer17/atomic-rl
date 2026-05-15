import pytest
import torch
import numpy as np
from tensordict import TensorDict
from functional.rollout_buffer import (
    init_rollout_buffer,
    store_rollout_step,
    record_truncations,
    get_rollout_next_values,
    flatten_rollout_buffer,
    yield_shuffled_minibatches,
    yield_sequential_minibatches,
)

pytestmark = pytest.mark.unit


def test_init_rollout_buffer():
    steps = 10
    num_envs = 4
    shapes = {"obs": (8,), "action": ()}
    device = "cpu"

    buffer = init_rollout_buffer(steps, num_envs, shapes, device=device)

    assert buffer.data["obs"].shape == (num_envs, steps, 8)
    assert buffer.data["action"].shape == (num_envs, steps)
    assert buffer.data["action"].dtype == torch.long
    assert len(buffer.truncation_records) == 0


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
    torch.testing.assert_close(buffer.data[:, step], transition)


def test_record_truncations():
    """Test recording of truncations in the rollout buffer."""
    steps = 5
    num_envs = 4
    shapes = {"obs": (4,)}
    buffer = init_rollout_buffer(steps, num_envs, shapes)

    # Simulation of example script logic:
    # 1. We have one ended env (1) that was truncated
    final_obs_1 = torch.ones(4, dtype=torch.float32)
    truncated_envs = torch.tensor([1], dtype=torch.long)
    final_observations = torch.stack([final_obs_1])

    record_truncations(
        buffer,
        step=2,
        truncated_envs=truncated_envs,
        final_observations=final_observations,
    )

    assert len(buffer.truncation_records) == 1
    step_rec, env_idx, obs = buffer.truncation_records[0]
    assert step_rec == 2
    assert env_idx == 1
    torch.testing.assert_close(obs, final_obs_1)


def test_get_rollout_next_values():
    """Test patching of next_values using truncation records."""
    steps = 3
    num_envs = 2
    shapes = {"values": ()}
    buffer = init_rollout_buffer(steps, num_envs, shapes)

    # Fill values: [B, T]
    # env 0: [1.0, 2.0, 3.0]
    # env 1: [1.1, 2.1, 3.1]
    buffer.data["values"] = torch.tensor([[1.0, 2.0, 3.0], [1.1, 2.1, 3.1]])

    last_values = torch.tensor([4.0, 4.1])  # [B]

    # Value function for patching: returns 100 * sum(obs)
    def get_value_fn(obs):
        return obs.sum(dim=-1, keepdim=True) * 100.0

    # Case 1: No truncations
    next_vals = get_rollout_next_values(buffer, last_values, get_value_fn, "cpu")
    # Expected: [values[:, 1:], last_values]
    # env 0: [2.0, 3.0, 4.0]
    # env 1: [2.1, 3.1, 4.1]
    expected = torch.tensor([[2.0, 3.0, 4.0], [2.1, 3.1, 4.1]])
    torch.testing.assert_close(next_vals, expected)

    # Case 2: With truncation at step 1, env 0
    obs_patch = torch.ones(4)
    buffer.truncation_records.append((1, 0, obs_patch))

    next_vals_patched = get_rollout_next_values(
        buffer, last_values, get_value_fn, "cpu"
    )
    # Expected: same as above, but next_vals[0, 1] = get_value_fn(ones) = 400.0 (env 0, step 1)
    expected_patched = expected.clone()
    expected_patched[0, 1] = 400.0
    torch.testing.assert_close(next_vals_patched, expected_patched)


def test_get_rollout_next_values_requires_final_obs_for_truncation():
    steps = 3
    num_envs = 2
    shapes = {"values": (), "truncated": ()}
    buffer = init_rollout_buffer(steps, num_envs, shapes)
    buffer.data["values"] = torch.tensor([[1.0, 2.0, 3.0], [1.1, 2.1, 3.1]])
    buffer.data["truncated"] = torch.zeros(num_envs, steps)
    buffer.data["truncated"][1, 2] = 1.0

    def get_value_fn(obs):
        return obs.sum(dim=-1, keepdim=True)

    with pytest.raises(RuntimeError, match="Missing final observations"):
        get_rollout_next_values(
            buffer,
            torch.tensor([4.0, 4.1]),
            get_value_fn,
            "cpu",
        )


def test_flatten_rollout_buffer():
    steps = 3
    num_envs = 2
    shapes = {"obs": (4,), "action": ()}
    buffer = init_rollout_buffer(steps, num_envs, shapes)

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
    assert torch.all(flat_data["obs"][0] == 0)
    assert torch.all(flat_data["obs"][1] == 1)
    assert torch.all(flat_data["obs"][2] == 2)
    assert torch.all(flat_data["obs"][3] == 0)
    assert torch.all(flat_data["obs"][4] == 1)
    assert torch.all(flat_data["obs"][5] == 2)


def test_yield_shuffled_minibatches():
    """Test yielding shuffled minibatches from a TensorDict."""
    total_size = 10
    minibatch_size = 3
    data = TensorDict(
        {
            "obs": torch.arange(total_size).float(),
            "action": torch.arange(total_size),
        },
        batch_size=[total_size],
    )

    # Use a fixed generator for determinism in the test
    generator = torch.Generator().manual_seed(42)

    batches = list(
        yield_shuffled_minibatches(data, minibatch_size, generator=generator)
    )

    # Check number of batches: ceil(10 / 3) = 4
    assert len(batches) == 4

    # Check batch sizes: 3, 3, 3, 1
    assert batches[0].batch_size[0] == 3
    assert batches[1].batch_size[0] == 3
    assert batches[2].batch_size[0] == 3
    assert batches[3].batch_size[0] == 1

    # Check that all data is present exactly once
    all_indices = torch.cat([b["action"] for b in batches])
    assert torch.all(torch.sort(all_indices)[0] == torch.arange(total_size))

    # Check shuffling: with seed 42, randperm(10) is [2, 0, 9, 8, 5, 1, 6, 3, 7, 4]
    # We just want to ensure it's not the original order
    assert not torch.all(all_indices == torch.arange(total_size))


def test_yield_sequential_minibatches():
    """Test yielding sequential minibatches for LSTM PPO."""
    num_envs = 4
    steps = 5
    num_minibatches = 2
    hidden_dim = 8
    num_layers = 1

    # buffer_data [envs, steps]
    data = TensorDict(
        {
            "obs": torch.zeros(num_envs, steps, 1),
        },
        batch_size=[num_envs, steps],
    )

    # Fill with recognizable values: data[env, step] = env * 10 + step
    for e in range(num_envs):
        for t in range(steps):
            data["obs"][e, t] = e * 10 + t

    initial_h = torch.randn(num_layers, num_envs, hidden_dim)
    initial_c = torch.randn(num_layers, num_envs, hidden_dim)
    initial_states = (initial_h, initial_c)

    generator = torch.Generator().manual_seed(42)

    batches = list(
        yield_sequential_minibatches(
            data, num_envs, num_minibatches, initial_states, generator=generator
        )
    )

    # num_envs=4, num_minibatches=2 -> 2 batches, each with 2 envs
    assert len(batches) == 2

    seen_envs = []
    for mb_flat, (mb_h, mb_c) in batches:
        # mb_flat should be [steps * envs_per_batch, ...] = [5 * 2, 1] = [10, 1]
        assert mb_flat.batch_size[0] == steps * 2
        assert mb_h.shape == (num_layers, 2, hidden_dim)
        assert mb_c.shape == (num_layers, 2, hidden_dim)

        # Reshape mb_flat to [steps, envs_per_batch]
        mb_reshaped = mb_flat["obs"].view(steps, 2, 1)

        # Verify that for each env in the batch, the steps are sequential: [env*10+0, env*10+1, ...]
        for i in range(2):
            env_data = mb_reshaped[:, i, 0]
            start_val = env_data[0].item()
            env_idx = int(start_val // 10)
            seen_envs.append(env_idx)
            expected = torch.tensor(
                [env_idx * 10 + t for t in range(steps)], dtype=torch.float32
            )
            torch.testing.assert_close(env_data, expected)

            # Verify LSTM states match the env_idx
            torch.testing.assert_close(mb_h[:, i], initial_h[:, env_idx])
            torch.testing.assert_close(mb_c[:, i], initial_c[:, env_idx])

    # Verify all envs were seen
    assert sorted(seen_envs) == list(range(num_envs))
