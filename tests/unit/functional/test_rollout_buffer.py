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

    # Info dict with two truncations
    final_obs_1 = np.ones(4, dtype=np.float32)
    final_obs_3 = np.ones(4, dtype=np.float32) * 3.0
    info = {
        "final_observation": [None, final_obs_1, None, final_obs_3],
        "_final_observation": np.array([False, True, False, True]),
    }
    # Only env 1 and 3 are in the info dict
    # But let's say only env 1 was actually 'truncated' (env 3 might be 'terminated')
    truncated = torch.tensor([False, True, False, False])

    record_truncations(buffer, step=2, info=info, truncated=truncated)

    assert len(buffer.truncation_records) == 1
    step_rec, env_idx, obs = buffer.truncation_records[0]
    assert step_rec == 2
    assert env_idx == 1
    torch.testing.assert_close(obs, torch.from_numpy(final_obs_1))


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

    last_values = torch.tensor([4.0, 4.1]) # [B]
    
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
    
    next_vals_patched = get_rollout_next_values(buffer, last_values, get_value_fn, "cpu")
    # Expected: same as above, but next_vals[0, 1] = get_value_fn(ones) = 400.0 (env 0, step 1)
    expected_patched = expected.clone()
    expected_patched[0, 1] = 400.0
    torch.testing.assert_close(next_vals_patched, expected_patched)


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
