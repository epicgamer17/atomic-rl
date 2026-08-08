import pytest
import torch
from tensordict import TensorDict

from atomic_rl.buffers.rollout import (
    init_rollout_buffer,
    record_truncations_,
    store_rollout_step_,
)

pytestmark = pytest.mark.unit


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

    store_rollout_step_(buffer, step, transition)
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

    record_truncations_(
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
