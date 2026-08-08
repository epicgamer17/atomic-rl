import pytest
import torch

from atomic_rl.buffers.rollout import init_rollout_buffer

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
