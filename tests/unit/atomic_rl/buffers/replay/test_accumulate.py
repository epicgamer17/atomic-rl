import pytest
import torch

from atomic_rl.buffers.replay import make_n_step_accumulator

pytestmark = pytest.mark.unit


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
    trans = process(
        obs[None],
        action[None],
        reward[None],
        next_obs[None],
        terminated[None],
        truncated[None],
    )
    assert len(trans) == 0

    # 2. Step until window full
    process(
        obs[None],
        action[None],
        reward[None],
        next_obs[None],
        terminated[None],
        truncated[None],
    )
    trans = process(
        obs[None],
        action[None],
        reward[None],
        next_obs[None],
        terminated[None],
        truncated[None],
    )
    assert len(trans) == 1
    # reward = 1 + 0.9*1 + 0.9^2 * 1 = 1 + 0.9 + 0.81 = 2.71
    torch.testing.assert_close(trans[0]["reward"], torch.tensor(2.71))
    torch.testing.assert_close(trans[0]["gamma"], torch.tensor(gamma**3))

    # 3. Terminate
    terminated_true = torch.tensor(1.0, dtype=torch.float32)
    trans_term = process(
        obs[None],
        action[None],
        reward[None],
        next_obs[None],
        terminated_true[None],
        truncated[None],
    )
    # history had 2 items left before termination, plus the terminal one = 3
    # it should flush all 3
    assert len(trans_term) == 3
    # last one should have terminated=1.0
    assert trans_term[-1]["terminated"] == 1.0
    # last one's reward should be just the last step's reward
    torch.testing.assert_close(trans_term[-1]["gamma"], torch.tensor(gamma**1))

    # 4. Reset
    process(
        obs[None],
        action[None],
        reward[None],
        next_obs[None],
        terminated[None],
        truncated[None],
    )
    reset()
    trans_after_reset = process(
        obs[None],
        action[None],
        reward[None],
        next_obs[None],
        terminated[None],
        truncated[None],
    )
    assert len(trans_after_reset) == 0  # history was cleared
