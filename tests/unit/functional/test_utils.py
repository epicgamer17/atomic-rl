import pytest
import torch
from functional.utils import (
    ema_update,
    standardize_tensor,
    scale_tensor_by_std,
    gnt_init_wrapper,
)

pytestmark = pytest.mark.unit


def test_ema_update():
    old = torch.tensor([1.0, 2.0])
    new = torch.tensor([3.0, 4.0])
    alpha = 0.1

    # (1-0.1)*1.0 + 0.1*3.0 = 0.9 + 0.3 = 1.2
    # (1-0.1)*2.0 + 0.1*4.0 = 1.8 + 0.4 = 2.2
    expected = torch.tensor([1.2, 2.2])

    res = ema_update(old, new, alpha)
    torch.testing.assert_close(res, expected)


def test_standardize_tensor():
    """Test mean-centering and scaling by standard deviation."""
    # 1. Normal case
    tensor = torch.tensor([1.0, 2.0, 3.0])
    # mean = 2.0, std = 1.0 (unbiased)
    # res = ([1-2]/1, [2-2]/1, [3-2]/1) = [-1, 0, 1]
    expected = torch.tensor([-1.0, 0.0, 1.0])
    res = standardize_tensor(tensor)
    torch.testing.assert_close(res, expected)

    # 2. Single element
    single = torch.tensor([5.0])
    res_single = standardize_tensor(single)
    assert torch.all(res_single == 0.0)
    assert res_single.shape == single.shape

    # 3. Low variance (std < eps)
    low_var = torch.tensor([1.0, 1.0, 1.0])
    res_low = standardize_tensor(low_var)
    # (1-1) / (0 + 1e-8) = 0
    assert torch.all(res_low == 0.0)


def test_scale_tensor_by_std():
    """Test scaling by standard deviation WITHOUT mean-centering."""
    # 1. Normal case
    tensor = torch.tensor([1.0, 3.0])
    # mean = 2.0, centered = [-1, 1], std = 1.4142
    # res = [1/1.4142, 3/1.4142] = [0.7071, 2.1213]
    std = tensor.std()
    expected = tensor / std
    res = scale_tensor_by_std(tensor)
    torch.testing.assert_close(res, expected)

    # 2. Single element
    single = torch.tensor([5.0])
    res_single = scale_tensor_by_std(single)
    assert torch.all(res_single == 0.0)

    # 3. Low variance (std < eps)
    # If the tensor is centered and has low variance, it should be all zeros.
    low_var = torch.tensor([0.0, 0.0, 0.0])
    res_low = scale_tensor_by_std(low_var)
    # 0 / (0 + 1e-8) = 0
    assert torch.all(res_low == 0.0)


def test_utils_assertions():
    """Test that utility functions raise assertions on invalid input shapes."""
    # ema_update
    with pytest.raises(AssertionError, match="EMA shape mismatch"):
        ema_update(torch.randn(2), torch.randn(3), alpha=0.1)


def test_extract_vector_env_final_obs():
    """Test extraction of final observations from Gymnasium info dict."""
    import numpy as np
    from functional.utils import extract_vector_env_final_obs

    # 1. No final observations
    info_none = {}
    indices, obs = extract_vector_env_final_obs(info_none)
    assert indices.size == 0
    assert obs.size == 0

    # 2. Final observations exist but mask is missing (should not happen in Gym, but for safety)
    info_no_mask = {"final_observation": [np.zeros(4)]}
    indices, obs = extract_vector_env_final_obs(info_no_mask)
    assert indices.size == 0
    assert obs.size == 0

    # 3. Proper Gymnasium format
    final_obs_1 = np.ones(4)
    final_obs_2 = np.ones(4) * 2.0
    info = {
        "final_observation": [None, final_obs_1, None, final_obs_2],
        "_final_observation": np.array([False, True, False, True]),
    }
    indices, obs = extract_vector_env_final_obs(info)

    assert np.array_equal(indices, np.array([1, 3]))
    assert obs.shape == (2, 4)
    assert np.array_equal(obs[0], final_obs_1)
    assert np.array_equal(obs[1], final_obs_2)

    # 4. Mask is all False
    info_all_false = {
        "final_observation": [None, None],
        "_final_observation": np.array([False, False]),
    }
    indices, obs = extract_vector_env_final_obs(info_all_false)
    assert indices.size == 0
    assert obs.size == 0


# TODO: change this into a test for pufferlib to verify it properly extracts on pufferlib (need to add that functionality though)
def test_extract_vector_env_final_obs_edge_cases():
    """Test edge cases for extract_vector_env_final_obs."""
    import numpy as np
    from functional.utils import extract_vector_env_final_obs

    # 1. Info is not a dict (e.g., PufferLib list)
    info_list = ["some", "data"]
    indices, obs = extract_vector_env_final_obs(info_list)
    assert indices.size == 0
    assert obs.size == 0

    # 2. Info is None
    indices, obs = extract_vector_env_final_obs(None)
    assert indices.size == 0
    assert obs.size == 0

    # 3. Missing _final_observation key but has final_observation
    info_missing_mask = {"final_observation": [np.zeros(4)]}
    indices, obs = extract_vector_env_final_obs(info_missing_mask)
    assert indices.size == 0
    assert obs.size == 0


def test_allocate_tensordict():
    from functional.utils import _allocate_tensordict

    shapes = {"obs": (4, 84, 84), "action": ()}
    batch_size = [2, 3]
    td = _allocate_tensordict(shapes, batch_size)

    assert list(td.shape) == [2, 3]
    assert td["obs"].shape == (2, 3, 4, 84, 84)
    assert td["obs"].dtype == torch.float32
    assert td["action"].shape == (2, 3)
    assert td["action"].dtype == torch.long


def test_add_dirichlet_noise():
    from functional.utils import add_dirichlet_noise

    probs = torch.tensor([[1.0, 0.0], [0.5, 0.5]])
    epsilon = 0.25
    alpha = 0.3

    torch.manual_seed(42)
    noisy_probs = add_dirichlet_noise(probs, epsilon, alpha)

    assert noisy_probs.shape == probs.shape
    torch.testing.assert_close(noisy_probs.sum(dim=-1), torch.ones(2))
    assert noisy_probs[0, 0] >= 0.75


def test_gnt_init_wrapper():
    import torch.nn as nn

    # Mock init_fn that sets weights to 1.0
    def mock_init(tensor):
        nn.init.constant_(tensor, 1.0)

    wrapped = gnt_init_wrapper(mock_init)

    # 1. Weight matrix (2D)
    weight = torch.zeros((2, 2))
    wrapped(weight)
    assert torch.all(weight == 1.0)

    # 2. Bias vector (1D)
    bias = torch.ones(2)
    wrapped(bias)
    assert torch.all(bias == 0.0)

    # 3. Higher dimensional weight (e.g., Conv2D)
    conv_weight = torch.zeros((2, 2, 3, 3))
    wrapped(conv_weight)
    assert torch.all(conv_weight == 1.0)
