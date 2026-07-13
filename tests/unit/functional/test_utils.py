import pytest
import torch
import numpy as np
from functional.utils import (
    ema_update,
    ema_update_,
    standardize_tensor,
    scale_tensor_by_std,
    to_tensor,
    to_numpy_action,
    update_welford_stats,
    normalize_features,
    add_dirichlet_noise,
    compute_tile_coding_features,
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


def test_ema_update_inplace():
    old = torch.tensor([1.0, 2.0])
    new = torch.tensor([3.0, 4.0])
    alpha = 0.1

    # (1-0.1)*1.0 + 0.1*3.0 = 0.9 + 0.3 = 1.2
    # (1-0.1)*2.0 + 0.1*4.0 = 1.8 + 0.4 = 2.2
    expected = torch.tensor([1.2, 2.2])

    res = ema_update_(old, new, alpha)
    torch.testing.assert_close(res, expected)
    torch.testing.assert_close(old, expected)  # check inplace


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

    # ema_update_
    with pytest.raises(AssertionError, match="EMA shape mismatch"):
        ema_update_(torch.randn(2), torch.randn(3), alpha=0.1)


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


# ==========================================
# Tests for to_tensor and to_numpy_action
# ==========================================


def test_to_tensor():
    """Verify conversion of python primitives and numpy structures to PyTorch tensors."""
    # Primitive int conversion
    t1 = to_tensor(5, dtype=torch.long)
    assert t1.dtype == torch.long
    assert t1.item() == 5

    # Numpy array conversion
    arr = np.array([1.0, 2.0], dtype=np.float64)
    t2 = to_tensor(arr, dtype=torch.float32)
    assert t2.dtype == torch.float32
    torch.testing.assert_close(t2, torch.tensor([1.0, 2.0]))


def test_to_numpy_action_discrete_flattening():
    """Verify discrete action vectors [B, 1] flatten out to 1D int32 numpy footprints."""
    # Discrete actions with shape [BatchSize=3, 1] -> standard for categorical policies
    act_discrete = torch.tensor([[0], [2], [1]], dtype=torch.long)
    res_discrete = to_numpy_action(act_discrete)

    assert res_discrete.dtype == np.int32
    assert res_discrete.ndim == 1
    np.testing.assert_array_equal(res_discrete, np.array([0, 2, 1], dtype=np.int32))


def test_to_numpy_action_continuous_preservation():
    """Verify continuous action layouts retain their multi-dimensional shape and precision properties."""
    # Continuous actions with shape [BatchSize=2, ActionDim=2]
    act_continuous = torch.tensor([[0.5, -0.5], [1.0, 0.0]], dtype=torch.float32)
    res_continuous = to_numpy_action(act_continuous)

    assert res_continuous.dtype == np.float32
    assert res_continuous.shape == (2, 2)
    np.testing.assert_array_equal(res_continuous, act_continuous.cpu().numpy())


# ==========================================
# Tests for Welford Online Statistics
# ==========================================


def test_update_welford_stats_and_normalization():
    """Verify running mean/variance updating math and downstream standardization transforms."""
    # Initialize zero running statistics for a 2-feature vector space
    mean = torch.zeros(2)
    var = torch.zeros(2)
    count = torch.tensor(0.0)

    # Pass a batch containing 2 instances
    batch = torch.tensor([[1.0, 10.0], [3.0, 20.0]])

    mean, var, count = update_welford_stats(mean, var, count, batch)

    # Math Verification:
    # count = 0 + 2 = 2
    # mean = [ (1+3)/2, (10+20)/2 ] = [2.0, 15.0]
    # var (M2 aggregate sum of squares) = (1-2)^2 + (3-2)^2 = 2.0 for feature 0
    #                                  = (10-15)^2 + (20-15)^2 = 50.0 for feature 1
    assert count.item() == 2.0
    torch.testing.assert_close(mean, torch.tensor([2.0, 15.0]))
    torch.testing.assert_close(var, torch.tensor([2.0, 50.0]))

    # Test feature normalization based on these running metrics
    features = torch.tensor([[2.0, 15.0]])
    # unbiased_var = var / (count - 1) = [2.0 / 1.0, 50.0 / 1.0] = [2.0, 50.0]
    # normalized = (features - mean) / sqrt(unbiased_var) -> exactly 0 since inputs match the mean
    normalized = normalize_features(features, mean, var, count)
    torch.testing.assert_close(normalized, torch.zeros(1, 2))


def test_update_welford_stats_assertion():
    """Verify that Welford stats update fails fast if the input batch is not 2D."""
    mean = torch.zeros(2)
    var = torch.zeros(2)
    count = torch.tensor(0.0)

    # Passing a unbatched 1D tensor should trigger the layout validation check
    bad_batch = torch.tensor([1.0, 2.0])
    with pytest.raises(AssertionError, match="Expected \\[Batch, Features\\]"):
        update_welford_stats(mean, var, count, bad_batch)


# ==========================================
# Tests for Action-Masked Dirichlet Noise
# ==========================================


def test_add_dirichlet_noise_with_action_mask():
    """Verify that Dirichlet noise is zeroed out for illegal masked actions and re-normalized."""
    probs = torch.tensor([[0.5, 0.0, 0.5]])
    # Action index 1 is blocked out by the environment mask
    mask = torch.tensor([[1, 0, 1]], dtype=torch.bool)

    noisy_probs = add_dirichlet_noise(probs, epsilon=0.5, alpha=0.3, mask=mask)

    assert noisy_probs.shape == probs.shape
    # Masked index must remain exactly 0.0
    assert noisy_probs[0, 1].item() == 0.0
    # Total probability must remain perfectly valid and normalized to 1.0
    torch.testing.assert_close(noisy_probs.sum(dim=-1), torch.ones(1))


# ==========================================
# Tests for Sparse Tile Coding Features
# ==========================================


def test_compute_tile_coding_features_structure():
    """Verify output boundaries and action indexing layouts for sparse tile coding blocks."""
    state = np.array([-0.5, 0.02])
    action = 1
    num_actions = 3
    num_tilings = 4
    tiles_per_tiling = 5

    phi = compute_tile_coding_features(
        state=state,
        action=action,
        num_actions=num_actions,
        num_tilings=num_tilings,
        tiles_per_tiling=tiles_per_tiling,
    )

    # Math Check on Sizes:
    # features_per_action = num_tilings * ((tiles_per_tiling + 1) ** 2) = 4 * (6 ** 2) = 144
    # total_features = 144 * 3 actions = 432
    assert phi.shape == (432,)
    assert phi.dtype == torch.float64

    # Exactly 1 active bit per tiling layer must equal 1.0
    assert phi.sum().item() == float(num_tilings)

    # Since action=1 was selected, active indices must fall entirely inside the action=1 chunk [144, 288)
    active_indices = torch.where(phi == 1.0)[0]
    for idx in active_indices:
        assert 144 <= idx.item() < 288
