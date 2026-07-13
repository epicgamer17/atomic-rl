import pytest
import torch
import torch.nn as nn
from typing import Tuple
from unittest.mock import Mock, patch

from functional.plasticity import (
    compute_gradient_utility,
    compute_magnitude_utility,
    get_threshold_pruning_mask,
    get_proportional_pruning_mask,
    reset_optimizer_states_elementwise_,
    apply_selective_weight_reinitialization,
    init_cbp_state,
    get_cbp_replacement_mask,
    apply_continual_backprop,
)

pytestmark = pytest.mark.unit

# ==========================================
# Tests for Core Utility Computations
# ==========================================


def test_compute_gradient_utility_correctness():
    """Verify Taylor approximation formula: |w * g_w|."""
    weight = torch.tensor([-2.0, 3.0, 0.5])
    grad = torch.tensor([4.0, -1.0, 0.0])

    utilities = compute_gradient_utility(weight, grad)
    expected = torch.tensor([8.0, 3.0, 0.0])
    torch.testing.assert_close(utilities, expected)


def test_compute_gradient_utility_mismatch():
    """Verify that a shape mismatch between weights and gradients fails immediately."""
    weight = torch.randn(2, 3)
    grad = torch.randn(2, 2)
    with pytest.raises(AssertionError, match="shapes must match"):
        compute_gradient_utility(weight, grad)


def test_compute_magnitude_utility():
    """Verify magnitude utility calculation: |w|."""
    weight = torch.tensor([-5.5, 0.0, 2.1])
    utilities = compute_magnitude_utility(weight)
    torch.testing.assert_close(utilities, torch.tensor([5.5, 0.0, 2.1]))


# ==========================================
# Tests for Pruning Masks & Stochastic Sampling
# ==========================================


def test_get_threshold_pruning_mask():
    """Verify threshold mask targets elements less than or equal to k * mean(utilities)."""
    utilities = torch.tensor([1.0, 2.0, 3.0, 4.0])  # mean = 2.5
    # threshold = 1.0 * 2.5 = 2.5 -> Elements <= 2.5 are pruned ([True, True, False, False])
    mask = get_threshold_pruning_mask(utilities, threshold_factor=1.0)
    torch.testing.assert_close(
        mask, torch.tensor([True, True, False, False], dtype=torch.bool)
    )


def test_get_proportional_pruning_mask_stochastic_round_down():
    """Force the Bernoulli sample to return 0, ensuring it only prunes the floor integer count."""
    utilities = torch.tensor([10.0, 20.0, 30.0, 40.0, 50.0])  # 5 elements

    # 0.3 * 5 = 1.5 -> Floor is 1.
    # Mocking bernoulli to return 0.0 means total pruned should be exactly 1.
    with patch(
        "torch.bernoulli", return_value=torch.tensor(0.0, device=utilities.device)
    ):
        mask = get_proportional_pruning_mask(utilities, proportional_factor=0.3)

    expected = torch.tensor([True, False, False, False, False], dtype=torch.bool)
    torch.testing.assert_close(mask, expected)


def test_get_proportional_pruning_mask_stochastic_round_up():
    """Force the Bernoulli sample to return 1, ensuring it rounds up to prune an extra item."""
    utilities = torch.tensor([10.0, 20.0, 30.0, 40.0, 50.0])  # 5 elements

    # 0.3 * 5 = 1.5 -> Floor is 1.
    # Mocking bernoulli to return 1.0 means total pruned should be 1 + 1 = 2.
    with patch(
        "torch.bernoulli", return_value=torch.tensor(1.0, device=utilities.device)
    ):
        mask = get_proportional_pruning_mask(utilities, proportional_factor=0.3)

    expected = torch.tensor([True, True, False, False, False], dtype=torch.bool)
    torch.testing.assert_close(mask, expected)


# ==========================================
# Tests for Elementwise Optimizer Resetting
# ==========================================


def test_reset_optimizer_states_elementwise():
    """Verify that only the masked index values inside the optimizer state are zeroed out."""
    param = nn.Parameter(torch.ones(2, 2))
    optimizer = torch.optim.Adam([param], lr=0.1)

    # Artificially populate Adam's state metrics cache
    optimizer.state[param] = {
        "exp_avg": torch.ones(2, 2) * 5.0,
        "exp_avg_sq": torch.ones(2, 2) * 10.0,
        "step": torch.tensor(1.0),  # Should be preserved (not in keys_to_reset)
    }

    mask = torch.tensor([[True, False], [False, True]], dtype=torch.bool)
    reset_optimizer_states_elementwise_(optimizer, param, mask)

    state = optimizer.state[param]
    expected_exp_avg = torch.tensor([[0.0, 5.0], [5.0, 0.0]])
    expected_exp_avg_sq = torch.tensor([[0.0, 10.0], [10.0, 0.0]])

    torch.testing.assert_close(state["exp_avg"], expected_exp_avg)
    torch.testing.assert_close(state["exp_avg_sq"], expected_exp_avg_sq)
    assert state["step"] == 1.0


# ==========================================
# Tests for SWR High-Level Orchestration
# ==========================================


def test_swr_raises_runtime_error_if_grad_missing():
    """Verify that SWR fails fast if gradient utility is selected before a backward pass."""
    param = nn.Parameter(torch.ones(2))
    optimizer = torch.optim.SGD([param], lr=0.1)

    # param.grad is explicitly None here
    with pytest.raises(
        RuntimeError, match="Gradient utility requested, but param.grad is None"
    ):
        apply_selective_weight_reinitialization(
            [param], optimizer, init_fn=nn.init.zeros_, utility_type="gradient"
        )


def test_swr_orchestration_execution_flow():
    """Verify SWR samples replacement values, copies over mask, and returns applied configurations."""
    param = nn.Parameter(torch.tensor([1.0, 10.0, 1.0]))
    param.grad = torch.tensor([0.1, 0.1, 0.1])
    optimizer = torch.optim.SGD([param], lr=0.1)

    # Utility = |w * g_w| = [0.1, 1.0, 0.1]. Mean = 0.4.
    # Threshold factor k = 0.5 -> threshold = 0.2.
    # Indicies 0 and 2 are <= 0.2 and should be reinitialized.
    def mock_init(tensor):
        tensor.fill_(-99.0)

    masks_applied = apply_selective_weight_reinitialization(
        [param],
        optimizer,
        init_fn=mock_init,
        k=0.5,
        utility_type="gradient",
        prune_type="threshold",
    )

    expected_param = torch.tensor([-99.0, 10.0, -99.0])
    torch.testing.assert_close(param, expected_param)
    assert masks_applied[param].tolist() == [True, False, True]


# ==========================================
# Tests for CBP Core Mechanics
# ==========================================


def test_init_cbp_state():
    """Verify shape registration and metric zero-allocation properties of cbp tracking states."""
    weight = torch.zeros(5, 10)
    state = init_cbp_state(weight)

    assert state["ages"].shape == (5,)
    assert state["utilities"].shape == (5,)
    assert state["avg_activations"].shape == (5,)
    assert torch.all(state["ages"] == 0.0)


def test_get_cbp_replacement_mask_excludes_ineligible():
    """Verify that low-utility features are skipped if they have not reached maturity age."""
    # 4 tracking features total
    utilities = torch.tensor([0.1, 0.2, 0.05, 0.8])
    # Feature index 2 has the lowest absolute utility (0.05), but it is marked ineligible!
    eligible_mask = torch.tensor([True, True, False, True], dtype=torch.bool)

    # replacement_rate = 0.25 -> requests 1 item to replace.
    # It must select index 0 (utility 0.1) because it's the lowest eligible item.
    mask = get_cbp_replacement_mask(utilities, eligible_mask, replacement_rate=0.25)
    expected = torch.tensor([True, False, False, False], dtype=torch.bool)
    torch.testing.assert_close(mask, expected)


# ==========================================
# Tests for CBP High-Level Orchestration
# ==========================================


def test_apply_continual_backprop_pipeline():
    """Verify CBP trace accumulation, consumer bias shifting, and structural weight zero-masking."""
    # 1. Setup a minimal 2-layer sequential mapping footprint
    # Layer 1: [Out_features=2, In_features=3]
    weight = nn.Parameter(torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]))
    bias = nn.Parameter(torch.tensor([0.1, 0.2]))

    # Layer 2: [Next_out_features=1, Next_in_features=2]
    next_weight = nn.Parameter(torch.tensor([[2.0, 3.0]]))
    next_bias = nn.Parameter(torch.tensor([0.5]))

    # 2. Setup corresponding activations and tracking frames
    activations = [torch.tensor([[1.0, 1.0]])]  # Batch=1, Features=2

    cbp_states = {
        weight: {
            "ages": torch.tensor([100000.0, 100000.0]),  # Reached maturity
            "utilities": torch.tensor([1.0, 1.0]),
            "avg_activations": torch.tensor([1.0, 1.0]),
        }
    }

    # Instantiate optimizer context
    optimizer = torch.optim.SGD([weight, bias, next_weight, next_bias], lr=0.1)

    # Configure deterministic replacement mask generation
    # Force feature 0 to be zeroed out completely
    cbp_states[weight]["utilities"] = torch.tensor([0.0001, 100.0])

    def mock_init(tensor):
        tensor.fill_(-5.0)

    # 3. Execute CBP sequence pass
    apply_continual_backprop(
        layer_pairs=[(weight, bias, next_weight, next_bias)],
        activations=activations,
        cbp_states=cbp_states,
        optimizer=optimizer,
        init_fn=mock_init,
        eta=0.99,
        maturity_threshold=100,
        replacement_rate=0.5,  # Requests 50% of 2 features = 1 replacement
    )

    # --- Structural Correctness Validations ---

    # Validation A: Input weights for feature 0 should be reinitialized to mock value (-5.0)
    # Row index 0 contains inputs for feature 0
    torch.testing.assert_close(weight[0], torch.tensor([-5.0, -5.0, -5.0]))
    # Row index 1 should remain untouched
    torch.testing.assert_close(weight[1], torch.tensor([4.0, 5.0, 6.0]))

    # Validation B: Input bias for feature 0 must be zeroed out
    assert bias[0].item() == 0.0
    assert bias[1].item() == pytest.approx(0.2)

    # Validation C: Replaced feature's outgoing connection in next_weight must be zeroed out
    # Column index 0 represents the outgoing connection from feature 0
    assert next_weight[0, 0].item() == 0.0
    assert next_weight[0, 1].item() == pytest.approx(3.0)

    # Validation D: Bias of the consumer (next_bias) must absorb the removed unit's historical contribution
    # next_bias = old_next_bias + (next_weight[:, mask] * f_hat[mask])
    # next_bias = 0.5 + (2.0 * 1.0) = 2.5
    assert next_bias.item() == pytest.approx(2.5)

    # Validation E: CBP metrics tracking state variables for feature 0 must be reset to zero
    assert cbp_states[weight]["ages"][0].item() == 0.0
    assert cbp_states[weight]["utilities"][0].item() == 0.0
    assert cbp_states[weight]["avg_activations"][0].item() == 0.0
