import pytest
import torch
import math
from torch.nn import Parameter

from functional.meta_optimization import (
    update_idbd_rates_,
    update_k1_rates_,
    update_k2_rates_,
    update_autostep_v_normalizer_,
    update_autostep_m_cap_,
    update_autostep_rates_,
    IDBD,
    Autostep,
    K1,
    K2,
)

pytestmark = pytest.mark.unit


# ==========================================
# Tests for IDBD Functional Core
# ==========================================


def test_idbd_rates_mathematical_direction():
    """
    Verify Sutton's core premise:
    Positive correlation between current and past updates increases beta.
    Negative correlation pushes beta down.
    """
    # Initialize log learning rates at 0.0 -> alpha = exp(0) = 1.0
    betas = torch.tensor([0.0, 0.0])
    # Feature 0 has positive trace history, Feature 1 has negative trace history
    h = torch.tensor([1.0, -1.0])
    inputs = torch.tensor([1.0, 1.0])
    error = torch.tensor(1.0)
    meta_lr = 0.1

    alphas = update_idbd_rates_(
        betas=betas, h=h, inputs=inputs, error=error, meta_lr=meta_lr
    )

    # delta_beta = inputs * h * meta_lr * error = [0.1, -0.1]
    # New betas should be [0.1, -0.1]
    torch.testing.assert_close(betas, torch.tensor([0.1, -0.1]))
    torch.testing.assert_close(alphas, torch.exp(torch.tensor([0.1, -0.1])))


def test_idbd_shape_assertions():
    """Verify that fail-fast checks catch mismatched parameters and errors."""
    betas = torch.tensor([0.0, 0.0])
    h = torch.tensor([0.0])  # Mismatched trace size
    inputs = torch.tensor([1.0, 1.0])
    error = torch.tensor(1.0)

    with pytest.raises(AssertionError, match="Betas and traces must match"):
        update_idbd_rates_(betas, h, inputs, error, meta_lr=0.01)

    # Batched mismatched error footprint checking
    h_correct = torch.tensor([0.0, 0.0])
    inputs_batched = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    error_flat = torch.tensor([1.0, 1.0])  # Must be [Batch, 1], not 1D [Batch]

    with pytest.raises(AssertionError, match="Batched error must have shape"):
        update_idbd_rates_(betas, h_correct, inputs_batched, error_flat, meta_lr=0.01)


# ==========================================
# Tests for K1 & K2 (Kalman Filter Approximations)
# ==========================================


def test_k1_normalization_and_gain():
    """Verify K1 scales the gain vector based on pseudo-covariance normalizer."""
    betas = torch.tensor([0.0, 0.0])  # p_hat = [1.0, 1.0]
    h = torch.tensor([0.0, 0.0])
    inputs = torch.tensor([2.0, 3.0])
    error = torch.tensor(1.0)
    r_hat = 1.0

    k_gain = update_k1_rates_(betas, h, inputs, error, meta_lr=0.01, r_hat=r_hat)

    # d_t = r_hat + sum(p_hat * inputs^2) = 1.0 + (1.0 * 4.0 + 1.0 * 9.0) = 14.0
    # k_gain = (p_hat * inputs) / d_t = [2.0 / 14.0, 3.0 / 14.0]
    expected_gain = torch.tensor([2.0 / 14.0, 3.0 / 14.0])
    torch.testing.assert_close(k_gain, expected_gain)


def test_k2_traceless_regression():
    """Verify K2 safely updates log steps using regression errors without trace matrices."""
    betas = torch.tensor([0.0])  # p_hat_old = 1.0
    inputs = torch.tensor([2.0])  # inputs^4 = 16.0
    error = torch.tensor([3.0])  # error^2 = 9.0

    # regression_error = error^2 - r_hat - (p_hat * inputs^2) = 9.0 - 1.0 - (1.0 * 4.0) = 4.0
    # beta_normalizer = 1.0 + inputs^4 = 1.0 + 16.0 = 17.0
    # delta_beta = inputs^2 * (meta_lr / beta_normalizer) * regression_error
    #            = 4.0 * (0.1 / 17.0) * 4.0 = 1.6 / 17.0 = 0.0941176
    meta_lr = 0.1
    r_hat = 1.0

    update_k2_rates_(betas, inputs, error, meta_lr=meta_lr, r_hat=r_hat)
    expected_beta = torch.tensor([0.0941176])
    torch.testing.assert_close(betas, expected_beta, rtol=1e-5, atol=1e-5)


# ==========================================
# Tests for Autostep Normalizers & Logic
# ==========================================


def test_autostep_v_normalizer_numerical_stability():
    """Verify running normalizer maximum doesn't divide or decay to invalid states when zeroed."""
    v = torch.tensor([0.0])
    abs_meta_grad = torch.tensor([5.0])
    alphas = torch.tensor([0.1])
    inputs = torch.tensor([2.0])

    update_autostep_v_normalizer_(v, abs_meta_grad, alphas, inputs, tau=10.0)
    # The output targets an element-wise torch.maximum operation with the running step value.
    # Therefore, v must be at least as large as the instant absolute meta-gradient.
    assert v.item() >= 5.0


def test_autostep_m_cap_overshoot_prevention():
    """Verify that Autostep caps the total effective step size to prevent explosion."""
    # Setup betas to yield extremely large alphas: exp(5.0) = 148.413
    betas = torch.tensor([5.0])
    inputs = torch.tensor([2.0])  # inputs^2 = 4.0

    # effective_step_size = 148.413 * 4.0 = 593.652
    # m = max(593.652, 1.0) = 593.652
    # Expected alpha returned = 148.413 / 593.652 = 0.25 (which is exactly 1 / inputs^2)
    alphas = update_autostep_m_cap_(betas, inputs)

    torch.testing.assert_close(alphas, torch.tensor([0.25]))
    # Beta should be shifted backwards by log(m)
    torch.testing.assert_close(betas, torch.tensor([5.0 - math.log(593.652)]))


# ==========================================
# Tests for PyTorch Optimizer Wrappers
# ==========================================


@pytest.mark.parametrize("optimizer_cls", [IDBD, Autostep, K1, K2])
def test_optimizer_lazy_initialization_and_step(optimizer_cls):
    """Verify that all custom meta-gradient optimizers handle lazy-state mapping and alter weights."""
    weight = Parameter(torch.tensor([1.0, 2.0]))
    inputs = torch.tensor([0.5, 0.5])
    error = torch.tensor(0.2)

    optimizer = optimizer_cls([weight], initial_lr=0.05, meta_lr=0.01)

    # Before step, state cache should be empty
    assert len(optimizer.state[weight]) == 0

    # Execute step update
    optimizer.step(inputs, error)

    # Confirm lazy allocations are complete
    state = optimizer.state[weight]
    assert "beta" in state
    assert state["beta"].shape == weight.shape

    # Check that weights moved away from their starting parameters
    assert not torch.equal(weight, torch.tensor([1.0, 2.0]))


def test_optimizer_linearity_mismatch_assertion():
    """Verify that optimizers throw value errors if the parameter configuration breaks linearity assumptions."""
    weight = Parameter(torch.zeros(1, 5))  # expects 5 input features
    inputs = torch.tensor([1.0, 2.0, 3.0])  # passes only 3 features
    error = torch.tensor(1.0)

    optimizer = IDBD([weight])
    with pytest.raises(ValueError, match="Linearity constraint violated"):
        optimizer.step(inputs, error)
