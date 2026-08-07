import pytest
import torch
import torch.nn as nn
import torch.optim as optim
from functional.optimizer import (
    ObGD,
    apply_gradients,
    obgd_td_update_,
    obgd_update_,
)

pytestmark = pytest.mark.unit


def test_apply_gradients_basic():
    model = nn.Linear(2, 1)
    optimizer = optim.SGD(model.parameters(), lr=0.1)

    # Save original weight
    original_weight = model.weight.clone().detach()

    # Dummy input and loss
    x = torch.randn(1, 2)
    y = model(x)
    loss = (y - 1.0) ** 2

    apply_gradients(optimizer, loss)

    # Weight should have changed
    assert not torch.equal(model.weight, original_weight)

    # Verify zero_grad(set_to_none=True) happened
    # After optimizer.step(), gradients should still be None if set_to_none=True was used correctly before backward
    # Wait, loss.backward() populates .grad. optimizer.step() doesn't clear it.
    # But apply_gradients calls zero_grad(set_to_none=True) FIRST.
    # So if we call it twice, after the second zero_grad, .grad should be None.
    optimizer.zero_grad(set_to_none=True)
    assert model.weight.grad is None


def test_apply_gradients_clipping():
    model = nn.Linear(1, 1)
    optimizer = optim.SGD(model.parameters(), lr=0.1)

    # Force a large gradient
    with torch.no_grad():
        model.weight.fill_(1.0)

    x = torch.tensor([[100.0]])  # Large input
    y = model(x)
    loss = y**2  # Loss = (100 * 1)^2 = 10000. Grad = 2 * 100 * 100 = 20000.

    clip_norm = 1.0
    apply_gradients(optimizer, loss, model=model, clip_grad_norm=clip_norm)

    # Total norm should be clipped to 1.0
    total_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=clip_norm)
    assert total_norm <= clip_norm + 1e-6


def test_apply_gradients_no_model_clipping_error():
    model = nn.Linear(1, 1)
    optimizer = optim.SGD(model.parameters(), lr=0.1)
    loss = torch.tensor(1.0, requires_grad=True)

    with pytest.raises(
        AssertionError, match="Model must be provided for gradient clipping"
    ):
        apply_gradients(optimizer, loss, clip_grad_norm=1.0)


# ==========================================
# ObGD global step size (Algorithm 3 of the Stream RL paper)
# ==========================================
#
# ObGD computes a SINGLE step size from the L1 norm of the whole (concatenated)
# update vector across all parameters, then applies that same step size to every
# parameter. These tests guard against regressing to the old per-parameter-tensor
# normalization (one step size per tensor).


def _expected_step_size(lr, scaling_factor, error, total_norm):
    """Reference ObGD step size: lr / max(1, lr * kappa * delta_bar * ||z||_1)."""
    delta_bar = max(abs(error.item()), 1.0) if error is not None else 1.0
    M = lr * scaling_factor * delta_bar * total_norm
    return lr / max(1.0, M)


def test_obgd_td_step_single_global_step_size():
    """Two params with different trace L1 norms must share ONE global step size."""
    p1 = nn.Parameter(torch.ones(5))
    p2 = nn.Parameter(torch.ones(3))
    opt = ObGD([p1, p2], lr=1.0, scaling_factor=2.0)

    e1 = torch.full((5,), 20.0)  # ||e1||_1 = 100
    e2 = torch.full((3,), 10.0 / 3)  # ||e2||_1 = 10
    traces = {p1: e1, p2: e2}
    error = torch.tensor(1.0)

    p1_before, p2_before = p1.clone(), p2.clone()
    opt.td_step(error=error, traces=traces)

    # Single global norm across ALL params, so both share the same step size.
    z_sum = 110.0
    step = _expected_step_size(
        lr=1.0, scaling_factor=2.0, error=error, total_norm=z_sum
    )

    # Updates must be proportional to each tensor's share of the total trace norm
    # (0.45 and 0.045), NOT equalized per-tensor (which would be 0.5 and 0.5).
    torch.testing.assert_close((p1 - p1_before).abs().sum(), torch.tensor(step * 100.0))
    torch.testing.assert_close((p2 - p2_before).abs().sum(), torch.tensor(step * 10.0))

    # Sign convention: theta += step_size * delta * trace.
    expected_p1 = p1_before + step * error * e1
    expected_p2 = p2_before + step * error * e2
    torch.testing.assert_close(p1, expected_p1)
    torch.testing.assert_close(p2, expected_p2)


def test_obgd_step_single_global_step_size():
    """Supervised step: one global step size from the sum of ALL gradient norms."""
    p1 = nn.Parameter(torch.ones(4))
    p2 = nn.Parameter(torch.ones(2))
    opt = ObGD([p1, p2], lr=1.0, scaling_factor=2.0)

    g1 = torch.full((4,), 0.5)  # ||g1||_1 = 2
    g2 = torch.full((2,), 3.0)  # ||g2||_1 = 6
    p1.grad = g1
    p2.grad = g2

    p1_before, p2_before = p1.clone(), p2.clone()
    opt.step()

    z_sum = 8.0
    step = _expected_step_size(lr=1.0, scaling_factor=2.0, error=None, total_norm=z_sum)

    # Gradient descent: theta -= step_size * grad.
    torch.testing.assert_close(p1, p1_before - step * g1)
    torch.testing.assert_close(p2, p2_before - step * g2)


def test_obgd_td_update_formula():
    """obgd_td_update_ applies theta += (lr / max(1, M)) * delta * trace."""
    theta = nn.Parameter(torch.zeros(3))
    trace = torch.tensor([0.5, 1.0, 2.0])
    error = torch.tensor(-2.0)
    lr, kappa = 1.0, 3.0
    total_norm = 3.5  # hypothetical global L1 norm across the whole network

    theta_before = theta.clone()
    obgd_td_update_(
        theta=theta,
        error=error,
        trace=trace,
        lr=lr,
        scaling_factor=kappa,
        total_norm=total_norm,
    )

    step = _expected_step_size(lr, kappa, error, total_norm)
    torch.testing.assert_close(theta, theta_before + step * error * trace)


def test_obgd_update_formula():
    """obgd_update_ applies theta -= (lr / max(1, M)) * grad."""
    theta = nn.Parameter(torch.zeros(3))
    grad = torch.tensor([1.0, -2.0, 3.0])
    lr, kappa = 0.5, 2.0
    total_norm = 6.0

    theta_before = theta.clone()
    obgd_update_(
        theta=theta,
        grad=grad,
        lr=lr,
        scaling_factor=kappa,
        total_norm=total_norm,
    )

    M = lr * kappa * total_norm
    step = lr / max(1.0, M)
    torch.testing.assert_close(theta, theta_before - step * grad)


def test_obgd_helpers_require_total_norm():
    """The per-tensor fallback was removed: total_norm is a required argument."""
    theta = nn.Parameter(torch.zeros(3))
    grad = torch.ones(3)
    error = torch.tensor(1.0)

    with pytest.raises(TypeError):
        obgd_update_(theta=theta, grad=grad, lr=1.0)
    with pytest.raises(TypeError):
        obgd_td_update_(theta=theta, error=error, trace=grad, lr=1.0)


def test_obgd_td_step_none_traces_skipped():
    """None traces are skipped and excluded from the global norm."""
    p1 = nn.Parameter(torch.ones(2))
    p2 = nn.Parameter(torch.ones(2))
    opt = ObGD([p1, p2], lr=1.0, scaling_factor=1.0)

    e1 = torch.full((2,), 1.0)  # ||e1||_1 = 2
    traces = {p1: e1, p2: None}
    error = torch.tensor(1.0)

    p1_before, p2_before = p1.clone(), p2.clone()
    opt.td_step(error=error, traces=traces)

    # Only p1's trace counts toward the global norm.
    step = _expected_step_size(lr=1.0, scaling_factor=1.0, error=error, total_norm=2.0)
    torch.testing.assert_close(p1, p1_before + step * error * e1)
    torch.testing.assert_close(p2, p2_before)  # untouched


def test_obgd_td_step_list_traces():
    """The list-input path also uses a single global step size."""
    p1 = nn.Parameter(torch.ones(2))
    p2 = nn.Parameter(torch.ones(2))
    opt = ObGD([p1, p2], lr=1.0, scaling_factor=2.0)

    e1 = torch.full((2,), 3.0)  # ||e1||_1 = 6
    e2 = torch.full((2,), 0.5)  # ||e2||_1 = 1
    error = torch.tensor(1.0)

    p1_before, p2_before = p1.clone(), p2.clone()
    opt.td_step(error=error, traces=[e1, e2])

    step = _expected_step_size(lr=1.0, scaling_factor=2.0, error=error, total_norm=7.0)
    torch.testing.assert_close(p1, p1_before + step * error * e1)
    torch.testing.assert_close(p2, p2_before + step * error * e2)


def test_obgd_td_step_missing_param_raises():
    """A param missing from the mapping raises KeyError (fail fast)."""
    p1 = nn.Parameter(torch.ones(2))
    p2 = nn.Parameter(torch.ones(2))
    opt = ObGD([p1, p2], lr=1.0)

    with pytest.raises(KeyError, match="Parameter trace not found"):
        opt.td_step(error=torch.tensor(1.0), traces={p1: torch.ones(2)})


def test_obgd_step_no_grads_noop():
    """step() with no gradients is a safe no-op."""
    p1 = nn.Parameter(torch.ones(2))
    p2 = nn.Parameter(torch.ones(2))
    opt = ObGD([p1, p2], lr=1.0)

    p1_before, p2_before = p1.clone(), p2.clone()
    opt.step()
    torch.testing.assert_close(p1, p1_before)
    torch.testing.assert_close(p2, p2_before)


# ==========================================
# AdaptiveObGD (Algorithm 11) Tests
# ==========================================


def test_adaptive_obgd_td_step_basic():
    """Verify second-moment accumulation and adaptive step calculation in AdaptiveObGD.td_step."""
    from functional.optimizer import AdaptiveObGD

    p = nn.Parameter(torch.tensor([1.0, 2.0]))
    opt = AdaptiveObGD([p], lr=1.0, scaling_factor=1.0, beta=0.9, eps=1e-8)

    trace = torch.tensor([0.5, 1.0])
    error = torch.tensor(2.0)

    p_before = p.clone()
    opt.td_step(error=error, traces={p: trace})

    # 1. Check second moment v: (1 - 0.9) * (error * trace)^2 = 0.1 * (2.0 * [0.5, 1.0])^2 = 0.1 * [1.0, 4.0] = [0.1, 0.4]
    v_expected = torch.tensor([0.1, 0.4])
    torch.testing.assert_close(opt.state[p]["v"], v_expected)

    # 2. Adjusted trace: trace / (sqrt(v) + eps)
    adj_trace = trace / (torch.sqrt(v_expected) + 1e-8)
    norm = torch.sum(torch.abs(adj_trace))  # total_norm

    # 3. Step calculation
    M = 1.0 * 1.0 * 2.0 * norm  # lr * kappa * |error| * norm
    step = 1.0 / max(1.0, M.item())

    expected_p = p_before + step * error * adj_trace
    torch.testing.assert_close(p, expected_p)


def test_adaptive_obgd_step_supervised():
    """Verify supervised AdaptiveObGD.step updating parameters via second moments."""
    from functional.optimizer import AdaptiveObGD

    p = nn.Parameter(torch.tensor([3.0, -1.0]))
    opt = AdaptiveObGD([p], lr=0.5, scaling_factor=2.0, beta=0.9, eps=1e-8)

    g = torch.tensor([1.0, -2.0])
    p.grad = g.clone()

    p_before = p.clone()
    opt.step()

    # Second moment v: 0.1 * [1.0, 4.0] = [0.1, 0.4]
    v_expected = torch.tensor([0.1, 0.4])
    torch.testing.assert_close(opt.state[p]["v"], v_expected)

    adj_grad = g / (torch.sqrt(v_expected) + 1e-8)
    norm = torch.sum(torch.abs(adj_grad))
    M = 0.5 * 2.0 * norm
    step = 0.5 / max(1.0, M.item())

    expected_p = p_before - step * adj_grad
    torch.testing.assert_close(p, expected_p)
