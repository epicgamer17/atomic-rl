import pytest
from functional.initialization import (
    _allocate_tensordict,
    make_gnt_init,
    layer_init,
    sparse_init_weight_,
    set_seed,
    make_sparse_init,
)
import torch.nn as nn
import torch
import random
import numpy as np

pytestmark = pytest.mark.unit


def test_layer_init():
    """Test orthogonal initialization of layers."""

    layer = nn.Linear(10, 5)
    std = 0.5
    bias_const = 0.1

    initialized_layer = layer_init(layer, std=std, bias_const=bias_const)

    # Check bias initialization
    torch.testing.assert_close(
        initialized_layer.bias, torch.full_like(layer.bias, bias_const)
    )

    # Orthogonal initialization is harder to check exactly,
    # but we can check if it's generally working (not all zeros/ones)
    assert not torch.all(initialized_layer.weight == 0.0)
    assert not torch.all(initialized_layer.weight == 1.0)

    # For a square matrix, W * W^T = std^2 * I
    # For non-square, we can check a smaller property or just that it changed
    # nn.init.orthogonal_ is a standard pytorch function, so we mostly test our wrapper
    assert initialized_layer is layer


def test_allocate_tensordict():

    shapes = {"obs": (4, 84, 84), "action": ()}
    batch_size = [2, 3]
    td = _allocate_tensordict(shapes, batch_size)

    assert list(td.shape) == [2, 3]
    assert td["obs"].shape == (2, 3, 4, 84, 84)
    assert td["obs"].dtype == torch.float32
    assert td["action"].shape == (2, 3)
    assert td["action"].dtype == torch.long


def test_make_gnt_init():

    # Mock init_fn that sets weights to 1.0
    def mock_init(tensor):
        nn.init.constant_(tensor, 1.0)

    wrapped = make_gnt_init(mock_init)

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


# ==========================================
# Tests for set_seed
# ==========================================


def test_set_seed_reproducibility():
    """Verify that setting the seed forces identical random generation sequences."""
    seed = 42

    # Run 1
    set_seed(seed)
    py_rand_1 = random.random()
    np_rand_1 = np.random.rand()
    torch_rand_1 = torch.rand(1).item()

    # Run 2
    set_seed(seed)
    py_rand_2 = random.random()
    np_rand_2 = np.random.rand()
    torch_rand_2 = torch.rand(1).item()

    assert py_rand_1 == py_rand_2
    assert np_rand_1 == np_rand_2
    assert torch_rand_1 == torch_rand_2


# ==========================================
# Tests for sparse_init_weight_
# ==========================================


def test_sparse_init_weight_2d():
    """Verify that exactly the requested fraction of columns (fan_in) is zeroed out uniformly."""
    # Matrix shape [out_features=5, in_features=10] -> fan_in = 10
    tensor = torch.ones(5, 10)
    sparsity = 0.4  # 40% of 10 = 4 columns should be zeroed out

    sparse_init_weight_(tensor, sparsity)

    # Columns are either fully zeroed out or fully untouched (all 1s)
    # Check each column sum
    col_sums = tensor.sum(dim=0)

    zero_cols = (col_sums == 0.0).sum().item()
    active_cols = (col_sums == 5.0).sum().item()

    assert zero_cols == 4
    assert active_cols == 6


def test_sparse_init_weight_high_dim():
    """Verify sparsity behavior on higher-dimensional tensors (e.g., Conv2D weights)."""
    # Shape [out_channels=2, in_channels=3, kernel_h=2, kernel_w=2]
    # fan_in = 3 * 2 * 2 = 12 elements
    tensor = torch.ones(2, 3, 2, 2)
    sparsity = 0.5  # 50% of 12 = 6 structural elements zeroed out

    sparse_init_weight_(tensor, sparsity)

    # Reshape to check output neuron profiles
    flat_view = tensor.view(2, -1)

    # Elements must be identically zeroed across all output channels
    channel_0_zeros = (flat_view[0] == 0.0).sum().item()
    channel_1_zeros = (flat_view[1] == 0.0).sum().item()

    assert channel_0_zeros == 6
    assert channel_1_zeros == 6
    # Ensure structural zero matching across channels
    torch.testing.assert_close(flat_view[0] == 0.0, flat_view[1] == 0.0)


def test_sparse_init_weight_edge_cases():
    """Verify boundary thresholds for zero sparsity and 1D tensors."""
    # Case 1: Sparsity evaluates to 0 elements zeroed out
    tensor_small = torch.ones(2, 5)
    sparse_init_weight_(tensor_small, sparsity=0.05)  # int(0.05 * 5) = 0
    assert torch.all(tensor_small == 1.0)

    # Case 2: 1D Tensor edge-case protection (no-op safety validation)
    tensor_1d = torch.ones(5)
    sparse_init_weight_(tensor_1d, sparsity=0.5)
    assert torch.all(tensor_1d == 1.0)  # Left untouched by conditional checks


# ==========================================
# Tests for make_sparse_init
# ==========================================


def test_make_sparse_init_factory():
    """Verify that the factory routes base-init and sparsity logic depending on dimension."""

    def mock_base_init(t: torch.Tensor) -> None:
        nn.init.constant_(t, 5.0)

    # Configure factory function
    configured_init = make_sparse_init(mock_base_init, sparsity=0.5)

    # 1. Evaluate 2D Weight routing (base init runs, then sparse masking executes)
    weight_matrix = torch.zeros(4, 6)  # fan_in = 6 -> 3 elements zeroed
    configured_init(weight_matrix)

    col_sums = weight_matrix.sum(dim=0)
    zero_cols = (col_sums == 0.0).sum().item()
    initialized_cols = (col_sums == (4 * 5.0)).sum().item()

    assert zero_cols == 3
    assert initialized_cols == 3

    # 2. Evaluate 1D Bias routing (forces zero initialization directly)
    bias_vector = torch.ones(4)
    configured_init(bias_vector)
    torch.testing.assert_close(bias_vector, torch.zeros(4))
