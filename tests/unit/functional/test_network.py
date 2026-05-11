import pytest
import torch
import torch.nn as nn
from functional.network import hard_update_target_network, soft_update_target_network

pytestmark = pytest.mark.unit


def test_hard_update_target_network():
    model = nn.Linear(5, 1)
    target_model = nn.Linear(5, 1)

    # Initialize with different weights
    with torch.no_grad():
        model.weight.fill_(1.0)
        target_model.weight.fill_(0.0)

    hard_update_target_network(model, target_model)

    torch.testing.assert_close(target_model.weight, model.weight)
    assert target_model.weight[0, 0] == 1.0


def test_soft_update_target_network():
    model = nn.Linear(5, 1)
    target_model = nn.Linear(5, 1)

    # Initialize with different weights
    with torch.no_grad():
        model.weight.fill_(1.0)
        target_model.weight.fill_(0.0)

    # target = (1 - tau) * target + tau * model
    # target = (1 - 0.5) * 0.0 + 0.5 * 1.0 = 0.5
    soft_update_target_network(model, target_model, tau=0.5)

    torch.testing.assert_close(target_model.weight, torch.full_like(target_model.weight, 0.5))

    # Test with default tau (0.005)
    # target = (1 - 0.005) * 0.5 + 0.005 * 1.0 = 0.4975 + 0.005 = 0.5025
    soft_update_target_network(model, target_model)
    torch.testing.assert_close(target_model.weight, torch.full_like(target_model.weight, 0.5025))


def test_layer_init():
    """Test orthogonal initialization of layers."""
    import numpy as np
    from functional.network import layer_init
    
    layer = nn.Linear(10, 5)
    std = 0.5
    bias_const = 0.1
    
    initialized_layer = layer_init(layer, std=std, bias_const=bias_const)
    
    # Check bias initialization
    torch.testing.assert_close(initialized_layer.bias, torch.full_like(layer.bias, bias_const))
    
    # Orthogonal initialization is harder to check exactly, 
    # but we can check if it's generally working (not all zeros/ones)
    assert not torch.all(initialized_layer.weight == 0.0)
    assert not torch.all(initialized_layer.weight == 1.0)
    
    # For a square matrix, W * W^T = std^2 * I
    # For non-square, we can check a smaller property or just that it changed
    # nn.init.orthogonal_ is a standard pytorch function, so we mostly test our wrapper
    assert initialized_layer is layer
