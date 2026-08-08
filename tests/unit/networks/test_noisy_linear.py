import pytest
import torch
import torch.nn as nn
from atomic_rl.networks.noisy_linear import NoisyLinear

pytestmark = pytest.mark.unit


def test_noisy_linear_init():
    in_features, out_features = 4, 8
    layer = NoisyLinear(in_features, out_features)

    assert layer.weight_mu.shape == (out_features, in_features)
    assert layer.weight_sigma.shape == (out_features, in_features)
    assert layer.bias_mu.shape == (out_features,)
    assert layer.bias_sigma.shape == (out_features,)

    # Buffers
    assert layer.weight_epsilon.shape == (out_features, in_features)
    assert layer.bias_epsilon.shape == (out_features,)


def test_noisy_linear_forward_eval():
    layer = NoisyLinear(4, 2)
    layer.eval()  # Switch to eval mode

    x = torch.randn(1, 4)

    # In eval mode, noise should be disabled
    out1 = layer(x)
    layer.reset_noise()  # Change noise buffers
    out2 = layer(x)

    # Output should be identical despite noise reset
    torch.testing.assert_close(out1, out2)

    # Manual verification: out should match linear(x, weight_mu, bias_mu)
    expected = torch.nn.functional.linear(x, layer.weight_mu, layer.bias_mu)
    torch.testing.assert_close(out1, expected)


def test_noisy_linear_forward_train():
    torch.manual_seed(42)
    layer = NoisyLinear(4, 2)
    layer.train()  # Switch to train mode

    x = torch.ones(1, 4)

    # 1. First forward
    out1 = layer(x)

    # 2. Reset noise and forward again
    layer.reset_noise()
    out2 = layer(x)

    # In train mode, output should change because epsilon changed
    assert not torch.allclose(out1, out2)


def test_noisy_linear_reset_noise_changes_buffers():
    layer = NoisyLinear(4, 2)

    eps_w_old = layer.weight_epsilon.clone()
    eps_b_old = layer.bias_epsilon.clone()

    layer.reset_noise()

    assert not torch.allclose(layer.weight_epsilon, eps_w_old)
    assert not torch.allclose(layer.bias_epsilon, eps_b_old)
