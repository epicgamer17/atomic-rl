import pytest
import torch
import torch.nn as nn
from atomic_rl.networks.resnet import (
    ResNetBlock2d,
    ResNetBlock1d,
    ResNetBackbone,
    ResNetBlock,
)

pytestmark = pytest.mark.unit


def test_resnet_block_2d_shapes_and_gradient():
    """Verify standard post-activation 2D residual block forward pass, dimensions, and gradients."""
    block = ResNetBlock2d(
        in_channels=16, out_channels=16, stride=1, pre_activation=False
    )
    x = torch.randn(4, 16, 8, 8, requires_grad=True)

    out = block(x)

    assert out.shape == (4, 16, 8, 8)
    loss = out.sum()
    loss.backward()
    assert x.grad is not None
    assert x.grad.shape == (4, 16, 8, 8)


def test_resnet_block_2d_projection_shortcut():
    """Verify that dimension or stride mismatch correctly triggers projection shortcut."""
    block = ResNetBlock2d(
        in_channels=16, out_channels=32, stride=2, pre_activation=False
    )
    x = torch.randn(2, 16, 16, 16)

    out = block(x)

    # Spatial dimensions halved due to stride=2, channels doubled to 32
    assert out.shape == (2, 32, 8, 8)


def test_resnet_block_2d_pre_activation():
    """Verify pre-activation ResNetBlock2d functionality."""
    block = ResNetBlock2d(in_channels=8, out_channels=8, pre_activation=True)
    x = torch.randn(2, 8, 4, 4)

    out = block(x)

    assert out.shape == (2, 8, 4, 4)


def test_resnet_block_1d_shapes():
    """Verify 1D residual block output dimensions for sequence signals."""
    block = ResNetBlock1d(in_channels=8, out_channels=16, stride=2)
    x = torch.randn(4, 8, 32)  # [Batch, Channels, Length]

    out = block(x)

    assert out.shape == (4, 16, 16)


def test_resnet_backbone():
    """Verify full ResNetBackbone feature extractor."""
    backbone2d = ResNetBackbone(in_channels=3, num_filters=16, num_blocks=2, dim=2)
    x2d = torch.randn(2, 3, 9, 9)
    out2d = backbone2d(x2d)
    assert out2d.shape == (2, 16, 9, 9)

    backbone1d = ResNetBackbone(in_channels=4, num_filters=8, num_blocks=3, dim=1)
    x1d = torch.randn(2, 4, 20)
    out1d = backbone1d(x1d)
    assert out1d.shape == (2, 8, 20)
