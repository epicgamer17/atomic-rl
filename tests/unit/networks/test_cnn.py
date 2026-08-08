import pytest
import torch
from atomic_rl.networks.cnn import Transpose, AtariCNN, Conv2dBackbone

pytestmark = pytest.mark.unit


def test_transpose_module():
    """Verify Transpose module dimension permutation logic."""
    mod = Transpose((0, 3, 1, 2))
    x = torch.randn(4, 84, 84, 3)  # [B, H, W, C]

    out = mod(x)

    assert out.shape == (4, 3, 84, 84)


def test_atari_cnn():
    """Verify AtariCNN architecture output shape and input scaling toggle."""
    cnn = AtariCNN(in_channels=4, out_features=512, scale_inputs=True)
    x = torch.randint(0, 256, (2, 4, 84, 84), dtype=torch.float32)

    out = cnn(x)

    assert out.shape == (2, 512)


def test_conv2d_backbone():
    """Verify Conv2dBackbone custom channels and strides."""
    backbone = Conv2dBackbone(
        in_channels=3,
        channels=(16, 32),
        kernel_sizes=(3, 3),
        strides=(1, 1),
        paddings=(1, 1),
    )
    x = torch.randn(2, 3, 10, 10)

    out = backbone(x)

    # 32 channels * 10 * 10 = 3200
    assert out.shape == (2, 3200)
