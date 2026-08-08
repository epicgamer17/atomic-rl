"""
Neural network components layer for reinforcement learning.

Exports common network building blocks (ResNets, CNN backbones, Noisy layers, Transformer skeletons)
while keeping algorithm orchestration in example scripts.
"""

from atomic_rl.networks.noisy_linear import NoisyLinear
from atomic_rl.networks.resnet import (
    ResNetBlock2d,
    ResNetBlock1d,
    ResNetBackbone,
    ResNetBlock,
)
from atomic_rl.networks.cnn import Transpose, AtariCNN, Conv2dBackbone
from atomic_rl.networks.transformer import (
    PositionalEncoding,
    MultiHeadSelfAttention,
    TransformerBlock,
    TransformerEncoder,
)

__all__ = [
    "NoisyLinear",
    "ResNetBlock2d",
    "ResNetBlock1d",
    "ResNetBackbone",
    "ResNetBlock",
    "Transpose",
    "AtariCNN",
    "Conv2dBackbone",
    "PositionalEncoding",
    "MultiHeadSelfAttention",
    "TransformerBlock",
    "TransformerEncoder",
]
