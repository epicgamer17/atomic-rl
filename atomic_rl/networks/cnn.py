"""
Convolutional Neural Network (CNN) building blocks and feature extractors.

Architectural Design & Behavior Notes:
-------------------------------------
- Transpose:
  Permutes tensor dimensions. Commonly used to convert channel-last environments (e.g., [B, H, W, C])
  to PyTorch channel-first format (e.g., [B, C, H, W]).

- AtariCNN (Nature CNN backbone, Mnih et al., 2015):
  Sequential architecture for processing 84x84 (or similar) image frame stacks:
      1. Conv2d(in_channels, 32, kernel_size=8, stride=4) -> ReLU
      2. Conv2d(32, 64, kernel_size=4, stride=2) -> ReLU
      3. Conv2d(64, 64, kernel_size=3, stride=1) -> ReLU
      4. Flatten
      5. Linear(64 * 7 * 7, out_features) -> ReLU
"""

import torch
import torch.nn as nn
from typing import Tuple, Sequence, Optional
from atomic_rl.initialization import layer_init_


class Transpose(nn.Module):
    """
    Permutes tensor dimensions.

    Commonly used to convert channel-last observation tensors (e.g. [B, H, W, C])
    to PyTorch's required channel-first format (e.g. [B, C, H, W]).

    Args:
        permutation (Tuple[int, ...]): Tuple of dimension indices specifying the output ordering.
            Example: (0, 3, 1, 2) converts [B, H, W, C] to [B, C, H, W].
    """

    def __init__(self, permutation: Tuple[int, ...]):
        super().__init__()
        self.permutation = permutation

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Permuted tensor.
        """
        return x.permute(self.permutation)


class AtariCNN(nn.Module):
    """
    Standard Atari Nature CNN feature extractor (Mnih et al., 2015, Human-level control through deep RL).

    Architecture:
        - Layer 1: Conv2d(in_channels, 32, kernel_size=8, stride=4) + ReLU
        - Layer 2: Conv2d(32, 64, kernel_size=4, stride=2) + ReLU
        - Layer 3: Conv2d(64, 64, kernel_size=3, stride=1) + ReLU
        - Layer 4: Flatten
        - Layer 5: Linear(64 * 7 * 7, out_features) + ReLU

    Args:
        in_channels (int): Number of input frame channels (e.g., 4 for stacked Atari frames). Defaults to 4.
        out_features (int): Dimensionality of the final dense feature embedding. Defaults to 512.
        scale_inputs (bool): If True, divides input tensors by 255.0 to normalize uint8 pixel values
            to [0.0, 1.0]. Defaults to False (assumes caller handles normalization).
    """

    def __init__(
        self, in_channels: int = 4, out_features: int = 512, scale_inputs: bool = False
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_features = out_features
        self.scale_inputs = scale_inputs

        self.network = nn.Sequential(
            layer_init_(nn.Conv2d(in_channels, 32, kernel_size=8, stride=4)),
            nn.ReLU(),
            layer_init_(nn.Conv2d(32, 64, kernel_size=4, stride=2)),
            nn.ReLU(),
            layer_init_(nn.Conv2d(64, 64, kernel_size=3, stride=1)),
            nn.ReLU(),
            nn.Flatten(),
            layer_init_(nn.Linear(64 * 7 * 7, out_features)),
            nn.ReLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): Input image tensor of shape [Batch, in_channels, 84, 84]. Not gray-scaled, values should be between 0 and 255.

        Returns:
            torch.Tensor: Feature vector of shape [Batch, out_features].

        NOTE: explicitly normalize pixels here to avoid passing huge floats in the buffer
        """
        if self.scale_inputs:
            x = x / 255.0
        return self.network(x)


class Conv2dBackbone(nn.Module):
    """
    Generic 2D Convolutional encoder backbone.

    Configures a sequence of 2D convolutions with optional normalization, activation, and pooling.

    Args:
        in_channels (int): Number of input channels.
        channels (Sequence[int]): Sequence of output channel counts for each Conv2d layer.
        kernel_sizes (Sequence[int]): Sequence of kernel sizes for each layer.
        strides (Sequence[int]): Sequence of strides for each layer.
        paddings (Optional[Sequence[int]]): Sequence of padding sizes. Defaults to 0.
        norm_type (Optional[str]): "batch" for BatchNorm2d or None. Defaults to None.
    """

    def __init__(
        self,
        in_channels: int,
        channels: Sequence[int] = (32, 64, 64),
        kernel_sizes: Sequence[int] = (8, 4, 3),
        strides: Sequence[int] = (4, 2, 1),
        paddings: Optional[Sequence[int]] = None,
        norm_type: Optional[str] = None,
    ):
        super().__init__()
        if paddings is None:
            paddings = [0] * len(channels)

        layers = []
        curr_channels = in_channels
        for out_c, k, s, p in zip(channels, kernel_sizes, strides, paddings):
            layers.append(
                layer_init_(
                    nn.Conv2d(curr_channels, out_c, kernel_size=k, stride=s, padding=p)
                )
            )
            if norm_type == "batch":
                layers.append(nn.BatchNorm2d(out_c))
            layers.append(nn.ReLU())
            curr_channels = out_c

        layers.append(nn.Flatten())
        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): Input tensor [Batch, in_channels, H, W].

        Returns:
            torch.Tensor: Flattened feature vector [Batch, FlattenedSize].
        """
        return self.network(x)
