"""
Residual Network (ResNet) components for 1D and 2D signals.

Architectural Design & Behavior Notes:
-------------------------------------
- ResNetBlock2d defaults to Post-Activation (He et al., 2016 Deep Residual Learning):
      x -> Conv2d -> BatchNorm2d -> ReLU -> Conv2d -> BatchNorm2d -> (+) residual -> ReLU
  If `pre_activation=True` (He et al., 2016 Identity Mappings):
      x -> BatchNorm2d -> ReLU -> Conv2d -> BatchNorm2d -> ReLU -> Conv2d -> (+) residual

- Shortcut / Downsampling:
  When `stride != 1` or `in_channels != out_channels`, a 1x1 Conv2d (+ Norm) projection shortcut is
  applied to match spatial dimensions and channel counts. Otherwise, an identity shortcut is used.

- ResNetBlock1d applies the equivalent sequence to 1D signals (e.g., temporal or sequential data).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Union, Sequence


class ResNetBlock2d(nn.Module):
    """
    2D Residual Block for spatial image/grid inputs (e.g., AlphaZero, MuZero, Atari, grid worlds).

    Architectural Flow:
        Standard (Post-Activation, default):
            residual = x
            out = Conv2d(in_channels, out_channels)(x)
            out = Norm2d(out_channels)(out)
            out = Activation(out)
            out = Conv2d(out_channels, out_channels)(out)
            out = Norm2d(out_channels)(out)
            if in_channels != out_channels or stride != 1:
                residual = ProjectionShortcut(x)
            out = out + residual
            return Activation(out)

        Pre-Activation (`pre_activation=True`):
            residual = x
            out = Norm2d(in_channels)(x)
            out = Activation(out)
            out = Conv2d(in_channels, out_channels)(out)
            out = Norm2d(out_channels)(out)
            out = Activation(out)
            out = Conv2d(out_channels, out_channels)(out)
            if in_channels != out_channels or stride != 1:
                residual = ProjectionShortcut(x)
            return out + residual

    Args:
        in_channels (int): Number of input channels.
        out_channels (int, optional): Number of output channels. Defaults to in_channels.
        stride (int): Stride of the first convolutional layer. Defaults to 1.
        pre_activation (bool): If True, use Pre-Activation layout. Defaults to False (Post-Activation).
        norm_type (str, optional): Normalization type: "batch" for BatchNorm2d, "group" for GroupNorm,
            "layer" for LayerNorm (spatial), or None for no normalization. Defaults to "batch".
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: Optional[int] = None,
        stride: int = 1,
        pre_activation: bool = False,
        norm_type: Optional[str] = "batch",
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels if out_channels is not None else in_channels
        self.stride = stride
        self.pre_activation = pre_activation
        self.norm_type = norm_type

        # First convolution layer
        self.conv1 = nn.Conv2d(
            self.in_channels,
            self.out_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=(norm_type is None),
        )
        self.norm1 = self._make_norm(self.in_channels if pre_activation else self.out_channels)

        # Second convolution layer
        self.conv2 = nn.Conv2d(
            self.out_channels,
            self.out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=(norm_type is None),
        )
        self.norm2 = self._make_norm(self.out_channels)

        # Shortcut projection when dimensions or stride mismatch
        if self.in_channels != self.out_channels or self.stride != 1:
            self.shortcut = nn.Sequential(
                nn.Conv2d(
                    self.in_channels,
                    self.out_channels,
                    kernel_size=1,
                    stride=stride,
                    bias=(norm_type is None),
                ),
                self._make_norm(self.out_channels),
            )
        else:
            self.shortcut = nn.Identity()

    def _make_norm(self, num_channels: int) -> nn.Module:
        if self.norm_type == "batch":
            return nn.BatchNorm2d(num_channels)
        elif self.norm_type == "group":
            return nn.GroupNorm(num_groups=min(32, num_channels), num_channels=num_channels)
        elif self.norm_type is None:
            return nn.Identity()
        else:
            raise ValueError(f"Unsupported norm_type '{self.norm_type}'. Expected 'batch', 'group', or None.")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): Input tensor of shape [Batch, in_channels, Height, Width].

        Returns:
            torch.Tensor: Output tensor of shape [Batch, out_channels, Height', Width'].
        """
        residual = self.shortcut(x)

        if self.pre_activation:
            out = self.norm1(x)
            out = F.relu(out)
            out = self.conv1(out)
            out = self.norm2(out)
            out = F.relu(out)
            out = self.conv2(out)
            return out + residual
        else:
            out = self.conv1(x)
            out = self.norm1(out)
            out = F.relu(out)
            out = self.conv2(out)
            out = self.norm2(out)
            out += residual
            return F.relu(out)


# Alias for backward compatibility / standard 2D usage
ResNetBlock = ResNetBlock2d


class ResNetBlock1d(nn.Module):
    """
    1D Residual Block for sequence or temporal vector signals.

    Architectural Flow (Post-Activation):
        residual = shortcut(x)
        out = Conv1d(in_channels, out_channels, kernel_size=3, padding=1)(x)
        out = Norm1d(out_channels)(out)
        out = ReLU(out)
        out = Conv1d(out_channels, out_channels, kernel_size=3, padding=1)(out)
        out = Norm1d(out_channels)(out)
        out += residual
        return ReLU(out)

    Args:
        in_channels (int): Number of input channels.
        out_channels (int, optional): Output channels. Defaults to in_channels.
        stride (int): Convolution stride. Defaults to 1.
        norm_type (str, optional): "batch" for BatchNorm1d or None. Defaults to "batch".
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: Optional[int] = None,
        stride: int = 1,
        norm_type: Optional[str] = "batch",
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels if out_channels is not None else in_channels
        self.stride = stride
        self.norm_type = norm_type

        self.conv1 = nn.Conv1d(
            self.in_channels,
            self.out_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=(norm_type is None),
        )
        self.norm1 = nn.BatchNorm1d(self.out_channels) if norm_type == "batch" else nn.Identity()

        self.conv2 = nn.Conv1d(
            self.out_channels,
            self.out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=(norm_type is None),
        )
        self.norm2 = nn.BatchNorm1d(self.out_channels) if norm_type == "batch" else nn.Identity()

        if self.in_channels != self.out_channels or self.stride != 1:
            self.shortcut = nn.Sequential(
                nn.Conv1d(
                    self.in_channels,
                    self.out_channels,
                    kernel_size=1,
                    stride=stride,
                    bias=(norm_type is None),
                ),
                nn.BatchNorm1d(self.out_channels) if norm_type == "batch" else nn.Identity(),
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): Input tensor of shape [Batch, in_channels, Length].

        Returns:
            torch.Tensor: Output tensor of shape [Batch, out_channels, Length'].
        """
        residual = self.shortcut(x)
        out = F.relu(self.norm1(self.conv1(x)))
        out = self.norm2(self.conv2(out))
        out += residual
        return F.relu(out)


class ResNetBackbone(nn.Module):
    """
    Configurable stack of Residual Blocks (1D or 2D) serving as a generic feature extractor backbone.

    Args:
        in_channels (int): Input channel dimension.
        num_filters (int): Base channel count across residual blocks.
        num_blocks (int): Number of stacked residual blocks.
        dim (int): Spatial dimension, 1 or 2. Defaults to 2.
        pre_activation (bool): Use pre-activation layout if True. Defaults to False.
    """

    def __init__(
        self,
        in_channels: int,
        num_filters: int = 64,
        num_blocks: int = 4,
        dim: int = 2,
        pre_activation: bool = False,
    ):
        super().__init__()
        self.dim = dim
        conv_cls = nn.Conv2d if dim == 2 else nn.Conv1d
        norm_cls = nn.BatchNorm2d if dim == 2 else nn.BatchNorm1d

        self.stem = nn.Sequential(
            conv_cls(in_channels, num_filters, kernel_size=3, padding=1, bias=False),
            norm_cls(num_filters),
            nn.ReLU(),
        )

        blocks = []
        for _ in range(num_blocks):
            if dim == 2:
                blocks.append(
                    ResNetBlock2d(
                        num_filters,
                        num_filters,
                        pre_activation=pre_activation,
                    )
                )
            else:
                blocks.append(ResNetBlock1d(num_filters, num_filters))
        self.blocks = nn.Sequential(*blocks)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): Input tensor [Batch, in_channels, ...].

        Returns:
            torch.Tensor: Feature maps [Batch, num_filters, ...].
        """
        x = self.stem(x)
        return self.blocks(x)
