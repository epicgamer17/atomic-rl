"""
Transformer sequence modeling components (Skeleton / API Specifications).

Architectural Design & Behavior Notes:
-------------------------------------
- TransformerBlock:
  Pre-LN Architecture (Default in modern Transformer architectures like Decision Transformer / GPT-2):
      x = x + SelfAttention(LayerNorm(x))
      x = x + FeedForward(LayerNorm(x))

  Post-LN Architecture (Original Vaswani et al., 2017 Attention Is All You Need):
      x = LayerNorm(x + SelfAttention(x))
      x = LayerNorm(x + FeedForward(x))

- MultiHeadSelfAttention:
  Projects inputs into Query, Key, Value representations across `num_heads` sub-spaces.
  Supports causal masking (preventing future token leakage in autoregressive RL decision models)
  and padding masks.

- PositionalEncoding:
  Adds positional context (sinusoidal or learned) to input embeddings across sequence dimension.

Status: Skeleton / Specification only.
TODO: Implement full tensor operations for Transformer components when needed by sequence RL algorithms.
"""

import torch
import torch.nn as nn
from typing import Optional


class PositionalEncoding(nn.Module):
    """
    Positional Encoding module (sinusoidal or learned embeddings).

    TODO: Implement sinusoidal / learned positional embedding tensor generation.

    Args:
        embed_dim (int): Embedding channel dimension.
        max_len (int): Maximum sequence horizon length. Defaults to 5000.
        learned (bool): If True, uses learnable parameters instead of static sinusoids. Defaults to False.
    """

    def __init__(self, embed_dim: int, max_len: int = 5000, learned: bool = False):
        super().__init__()
        self.embed_dim = embed_dim
        self.max_len = max_len
        self.learned = learned
        # TODO: Initialize positional embedding weights or static sinusoidal buffer

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): Input embeddings of shape [Batch, SeqLen, EmbedDim] or [SeqLen, Batch, EmbedDim].

        Returns:
            torch.Tensor: Positionally encoded embeddings of matching shape.
        """
        raise NotImplementedError("TODO: Implement PositionalEncoding component in networks/transformer.py")


class MultiHeadSelfAttention(nn.Module):
    """
    Multi-Head Self-Attention (MHA) module.

    TODO: Implement QKV projection, scaled dot-product attention, causal masking, and output projection.

    Args:
        embed_dim (int): Input and output embedding feature dimension.
        num_heads (int): Number of parallel attention heads.
        dropout (float): Attention dropout probability. Defaults to 0.0.
        causal (bool): If True, applies lower-triangular causal mask to prevent attending to future tokens. Defaults to False.
    """

    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.0, causal: bool = False):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.causal = causal
        # TODO: Initialize Q, K, V linear projections and output projection layer

    def forward(
        self,
        x: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
        key_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): Input sequence tensor of shape [Batch, SeqLen, EmbedDim].
            attn_mask (Optional[torch.Tensor]): Causal or custom attention mask.
            key_padding_mask (Optional[torch.Tensor]): Mask indicating padding tokens to ignore.

        Returns:
            torch.Tensor: Attended representation of shape [Batch, SeqLen, EmbedDim].
        """
        raise NotImplementedError("TODO: Implement MultiHeadSelfAttention component in networks/transformer.py")


class TransformerBlock(nn.Module):
    """
    Single Transformer Layer.

    Architectural Behavior:
        Pre-LN (pre_ln=True, default):
            h = x + MultiHeadAttention(LayerNorm(x))
            out = h + FeedForward(LayerNorm(h))

        Post-LN (pre_ln=False):
            h = LayerNorm(x + MultiHeadAttention(x))
            out = LayerNorm(h + FeedForward(h))

    Args:
        embed_dim (int): Model feature dimension.
        num_heads (int): Number of attention heads.
        ff_dim (Optional[int]): Feed-forward hidden dimension. Defaults to 4 * embed_dim.
        dropout (float): Dropout rate. Defaults to 0.1.
        pre_ln (bool): If True, use Pre-LayerNorm configuration. Defaults to True.
        causal (bool): If True, apply causal sequence mask in attention layer. Defaults to False.
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        ff_dim: Optional[int] = None,
        dropout: float = 0.1,
        pre_ln: bool = True,
        causal: bool = False,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.ff_dim = ff_dim if ff_dim is not None else 4 * embed_dim
        self.dropout = dropout
        self.pre_ln = pre_ln
        self.causal = causal
        # TODO: Initialize MHA, LayerNorms, and MLP feedforward layers

    def forward(self, x: torch.Tensor, attn_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): Sequence tensor [Batch, SeqLen, EmbedDim].
            attn_mask (Optional[torch.Tensor]): Attention mask.

        Returns:
            torch.Tensor: Output sequence tensor [Batch, SeqLen, EmbedDim].
        """
        raise NotImplementedError("TODO: Implement TransformerBlock component in networks/transformer.py")


class TransformerEncoder(nn.Module):
    """
    Configurable stack of Transformer Blocks.

    TODO: Implement stack of TransformerBlocks with optional positional encoding.

    Args:
        embed_dim (int): Model embedding dimension.
        num_heads (int): Attention head count.
        num_layers (int): Number of TransformerBlocks to stack.
        ff_dim (Optional[int]): Feedforward dimension.
        dropout (float): Dropout probability.
        pre_ln (bool): Pre-LN or Post-LN toggle.
        causal (bool): Causal mask toggle.
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        num_layers: int = 4,
        ff_dim: Optional[int] = None,
        dropout: float = 0.1,
        pre_ln: bool = True,
        causal: bool = False,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        # TODO: Initialize stack of TransformerBlock modules

    def forward(self, x: torch.Tensor, attn_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): Sequence tensor [Batch, SeqLen, EmbedDim].
            attn_mask (Optional[torch.Tensor]): Mask tensor.

        Returns:
            torch.Tensor: Transformed sequence tensor [Batch, SeqLen, EmbedDim].
        """
        raise NotImplementedError("TODO: Implement TransformerEncoder component in networks/transformer.py")
