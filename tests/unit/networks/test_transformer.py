import pytest
import torch
from atomic_rl.networks.transformer import (
    PositionalEncoding,
    MultiHeadSelfAttention,
    TransformerBlock,
    TransformerEncoder,
)

pytestmark = pytest.mark.unit


def test_transformer_skeleton_not_implemented():
    """Verify that skeleton Transformer components raise NotImplementedError when called."""
    pos_enc = PositionalEncoding(embed_dim=64)
    with pytest.raises(NotImplementedError, match="TODO: Implement PositionalEncoding"):
        pos_enc(torch.randn(2, 10, 64))

    mha = MultiHeadSelfAttention(embed_dim=64, num_heads=4)
    with pytest.raises(
        NotImplementedError, match="TODO: Implement MultiHeadSelfAttention"
    ):
        mha(torch.randn(2, 10, 64))

    block = TransformerBlock(embed_dim=64, num_heads=4)
    with pytest.raises(NotImplementedError, match="TODO: Implement TransformerBlock"):
        block(torch.randn(2, 10, 64))

    encoder = TransformerEncoder(embed_dim=64, num_heads=4, num_layers=2)
    with pytest.raises(NotImplementedError, match="TODO: Implement TransformerEncoder"):
        encoder(torch.randn(2, 10, 64))
