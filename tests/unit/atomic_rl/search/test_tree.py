import pytest
import torch

from atomic_rl.search import init_mcts_tree

pytestmark = pytest.mark.unit


def test_init_mcts_tree_geometry():
    """Verify that tensor buffers are pre-allocated with correct dimensions and types."""
    batch_size = 2
    num_simulations = 4
    num_actions = 3
    embedding_dim = 8
    root_embeddings = torch.randn(batch_size, embedding_dim)

    tree = init_mcts_tree(root_embeddings, num_simulations, num_actions)
    max_nodes = num_simulations + 1  # 5 slots

    assert tree["embeddings"].shape == (batch_size, max_nodes, embedding_dim)
    assert tree["children_index"].shape == (batch_size, max_nodes, num_actions)
    assert tree["is_terminal"].shape == (batch_size, max_nodes)
    assert torch.all(tree["children_index"] == -1)
    assert tree["node_counts"].tolist() == [1, 1]  # Only the root is occupied initially
    torch.testing.assert_close(tree["embeddings"][:, 0], root_embeddings)
