import torch
from tensordict import TensorDict
from typing import Tuple, Callable, List, Optional
from ..utils import add_dirichlet_noise


def init_mcts_tree(
    root_embeddings: torch.Tensor,
    num_simulations: int,
    num_actions: int,
) -> TensorDict:
    """
    Initializes the MCTS tree structure as a TensorDict.

    Args:
        root_embeddings: Initial state representations [B, D]
        num_simulations: Number of simulations to perform.
        num_actions: Number of possible actions.

    Returns:
        A TensorDict representing the initial tree state.
    """
    batch_size = root_embeddings.shape[0]
    max_nodes = num_simulations + 1  # Root + 1 node per simulation

    # We pre-allocate the tree to avoid dynamic resizing (Torch Compile friendly)
    tree = TensorDict(
        {
            "embeddings": root_embeddings.new_zeros(
                (batch_size, max_nodes, *root_embeddings.shape[1:])
            ),
            "children_index": root_embeddings.new_full(
                (batch_size, max_nodes, num_actions),
                -1,
                dtype=torch.long,
            ),
            "children_prior": root_embeddings.new_zeros(
                (batch_size, max_nodes, num_actions)
            ),
            "children_visits": root_embeddings.new_zeros(
                (batch_size, max_nodes, num_actions)
            ),
            "children_rewards": root_embeddings.new_zeros(
                (batch_size, max_nodes, num_actions)
            ),
            "children_q_values": root_embeddings.new_zeros(
                (batch_size, max_nodes, num_actions)
            ),
            "node_counts": root_embeddings.new_ones(batch_size, dtype=torch.long),
            "to_play": root_embeddings.new_zeros(
                batch_size, max_nodes, dtype=torch.long
            ),
            "is_terminal": root_embeddings.new_zeros(
                (batch_size, max_nodes), dtype=torch.bool
            ),
            "min_q": root_embeddings.new_full((batch_size,), 1e9),
            "max_q": root_embeddings.new_full((batch_size,), -1e9),
        },
        batch_size=[batch_size],
    )

    # Initialize root (index 0)
    tree["embeddings"][:, 0] = root_embeddings
    return tree
