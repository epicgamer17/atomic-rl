import torch
from tensordict import TensorDict
from typing import Tuple, Callable, List, Optional
from ..utils import add_dirichlet_noise


# TODO: legal move masking for AlphaZero and terminal nodes for AlphaZero
# TODO: Sampled MuZero
# TODO: initial value for unvisited nodes, allow options, AlphaZero and MuZero: 0, Gumbel Muzero, v_mix, EfficientZero and Batch MCTS mean score
def expand_node_(
    tree: TensorDict,
    parent_nodes: torch.Tensor,
    actions_taken: torch.Tensor,
    policy_logits: torch.Tensor,
    rewards: torch.Tensor,  # From dynamics_fn
    next_embeddings: torch.Tensor,  # From dynamics_fn
    next_to_play: torch.Tensor,  # From dynamics_fn
    is_terminal: torch.Tensor = None,
):
    """
    Adds newly evaluated nodes to the tree.

    Args:
        tree: The MCTS tree TensorDict.
        parent_nodes: Indices of the parent nodes [B].
        actions_taken: Actions taken from parents that led to new nodes [B].
        policy_logits: Predicted policy logits for the new nodes [B, A].
        rewards: Rewards received during transition [B].
        next_embeddings: Embeddings of the new nodes [B, D].
        next_to_play: The player whose turn it is in the new node [B].
        is_terminal: Boolean mask indicating terminal nodes [B].
    """
    batch_size = tree.batch_size[0]
    device = tree.device
    batch_range = torch.arange(batch_size, device=device)

    # 1. Get the next available indices for each batch element
    new_node_indices = tree["node_counts"]

    # 2. Update the parent to point to the new children
    tree["children_index"][batch_range, parent_nodes, actions_taken] = new_node_indices

    # 3. Store the new node data
    tree["embeddings"][batch_range, new_node_indices] = next_embeddings
    tree["children_rewards"][batch_range, parent_nodes, actions_taken] = rewards
    tree["to_play"][batch_range, new_node_indices] = next_to_play
    if is_terminal is not None:
        tree["is_terminal"][batch_range, new_node_indices] = is_terminal

    # 4. Initialize priors for the new node (apply softmax to logits)
    priors = torch.softmax(policy_logits, dim=-1)
    tree["children_prior"][batch_range, new_node_indices] = priors

    # 5. Increment node counts
    tree["node_counts"] += 1
