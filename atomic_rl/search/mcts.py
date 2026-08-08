import torch
from tensordict import TensorDict
from typing import Tuple, Callable, List, Optional
from ..utils import add_dirichlet_noise
from .backpropagation import backpropagate_
from .expansion import expand_node_
from .selection import select_leaf
from .tree import init_mcts_tree


# TODO: remember we eventually want gumbel sequential halving and possibly other search methods too.
# TODO: dont hard code dirichlet params, pass em in as optional arguments
# TODO: avoid flags like is_zero_sum
def mcts_search(
    root_embeddings: torch.Tensor,
    num_simulations: int,
    num_actions: int,
    expansion_fn: Callable,  # Returns (policy_logits, value)
    dynamics_fn: Callable,  # Returns (next_embedding, reward) or (next_embedding, reward, next_to_play) or (next_embedding, reward, next_to_play, is_terminal) or (next_embedding, reward, next_to_play, is_terminal, next_legal_mask)
    pb_c_base: float = 19652,
    pb_c_init: float = 1.25,
    gamma: float = 0.99,
    dirichlet_epsilon: float = 0.25,
    dirichlet_alpha: float = 0.3,
    root_to_play: torch.Tensor = None,
    root_legal_mask: Optional[torch.Tensor] = None,
) -> TensorDict:
    """
    Orchestrates a batched MCTS search.

    Args:
        root_embeddings: Initial state representations [B, D]
        num_simulations: Number of simulations to perform.
        num_actions: Number of possible actions.
        expansion_fn: Function to get policy/value from embeddings.
        dynamics_fn: Learned model for MuZero (or simulator for AlphaZero).
        pb_c_base: Base constant for PUCT.
        pb_c_init: Additive constant for PUCT.
        gamma: Discount factor for rewards.
        dirichlet_epsilon: Weight of Dirichlet noise at the root.
        dirichlet_alpha: Concentration parameter for Dirichlet noise.
        root_to_play: Optional initial player array [B].
        root_legal_mask: Optional explicit boolean mask [B, num_actions] of legal actions at the root environment state.
    """
    device = root_embeddings.device
    batch_size = root_embeddings.shape[0]
    batch_range = torch.arange(batch_size, device=device)

    # 1. Initialize Tree State
    tree = init_mcts_tree(root_embeddings, num_simulations, num_actions)
    if root_to_play is not None:
        tree["to_play"][:, 0] = root_to_play

    # 2. Initial Evaluation (Root)
    policy_logits, _ = expansion_fn(root_embeddings)
    if root_legal_mask is None:
        import warnings

        warnings.warn(
            "root_legal_mask was not provided to mcts_search. Illegal actions at the root will not be masked during search or Dirichlet noise calculation.",
            UserWarning,
            stacklevel=2,
        )
    else:
        policy_logits = torch.where(root_legal_mask, policy_logits, -1e9)

    priors = torch.softmax(policy_logits, dim=-1)

    # 3. Add Dirichlet Noise (Root exploration, using explicit legal action mask)
    if dirichlet_epsilon > 0:
        priors = add_dirichlet_noise(
            priors, dirichlet_epsilon, dirichlet_alpha, mask=root_legal_mask
        )

    tree["children_prior"][:, 0] = priors

    for _ in range(num_simulations):
        # A. Selection: Find the best leaf using PUCT score
        leaf_indices, trajectory = select_leaf(tree, pb_c_base, pb_c_init)

        # The expansion happens at the end of the trajectory
        parent_nodes, actions_taken = trajectory[-1][0], trajectory[-1][1]

        # B. Dynamics (MuZero style / Simulator): Transition to next state
        dyn_output = dynamics_fn(
            tree["embeddings"][batch_range, parent_nodes], actions_taken
        )
        next_legal_mask = None
        if len(dyn_output) == 5:
            next_embeddings, rewards, next_to_play, is_terminal, next_legal_mask = (
                dyn_output
            )
        elif len(dyn_output) == 4:
            next_embeddings, rewards, next_to_play, is_terminal = dyn_output
        else:
            next_embeddings, rewards, next_to_play = dyn_output
            is_terminal = root_embeddings.new_zeros(batch_size, dtype=torch.bool)

        # C. Expansion & Evaluation: Predict policy and value for the leaf
        policy_logits, value = expansion_fn(next_embeddings)
        if next_legal_mask is not None:
            policy_logits = torch.where(next_legal_mask, policy_logits, -1e9)

        # For terminal states, value should be 0.0 (terminal state has no future expected return)
        value = torch.where(is_terminal, torch.zeros_like(value), value)

        # D. Expand Tree: Add the new node
        expand_node_(
            tree,
            parent_nodes,
            actions_taken,
            policy_logits,
            rewards,
            next_embeddings,
            next_to_play,
            is_terminal=is_terminal,
        )

        # E. Backpropagation: Update value/visit counts up the trajectory
        backpropagate_(tree, trajectory, value, gamma)

    return tree
