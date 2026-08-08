import torch
from tensordict import TensorDict
from typing import Tuple, Callable, List, Optional
from ..utils import add_dirichlet_noise


# TODO: make this work with alternating and single player games. also make work for catan (inconsistent turn ordering, ie p1 twice then p2 3 times, then p3 once)
# TODO: make it work with more than 2 players
def backpropagate_(
    tree: TensorDict,
    trajectory: List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]],
    leaf_value: torch.Tensor,
    gamma: float = 0.99,
):
    """
    Backpropagates the leaf value up the search trajectory.

    Args:
        tree: The MCTS tree TensorDict.
        trajectory: List of (node_idx, action_idx, mask) tuples.
        leaf_value: The predicted value of the leaf node [B].
        gamma: Discount factor.
    """
    batch_size = tree.batch_size[0]
    device = tree.device
    batch_range = torch.arange(batch_size, device=device)

    # Iterate backwards through the trajectory
    running_value = leaf_value
    for node_idx, action_idx, mask in reversed(trajectory):
        # 1. Select only active elements for this step
        # This prevents over-counting visits for elements that hit a leaf early.
        b_idx = batch_range[mask]
        n_idx = node_idx[mask]
        a_idx = action_idx[mask]

        # 2. Update visit counts
        tree["children_visits"][b_idx, n_idx, a_idx] += 1

        # 3. Update Q-value (Running Mean)
        q = tree["children_q_values"][b_idx, n_idx, a_idx]
        n = tree["children_visits"][b_idx, n_idx, a_idx]

        # 4. MuZero style: include discounted reward in the return
        # If parent and child are the same player, values align.
        # If they are different (zero-sum), we flip the perspective.
        next_n_idx = tree["children_index"][b_idx, n_idx, a_idx]
        parent_player = tree["to_play"][b_idx, n_idx]
        child_player = tree["to_play"][b_idx, next_n_idx]

        is_same_player = (parent_player == child_player).float()
        perspective_multiplier = is_same_player * 1.0 + (1.0 - is_same_player) * -1.0

        reward = tree["children_rewards"][b_idx, n_idx, a_idx]
        running_value_masked = (
            reward + gamma * running_value[mask] * perspective_multiplier
        )
        running_value[mask] = running_value_masked

        # 5. Update Q-value
        new_q = (q * (n - 1) + running_value[mask]) / n
        tree["children_q_values"][b_idx, n_idx, a_idx] = new_q

        # 6. Update Min-Max Q-values for scaling
        # We update the min/max for the whole batch element based on the new Q-value
        # Use scatter_reduce or a loop? For simplicity and since b_idx is often small,
        # we can use a simpler approach.
        # Actually, running_value[mask] is the new G, and new_q is the mean.
        # MuZero updates min/max using the Q-values.

        # We need to handle the case where multiple elements in the batch have the same index?
        # In MCTS, b_idx is unique within a step.
        tree["min_q"][b_idx] = torch.min(tree["min_q"][b_idx], new_q)
        tree["max_q"][b_idx] = torch.max(tree["max_q"][b_idx], new_q)
