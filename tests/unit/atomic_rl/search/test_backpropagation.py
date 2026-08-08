import pytest
import torch

from atomic_rl.search import backpropagate, init_mcts_tree

pytestmark = pytest.mark.unit


def test_backpropagate_single_player():
    pass


def test_backpropagate_alternating_players():
    """Verify that value tracking inverts across perspectives when players alternate."""
    # Setup a 1-batch tree manually to isolate the math of the backward pass
    root_embeddings = torch.zeros(1, 2)
    tree = init_mcts_tree(root_embeddings, num_simulations=5, num_actions=2)

    # Construct a simple sequential path: Node 0 -> Node 1
    tree["children_index"][0, 0, 0] = 1
    tree["children_rewards"][0, 0, 0] = 0.5  # Reward obtained along the edge

    # Scenario: Alternating zero-sum game
    tree["to_play"][0, 0] = 0  # Parent is Player 0
    tree["to_play"][0, 1] = 1  # Child is Player 1

    # Trajectory format: [(node_idx, action_idx, active_mask)]
    trajectory = [(torch.tensor([0]), torch.tensor([0]), torch.tensor([True]))]
    leaf_value = torch.tensor(
        [1.0]
    )  # Value evaluation out of Node 1 from Player 1's perspective
    gamma = 1.0

    backpropagate(tree, trajectory, leaf_value, gamma)

    # Math:
    # parent_player (0) != child_player (1) -> perspective_multiplier = -1.0
    # G = Reward + gamma * leaf_value * multiplier = 0.5 + 1.0 * 1.0 * (-1.0) = -0.5
    # Expected target Q value at root edge = -0.5
    assert tree["children_visits"][0, 0, 0].item() == 1
    assert tree["children_q_values"][0, 0, 0].item() == -0.5
    assert tree["min_q"][0].item() == -0.5
    assert tree["max_q"][0].item() == -0.5
