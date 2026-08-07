import pytest
import torch
from tensordict import TensorDict
from unittest.mock import Mock

from functional.mcts import (
    mcts_search,
    init_mcts_tree,
    normalize_q_values,
    puct_score,
    select_leaf,
    expand_node,
    backpropagate,
    get_mcts_visit_policy,
)

pytestmark = pytest.mark.unit

# ==========================================
# Tests for Q-Value Normalization
# ==========================================


def test_normalize_q_values_standard():
    """Verify standard min-max mapping to the [0, 1] interval."""
    q_values = torch.tensor([[0.0, 5.0, 10.0], [2.0, 3.0, 4.0]])
    min_q = torch.tensor([0.0, 2.0])
    max_q = torch.tensor([10.0, 4.0])

    normalized = normalize_q_values(q_values, min_q, max_q)
    expected = torch.tensor([[0.0, 0.5, 1.0], [0.0, 0.5, 1.0]])
    torch.testing.assert_close(normalized, expected)


def test_normalize_q_values_division_by_zero():
    """Verify numerical safety adjustments when min_q equals max_q."""
    q_values = torch.tensor([[5.0, 5.0], [0.0, 0.0]])
    min_q = torch.tensor([5.0, 0.0])
    max_q = torch.tensor([5.0, 0.0])  # Span is 0.0

    # Should safely treat span as 1.0 to map (q_values - min_q) / 1.0 -> 0.0
    normalized = normalize_q_values(q_values, min_q, max_q)
    torch.testing.assert_close(normalized, torch.zeros_like(q_values))


# ==========================================
# Tests for PUCT Scores
# ==========================================


def test_puct_score_assertion():
    """Verify that the implementation fails fast if the policy priors do not sum to 1."""
    q_values = torch.tensor([[1.0, 2.0]])
    priors_invalid = torch.tensor([[0.5, 0.8]])  # Sums to 1.3
    visits = torch.tensor([[0, 0]])
    total_visits = 0
    min_q = torch.tensor([0.0])
    max_q = torch.tensor([2.0])

    with pytest.raises(AssertionError, match="Policy prior must be normalized"):
        puct_score(q_values, priors_invalid, visits, total_visits, min_q, max_q)


def test_puct_score_mathematical_correctness():
    """
    Direct analytical check of the PUCT formula.
    Formula: Q_norm + c * P * sqrt(N_total) / (1 + N_action)
    where c = log((N_total + base + 1) / base) + init
    """
    # Inputs chosen for clean evaluation tracking
    q_values = torch.tensor([[2.0, 6.0]])
    priors = torch.tensor([[0.4, 0.6]])
    visits = torch.tensor([[1.0, 3.0]])
    total_visits = torch.tensor([4.0])

    # Static min/max to ensure predictable normalization output:
    # Action 0: (2.0 - 0.0) / 8.0 = 0.25
    # Action 1: (6.0 - 0.0) / 8.0 = 0.75
    min_q = torch.tensor([0.0])
    max_q = torch.tensor([8.0])

    pb_c_base = 100.0
    pb_c_init = 2.0

    # Execute system function
    calculated_scores = puct_score(
        q_values, priors, visits, total_visits, min_q, max_q, pb_c_base, pb_c_init
    )

    # Manual step-by-step verification oracle
    expected_norm_q = torch.tensor([[0.25, 0.75]])

    # c = log((4 + 100 + 1) / 100) + 2.0 = log(1.05) + 2.0
    expected_c = torch.log(torch.tensor(1.05)) + 2.0

    # exploration term factor = c * sqrt(4) / (visit_counts + 1)
    exploration_factor = expected_c * torch.sqrt(total_visits) / (visits + 1)
    expected_scores = expected_norm_q + exploration_factor * priors

    torch.testing.assert_close(calculated_scores, expected_scores, atol=1e-6, rtol=1e-6)


def test_puct_score_zero_prior_guard():
    """Verify that actions with prior=0 (e.g. masked illegal actions) receive -1e9 penalty."""
    q_values = torch.tensor([[10.0, 5.0]])
    priors = torch.tensor([[0.0, 1.0]])  # Action 0 masked (prior=0)
    visits = torch.tensor([[0, 0]])
    total_visits = torch.tensor([0])
    min_q = torch.tensor([0.0])
    max_q = torch.tensor([10.0])

    scores = puct_score(q_values, priors, visits, total_visits, min_q, max_q)
    assert scores[0, 0].item() == -1e9
    assert scores[0, 1].item() > 0.0


# ==========================================
# Tests for Tree Initialization
# ==========================================


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


# ==========================================
# Tests for Node Expansion
# ==========================================


def test_expand_node():
    """Verify tree expansion correctly links parents to children and registers state transitions."""
    batch_size = 1
    num_simulations = 2
    num_actions = 2
    root_embeddings = torch.zeros(batch_size, 4)

    tree = init_mcts_tree(root_embeddings, num_simulations, num_actions)

    parent_nodes = torch.tensor([0])
    actions_taken = torch.tensor([1])
    policy_logits = torch.tensor([[0.0, 0.0]])  # Softmax will give [0.5, 0.5]
    rewards = torch.tensor([1.5])
    next_embeddings = torch.ones(batch_size, 4)
    next_to_play = torch.tensor([1])
    is_terminal = torch.tensor([True])

    # Run expansion
    expand_node(
        tree,
        parent_nodes,
        actions_taken,
        policy_logits,
        rewards,
        next_embeddings,
        next_to_play,
        is_terminal=is_terminal,
    )

    # Assert structural mutations
    assert tree["children_index"][0, 0, 1].item() == 1  # Parent points to child index 1
    assert tree["node_counts"][0].item() == 2  # Total nodes incremented to 2
    torch.testing.assert_close(tree["embeddings"][0, 1], next_embeddings[0])
    assert tree["children_rewards"][0, 0, 1].item() == 1.5
    assert tree["to_play"][0, 1].item() == 1
    assert tree["is_terminal"][0, 1].item() is True
    torch.testing.assert_close(tree["children_prior"][0, 1], torch.tensor([0.5, 0.5]))


# ==========================================
# Tests for Policy Extraction
# ==========================================


def test_get_mcts_visit_policy_temperature_1():
    """Verify temperature tau=1.0 yields visit count proportional policy."""
    visit_counts = torch.tensor([[10.0, 30.0, 60.0]])
    policy = get_mcts_visit_policy(visit_counts, temperature=1.0)
    expected = torch.tensor([[0.1, 0.3, 0.6]])
    torch.testing.assert_close(policy, expected)


def test_get_mcts_visit_policy_temperature_greedy():
    """Verify temperature tau=0.0 yields greedy one-hot policy."""
    visit_counts = torch.tensor([[10.0, 30.0, 60.0]])
    policy = get_mcts_visit_policy(visit_counts, temperature=0.0)
    expected = torch.tensor([[0.0, 0.0, 1.0]])
    torch.testing.assert_close(policy, expected)


def test_get_mcts_visit_policy_negative_temperature():
    """Verify fail-fast assertion on negative temperature."""
    visit_counts = torch.tensor([[10.0, 30.0]])
    with pytest.raises(AssertionError, match="Temperature must be non-negative"):
        get_mcts_visit_policy(visit_counts, temperature=-0.5)


# ==========================================
# Tests for Backpropagation & Turn Mechanics
# ==========================================


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


# ==========================================
# Tests for Selection Mechanics
# ==========================================


def test_select_leaf_early_termination():
    """Verify that leaf selection halts individual batch components when they hit unexpanded slots."""
    batch_size = 2
    root_embeddings = torch.zeros(batch_size, 2)
    tree = init_mcts_tree(root_embeddings, num_simulations=5, num_actions=2)

    # Setup priors to perfectly direct the deterministic argmax choice
    tree["children_prior"][:, 0, :] = torch.tensor(
        [1.0, 0.0]
    )  # Forces Action 0 at root

    # Environment 0 has a child already expanded at index 1
    tree["children_index"][0, 0, 0] = 1
    tree["children_prior"][0, 1, :] = torch.tensor(
        [0.0, 1.0]
    )  # Forces Action 1 at node 1

    # Environment 1 has no children expanded at all (remains -1 at root)

    leaf_indices, trajectory = select_leaf(
        tree, pb_c_base=100.0, pb_c_init=1.0, max_depth=5
    )

    # Env 0 should fall deep into slot 1. Env 1 should stop immediately at root (slot 0)
    torch.testing.assert_close(leaf_indices, torch.tensor([1, 0], dtype=torch.long))

    # Validate the trajectory steps
    # Step 1: Both tracking routes were active
    assert torch.equal(trajectory[0][2], torch.tensor([True, True]))
    # Step 2: Env 1 hit a leaf node on step 1, turning inactive for depth level 2
    assert torch.equal(trajectory[1][2], torch.tensor([True, False]))


# ==========================================
# Deterministic Selection Trajectory Routing
# ==========================================


def test_select_leaf_deterministic_path():
    """
    Forces select_leaf down a strict, multi-step structural path
    by overwhelming the exploration constants with hand-crafted Q-values.

    Path target: Node 0 (Root) -> Action 1 -> Node 2 -> Action 0 -> Leaf (-1)
    """
    batch_size = 1
    root_embeddings = torch.zeros(batch_size, 4)

    # Allocate standard tree structure
    tree = init_mcts_tree(root_embeddings, num_simulations=4, num_actions=2)

    # Establish topology linking Node 0 to Node 2 via Action 1
    tree["children_index"][0, 0, 1] = 2

    # Configure Node 0 (Root): Distort values to guarantee selection of Action 1
    tree["children_q_values"][0, 0, 0] = 0.0
    tree["children_q_values"][0, 0, 1] = 50.0  # Dominates entirely
    tree["children_prior"][0, 0, :] = torch.tensor([0.5, 0.5])
    tree["children_visits"][0, 0, :] = torch.tensor([0.0, 0.0])

    # Configure Node 2: Distort values to guarantee selection of Action 0
    # Node 2's children arrays default to -1, making whatever action is selected a leaf
    tree["children_q_values"][0, 2, 0] = 50.0  # Dominates entirely
    tree["children_q_values"][0, 2, 1] = 0.0
    tree["children_prior"][0, 2, :] = torch.tensor([0.5, 0.5])
    tree["children_visits"][0, 2, :] = torch.tensor([0.0, 0.0])

    # Keep scaling properties linear and constant
    tree["min_q"][0] = 0.0
    tree["max_q"][0] = 50.0

    # Execute selection pass
    leaf_idx, trajectory = select_leaf(
        tree, pb_c_base=19652, pb_c_init=1.25, max_depth=5
    )

    # 1. The search loop should break and target Node 2 as the expansion candidate
    assert leaf_idx.item() == 2

    # 2. Verify chronological sequence preservation inside trajectory tracking
    assert len(trajectory) == 2

    # Step 1 checking: At Node 0, Action 1 chosen
    node_step_1, action_step_1, mask_step_1 = trajectory[0]
    assert node_step_1.item() == 0
    assert action_step_1.item() == 1
    assert mask_step_1.item() is True

    # Step 2 checking: At Node 2, Action 0 chosen
    node_step_2, action_step_2, mask_step_2 = trajectory[1]
    assert node_step_2.item() == 2
    assert action_step_2.item() == 0
    assert mask_step_2.item() is True
