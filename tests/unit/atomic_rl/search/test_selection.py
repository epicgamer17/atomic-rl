import pytest
import torch

from atomic_rl.search import init_mcts_tree, puct_score, select_leaf
from atomic_rl.search.selection import _normalize_q_values

pytestmark = pytest.mark.unit

# ==========================================
# Tests for Q-Value Normalization
# ==========================================


def test_normalize_q_values_standard():
    """Verify standard min-max mapping to the [0, 1] interval."""
    q_values = torch.tensor([[0.0, 5.0, 10.0], [2.0, 3.0, 4.0]])
    min_q = torch.tensor([0.0, 2.0])
    max_q = torch.tensor([10.0, 4.0])

    normalized = _normalize_q_values(q_values, min_q, max_q)
    expected = torch.tensor([[0.0, 0.5, 1.0], [0.0, 0.5, 1.0]])
    torch.testing.assert_close(normalized, expected)


def test_normalize_q_values_division_by_zero():
    """Verify numerical safety adjustments when min_q equals max_q."""
    q_values = torch.tensor([[5.0, 5.0], [0.0, 0.0]])
    min_q = torch.tensor([5.0, 0.0])
    max_q = torch.tensor([5.0, 0.0])  # Span is 0.0

    # Should safely treat span as 1.0 to map (q_values - min_q) / 1.0 -> 0.0
    normalized = _normalize_q_values(q_values, min_q, max_q)
    torch.testing.assert_close(normalized, torch.zeros_like(q_values))


# ==========================================
# Tests for PUCT Scores
# ==========================================


def test_puct_score_assertion():
    """Verify that the implementation fails fast if the policy priors do not sum to 1."""
    q_values = torch.tensor([[1.0, 2.0]])
    priors_invalid = torch.tensor([[0.5, 0.8]])  # Sums to 1.3
    visits = torch.tensor([[0, 0]])
    total_visits = torch.tensor([[0]])
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
    total_visits = torch.tensor([[4.0]])

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
    total_visits = torch.tensor([[0]])
    min_q = torch.tensor([0.0])
    max_q = torch.tensor([10.0])

    scores = puct_score(q_values, priors, visits, total_visits, min_q, max_q)
    assert scores[0, 0].item() == -1e9
    assert scores[0, 1].item() > 0.0


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
