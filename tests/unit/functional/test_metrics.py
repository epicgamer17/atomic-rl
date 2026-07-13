import pytest
import torch
import torch.nn as nn

from functional.metrics import (
    compute_dead_units_proportion,
    compute_average_weight_magnitude,
    compute_average_gradient_magnitude,
    compute_stable_rank,
)

pytestmark = pytest.mark.unit


# ==========================================
# Tests for compute_dead_units_proportion
# ==========================================


def test_dead_units_none_dead():
    # All activations are strictly positive -> 0% dead units
    activations = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    assert compute_dead_units_proportion(activations) == 0.0


def test_dead_units_all_dead():
    # All elements are 0 or negative -> 100% dead units
    activations = torch.tensor([[0.0, -1.0], [0.0, 0.0]])
    assert compute_dead_units_proportion(activations) == 1.0


def test_dead_units_partial():
    # Column 0 is dead (all <= 0), Column 1 is alive (has at least one > 0)
    activations = torch.tensor([[0.0, 2.0], [-0.5, 0.0]])
    # 1 out of 2 features is dead -> 0.5
    assert compute_dead_units_proportion(activations) == 0.5


# ==========================================
# Tests for compute_average_weight_magnitude
# ==========================================


def test_weight_magnitude_empty():
    # Empty iterable should safely return 0.0
    assert compute_average_weight_magnitude([]) == 0.0


def test_weight_magnitude_standard():
    p1 = nn.Parameter(torch.tensor([1.0, -2.0]))
    p2 = nn.Parameter(torch.tensor([3.0]))
    # Total absolute sum: 1.0 + 2.0 + 3.0 = 6.0
    # Total elements: 3
    # Expected: 6.0 / 3 = 2.0
    assert compute_average_weight_magnitude([p1, p2]) == pytest.approx(2.0)


# ==========================================
# Tests for compute_average_gradient_magnitude
# ==========================================


def test_gradient_magnitude_empty_or_none():
    # Case 1: Empty list
    assert compute_average_gradient_magnitude([]) == 0.0

    # Case 2: Parameters exist but their .grad fields are None
    p1 = nn.Parameter(torch.tensor([1.0, 2.0]))
    assert compute_average_gradient_magnitude([p1]) == 0.0


def test_gradient_magnitude_mixed():
    p1 = nn.Parameter(torch.tensor([1.0, 2.0]))
    p1.grad = torch.tensor([1.5, -2.5])

    p2 = nn.Parameter(torch.tensor([3.0]))  # No gradient populated

    # Only p1 should be counted. Total absolute grad sum = 1.5 + 2.5 = 4.0
    # Total graded elements = 2
    # Expected: 4.0 / 2 = 2.0
    assert compute_average_gradient_magnitude([p1, p2]) == pytest.approx(2.0)


# ==========================================
# Tests for compute_stable_rank
# ==========================================


def test_stable_rank_zero_variance():
    # When all activations are completely identical across the batch,
    # the centered matrix becomes zero, triggering the < 1e-8 safety check.
    activations = torch.tensor([[3.0, 3.0], [3.0, 3.0]])
    assert compute_stable_rank(activations) == 0.0


def test_stable_rank_rank_one():
    # Hand-calculating a controlled matrix:
    # Let activations be a 2x2 matrix where centering leaves a clear Rank-1 structure.
    activations = torch.tensor([[1.0, 0.0], [-1.0, 0.0]])
    # Mean along dim 0 is [0.0, 0.0], so centered == activations.
    # Singular values of this matrix are sqrt(2) and 0.
    # Squared singular values = 2.0 and 0.0.
    # Sum = 2.0, Max = 2.0 -> Stable Rank = 2.0 / 2.0 = 1.0
    assert compute_stable_rank(activations) == pytest.approx(1.0)


def test_stable_rank_orthogonal():
    # For isotropic/orthogonal configurations, the dimensions should express higher rank
    activations = torch.tensor([[1.0, 0.0], [0.0, 1.0], [-1.0, -1.0]])
    # This yields well-distributed singular values across its dimensions.
    rank = compute_stable_rank(activations)
    assert rank > 1.0
