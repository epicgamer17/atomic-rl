import torch
from tensordict import TensorDict
from typing import Tuple, Callable, List, Optional
from ..utils import add_dirichlet_noise


# TODO: should we merge this with select_leaf?
def puct_score(
    q_values: torch.Tensor,
    policy_prior: torch.Tensor,
    visit_counts: torch.Tensor,
    total_visit_counts: torch.Tensor,
    min_q: torch.Tensor,
    max_q: torch.Tensor,
    pb_c_base: float = 19652,
    pb_c_init: float = 1.25,
) -> torch.Tensor:
    """
    The PUCT score.

    The formula is: Q(s,a) + U(s,a)
    where U(s,a) = C * P(s,a) * sqrt(N(s)) / (1 + N(s,a))

    Args:
        q_values: Q-values of the actions.
        policy_prior: Prior probabilities of the actions. Not logits!
        visit_counts: Visit counts for each action.
        total_visit_counts: Total visit count for the parent state.
        pb_c_base: Base constant for PUCT.
        pb_c_init: Additive constant for PUCT (used for virtual exploration).
    """
    # 1. Fail Fast: Ensure shape contracts match expected [B, num_actions] and [B, 1] dimensions
    assert (
        q_values.shape == policy_prior.shape
    ), f"q_values shape {q_values.shape} must match policy_prior shape {policy_prior.shape}"
    assert (
        q_values.shape == visit_counts.shape
    ), f"q_values shape {q_values.shape} must match visit_counts shape {visit_counts.shape}"
    assert (
        total_visit_counts.shape[:-1] == q_values.shape[:-1]
    ), f"total_visit_counts batch shape {total_visit_counts.shape[:-1]} must match q_values batch shape {q_values.shape[:-1]}"

    # Ensure policy prior is normalized (sums to 1)
    assert torch.allclose(
        policy_prior.sum(dim=-1),
        torch.ones_like(policy_prior.sum(dim=-1)),
        atol=1e-5,
    ), "Policy prior must be normalized (sum to 1) for PUCT calculation."

    tot_visits_t = torch.as_tensor(
        total_visit_counts, dtype=q_values.dtype, device=q_values.device
    )

    pb_c = torch.log((tot_visits_t + pb_c_base + 1) / pb_c_base) + pb_c_init
    pb_c = pb_c * (torch.sqrt(tot_visits_t) / (visit_counts + 1))

    # MuZero: Normalize Q-values to [0, 1] before adding the PUCT term
    normalized_q = normalize_q_values(q_values, min_q, max_q)
    raw_puct = normalized_q + pb_c * policy_prior

    # Zero-prior guard: Actions with 0 prior (e.g. masked illegal actions) receive -1e9 penalty
    return torch.where(policy_prior > 0, raw_puct, raw_puct.new_tensor(-1e9))


# TODO: work with Batched MCTS, batch_mcts.pdf
# TODO: work with Vectorized MCTS
# TODO: work iwth Batched + Vectorized MCTS
# TODO: can we reuse our action_selection.py methods/functions?
# TODO: Stochastic MuZero
def select_leaf(
    tree: TensorDict, pb_c_base: float, pb_c_init: float, max_depth: int = 512
) -> Tuple[torch.Tensor, List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]]:
    """
    Selects a leaf node to expand by following the PUCT policy.

    Args:
        tree: The MCTS tree TensorDict.
        pb_c_base: PUCT base constant.
        pb_c_init: PUCT init constant.
        max_depth: Maximum depth to search to avoid infinite loops. For default AlphaZero and MuZero behaviour simply set this to num_simulations.

    Returns:
        A tuple containing:
            - leaf_indices: The indices of the selected leaf nodes [B].
            - trajectory: A list of (node_idx, action_idx, mask) tuples for backpropagation.
    """
    batch_size = tree.batch_size[0]
    device = tree.device
    batch_range = torch.arange(batch_size, device=device)

    current_node = tree["min_q"].new_zeros(batch_size, dtype=torch.long)
    trajectory = []  # List of (node_idx, action_idx, mask)

    # Track which batch elements are still descending the tree
    active_mask = tree["min_q"].new_ones(batch_size, dtype=torch.bool)

    # The search depth is naturally bounded by the number of nodes or a safety limit
    for _ in range(max_depth):
        # 1. Get stats for current nodes
        q_values = tree["children_q_values"][batch_range, current_node]
        priors = tree["children_prior"][batch_range, current_node]
        visits = tree["children_visits"][batch_range, current_node]
        total_visits = visits.sum(dim=-1, keepdim=True)

        # 2. Calculate PUCT scores
        scores = puct_score(
            q_values,
            priors,
            visits,
            total_visits,
            tree["min_q"],
            tree["max_q"],
            pb_c_base,
            pb_c_init,
        )

        # 3. Select best action
        # TODO: should we use action_selection.py here? Does it make sense to use it?
        action = torch.argmax(scores, dim=-1)

        # 4. Check for leaf (child index is -1) or terminal node
        next_node = tree["children_index"][batch_range, current_node, action]
        is_leaf = next_node == -1
        is_term = tree["is_terminal"][batch_range, current_node]

        # 5. Record to trajectory (only for elements that were active at the START of this step)
        trajectory.append((current_node.clone(), action.clone(), active_mask.clone()))

        # 6. Update active mask: those who hit a leaf or terminal node are no longer active for the NEXT step
        active_mask = active_mask & (~is_leaf) & (~is_term)

        if not active_mask.any():
            break

        # Update current_node for elements that haven't hit a leaf/terminal
        current_node = torch.where(active_mask, next_node, current_node)

    return current_node, trajectory


# TODO: soft min max stats? efficient_zero.pdf
def normalize_q_values(
    q_values: torch.Tensor, min_q: torch.Tensor, max_q: torch.Tensor
) -> torch.Tensor:
    """
    Normalizes Q-values to [0, 1] using the min/max observed in the tree.

    Args:
        q_values: Q-values to normalize [..., Num_Actions].
        min_q: Minimum observed Q-value per batch element [B].
        max_q: Maximum observed Q-value per batch element [B].
    """
    # Reshape min/max for broadcasting if q_values is [B, A]
    if q_values.ndim > min_q.ndim:
        min_q = min_q.view(-1, 1)
        max_q = max_q.view(-1, 1)

    span = max_q - min_q
    # Protect against division by zero and handle uninitialized min/max
    # TODO: is this justified by the original paper or is this soft min max stats from efficient zero bleeding through?
    span = torch.where(span > 1e-6, span, torch.ones_like(span))
    return (q_values - min_q) / span
