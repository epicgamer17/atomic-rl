import torch
from tensordict import TensorDict
from typing import Tuple, Callable, List
from .utils import add_dirichlet_noise


# TODO: remember we eventually want gumbel sequential halving and possibly other search methods too.
# TODO: dont hard code dirichlet params, pass em in as optional arguments
# TODO: avoid flags like is_zero_sum
# TODO: is there a better way to do masking than what we are doing, which is somewhat heuristic based.
def mcts_search(
    root_embeddings: torch.Tensor,
    num_simulations: int,
    num_actions: int,
    expansion_fn: Callable,  # Returns (policy_logits, value)
    dynamics_fn: Callable,  # Returns (next_embedding, reward) or (next_embedding, reward, next_to_play) or (next_embedding, reward, next_to_play, is_terminal)
    pb_c_base: float = 19652,
    pb_c_init: float = 1.25,
    gamma: float = 0.99,
    dirichlet_epsilon: float = 0.25,
    dirichlet_alpha: float = 0.3,
    root_to_play: torch.Tensor = None,
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
    priors = torch.softmax(policy_logits, dim=-1)

    # 3. Add Dirichlet Noise (Root exploration, masked to legal actions only)
    if dirichlet_epsilon > 0:
        root_legal_mask = priors > 1e-8
        priors = add_dirichlet_noise(
            priors, dirichlet_epsilon, dirichlet_alpha, mask=root_legal_mask
        )

    tree["children_prior"][:, 0] = priors

    for _ in range(num_simulations):
        # A. Selection: Find the best leaf using PUCT score
        leaf_indices, trajectory = select_leaf(tree, pb_c_base, pb_c_init)

        # The expansion happens at the end of the trajectory
        parent_nodes, actions_taken = trajectory[-1][0], trajectory[-1][1]

        # B. Dynamics (MuZero style): Transition to next state
        dyn_output = dynamics_fn(
            tree["embeddings"][batch_range, parent_nodes], actions_taken
        )
        if len(dyn_output) == 4:
            next_embeddings, rewards, next_to_play, is_terminal = dyn_output
        else:
            next_embeddings, rewards, next_to_play = dyn_output
            is_terminal = root_embeddings.new_zeros(batch_size, dtype=torch.bool)

        # C. Expansion & Evaluation: Predict policy and value for the leaf
        policy_logits, value = expansion_fn(next_embeddings)

        # For terminal states, value should be 0.0 (terminal state has no future expected return)
        value = torch.where(is_terminal, torch.zeros_like(value), value)

        # D. Expand Tree: Add the new node
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

        # E. Backpropagation: Update value/visit counts up the trajectory
        backpropagate(tree, trajectory, value, gamma)

    return tree


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
    span = torch.where(span > 1e-6, span, torch.ones_like(span))
    return (q_values - min_q) / span


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


# TODO: legal move masking for AlphaZero and terminal nodes for AlphaZero
# TODO: Sampled MuZero
# TODO: initial value for unvisited nodes, allow options, AlphaZero and MuZero: 0, Gumbel Muzero, v_mix, EfficientZero and Batch MCTS mean score
def expand_node(
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


# TODO: make this work with alternating and single player games. also make work for catan (inconsistent turn ordering, ie p1 twice then p2 3 times, then p3 once)
# TODO: make it work with more than 2 players
def backpropagate(
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


def get_mcts_visit_policy(
    visit_counts: torch.Tensor,
    temperature: float = 1.0,
) -> torch.Tensor:
    """
    Computes a target policy distribution from MCTS visit counts.

    Formula:
        for tau > 0: pi(a|s) = N(s, a)^(1/tau) / sum_b N(s, b)^(1/tau)
        for tau = 0: pi(a|s) = one_hot(argmax_a N(s, a))

    Args:
        visit_counts: Tensor of visit counts [B, A] or [A].
        temperature: Temperature parameter tau >= 0.

    Returns:
        torch.Tensor: Target policy probability distribution with same shape as visit_counts.
    """
    assert temperature >= 0.0, f"Temperature must be non-negative, got {temperature}"

    if temperature == 0.0:
        is_max = (
            visit_counts == torch.max(visit_counts, dim=-1, keepdim=True).values
        ).float()
        return is_max / is_max.sum(dim=-1, keepdim=True)

    if temperature == 1.0:
        total_visits = visit_counts.sum(dim=-1, keepdim=True)
        total_visits = torch.where(
            total_visits > 0, total_visits, torch.ones_like(total_visits)
        )
        return visit_counts / total_visits

    exponent = 1.0 / temperature
    scaled_visits = torch.pow(visit_counts.float(), exponent)
    total_scaled = scaled_visits.sum(dim=-1, keepdim=True)
    total_scaled = torch.where(
        total_scaled > 0, total_scaled, torch.ones_like(total_scaled)
    )
    return scaled_visits / total_scaled
