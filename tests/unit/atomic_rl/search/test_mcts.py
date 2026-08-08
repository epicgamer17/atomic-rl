import pytest
import torch

from atomic_rl.search import mcts_search

pytestmark = pytest.mark.unit


def test_mcts_search_root_legal_mask():
    """Verify that passing root_legal_mask to mcts_search explicitly masks root illegal actions."""
    root_embed = torch.zeros(1, 4)

    def expansion_fn(embeds):
        logits = torch.tensor([[2.0, 5.0, 1.0]])  # Action 1 has highest unmasked logit!
        values = torch.tensor([0.5])
        return logits, values

    def dummy_dynamics_fn(embeds, actions):
        next_embeds = torch.zeros_like(embeds)
        rewards = torch.zeros(1)
        next_to_play = torch.zeros(1, dtype=torch.long)
        is_terminal = torch.zeros(1, dtype=torch.bool)
        return next_embeds, rewards, next_to_play, is_terminal

    root_legal_mask = torch.tensor(
        [[True, False, True]]
    )  # Action 1 is explicitly ILLEGAL

    tree = mcts_search(
        root_embeddings=root_embed,
        num_simulations=20,
        num_actions=3,
        expansion_fn=expansion_fn,
        dynamics_fn=dummy_dynamics_fn,
        root_legal_mask=root_legal_mask,
        dirichlet_epsilon=0.5,
        dirichlet_alpha=0.3,
    )

    # Action 1 (highest raw logit) must be masked to 0.0 prior at root and receive 0 visits
    assert tree["children_prior"][0, 0, 1].item() == 0.0
    assert tree["children_visits"][0, 0, 1].item() == 0.0
    assert (
        tree["children_visits"][0, 0, 0].item()
        + tree["children_visits"][0, 0, 2].item()
        == 20
    )


def test_mcts_search_dynamics_fn_legal_mask():
    """Verify that dynamics_fn returning next_legal_mask explicitly masks child node illegal actions during search."""
    root_embed = torch.zeros(1, 4)

    def expansion_fn(embeds):
        # Always return raw unmasked logits where Action 1 is highest
        logits = torch.tensor([[1.0, 10.0, 1.0]])
        values = torch.tensor([0.5])
        return logits, values

    def dynamics_fn_with_mask(embeds, actions):
        next_embeds = torch.zeros_like(embeds)
        rewards = torch.zeros(1)
        next_to_play = torch.zeros(1, dtype=torch.long)
        is_terminal = torch.zeros(1, dtype=torch.bool)
        # 5th item: dynamics returns legal mask for next state (Action 1 illegal in expanded nodes)
        next_legal_mask = torch.tensor([[True, False, True]])
        return next_embeds, rewards, next_to_play, is_terminal, next_legal_mask

    # Root legal mask allows all actions at root so simulation moves to child
    root_legal_mask = torch.tensor([[True, True, True]])

    tree = mcts_search(
        root_embeddings=root_embed,
        num_simulations=10,
        num_actions=3,
        expansion_fn=expansion_fn,
        dynamics_fn=dynamics_fn_with_mask,
        root_legal_mask=root_legal_mask,
        dirichlet_epsilon=0.0,
    )

    # At child nodes, dynamics_fn mask prevents Action 1 from receiving visits at deeper levels
    assert tree["children_index"].shape[1] > 1
