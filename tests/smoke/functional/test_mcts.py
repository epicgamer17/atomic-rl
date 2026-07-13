import pytest
from tensordict import TensorDict
from functional.mcts import mcts_search
import torch

pytestmark = pytest.mark.smoke


# ==========================================
# End-to-End Search Orchestration (Smoke Test)
# ==========================================


def test_mcts_search_loop():
    """Run an end-to-end integration loop with functional mocks to catch unexpected crashes."""
    batch_size = 2
    num_simulations = 3
    num_actions = 2
    root_embeddings = torch.randn(batch_size, 4)

    # Mock neural network output dynamics
    def dummy_expansion(embeddings):
        logits = torch.ones(
            embeddings.shape[0], num_actions
        )  # Uniform likelihood distributions
        values = torch.zeros(embeddings.shape[0])
        return logits, values

    def dummy_dynamics(embeddings, actions):
        next_emb = embeddings.clone()
        rewards = torch.ones(embeddings.shape[0]) * 0.1
        next_to_play = torch.zeros(embeddings.shape[0], dtype=torch.long)
        return next_emb, rewards, next_to_play

    # Explicitly deactivate dirichlet noise injection to avoid mock patching utils modules
    tree = mcts_search(
        root_embeddings=root_embeddings,
        num_simulations=num_simulations,
        num_actions=num_actions,
        expansion_fn=dummy_expansion,
        dynamics_fn=dummy_dynamics,
        dirichlet_epsilon=0.0,
        gamma=0.95,
    )

    assert isinstance(tree, TensorDict)
    # 1 root node + 3 expansion cycles = 4 occupied node slots tracking records
    assert tree["node_counts"][0].item() == 4
    assert tree["node_counts"][1].item() == 4
