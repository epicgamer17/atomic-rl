import pytest
import torch
import numpy as np
import random


@pytest.fixture(autouse=True)
def seed_everything():
    """Seed all random number generators for reproducibility."""
    seed = 42
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # Ensure deterministic behavior for some ops if needed
    # torch.use_deterministic_algorithms(True)
    # NOTE: torch.use_deterministic_algorithms can be too restrictive/slow for some tests
