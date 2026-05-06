import pytest
import torch
import torch.nn as nn
from functional.network import hard_update_target_network, soft_update_target_network

pytestmark = pytest.mark.unit


def test_hard_update_target_network():
    model = nn.Linear(5, 1)
    target_model = nn.Linear(5, 1)

    # Initialize with different weights
    with torch.no_grad():
        model.weight.fill_(1.0)
        target_model.weight.fill_(0.0)

    hard_update_target_network(model, target_model)

    torch.testing.assert_close(target_model.weight, model.weight)
    assert target_model.weight[0, 0] == 1.0


def test_soft_update_target_network():
    model = nn.Linear(5, 1)
    target_model = nn.Linear(5, 1)

    # Initialize with different weights
    with torch.no_grad():
        model.weight.fill_(1.0)
        target_model.weight.fill_(0.0)

    # target = (1 - tau) * target + tau * model
    # target = (1 - 0.5) * 0.0 + 0.5 * 1.0 = 0.5
    soft_update_target_network(model, target_model, tau=0.5)

    torch.testing.assert_close(target_model.weight, torch.full_like(target_model.weight, 0.5))

    # Test with default tau (0.005)
    # target = (1 - 0.005) * 0.5 + 0.005 * 1.0 = 0.4975 + 0.005 = 0.5025
    soft_update_target_network(model, target_model)
    torch.testing.assert_close(target_model.weight, torch.full_like(target_model.weight, 0.5025))
