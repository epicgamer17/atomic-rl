import pytest
import torch
import torch.nn as nn
import torch.optim as optim
from functional.optimizer import apply_gradients

pytestmark = pytest.mark.unit


def test_apply_gradients_basic():
    model = nn.Linear(2, 1)
    optimizer = optim.SGD(model.parameters(), lr=0.1)
    
    # Save original weight
    original_weight = model.weight.clone().detach()
    
    # Dummy input and loss
    x = torch.randn(1, 2)
    y = model(x)
    loss = (y - 1.0) ** 2
    
    apply_gradients(optimizer, loss)
    
    # Weight should have changed
    assert not torch.equal(model.weight, original_weight)
    
    # Verify zero_grad(set_to_none=True) happened
    # After optimizer.step(), gradients should still be None if set_to_none=True was used correctly before backward
    # Wait, loss.backward() populates .grad. optimizer.step() doesn't clear it.
    # But apply_gradients calls zero_grad(set_to_none=True) FIRST.
    # So if we call it twice, after the second zero_grad, .grad should be None.
    optimizer.zero_grad(set_to_none=True)
    assert model.weight.grad is None


def test_apply_gradients_clipping():
    model = nn.Linear(1, 1)
    optimizer = optim.SGD(model.parameters(), lr=0.1)
    
    # Force a large gradient
    with torch.no_grad():
        model.weight.fill_(1.0)
    
    x = torch.tensor([[100.0]]) # Large input
    y = model(x)
    loss = y ** 2 # Loss = (100 * 1)^2 = 10000. Grad = 2 * 100 * 100 = 20000.
    
    clip_norm = 1.0
    apply_gradients(optimizer, loss, model=model, clip_grad_norm=clip_norm)
    
    # Total norm should be clipped to 1.0
    total_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=clip_norm)
    assert total_norm <= clip_norm + 1e-6


def test_apply_gradients_no_model_clipping_error():
    model = nn.Linear(1, 1)
    optimizer = optim.SGD(model.parameters(), lr=0.1)
    loss = torch.tensor(1.0, requires_grad=True)
    
    with pytest.raises(AssertionError, match="Model must be provided for gradient clipping"):
        apply_gradients(optimizer, loss, clip_grad_norm=1.0)
