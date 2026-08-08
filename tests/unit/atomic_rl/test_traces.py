import pytest
import torch
from atomic_rl.traces import compute_accumulating_traces, compute_replacing_traces

pytestmark = pytest.mark.unit

def test_update_accumulating_traces():
    # traces: [2, 3], gradients: [2, 3], terminated: [2]
    traces = torch.tensor([[1.0, 1.0, 1.0], [1.0, 1.0, 1.0]])
    gradients = torch.tensor([[0.5, 0.0, 0.0], [0.0, 0.5, 0.0]])
    terminated = torch.tensor([0.0, 1.0])
    gamma = 0.9
    lam = 0.9
    
    # For batch 0 (not terminated): 0.9 * 0.9 * 1.0 + grad = 0.81 + grad
    # [0.81 + 0.5, 0.81 + 0.0, 0.81 + 0.0] = [1.31, 0.81, 0.81]
    
    # For batch 1 (terminated): trace reset to 0 + grad
    # [0.0, 0.5, 0.0]
    
    expected = torch.tensor([
        [1.31, 0.81, 0.81],
        [0.0, 0.5, 0.0]
    ])
    
    res = compute_accumulating_traces(traces, gradients, gamma, lam, terminated)
    torch.testing.assert_close(res, expected)

def test_update_replacing_traces():
    traces = torch.tensor([[1.0, 1.0, 1.0], [1.0, 1.0, 1.0]])
    features = torch.tensor([[0.0, 1.0, 0.0], [0.0, 1.0, 0.0]])
    terminated = torch.tensor([0.0, 1.0])
    gamma = 0.9
    lam = 0.9
    
    # For batch 0 (not terminated): 0.9 * 0.9 * 1.0 = 0.81
    # max(0.81, feature)
    # feature = [0.0, 1.0, 0.0] -> max -> [0.81, 1.0, 0.81]
    
    # For batch 1 (terminated): trace reset to 0
    # max(0.0, feature)
    # feature = [0.0, 1.0, 0.0] -> max -> [0.0, 1.0, 0.0]
    
    expected = torch.tensor([
        [0.81, 1.0, 0.81],
        [0.0, 1.0, 0.0]
    ])
    
    res = compute_replacing_traces(traces, features, gamma, lam, terminated)
    torch.testing.assert_close(res, expected)

def test_traces_assertions():
    with pytest.raises(AssertionError, match="Trace and gradient shapes must match"):
        compute_accumulating_traces(torch.randn(2, 3), torch.randn(2, 2), 0.9, 0.9, torch.randn(2))
