import pytest
import torch
import torch.nn as nn
from typing import Tuple
from atomic_rl.network import (
    hard_update_target_network_,
    soft_update_target_network_,
    unroll_rnn,
    make_burn_in_evaluator,
)

pytestmark = pytest.mark.unit


# ==========================================
# Mock Components for Isolated Behavior Tests
# ==========================================


class MockRNNCell(nn.Module):
    """A dummy RNN cell that captures the hidden state passed to it at every timestep."""

    def __init__(self):
        super().__init__()
        self.recorded_states = []

    def forward(self, x: torch.Tensor, state: Tuple[torch.Tensor, torch.Tensor]):
        # Clone states to capture snapshot before in-place modifications happen down the line
        cloned_state = torch.utils._pytree.tree_map(lambda s: s.clone(), state)
        self.recorded_states.append(cloned_state)
        # Return dummy outputs matching shape [SeqLen, Batch, HiddenSize]
        # matching input's [SeqLen, Batch, Features] shape
        seq_len, batch_size, _ = x.shape
        return torch.zeros(seq_len, batch_size, 4, device=x.device), state


class DummyDRQN(nn.Module):
    """Mock architecture mimicking a DRQN with an explicit LSTM sub-attribute structure."""

    def __init__(self, num_layers=2, hidden_size=8):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=4, hidden_size=hidden_size, num_layers=num_layers
        )
        self.recorded_forward_args = {}

    def forward(self, x, lstm_state, dones, batch_size):
        self.recorded_forward_args = {
            "x": x,
            "lstm_state": lstm_state,
            "dones": dones,
            "batch_size": batch_size,
        }
        # Return mock Q-values [Batch, Actions]
        return torch.ones(batch_size, 3, device=x.device), lstm_state


def test_hard_update_target_network():
    model = nn.Linear(5, 1)
    target_model = nn.Linear(5, 1)

    # Initialize with different weights
    with torch.no_grad():
        model.weight.fill_(1.0)
        target_model.weight.fill_(0.0)

    hard_update_target_network_(model, target_model)

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
    soft_update_target_network_(model, target_model, tau=0.5)

    torch.testing.assert_close(
        target_model.weight, torch.full_like(target_model.weight, 0.5)
    )

    # Test with default tau (0.005)
    # target = (1 - 0.005) * 0.5 + 0.005 * 1.0 = 0.4975 + 0.005 = 0.5025
    soft_update_target_network_(model, target_model)
    torch.testing.assert_close(
        target_model.weight, torch.full_like(target_model.weight, 0.5025)
    )


# ==========================================
# Tests for unroll_rnn
# ==========================================


def test_unroll_rnn_fast_path_batch_first():
    """Verify standard native fast-path execution loop with batch-major dimensions."""
    cell = nn.GRU(input_size=4, hidden_size=6, num_layers=1, batch_first=True)
    inputs = torch.randn(2, 3, 4)  # [Batch=2, Time=3, Features=4]
    h0 = torch.zeros(1, 2, 6)

    out, hn = unroll_rnn(cell, inputs, h0, dones=None, batch_first=True)

    assert out.shape == (2, 3, 6)
    assert out.is_contiguous()


def test_unroll_rnn_fast_path_sequence_first():
    """Verify native fast-path sequence alignment transformations and memory continuity checks."""
    cell = nn.GRU(input_size=4, hidden_size=6, num_layers=1, batch_first=False)
    inputs = torch.randn(2, 3, 4)  # [Batch=2, Time=3, Features=4]
    h0 = torch.zeros(1, 2, 6)

    out, hn = unroll_rnn(cell, inputs, h0, dones=None, batch_first=False)

    assert out.shape == (2, 3, 6)
    assert out.is_contiguous()


def test_unroll_rnn_mid_sequence_resets():
    """Verify that incoming tracking traces zero out recurrent hidden states when done flags hit."""
    cell = MockRNNCell()

    # Batch size 2, Time horizon 3, Feature size 2
    inputs = torch.randn(2, 3, 2)

    # Element 0: Done triggers at step 1. Element 1: Remains clean throughout sequence.
    dones = torch.tensor([[0.0, 1.0, 0.0], [0.0, 0.0, 0.0]])

    # Initialize with clean ones to monitor exact degradation effects
    h0 = torch.ones(1, 2, 4)  # [Layers, Batch, Hidden]
    c0 = torch.ones(1, 2, 4)
    initial_state = (h0, c0)

    out, final_state = unroll_rnn(cell, inputs, initial_state, dones=dones)

    assert out.shape == (2, 3, 4)
    assert out.is_contiguous()

    # Evaluate step states from the captured mock arrays
    # Timestep 0: No resets active yet
    h_t0, c_t0 = cell.recorded_states[0]
    assert torch.all(h_t0 == 1.0)

    # Timestep 1: Element 0 is marked done -> state indices must clear to 0.0
    h_t1, c_t1 = cell.recorded_states[1]
    assert torch.all(h_t1[:, 0, :] == 0.0)  # Batch element 0 reset
    assert torch.all(h_t1[:, 1, :] == 1.0)  # Batch element 1 preserved


# ==========================================
# Tests for make_burn_in_evaluator
# ==========================================


def test_make_burn_in_evaluator_execution():
    """Verify state initialization and parameter pipeline inside the higher-order closure."""
    model = DummyDRQN(num_layers=2, hidden_size=8)
    dones = torch.zeros(2, 4)
    batch_size = 2

    evaluator = make_burn_in_evaluator(model, dones, batch_size)
    obs = torch.randn(batch_size, 4)

    q_values = evaluator(obs)

    # Validate correct structure mappings
    assert q_values.shape == (2, 3)

    # Verify the automatic generation of fresh zeroed states for burn-in tasks
    h0, c0 = model.recorded_forward_args["lstm_state"]
    assert h0.shape == (2, 2, 8)  # [num_layers, batch_size, hidden_size]
    assert torch.all(h0 == 0.0)
    assert torch.all(c0 == 0.0)


def test_make_burn_in_evaluator_fail_fast():
    """Verify that the closure enforces input dimensionality matching rules immediately."""
    model = DummyDRQN()
    evaluator = make_burn_in_evaluator(model, torch.zeros(1), batch_size=1)

    # Passing an unbatched/flat scalar input (0D tensor) should trigger the assertion error
    with pytest.raises(AssertionError, match="Expected flat batched obs"):
        evaluator(torch.tensor(1.5))
