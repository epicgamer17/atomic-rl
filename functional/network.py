import torch
import torch.nn as nn
import numpy as np
from typing import Any, Optional, Tuple, Callable
from functional.utils import ema_update


# NOTE: This is a stateful function which is not ideal for the functional paradigm but follows standard pytorch and ml conventions.
def soft_update_target_network_(
    model: nn.Module, target_model: nn.Module, tau: float = 0.005
) -> None:
    """
    Soft update of target network parameters.
    target_params = (1 - tau) * target_params + tau * params

    Args:
        model (nn.Module): The model to update the target network from.
        target_model (nn.Module): The target network to update.
        tau (float): The soft update coefficient.
    """
    with torch.no_grad():
        for target_param, param in zip(target_model.parameters(), model.parameters()):
            target_param.copy_(ema_update(target_param, param, tau))


# NOTE: This is a stateful function which is not ideal for the functional paradigm but follows standard pytorch and ml conventions.
def hard_update_target_network_(model: nn.Module, target_model: nn.Module) -> None:
    """
    Hard update of target network parameters.
    target_params = params

    Args:
        model (nn.Module): The model to update the target network from.
        target_model (nn.Module): The target network to update.
    """
    target_model.load_state_dict(model.state_dict())


# TODO: clean up this a little maybe the api could be nicer.
# TODO: should we split the online and offline logic? they provide the same functionality, just one can have resetting of the LSTM state in the middle of the sequence and the other cant.
# NOTE: EfficientZero Reward LSTM uses similar to PPO, fixed horizon that resets periodically (every 5 or 3 steps), despite being an offline/off-policy algorithm.
# TODO: should R2D2 have an initial LSTM state from the buffer to kick off the burn in?
# TODO: what is BPTT?
def unroll_rnn(
    cell: nn.Module,
    inputs: torch.Tensor,
    initial_state: Any,
    dones: Optional[torch.Tensor] = None,
    batch_first: bool = False,
) -> Tuple[torch.Tensor, Any]:
    """
    Vectorized RNN unrolling with support for mid-sequence state resets.

    Args:
        cell (nn.Module): The RNN module (e.g., nn.LSTM, nn.GRU).
        inputs (torch.Tensor): Input tensor of shape [Batch, Time, Features].
        initial_state (Any): Initial recurrent state(s).
        dones (Optional[torch.Tensor]): Optional binary mask of shape [Batch, Time].
            1.0 indicates the start of a new episode (reset state).
        batch_first (bool): Indicates if the cell expects batch-major inputs.

    Returns:
        Tuple[torch.Tensor, Any]:
            - outputs: [Batch, Time, HiddenSize]
            - final_state: Final recurrent state(s).
    """
    if dones is None:
        # Fast path for natively compiled unrolling (e.g., DRQN random updates)
        if batch_first:
            out, state = cell(inputs, initial_state)
            return out.contiguous(), state  # <--- Fix applied here

        # [B, T, F] -> [T, B, F]
        out, state = cell(inputs.transpose(0, 1), initial_state)
        # .contiguous() is fundamentally required here because the transpose makes the sequence-major output non-contiguous.
        # If the caller attempts to .view() or flatten the sequence dimensions for MLP processing, it will trigger a RuntimeError.
        return out.transpose(0, 1).contiguous(), state

    # Slower path for sequences with mid-stream resets (e.g., PPO rollouts)
    outputs = []
    state = initial_state

    # Transpose for time-major iteration: [B, T, ...] -> [T, B, ...]
    inputs_t = inputs.transpose(0, 1)
    dones_t = dones.transpose(0, 1)

    for x_t, d_t in zip(inputs_t, dones_t):
        # mask shape: [1, B, 1] to broadcast across [Layers, Batch, Hidden]
        mask = (1.0 - d_t).view(1, -1, 1)

        # Use tree_map to structurally mask all nested state tensors simultaneously (handles LSTM tuples vs GRU scalars cleanly)
        state = torch.utils._pytree.tree_map(lambda s: s * mask, state)

        # cell expects [Seq, Batch, Features], so we unsqueeze(0) for seq_len=1
        out, state = cell(x_t.unsqueeze(0), state)
        outputs.append(out)

    # .contiguous() is fundamentally required here because the transpose makes the sequence-major output non-contiguous.
    # If the caller attempts to .view() or flatten the sequence dimensions for MLP processing, it will trigger a RuntimeError.
    return torch.cat(outputs, dim=0).transpose(0, 1).contiguous(), state

    # NOTE/TODO: there are many ways of handling recurrent states in learning. they tend to differ between online and offline. online simply uses the hidden states from the collection steps, so no burn in is requred. the below method is one way of handling this for offline learning which requires a burn in. Make this more clear.
    # TODO: thoughts on this API:
    # hidden_state = online_net.init_hidden(batch_size, batch.device)
    # target_hidden_state = target_net.init_hidden(batch_size, batch.device)
    # and having the user sort of handle the unrolling?


def make_burn_in_evaluator(
    model: torch.nn.Module, dones: torch.Tensor, batch_size: int
) -> Callable[[torch.Tensor], torch.Tensor]:
    """
    Higher-order function that creates a stateless evaluator for Q-learning.
    It automatically initializes the zeroed hidden states required to correctly do the burn in phase for offline recurrent learning algorithms (like DRQN) and handles the recurrent forward pass.
    """

    def evaluator(obs: torch.Tensor) -> torch.Tensor:
        # Fail Fast: Ensure we have the flat tensor expected
        assert obs.ndim >= 2, f"Expected flat batched obs, got {obs.shape}"

        # Dynamically pull dimensions from the model's LSTM
        num_layers = model.lstm.num_layers
        hidden_size = model.lstm.hidden_size
        device = obs.device

        # Initialize zero states (DRQN Bootstrapped Random Update rule)
        zero_state = (
            obs.new_zeros(num_layers, batch_size, hidden_size),
            obs.new_zeros(num_layers, batch_size, hidden_size),
        )

        # Forward pass
        q_values, _ = model(
            x=obs, lstm_state=zero_state, dones=dones, batch_size=batch_size
        )
        return q_values

    return evaluator
