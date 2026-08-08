# TODO: clean up this a little maybe the api could be nicer.
# TODO: should we split the online and offline logic? they provide the same functionality, just one can have resetting of the LSTM state in the middle of the sequence and the other cant.
# NOTE: EfficientZero Reward LSTM uses similar to PPO, fixed horizon that resets periodically (every 5 or 3 steps), despite being an offline/off-policy algorithm.
# TODO: should R2D2 have an initial LSTM state from the buffer to kick off the burn in?
# TODO: what is BPTT?
# TODO: should the user handle the unrolling?
import torch
import torch.nn as nn
from typing import Any, Optional, Tuple


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
