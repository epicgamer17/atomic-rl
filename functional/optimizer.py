import torch
import torch.nn as nn
from typing import Callable
from torch.optim.optimizer import Optimizer


def apply_gradients(
    optimizer: Optimizer,
    loss: torch.Tensor,
    model: nn.Module = None,
    clip_grad_norm: float = None,
):
    """
    Applies the gradients to the model.
    Args:
        optimizer (Optimizer): The optimizer.
        loss (torch.Tensor): The loss.
        model (nn.Module, optional): The model. Defaults to None.
        clip_grad_norm (float, optional): The gradient clipping norm. Defaults to None.
    Returns:
        Optimizer: The updated optimizer.
    """
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    if clip_grad_norm is not None:
        assert model is not None, "Model must be provided for gradient clipping"
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=clip_grad_norm)
    optimizer.step()
    return optimizer
