import torch
import torch.nn as nn
import numpy as np
from typing import Any, Optional, Tuple, Callable
from .utils import ema_update


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
