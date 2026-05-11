import torch
import torch.nn as nn
import numpy as np
from functional.utils import exponential_moving_average


def layer_init(
    layer: nn.Module, std: float = np.sqrt(2), bias_const: float = 0.0
) -> nn.Module:
    """
    Orthogonal initialization of weights and constant initialization of biases.
    Standard for PPO and other policy gradient methods in the CleanRL style.

    Args:
        layer (nn.Module): The layer to initialize.
        std (float): The scaling factor (gain) for orthogonal initialization.
        bias_const (float): The constant value to initialize the bias with.

    Returns:
        nn.Module: The initialized layer.
    """
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


# NOTE: This is a stateful function which is not ideal for the functional paradigm but follows standard pytorch and ml conventions.
def soft_update_target_network(
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
            target_param.copy_(exponential_moving_average(target_param, param, tau))


# NOTE: This is a stateful function which is not ideal for the functional paradigm but follows standard pytorch and ml conventions.
def hard_update_target_network(model: nn.Module, target_model: nn.Module) -> None:
    """
    Hard update of target network parameters.
    target_params = params

    Args:
        model (nn.Module): The model to update the target network from.
        target_model (nn.Module): The target network to update.
    """
    target_model.load_state_dict(model.state_dict())
