import torch
import torch.nn as nn
from functional.utils import exponential_moving_average


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


# TODO: is it okay that the function is not pure? ie it has side-effects
def hard_update_target_network(model: nn.Module, target_model: nn.Module) -> None:
    """
    Hard update of target network parameters.
    target_params = params

    Args:
        model (nn.Module): The model to update the target network from.
        target_model (nn.Module): The target network to update.
    """
    target_model.load_state_dict(model.state_dict())
