import torch
import torch.nn as nn
from typing import Callable, List
from torch.optim.optimizer import Optimizer

# TODO: CHANGE THE OPTIMIZER API HERE TO MATCH WITH IDBD, CBP, and ObGD for adam and SGD
# TODO: Modern PyTorch actually implements its optimizers using a functional core (available via torch.optim._functional or by directly using the stateless operations). You can create lightweight functional wrappers for Adam and SGD that perfectly match the signature of your adaptive_obgd_update_ without sacrificing PyTorch's C++ speed.
# TODO: Because functional optimizers require explicit state initialization (which torch.optim.Adam hides from you), you can create a single helper function in functional/initialization.py or functional/utils.py to instantiate these states, ensuring you don't violate the "Minimize the amount of code a person has to write" rule.


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


# TODO: ObGD, should it be a torch.optim.Optimizer subclass? or a functional approach? If we make it an Optimizer subclass should we try and do the same for the IDBD stuff? or the CBP?
"""
ObGD

Require: Eligibility trace zw, weight vector w, error δ, step size α, scaling factor κ
¯δ = max(|δ|, 1)
M ← ακ¯δ∥zw∥1 ▷ Note that zw = ∇wf for supervised learning
α ← min(α/M, α)
w ← w + αδzw
return w

NOTE: i think this is specifically for TD learning with semi gradient. May need to be adjusted for other TD learning methods. Additionally, I think as noted different for supervised learning. 

TODO: will need an ObGD for Adam and SGD seperately. Adam is in Appendix B of the Stream RL paper.
"""


# TODO: the paper defines one obgd_update and just passes in the grad as a trace, is that a better approach?
def obgd_update_(
    theta: torch.Tensor,
    grad: torch.Tensor,
    lr: float,
    scaling_factor: float = 1.0,
) -> None:
    """
    Standard Observation-based Gradient Descent step (supervised).
    """
    with torch.no_grad():
        norm_grad = torch.sum(torch.abs(grad))
        M = lr * scaling_factor * norm_grad
        new_step_size = lr / M.clamp(min=1.0)
        theta.sub_(grad, alpha=new_step_size)


def obgd_td_update_(
    theta: torch.Tensor,
    error: torch.Tensor,
    trace: torch.Tensor,
    lr: float,
    scaling_factor: float = 1.0,
) -> None:
    """
    Observation-based Gradient Descent driven by TD-error and eligibility traces.
    """
    with torch.no_grad():
        effective_error = torch.abs(error).clamp(min=1.0)
        norm_trace = torch.sum(torch.abs(trace))
        M = lr * scaling_factor * effective_error * norm_trace
        new_step_size = lr / M.clamp(min=1.0)
        theta.add_(trace, alpha=new_step_size * error)


class ObGD(Optimizer):
    """
    Observation-based Gradient Descent
    """

    def __init__(self, params, lr=1e-2, scaling_factor=1.0):
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        defaults = dict(lr=lr, scaling_factor=scaling_factor)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        """
        Performs a single optimization step (using p.grad).
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                obgd_update_(
                    theta=p,
                    grad=p.grad,
                    lr=group["lr"],
                    scaling_factor=group["scaling_factor"],
                )
        return loss

    # TODO: for now our solution. May want to better handle all TD methods, and its possible this is unecessary with some ways we pass gradients and stuff.
    @torch.no_grad()
    def td_step(self, error: torch.Tensor, traces: List[torch.Tensor], closure=None):
        """
        Performs a single temporal difference optimization step.
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        global_p_idx = 0
        for group in self.param_groups:
            for p in group["params"]:
                trace = traces[global_p_idx]
                global_p_idx += 1
                if trace is None:
                    continue
                obgd_td_update_(
                    theta=p,
                    error=error,
                    trace=trace,
                    lr=group["lr"],
                    scaling_factor=group["scaling_factor"],
                )
        return loss
