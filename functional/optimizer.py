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


def obgd_update_(
    params: List[torch.Tensor],
    traces: List[
        torch.Tensor
    ],  # Add or document the trace used for supervised learning.
    error: torch.Tensor,
    base_lr: float,
    scaling_factor: float = 1.0,
):
    """ """
    with torch.no_grad():
        effective_error = torch.abs(error).clamp(min=1.0)

        for param, trace in zip(params, traces):
            norm_trace = torch.sum(torch.abs(trace))

            M = base_lr * scaling_factor * effective_error * norm_trace

            # Clip step size
            new_step_size = base_lr / M.clamp(min=1.0)

            # Update weights
            param.add_(new_step_size * error * trace)


# TODO: improve to make this work better with td learning and also have the standard step() api. brainstorm solutions. One possible solution is a new td_step API.
class ObGD(Optimizer):
    """
    Observation-based Gradient Descent
    """

    def __init__(self, params, lr=1e-2, scaling_factor=1.0):
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        defaults = dict(lr=lr, scaling_factor=scaling_factor)
        super().__init__(params, defaults)

    # TODO: make error and traces optional?
    @torch.no_grad()
    def step(
        self, error: torch.Tensor, traces: List[torch.Tensor] = None, closure=None
    ):
        """
        Performs a single optimization step.

        Args:
            error (torch.Tensor): The scalar TD error or supervised loss (δ).
            traces (List[torch.Tensor], optional): The eligibility traces. If None, uses p.grad.
            # TODO: should it be a list of torch.Tensor or just a torch.Tensor?
        """

        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        global_p_idx = 0
        for group in self.param_groups:
            params_with_grad = []
            grads_or_traces = []

            for p in group["params"]:
                if traces is not None:
                    trace = traces[global_p_idx]
                else:
                    trace = p.grad

                global_p_idx += 1

                if trace is None:
                    continue

                params_with_grad.append(p)
                grads_or_traces.append(trace)

            # Delegate the actual math to our purely functional, stateless core!
            obgd_update_(
                params=params_with_grad,
                traces=grads_or_traces,
                error=error,
                base_lr=group["lr"],
                scaling_factor=group["scaling_factor"],
            )

        return loss
