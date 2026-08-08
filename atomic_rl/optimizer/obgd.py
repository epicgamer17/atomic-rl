import torch
from typing import List, Union, Mapping
from torch.optim.optimizer import Optimizer

# The authors' official implementation of ObGD / AdaptiveObGD is in
# https://github.com/mohmdelsayed/streaming-drl/blob/main/src/optim.py — use it as a
# reference when verifying these update rules against the released code.
# TODO: CHANGE THE OPTIMIZER API HERE TO MATCH WITH IDBD, CBP, and ObGD for adam and SGD
# TODO: Modern PyTorch actually implements its optimizers using a functional core (available via torch.optim._functional or by directly using the stateless operations). You can create lightweight functional wrappers for Adam and SGD that perfectly match the signature of your adaptive_obgd_update_ without sacrificing PyTorch's C++ speed.
# TODO: Because functional optimizers require explicit state initialization (which torch.optim.Adam hides from you), you can create a single helper function in atomic_rl/initialization.py or atomic_rl/utils.py to instantiate these states, ensuring you don't violate the "Minimize the amount of code a person has to write" rule.


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
    total_norm: float | torch.Tensor,
    scaling_factor: float = 1.0,
) -> None:
    """
    Standard Overshooting-bounded Gradient Descent step (supervised).

    Args:
        theta (torch.Tensor): A parameter tensor of the network (modified in-place).
        grad (torch.Tensor): The gradient tensor for theta.
        lr (float): The base step size (alpha).
        total_norm (float | torch.Tensor): The L1 norm of the gradients of the
            WHOLE network (Algorithm 3 of the Stream RL paper). This is a single
            global norm shared by every parameter.
        scaling_factor (float): The overshooting scaling factor (kappa).

    Returns:
        None
    """
    # TODO: add shape assertions

    with torch.no_grad():
        # Paper Algorithm 3 normalizes by the L1 norm of the whole (concatenated)
        # gradient vector, giving the network a single shared step size.
        norm = torch.as_tensor(total_norm, dtype=torch.float32, device=theta.device)
        M = lr * scaling_factor * norm
        new_step_size = lr / M.clamp(min=1.0)
        theta.sub_(grad, alpha=new_step_size)


def obgd_td_update_(
    theta: torch.Tensor,
    error: torch.Tensor,
    trace: torch.Tensor,
    lr: float,
    total_norm: float | torch.Tensor,
    scaling_factor: float = 1.0,
) -> None:
    """
    Overshooting-bounded Gradient Descent driven by TD-error and eligibility traces.

    Args:
        theta (torch.Tensor): A parameter tensor of the network (modified in-place).
        error (torch.Tensor): The scalar TD-error (delta).
        trace (torch.Tensor): The eligibility trace tensor for theta.
        lr (float): The base step size (alpha).
        total_norm (float | torch.Tensor): The L1 norm of the eligibility traces of the
            WHOLE network (Algorithm 3 of the Stream RL paper). This is a single global norm.
        scaling_factor (float): The overshooting scaling factor (kappa).

    Returns:
        None
    """

    # TODO: add shape assertions

    with torch.no_grad():
        # Paper Algorithm 3 normali
        effective_error = torch.abs(error).clamp(min=1.0)
        norm = torch.as_tensor(total_norm, dtype=torch.float32, device=theta.device)
        M = lr * scaling_factor * effective_error * norm
        new_step_size = lr / M.clamp(min=1.0)
        theta.add_(trace, alpha=new_step_size * error)


# TODO: should we update this to use foreach or is it already efficient.
# TODO: not sure about the td_step API. brainstorming the idea of making a trace class. one for each of our different trace methods, and you call it before your optimizer step, after the backwards step. it sets a p.trace. that way we could have a standard step() api and not need td_step(). trouble is how does this work for optimizers that can do both TD and non TD learning (definitely ObGD and i think thats even IDBD which has a TD version but is not forced to do TD only? is that true?). The step function now performs two functions, either a TD step or a standard gradient step. how would it be clear to the user which behaviour is happening? Is the solution to have the trace class replace the p.grad instead? This has the benefit of allowing standard optimizers like Adam to also perform TD learning i think (though do we even want that)? Is that viable for ObGD or IDBD do they need both the gradient and TD error? think about this more.
# NOTE: a benefit of the step and td_step api (among many): Preserves standard step(). If your agent has a separate auxiliary module (like a world model or an autoencoder) trained via standard supervised backpropagation, you can pass its parameters to the exact same optimizer and call optimizer.step() normally.
class ObGD(Optimizer):
    """
    Overshooting-bounded Gradient Descent (ObGD - SGD variant).
    Implementation of Algorithm 3 from Elsayed et al. (2024).
    """

    def __init__(self, params, lr=1.0, scaling_factor=1.0):
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        defaults = dict(lr=lr, scaling_factor=scaling_factor)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        """
        Performs a single supervised optimization step.
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        device = (
            next(
                p.device
                for group in self.param_groups
                for p in group["params"]
                if p.grad is not None
            )
            if any(
                p.grad is not None
                for group in self.param_groups
                for p in group["params"]
            )
            else None
        )
        total_norm = (
            torch.tensor(0.0, device=device)
            if device is not None
            else torch.tensor(0.0)
        )
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                total_norm += torch.sum(torch.abs(p.grad))

        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                obgd_update_(
                    theta=p,
                    grad=p.grad,
                    lr=group["lr"],
                    scaling_factor=group["scaling_factor"],
                    total_norm=total_norm,
                )
        return loss

    # TODO: for now our solution. May want to better handle all TD methods, and its possible this is unecessary with some ways we pass gradients and stuff.
    @torch.no_grad()
    def td_step(
        self,
        error: torch.Tensor,
        traces: Union[List[torch.Tensor], Mapping[torch.Tensor, torch.Tensor]],
        closure=None,
    ):
        """
        Performs a single temporal difference optimization step.
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        def resolve_trace(p: torch.Tensor, idx: int):
            if isinstance(traces, Mapping):
                if p not in traces:
                    raise KeyError(
                        f"Parameter trace not found in traces mapping for param: {p}"
                    )
                return traces[p], idx
            return traces[idx], idx + 1

        device = (
            next(p.device for group in self.param_groups for p in group["params"])
            if self.param_groups and self.param_groups[0]["params"]
            else None
        )
        total_norm = (
            torch.tensor(0.0, device=device)
            if device is not None
            else torch.tensor(0.0)
        )
        idx = 0
        for group in self.param_groups:
            for p in group["params"]:
                trace, idx = resolve_trace(p, idx)
                if trace is None:
                    continue
                total_norm += torch.sum(torch.abs(trace))

        idx = 0
        for group in self.param_groups:
            for p in group["params"]:
                trace, idx = resolve_trace(p, idx)
                if trace is None:
                    continue
                obgd_td_update_(
                    theta=p,
                    error=error,
                    trace=trace,
                    lr=group["lr"],
                    scaling_factor=group["scaling_factor"],
                    total_norm=total_norm,
                )
        return loss
