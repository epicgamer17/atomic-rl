import torch
from typing import List, Union, Mapping
from torch.optim.optimizer import Optimizer

# The authors' official implementation of ObGD / AdaptiveObGD is in
# https://github.com/mohmdelsayed/streaming-drl/blob/main/src/optim.py — use it as a
# reference when verifying these update rules against the released code.
# TODO: CHANGE THE OPTIMIZER API HERE TO MATCH WITH IDBD, CBP, and ObGD for adam and SGD
# TODO: Modern PyTorch actually implements its optimizers using a functional core (available via torch.optim._functional or by directly using the stateless operations). You can create lightweight functional wrappers for Adam and SGD that perfectly match the signature of your adaptive_obgd_update_ without sacrificing PyTorch's C++ speed.
# TODO: Because functional optimizers require explicit state initialization (which torch.optim.Adam hides from you), you can create a single helper function in atomic_rl/initialization.py or atomic_rl/utils.py to instantiate these states, ensuring you don't violate the "Minimize the amount of code a person has to write" rule.


# TODO: the paper defines one obgd_update and just passes in the grad as a trace, is that a better approach?
def adaptive_obgd_update_(
    theta: torch.Tensor,
    grad: torch.Tensor,
    v: torch.Tensor,
    lr: float,
    total_norm: float | torch.Tensor,
    scaling_factor: float = 1.0,
    eps: float = 1e-8,
) -> None:
    """
    Adaptive Overshooting-bounded Gradient Descent step (supervised).
    Follows Appendix B Algorithm 11 of Elsayed et al. (2024).

    Args:
        theta (torch.Tensor): A parameter tensor of the network (modified in-place).
        grad (torch.Tensor): The gradient tensor for theta.
        v (torch.Tensor): Second moment vector for theta used for normalization.
            To match the reference implementation, pass the bias-corrected
            v_hat = v / (1 - beta^step) here.
        lr (float): Base step size (alpha).
        total_norm (float | torch.Tensor): The global L1 norm of normalized gradients ||g / sqrt(v + eps)||_1
            summed across the ENTIRE network.
        scaling_factor (float): Overshooting scaling factor (kappa).
        eps (float): Numerical stability constant (default 1e-8).

    Returns:
        None
    """
    # TODO: add shape assertions

    with torch.no_grad():
        norm = torch.as_tensor(total_norm, dtype=torch.float32, device=theta.device)
        M = lr * scaling_factor * norm
        new_step_size = lr / M.clamp(min=1.0)
        adj_grad = grad / torch.sqrt(v + eps)
        theta.sub_(adj_grad, alpha=new_step_size)


def adaptive_obgd_td_update_(
    theta: torch.Tensor,
    error: torch.Tensor,
    trace: torch.Tensor,
    v: torch.Tensor,
    lr: float,
    total_norm: float | torch.Tensor,
    scaling_factor: float = 1.0,
    eps: float = 1e-8,
) -> None:
    """
    Adaptive Overshooting-bounded Gradient Descent step for TD learning (semi-gradient with eligibility traces).
    Follows Appendix B Algorithm 11 of Elsayed et al. (2024).

    Args:
        theta (torch.Tensor): A parameter tensor of the network (modified in-place).
        error (torch.Tensor): Scalar TD error (delta).
        trace (torch.Tensor): Eligibility trace tensor for theta (z_w).
        v (torch.Tensor): Second moment vector for theta used for normalization.
            To match the reference implementation, pass the bias-corrected
            v_hat = v / (1 - beta^step) here.
        lr (float): Base step size (alpha).
        total_norm (float | torch.Tensor): The global L1 norm of normalized traces ||z_w / sqrt(v + eps)||_1
            summed across the ENTIRE network.
        scaling_factor (float): Overshooting scaling factor (kappa).
        eps (float): Numerical stability constant (default 1e-8).

    Returns:
        None
    """

    # TODO: add shape assertions

    with torch.no_grad():
        effective_error = torch.abs(error).clamp(min=1.0)
        norm = torch.as_tensor(total_norm, dtype=torch.float32, device=theta.device)
        M = lr * scaling_factor * effective_error * norm
        new_step_size = lr / M.clamp(min=1.0)
        adj_trace = trace / torch.sqrt(v + eps)
        theta.add_(adj_trace, alpha=new_step_size * error)


class AdaptiveObGD(Optimizer):
    """
    Adaptive Overshooting-bounded Gradient Descent (ObGD Adam).
    Implementation of Algorithm 11 (Appendix B) from Elsayed et al. (2024).

    NOTE (reference vs. paper): We intentionally match the authors' released code (https://github.com/mohmdelsayed/streaming-drl/blob/main/src/optim.py) rather than the algorithm as written in the paper. The reference `AdaptiveObGD.step` applies a bias correction v_hat = v / (1 - beta^step) to the EMA second moment before normalizing; this correction is NOT in paper Algorithm 11. We follow the reference and apply it — a conscious and intentional decision for parity.

    NOTE (intentional divergence from the reference): The reference keeps eligibility traces internally (constructor takes gamma, lamda, kappa) and exposes a single .step(); we intentionally split trace management into atomic_rl.traces + td_step(error, traces).
    """

    def __init__(
        self,
        params,
        lr: float = 1.0,
        scaling_factor: float = 1.0,
        beta: float = 0.999,
        eps: float = 1e-8,
    ):
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not 0.0 <= beta < 1.0:
            raise ValueError(f"Invalid beta parameter: {beta}")
        if eps <= 0.0:
            raise ValueError(f"Invalid epsilon parameter: {eps}")

        defaults = dict(lr=lr, scaling_factor=scaling_factor, beta=beta, eps=eps)
        super().__init__(params, defaults)
        self.counter = (
            0  # bias-correction step counter (reference optim.py self.counter)
        )

    @torch.no_grad()
    def step(self, closure=None):
        """
        Performs a single supervised optimization step (Algorithm 11).
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        self.counter += 1

        device = next(
            (
                p.device
                for group in self.param_groups
                for p in group["params"]
                if p.grad is not None
            ),
            None,
        )
        total_norm = (
            torch.tensor(0.0, device=device)
            if device is not None
            else torch.tensor(0.0)
        )

        for group in self.param_groups:
            beta = group["beta"]
            eps = group["eps"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                state = self.state[p]
                if "v" not in state:
                    state["v"] = torch.zeros_like(p)
                v = state["v"]

                # v <- beta * v + (1 - beta) * (grad)^2
                v.mul_(beta).addcmul_(p.grad, p.grad, value=1.0 - beta)

                # Bias correction (reference optim.py): v_hat = v / (1 - beta^step).
                v_hat = v / (1.0 - beta**self.counter)

                # || grad / sqrt(v_hat + eps) ||_1
                adj_grad = p.grad / torch.sqrt(v_hat + eps)
                total_norm += torch.sum(torch.abs(adj_grad))

        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                state = self.state[p]
                beta = group["beta"]
                v_hat = state["v"] / (1.0 - beta**self.counter)
                adaptive_obgd_update_(
                    theta=p,
                    grad=p.grad,
                    v=v_hat,
                    lr=group["lr"],
                    scaling_factor=group["scaling_factor"],
                    total_norm=total_norm,
                    eps=group["eps"],
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
        Performs a single temporal difference optimization step (Algorithm 11).
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        self.counter += 1

        def resolve_trace(p: torch.Tensor, idx: int):
            if isinstance(traces, Mapping):
                if p not in traces:
                    raise KeyError(
                        f"Parameter trace not found in traces mapping for param: {p}"
                    )
                return traces[p], idx
            return traces[idx], idx + 1

        device = next(
            (p.device for group in self.param_groups for p in group["params"]),
            None,
        )
        total_norm = (
            torch.tensor(0.0, device=device)
            if device is not None
            else torch.tensor(0.0)
        )

        idx = 0
        for group in self.param_groups:
            beta = group["beta"]
            eps = group["eps"]
            for p in group["params"]:
                trace, idx = resolve_trace(p, idx)
                if trace is None:
                    continue
                state = self.state[p]
                if "v" not in state:
                    state["v"] = torch.zeros_like(p)
                v = state["v"]

                # v <- beta * v + (1 - beta) * (error * trace)^2
                semi_grad = error * trace
                v.mul_(beta).addcmul_(semi_grad, semi_grad, value=1.0 - beta)

                # Bias correction (reference optim.py): v_hat = v / (1 - beta^step).
                v_hat = v / (1.0 - beta**self.counter)

                # || trace / sqrt(v_hat + eps) ||_1
                adj_trace = trace / torch.sqrt(v_hat + eps)
                total_norm += torch.sum(torch.abs(adj_trace))

        idx = 0
        for group in self.param_groups:
            for p in group["params"]:
                trace, idx = resolve_trace(p, idx)
                if trace is None:
                    continue
                state = self.state[p]
                beta = group["beta"]
                v_hat = state["v"] / (1.0 - beta**self.counter)
                adaptive_obgd_td_update_(
                    theta=p,
                    error=error,
                    trace=trace,
                    v=v_hat,
                    lr=group["lr"],
                    scaling_factor=group["scaling_factor"],
                    total_norm=total_norm,
                    eps=group["eps"],
                )
        return loss
