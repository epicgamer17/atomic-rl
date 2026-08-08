"""
Extensions from "Gain Adaptation Beats Least Squares?" (Sutton, 1992b):

1. The Kalman Filter Connection:
Standard Least Squares and Kalman Filters are mathematically optimal but have O(n^2) computational and memory complexity, making them too slow for large networks. IDBD, K1, and K2 are O(n) approximations of the Kalman Filter. Sutton showed that if a Kalman filter's prior knowledge about how the system drifts is even slightly wrong, adaptive O(n) methods (like IDBD/K1/K2) will actually outperform it.

2. The K1 Algorithm (IDBD for NLMS):
While IDBD extends standard LMS (Stochastic Gradient Descent), K1 extends Normalized LMS (NLMS). It treats the exponentiated betas as the diagonal entries of a pseudo-covariance matrix to compute a normalized gain vector. This makes the step sizes inherently stable against input scale variations. It retains the 3n memory requirement (Weight, Beta, Trace).

3. The K2 Algorithm (Dropping the Trace):
K2 achieves IDBD-like performance but eliminates the memory trace (h) entirely, reducing memory requirements from 3n to 2n (just Weight and Beta). Instead of using meta-gradient descent on the correlation of past gradients, K2 adapts beta using an incremental regression of the squared error onto the squared inputs.
"""

import math
import torch

from torch.optim.optimizer import Optimizer


# TODO: make stateful or pass args around.
# TODO: Adam-HD? is this even desired?
# TODO extend to non linear case
# TODO: a little messy and hard to use/drop in, try and clean up the code.
# TODO: more work needed to make this actually useful/general. Work up to MetaOptize from Sutton.


@torch.no_grad()
def update_k1_rates_(
    betas: torch.Tensor,
    h: torch.Tensor,
    inputs: torch.Tensor,
    error: torch.Tensor,
    meta_lr: float,
    r_hat: float = 1.0,
    min_beta: float = -10.0,
    max_beta_change: float = 2.0,
) -> torch.Tensor:
    """
    Computes a single step of the K1 algorithm (IDBD for Normalized LMS). Updates betas and h in-place.

    K1 is an O(n) approximation of the Kalman Filter. It treats the exponentiated
    betas as the diagonal entries of a pseudo-covariance matrix to compute a
    normalized gain vector, making it more robust than vanilla IDBD.

    Args:
        betas (torch.Tensor): Log-learning rates. Shape must match inputs.
        h (torch.Tensor): Memory trace of past updates. Shape must match inputs.
        inputs (torch.Tensor): Input features. Shape must match betas.
        error (torch.Tensor): Prediction error (Target - Prediction). Shape: [..., 1] or [].
        meta_lr (float): The meta-learning rate (mu).
        r_hat (float): Estimate of the observation noise variance (R).
        min_beta (float): Lower bound for beta to prevent underflow.
        max_beta_change (float): Clips the update to beta to prevent explosion.

    Returns:
        torch.Tensor: k_gain (the normalized gain vector used for the update).

    NOTE/TODO: Strictly linear at the moment
    """
    # NOTE: By checking that inputs is broadcastable/matches the last dimension of betas (rather than being completely identical in shape),
    # this allows the algorithm to run in parallel across hundreds of output neurons in a standard PyTorch nn.Linear layer without altering the underlying math.
    assert betas.shape == h.shape, "Betas and traces must match weight shapes."
    assert inputs.shape[-1] == betas.shape[-1], (
        "Input features must match weight in_features."
    )
    if inputs.dim() > 1:
        assert error.dim() == inputs.dim() and error.shape[-1] == 1, (
            f"Batched error must have shape [..., 1], got {error.shape}"
        )

    # 1. Update betas
    # TODO: To make these composable with Neural Networks in the future (like Hypergradient Descent or Adam-HD),
    # you should gradually transition the API to accept gradients rather than inputs and error separately.
    # Right now, you have: \Delta \beta = \theta \cdot \delta \cdot x \cdot h
    delta_beta = (
        (inputs * h).mul_(meta_lr * error).clamp_(-max_beta_change, max_beta_change)
    )
    betas.add_(delta_beta).clamp_(min=min_beta)

    # 2. Compute p_hat (diagonal of pseudo-covariance matrix)
    p_hat = torch.exp(betas)

    # 3. Compute the normalizer D(t) = R_hat + sum(p_hat * inputs^2)
    # sum is taken over the feature dimension (dim=-1)
    d_t = r_hat + torch.sum(p_hat * inputs.square(), dim=-1, keepdim=True)

    # 4. Compute normalized gain vector K(t)
    k_gain = (p_hat * inputs).div_(d_t)

    # 5. Update trace h
    # Equation: h(t+1) = [h(t) + k(t) * error] * max(1 - k(t) * input, 0)
    decay = (1.0 - k_gain * inputs).clamp_(min=0.0)
    h.add_(k_gain * error).mul_(decay)

    return k_gain


@torch.no_grad()
def update_k2_rates_(
    betas: torch.Tensor,
    inputs: torch.Tensor,
    error: torch.Tensor,
    meta_lr: float,
    r_hat: float = 1.0,
    min_beta: float = -10.0,
    max_beta_change: float = 2.0,
) -> torch.Tensor:
    """
    Computes a single step of the K2 algorithm.

    K2 drops the memory trace `h` entirely, reducing memory complexity from 3n to 2n.
    It adapts betas using an incremental regression of the squared error onto the
    squared inputs.

    Args:
        betas (torch.Tensor): Log-learning rates. Shape must match inputs.
        inputs (torch.Tensor): Input features. Shape must match betas.
        error (torch.Tensor): Prediction error (Target - Prediction). Shape: [..., 1] or [].
        meta_lr (float): The meta-learning rate (mu).
        r_hat (float): Estimate of the observation noise variance (R).
        min_beta (float): Lower bound for beta to prevent underflow.
        max_beta_change (float): Clips the update to beta to prevent explosion.

    Returns:
        torch.Tensor: k_gain (the normalized gain vector used for the update).

    NOTE/TODO: Strictly linear at the moment
    """
    # NOTE: By checking that inputs is broadcastable/matches the last dimension of betas (rather than being completely identical in shape),
    # this allows the algorithm to run in parallel across hundreds of output neurons in a standard PyTorch nn.Linear layer without altering the underlying math.
    assert inputs.shape[-1] == betas.shape[-1], (
        "Input features must match weight in_features."
    )
    if inputs.dim() > 1:
        assert error.dim() == inputs.dim() and error.shape[-1] == 1, (
            f"Batched error must have shape [..., 1], got {error.shape}"
        )

    # 1. Compute regression error: delta^2 - R_hat - sum(p_hat_old * inputs^2)
    p_hat_old = torch.exp(betas)
    predicted_variance = torch.sum(p_hat_old * inputs.square(), dim=-1, keepdim=True)
    regression_error = error.square() - r_hat - predicted_variance

    # 2. Compute K2 beta normalizer: 1 + sum(inputs^4)
    beta_normalizer = 1.0 + torch.sum(inputs.pow(4), dim=-1, keepdim=True)

    # 3. Update betas via incremental regression
    # TODO: To make these composable with Neural Networks in the future (like Hypergradient Descent or Adam-HD),
    # you should gradually transition the API to accept gradients rather than inputs and error separately.
    # Right now, you have: \Delta \beta = \theta \cdot \delta \cdot x \cdot h
    delta_beta = inputs.square().mul_(meta_lr / beta_normalizer).mul_(regression_error)
    delta_beta.clamp_(-max_beta_change, max_beta_change)
    betas.add_(delta_beta).clamp_(min=min_beta)

    # 4. Compute new p_hat for the weight update
    p_hat_new = torch.exp(betas)

    # 5. Compute normalized gain vector K(t)
    d_t = r_hat + torch.sum(p_hat_new * inputs.square(), dim=-1, keepdim=True)
    k_gain = (p_hat_new * inputs).div_(d_t)

    return k_gain


class K1(Optimizer):
    """
    K1 Optimizer: O(n) Kalman Filter approximation extending NLMS.
    """

    def __init__(
        self,
        params,
        initial_lr: float = 0.01,
        meta_lr: float = 0.01,
        r_hat: float = 1.0,
    ):
        defaults = dict(initial_lr=initial_lr, meta_lr=meta_lr, r_hat=r_hat)
        super().__init__(params, defaults)

    @torch.no_grad()
    # TODO: To make these composable with Neural Networks in the future (like Hypergradient Descent or Adam-HD),
    # you should gradually transition the API to accept gradients rather than inputs and error separately.
    # Right now, you have: \Delta \beta = \theta \cdot \delta \cdot x \cdot h
    def step(self, inputs: torch.Tensor, error: torch.Tensor):
        for group in self.param_groups:
            for p in group["params"]:
                if p.shape[-1] != inputs.shape[-1]:
                    raise ValueError(
                        f"Linearity constraint violated: Parameter in_features {p.shape[-1]} does not "
                        f"match input in_features {inputs.shape[-1]}."
                    )
                state = self.state[p]

                if len(state) == 0:
                    state["beta"] = torch.full_like(p, math.log(group["initial_lr"]))
                    state["h"] = torch.zeros_like(p)

                k_gain = update_k1_rates_(
                    betas=state["beta"],
                    h=state["h"],
                    inputs=inputs,
                    error=error,
                    meta_lr=group["meta_lr"],
                    r_hat=group["r_hat"],
                )

                # Apply Weight Update: w <- w + k * δ
                p.add_(k_gain * error)


class K2(Optimizer):
    """
    K2 Optimizer: Traceless IDBD approximation.
    """

    def __init__(
        self,
        params,
        initial_lr: float = 0.01,
        meta_lr: float = 0.01,
        r_hat: float = 1.0,
    ):
        defaults = dict(initial_lr=initial_lr, meta_lr=meta_lr, r_hat=r_hat)
        super().__init__(params, defaults)

    @torch.no_grad()
    # TODO: To make these composable with Neural Networks in the future (like Hypergradient Descent or Adam-HD),
    # you should gradually transition the API to accept gradients rather than inputs and error separately.
    # Right now, you have: \Delta \beta = \theta \cdot \delta \cdot x \cdot h
    def step(self, inputs: torch.Tensor, error: torch.Tensor):
        for group in self.param_groups:
            for p in group["params"]:
                if p.shape[-1] != inputs.shape[-1]:
                    raise ValueError(
                        f"Linearity constraint violated: Parameter in_features {p.shape[-1]} does not "
                        f"match input in_features {inputs.shape[-1]}."
                    )
                state = self.state[p]

                if len(state) == 0:
                    state["beta"] = torch.full_like(p, math.log(group["initial_lr"]))

                k_gain = update_k2_rates_(
                    betas=state["beta"],
                    inputs=inputs,
                    error=error,
                    meta_lr=group["meta_lr"],
                    r_hat=group["r_hat"],
                )

                # Apply Weight Update: w <- w + k * δ
                p.add_(k_gain * error)
