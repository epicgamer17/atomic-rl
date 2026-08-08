"""
4. Autostep:
IDBD and other methods are sensitive to initialization and the choice of meta_lr, autostep removes that sensitivity by normalizing the step sizes.

Autostep can be extended in interesting ways. Extensions to nonlinear settings would make tuning-free step-size adaptation possible for artificial neural networks. Extensions to reinforcement learning would be natural given that many reinforcement learning problems are inherently nonstationary.
In the paper they state that "[They] are currently exploring extensions of Autostep to online temporal-difference learning." (TODO)

The original IDBD algorithm proved that you could adapt individual step sizes for each feature, but it introduced a new problem: you had to manually tune the meta-step-size parameter. If you got it wrong, the algorithm became unstable.Autostep solves this by introducing normalizations to the step-size updates. This is crucial because the Alberta Plan explicitly states that normalization of features "has a powerful effect on the speed of learning" and that "online, continual normalization, has been little studied". Reading Autostep right after IDBD allows you to see exactly how to take a theoretically sound meta-gradient method and make it stable enough for the "continual supervised learning" required in Step 1.


"K1 and K2 were introduced by Sutton as $O(n)$ approximations to the Kalman Filter. By the time Autostep was formalized, researchers realized that if you just normalize IDBD properly (creating Autostep), you get stability comparable to K1/K2 without needing the Kalman-style pseudo-covariance matrices." - Gemini
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
def update_autostep_v_normalizer_(
    v: torch.Tensor,
    abs_meta_grad: torch.Tensor,
    alphas: torch.Tensor,
    inputs: torch.Tensor,
    tau: float = 10000.0,
):
    """
    Computes the updated running maximum of the absolute meta-gradient.

    This is the first core idea of Autostep: Unit Normalization. It can be applied to ANY meta-gradient to make the resulting update unitless and scale-invariant.

    Args:
        v (torch.Tensor): Current running maximums. Shape: [..., Features]
        abs_meta_grad (torch.Tensor): The absolute value of the current step's meta-gradient.
        alphas (torch.Tensor): Current exponentiated learning rates.
        inputs (torch.Tensor): Current input features.
        tau (float): Time constant for the normalizer decay.

    Returns:
        torch.Tensor: The updated running maximums (new_v).

    NOTE/TODO: Strictly linear at the moment
    """
    # Equation: v_decay = (alpha * x^2) / tau
    v_decay_rate = (alphas * inputs.square()).div_(tau)

    # Equation: v_update = v + v_decay * (|meta_grad| - v)
    v_update = (abs_meta_grad - v).mul_(v_decay_rate).add_(v)

    # Equation: new_v = max(|meta_grad|, v_update)
    torch.maximum(abs_meta_grad, v_update, out=v)


# TODO: can this be merged or reused for the ObGD logic?
@torch.no_grad()
def update_autostep_m_cap_(
    betas: torch.Tensor,
    inputs: torch.Tensor,
) -> torch.Tensor:
    """
    Computes the effective step size cap (M) and normalizes the log-learning rates.

    This is the second core idea of Autostep: Overshoot Prevention. It ensures
    the sum of the effective step sizes does not exceed 1.0.

    Args:
        temp_betas (torch.Tensor): The proposed new log-learning rates before capping.
        inputs (torch.Tensor): Current input features.

    Returns:
        torch.Tensor: The normalized actual learning rates.

    NOTE/TODO: Strictly linear at the moment
    """
    temp_alphas = torch.exp(betas)

    # Calculate effective step size: sum(alpha_i * x_i^2)
    # NOTE: In multi-output mode, Autostep normalization is applied independently per output unit (per row of the weight matrix), equivalent to running separate scalar predictors sharing the same input features.
    effective_step_size = torch.sum(temp_alphas * inputs.square(), dim=-1, keepdim=True)

    # Cap at minimum 1.0 (if it's less than 1.0, m = 1.0 and no scaling occurs)
    m = torch.clamp(effective_step_size, min=1.0)

    # Scale alphas down by M, and shift betas down by log(M)
    betas.sub_(torch.log(m))

    return temp_alphas.div_(m)


@torch.no_grad()
def update_autostep_rates_(
    betas: torch.Tensor,
    h: torch.Tensor,
    v: torch.Tensor,
    inputs: torch.Tensor,
    error: torch.Tensor,
    meta_lr: float = 0.01,
    tau: float = 10000.0,
    min_beta: float = -10.0,
    max_beta_change: float = 2.0,
) -> torch.Tensor:
    """
    Decoupled Autostep: Computes the normalized learning rates and updates meta-states.
    Does NOT modify weights.

    Initialization:
        Set µ and τ as appropriate (e.g., 10^-2 and 10^4)
        for i = 1, . . . , n:
            hi ← vi ← 0
        Initialize wi and αi as desired (e.g., 0 and 0.1)
    for each new data sample (x1, . . . , xn, y):
        δ ← y − SUM(n, i=1, wixi)
        for i = 1, . . . , n:
            vi ← max(|δxihi|, vi + (1/τ) * αix^2i * (|δxihi| − vi))
            if vi 6= 0:
                αi ← αi exp (µ * δxihi / vi)

        M ← max(SUM(n, i=1, αix^2i), 1)
        for i = 1, . . . , n:
            αi ← αi / M
            hi ← hi * (1 - αix^2i) + αiδxi

    Returns:
        torch.Tensor: The normalized actual learning rates.

    NOTE/TODO: Strictly linear at the moment
    """
    # NOTE: By checking that inputs is broadcastable/matches the last dimension of betas (rather than being completely identical in shape),
    # this allows the algorithm to run in parallel across hundreds of output neurons in a standard PyTorch nn.Linear layer without altering the underlying math.
    assert betas.shape == h.shape == v.shape, (
        "Betas, traces, and normalizers must match weight shapes."
    )
    assert inputs.shape[-1] == betas.shape[-1], (
        "Input features must match weight in_features."
    )

    if inputs.dim() > 1:
        assert error.dim() == inputs.dim() and error.shape[-1] == 1, (
            f"Batched error must have shape [..., 1], got {error.shape}"
        )

    # 1. Calculate base IDBD meta-gradient
    # TODO: To make these composable with Neural Networks in the future (like Hypergradient Descent or Adam-HD),
    # you should gradually transition the API to accept gradients rather than inputs and error separately.
    # Right now, you have: \Delta \beta = \theta \cdot \delta \cdot x \cdot h
    meta_grad = error * inputs * h
    abs_meta_grad = torch.abs(meta_grad)
    alphas = torch.exp(betas)

    # 2. Apply Autostep Idea 1: Unit Normalization
    update_autostep_v_normalizer_(v, abs_meta_grad, alphas, inputs, tau)
    normalized_meta_grad = torch.where(
        v != 0, meta_grad / v, torch.zeros_like(meta_grad)
    )

    # 3. Propose new betas
    delta_beta = normalized_meta_grad.mul_(meta_lr).clamp_(
        -max_beta_change, max_beta_change
    )
    betas.add_(delta_beta).clamp_(min=min_beta)

    # 4. Apply Autostep Idea 2: Overshoot Prevention
    new_alphas = update_autostep_m_cap_(betas, inputs)

    # 5. Update trace using the safe alphas
    # NOTE: the autostep paper pseudocode does not do a positive only clamp for 1.0 - new_alphas * (inputs**2) like IDBD does. As shown in Table 1. They correctly have the clamp in the IDBD pseudocode but not their Autostep pseudocode so it is assumed to be intentional.
    decay = 1.0 - new_alphas * inputs.square()
    h.mul_(decay).add_(inputs * new_alphas * error)

    return new_alphas


class Autostep(Optimizer):
    """
    Autostep Optimizer: A normalized, tuning-free extension of IDBD.
    """

    def __init__(
        self,
        params,
        initial_lr: float = 0.1,
        meta_lr: float = 0.01,
        tau: float = 10000.0,
    ):
        defaults = dict(initial_lr=initial_lr, meta_lr=meta_lr, tau=tau)
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
                    state["v"] = torch.zeros_like(p)

                alphas = update_autostep_rates_(
                    betas=state["beta"],
                    h=state["h"],
                    v=state["v"],
                    inputs=inputs,
                    error=error,
                    meta_lr=group["meta_lr"],
                    tau=group["tau"],
                )

                # Apply Weight Update
                update = alphas * inputs * error
                p.add_(update)
