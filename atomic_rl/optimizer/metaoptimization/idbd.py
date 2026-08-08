"""
Notes on IDBD (Incremental Delta-Bar-Delta):

Idea: Don't learn on useless information (or inputs). Try to judge what is useful based on past experiences. Update the weight specific learning rates to do this. The original Delta-Bar-Delta (Jacobs, 1988) solved the learning rate problem, but it required batch learning (looking at the whole dataset at once). Sutton's major contribution with IDBD was making this math work incrementally (online, step-by-step), making it usable for real-time tracking and Reinforcement Learning

Global learning rates (or simple time-based decays) fail when different features require different levels of plasticity. IDBD maintains a distinct learning rate (alpha) per weight. It performs meta-gradient descent on the log-learning rate (beta). A trace (h) accumulates past gradient information to estimate the long-term derivative of the error with respect to beta. In other words h tracks the correlation between current gradient updates and recent past gradient updates.

IDBD is designed for non-stationary tracking tasks. In Reinforcement Learning, the targets (like TD-targets or Value Estimates) are constantly moving as the agent's policy improves or the environment shifts. Standard algorithms decay the learning rate to 0 (which stops learning) or keep it constant (which introduces permanent noise). IDBD dynamically figures out which weights track moving targets and keeps their learning rates high, while dropping the learning rates of weights connected to static or noisy features.

Because IDBD adapts `beta` (the log learning rate), you can directly inspect the resulting `alpha` vector. If `alpha` for a specific feature drops to near-zero, the algorithm has determined that feature is useless or pure noise. You can use this as a direct feature-selection/pruning metric. If a weight keeps being pushed in the same direction over multiple steps (positive correlation), the target is moving, so IDBD increases the learning rate. If a weight is pushed back and forth (negative correlation), it means it's overshooting or tracking noise, so IDBD decreases the learning rate. By defining $\alpha = e^\beta$, it guarantees that the learning rate $\alpha$ can never become negative (which would cause the network to learn backwards and immediately explode). It also allows the learning rates to move in geometric proportions (e.g., doubling or halving) rather than taking fixed-size linear steps, which makes it much faster to adapt to drastic changes.

Use cases:
- Non stationary tracking tasks. For example, RL where target is constantly changing.
- Automatic Feature Selection & Pruning
- Eliminating Manual Learning Rate Schedules

The original derivations assume linear function approximation. Extensions to nonlinear neural networks are substantially more difficult and generally require different approximations or hypergradient methods.Must custom write forward and backward loops to accomidate the extra parameters, as vanilla pytorch autograd does not support this.
Still must tune the meta learning rate. Though this may be less important than tuning the base learning rate in traditional SGD.
Risk of divergence if there is a large noisy error.
Triple the memory requirements compared to traditional SGD.
Difficult to extend beyond linear TD learning (AdaGain paper).

Further Papers/Improvements (TODO):
- Autostep (Mahmood et al., 2012): A normalized version of IDBD that doesn't require tuning the meta-learning rate as carefully.
- Hypergradient Descent (Maclaurin et al., 2015 / Baydin et al., 2017): Applies the IDBD concept of learning the learning-rate to deep, non-linear networks using backpropagation.
- TIDBD: Adapts IDBD specifically for temporal-difference (TD) learning traces.

NOTE: future work on instance based methods suggested in paper. Instance-based methods (like k-Nearest Neighbors or Episodic Memory in RL) don't learn weights; they store raw data points. An IDBD equivalent here wouldn't adjust weight learning rates, but rather adjust the *distance metric* or *bandwidth* (how much weight to give specific features when measuring similarity). While conceptually discussed, it is rare in practice compared to parametric meta-learning.
NOTE: On the task this paper is solving, learning time is proportional to the sum of squares of learning rates, so thats why learning rates must be distributed carefully.
NOTE: IDBD is fundamentally a meta-learning algorithm. It literally performs gradient descent on the learning rate itself. It computes the derivative of the squared prediction error with respect to the learning-rate parameter ($\beta$).
NOTE: Recommended to look at k1_k2.py and autostep.py
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
def update_idbd_rates_(
    betas: torch.Tensor,
    h: torch.Tensor,
    inputs: torch.Tensor,
    error: torch.Tensor,
    meta_lr: float,
    min_beta: float = -10.0,
    max_beta_change: float = 2.0,
) -> torch.Tensor:
    """
    Decoupled IDBD: Updates betas and h in-place. Returns the computed alphas.
    IDBD adapts the learning rate (alpha) of each individual weight based on the
    correlation between the current gradient and a trace of past gradient changes (h).
    It operates on the log-learning rate (beta) to ensure step sizes remain strictly positive.
    Does NOT modify weights.

    Args:
        betas (torch.Tensor): Log-learning rates (ln(alpha)). Proportional to the correlation between current and recent weight changes. Shape must match inputs.
        h (torch.Tensor): Decaying trace of the cumulative sum of recent changes to weights. Shape must match inputs.
        inputs (torch.Tensor): The input features (x). Shape must match betas.
        error (torch.Tensor): The scalar prediction error (Target - Prediction).
            Shape: [Batch, 1] (if batched) or [] (if unbatched). MUST be broadcastable to inputs.
        meta_lr (float): The meta-learning rate (theta) controlling how fast betas update.
        min_beta (float): Lower bound for beta to prevent underflow.
        max_beta_change (float): Clips the update to beta to prevent explosion.

    Returns:
        torch.Tensor: alphas (the exponentiated learning rates for this step).

    NOTE/TODO: Strictly linear at the moment
    """

    # Fail Fast: Strict shape assertions to catch mismatched dimensions immediately.
    # NOTE: By checking that inputs is broadcastable/matches the last dimension of betas (rather than being completely identical in shape),
    # this allows the algorithm to run in parallel across hundreds of output neurons in a standard PyTorch nn.Linear layer without altering the underlying math.
    assert betas.shape == h.shape, "Betas and traces must match weight shapes."
    assert inputs.shape[-1] == betas.shape[-1], (
        "Input features must match weight in_features."
    )

    # Fail Fast: Ensure error is explicitly broadcastable to the feature dimension.
    if inputs.dim() > 1:
        assert error.dim() == inputs.dim() and error.shape[-1] == 1, (
            f"Batched error must have shape [Batch, 1], got {error.shape}"
        )

    # 1. Update betas
    # TODO: To make these composable with Neural Networks in the future (like Hypergradient Descent or Adam-HD),
    # you should gradually transition the API to accept gradients rather than inputs and error separately.
    # Right now, you have: \Delta \beta = \theta \cdot \delta \cdot x \cdot h
    delta_beta = (
        (inputs * h).mul_(meta_lr * error).clamp_(-max_beta_change, max_beta_change)
    )
    betas.add_(delta_beta).clamp_(min=min_beta)

    # 2. Compute alphas (actual learning rates)
    # Equation: alpha(t+1) = exp(beta(t+1))
    alphas = torch.exp(betas)

    # 3. Update the trace (h) without needing weights
    decay = (1.0 - alphas * inputs.square()).clamp_(min=0.0)

    # Equation: h(t+1) = h(t) * decay + alpha(t+1) * error * input
    h.mul_(decay).add_(inputs * alphas * error)

    return alphas


class IDBD(Optimizer):
    """
    Incremental Delta-Bar-Delta (IDBD) Optimizer for linear models.
    """

    def __init__(self, params, initial_lr: float = 0.01, meta_lr: float = 0.01):
        if initial_lr <= 0.0:
            raise ValueError(f"Invalid initial_lr: {initial_lr}")
        defaults = dict(initial_lr=initial_lr, meta_lr=meta_lr)
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

                # Lazy State Initialization
                if len(state) == 0:
                    state["beta"] = torch.full_like(p, math.log(group["initial_lr"]))
                    state["h"] = torch.zeros_like(p)

                # 1. Compute new rates and traces via pure functional core
                alphas = update_idbd_rates_(
                    betas=state["beta"],
                    h=state["h"],
                    inputs=inputs,
                    error=error,
                    meta_lr=group["meta_lr"],
                )

                # 3. Apply Weight Update: w <- w + α * δ * x
                # Note: error is assumed to be a scalar or properly broadcastable
                update = alphas * inputs * error
                p.add_(update)
