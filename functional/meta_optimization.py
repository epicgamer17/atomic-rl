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

Extensions from "Gain Adaptation Beats Least Squares?" (Sutton, 1992b):

1. The Kalman Filter Connection:
Standard Least Squares and Kalman Filters are mathematically optimal but have O(n^2) computational and memory complexity, making them too slow for large networks. IDBD, K1, and K2 are O(n) approximations of the Kalman Filter. Sutton showed that if a Kalman filter's prior knowledge about how the system drifts is even slightly wrong, adaptive O(n) methods (like IDBD/K1/K2) will actually outperform it.

2. The K1 Algorithm (IDBD for NLMS):
While IDBD extends standard LMS (Stochastic Gradient Descent), K1 extends Normalized LMS (NLMS). It treats the exponentiated betas as the diagonal entries of a pseudo-covariance matrix to compute a normalized gain vector. This makes the step sizes inherently stable against input scale variations. It retains the 3n memory requirement (Weight, Beta, Trace).

3. The K2 Algorithm (Dropping the Trace):
K2 achieves IDBD-like performance but eliminates the memory trace (h) entirely, reducing memory requirements from 3n to 2n (just Weight and Beta). Instead of using meta-gradient descent on the correlation of past gradients, K2 adapts beta using an incremental regression of the squared error onto the squared inputs.

4. Autostep:
IDBD and other methods are sensitive to initialization and the choice of meta_lr, autostep removes that sensitivity by normalizing the step sizes.

Autostep can be extended in interesting ways. Extensions to nonlinear settings would make tuning-free step-size adaptation possible for artificial neural networks. Extensions to reinforcement learning would be natural given that many reinforcement learning problems are inherently nonstationary.
In the paper they state that "[They] are currently exploring extensions of Autostep to online temporal-difference learning." (TODO)

The original IDBD algorithm proved that you could adapt individual step sizes for each feature, but it introduced a new problem: you had to manually tune the meta-step-size parameter. If you got it wrong, the algorithm became unstable.Autostep solves this by introducing normalizations to the step-size updates. This is crucial because the Alberta Plan explicitly states that normalization of features "has a powerful effect on the speed of learning" and that "online, continual normalization, has been little studied". Reading Autostep right after IDBD allows you to see exactly how to take a theoretically sound meta-gradient method and make it stable enough for the "continual supervised learning" required in Step 1.

"K1 and K2 were introduced by Sutton as $O(n)$ approximations to the Kalman Filter. By the time Autostep was formalized, researchers realized that if you just normalize IDBD properly (creating Autostep), you get stability comparable to K1/K2 without needing the Kalman-style pseudo-covariance matrices." - Gemini

Further Papers/Improvements (TODO):
- Autostep (Mahmood et al., 2012): A normalized version of IDBD that doesn't require tuning the meta-learning rate as carefully.
- Hypergradient Descent (Maclaurin et al., 2015 / Baydin et al., 2017): Applies the IDBD concept of learning the learning-rate to deep, non-linear networks using backpropagation.
- TIDBD: Adapts IDBD specifically for temporal-difference (TD) learning traces.

NOTE: future work on instance based methods suggested in paper. Instance-based methods (like k-Nearest Neighbors or Episodic Memory in RL) don't learn weights; they store raw data points. An IDBD equivalent here wouldn't adjust weight learning rates, but rather adjust the *distance metric* or *bandwidth* (how much weight to give specific features when measuring similarity). While conceptually discussed, it is rare in practice compared to parametric meta-learning.
NOTE: On the task this paper is solving, learning time is proportional to the sum of squares of learning rates, so thats why learning rates must be distributed carefully.
NOTE: IDBD is fundamentally a meta-learning algorithm. It literally performs gradient descent on the learning rate itself. It computes the derivative of the squared prediction error with respect to the learning-rate parameter ($\beta$).
"""

"""
Notes on AdaGain: 

"Consider a learning system whose objective is to learn a large collection of predictions about an agent’s future interactions with the world. The predictions specify the value of some signal many steps in the future, given that the agent follows some specific course of action. There are many examples of such prediction learning systems including:
- Predictive State Representations (Littman, Sutton, and Singh 2001)
- Observable Operator Models (Jaeger 2000)
- Temporal-difference Networks (Sutton and Tanner 2004)
- General Value Functions (Sutton et al. 2011)

In our setting, the agent continually interacts with the world, making new predictions about the future, and revising its previous predictions as new outcomes are revealed. Occasionally, partially due to changes in the world and partially due to changes in the agent’s own behaviour, the targets may change and the agent must refine its predictions" 

AdaGain is a new meta-descent algorithm that attempts to optimize the stability of the base learner, rather than convergence to a fixed point (TODO what does that mean). It is built in a way that allows it to be easily combined with a variety of base-learners including SGD, TD learning, and even AMSGrad! 

The problem they use in the paper is given an imperfect observation, agent must make a prediction using that observation. Observations come in a continual online stream. Agent wants to learn a predictor, this could be discounted returns, or predict next observation from current observation, etc. In this setting the agent updates its weights after each new observation (not in batches like is done traditionally). 

Extensions from "Meta-descent for Online, Continual Prediction" (Jacobsen et al., 2019):

1. Optimization for Stability vs. Convergence:
Traditional meta-descent (IDBD/Autostep) minimizes the squared prediction error. AdaGain takes a dynamical systems approach and instead minimizes the squared *norm of the update vector* (||Delta||^2). By keeping the updates small, it explicitly optimizes for system stability in non-stationary environments, rather than convergence to a fixed point.

2. Algorithm Agnostic (Composability):
Because IDBD is derived specifically for the LMS/squared-error objective, it cannot easily wrap other optimizers. AdaGain's derivation allows it to be layered on top of *any* base update vector, including quasi-second-order methods like RMSProp or semi-gradient updates like Temporal Difference (TD) learning. 

3. The Finite Difference Approximation:
To wrap any generic algorithm, AdaGain needs the Jacobian of the update function. The authors propose a linear-complexity, finite-difference approximation. It evaluates the base update function three times per step (at current weights, w + r*update, and w - r*update) to approximate the curvature and adapt the step sizes.

4. Linear TD AdaGain:
For linear Reinforcement Learning, the analytic Jacobian can be computed directly without finite differences, resulting in a highly efficient meta-descent algorithm specifically tailored for TD(lambda) (TODO: an experiment showing this and its benefits)

TODO: learn the gradient math of AdaGain better.

"""

# TODO: make stateful or pass args around.
# TODO: Adam-HD? is this even desired?
# TODO extend to non linear case
# TODO: merge with functional/optimizer.py potentially.
# TODO: a little messy and hard to use/drop in, try and clean up the code.
# TODO: more work needed to make this actually useful/general. Work up to MetaOptize from Sutton.

import torch
from typing import Tuple

import math
import torch
import torch.nn as nn
from typing import Dict, Any


def compute_idbd_rates(
    betas: torch.Tensor,
    h: torch.Tensor,
    inputs: torch.Tensor,
    error: torch.Tensor,
    meta_lr: float,
    min_beta: float = -10.0,
    max_beta_change: float = 2.0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Decoupled IDBD: Only calculates new step sizes and updates meta-states.
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
        tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
            - new_betas: The updated log-learning rates.
            - new_h: The updated trace of past changes.
            - alphas: The exponentiated learning rates (alpha = exp(beta)) used for this step.
    """

    # Fail Fast: Strict shape assertions to catch mismatched dimensions immediately.
    assert (
        betas.shape == h.shape == inputs.shape
    ), f"Shape mismatch: betas({betas.shape}), h({h.shape}), inputs({inputs.shape})"

    # Fail Fast: Ensure error is explicitly broadcastable to the feature dimension.
    if inputs.dim() > 1:
        assert (
            error.dim() == inputs.dim() and error.shape[-1] == 1
        ), f"Batched error must have shape [Batch, 1], got {error.shape}"

    # 1. Update betas
    delta_beta = meta_lr * error * inputs * h
    delta_beta = torch.clamp(delta_beta, min=-max_beta_change, max=max_beta_change)
    new_betas = torch.clamp(betas + delta_beta, min=min_beta)

    # 2. Compute alphas (actual learning rates)
    # Equation: alpha(t+1) = exp(beta(t+1))
    alphas = torch.exp(new_betas)

    # 3. Update the trace (h) without needing weights
    decay = torch.clamp(1.0 - alphas * (inputs**2), min=0.0)

    # Equation: h(t+1) = h(t) * decay + alpha(t+1) * error * input
    new_h = h * decay + alphas * error * inputs

    return new_betas, new_h, alphas


def compute_k1_rates(
    betas: torch.Tensor,
    h: torch.Tensor,
    inputs: torch.Tensor,
    error: torch.Tensor,
    meta_lr: float,
    r_hat: float = 1.0,
    min_beta: float = -10.0,
    max_beta_change: float = 2.0,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Computes a single step of the K1 algorithm (IDBD for Normalized LMS).

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
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
            - new_betas: The updated log-learning rates.
            - new_h: The updated memory trace.
            - k_gain: The normalized gain vector used for the update.
    """
    assert (
        betas.shape == h.shape == inputs.shape
    ), f"Shape mismatch: betas({betas.shape}), h({h.shape}), inputs({inputs.shape})"
    if inputs.dim() > 1:
        assert (
            error.dim() == inputs.dim() and error.shape[-1] == 1
        ), f"Batched error must have shape [..., 1], got {error.shape}"

    # 1. Update betas
    delta_beta = meta_lr * error * inputs * h
    delta_beta = torch.clamp(delta_beta, min=-max_beta_change, max=max_beta_change)
    new_betas = torch.clamp(betas + delta_beta, min=min_beta)

    # 2. Compute p_hat (diagonal of pseudo-covariance matrix)
    p_hat = torch.exp(new_betas)

    # 3. Compute the normalizer D(t) = R_hat + sum(p_hat * inputs^2)
    # sum is taken over the feature dimension (dim=-1)
    d_t = r_hat + torch.sum(p_hat * (inputs**2), dim=-1, keepdim=True)

    # 4. Compute normalized gain vector K(t)
    k_gain = (p_hat * inputs) / d_t

    # 5. Update trace h
    # Equation: h(t+1) = [h(t) + k(t) * error] * max(1 - k(t) * input, 0)
    decay = torch.clamp(1.0 - k_gain * inputs, min=0.0)
    new_h = (h + k_gain * error) * decay

    return new_betas, new_h, k_gain


def compute_k2_rates(
    betas: torch.Tensor,
    inputs: torch.Tensor,
    error: torch.Tensor,
    meta_lr: float,
    r_hat: float = 1.0,
    min_beta: float = -10.0,
    max_beta_change: float = 2.0,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
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
        Tuple[torch.Tensor, torch.Tensor]:
            - new_betas: The updated log-learning rates.
            - k_gain: The normalized gain vector used for the update.
    """
    assert (
        betas.shape == inputs.shape
    ), f"Shape mismatch: betas({betas.shape}), inputs({inputs.shape})"
    if inputs.dim() > 1:
        assert (
            error.dim() == inputs.dim() and error.shape[-1] == 1
        ), f"Batched error must have shape [..., 1], got {error.shape}"

    # 1. Compute regression error: delta^2 - R_hat - sum(p_hat_old * inputs^2)
    p_hat_old = torch.exp(betas)
    predicted_variance = torch.sum(p_hat_old * (inputs**2), dim=-1, keepdim=True)
    regression_error = (error**2) - r_hat - predicted_variance

    # 2. Compute K2 beta normalizer: 1 + sum(inputs^4)
    beta_normalizer = 1.0 + torch.sum(inputs**4, dim=-1, keepdim=True)

    # 3. Update betas via incremental regression
    delta_beta = (meta_lr * (inputs**2) / beta_normalizer) * regression_error
    delta_beta = torch.clamp(delta_beta, min=-max_beta_change, max=max_beta_change)
    new_betas = torch.clamp(betas + delta_beta, min=min_beta)

    # 4. Compute new p_hat for the weight update
    p_hat_new = torch.exp(new_betas)

    # 5. Compute normalized gain vector K(t)
    d_t = r_hat + torch.sum(p_hat_new * (inputs**2), dim=-1, keepdim=True)
    k_gain = (p_hat_new * inputs) / d_t

    return new_betas, k_gain


import torch


def compute_autostep_v_normalizer(
    v: torch.Tensor,
    abs_meta_grad: torch.Tensor,
    alphas: torch.Tensor,
    inputs: torch.Tensor,
    tau: float = 10000.0,
) -> torch.Tensor:
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
    """
    # Equation: v_decay = (alpha * x^2) / tau
    v_decay_rate = (alphas * (inputs**2)) / tau

    # Equation: v_update = v + v_decay * (|meta_grad| - v)
    v_update = v + v_decay_rate * (abs_meta_grad - v)

    # Equation: new_v = max(|meta_grad|, v_update)
    new_v = torch.maximum(abs_meta_grad, v_update)
    return new_v


def compute_autostep_m_cap(
    temp_betas: torch.Tensor,
    inputs: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Computes the effective step size cap (M) and normalizes the log-learning rates.

    This is the second core idea of Autostep: Overshoot Prevention. It ensures
    the sum of the effective step sizes does not exceed 1.0.

    Args:
        temp_betas (torch.Tensor): The proposed new log-learning rates before capping.
        inputs (torch.Tensor): Current input features.

    Returns:
        Tuple[torch.Tensor, torch.Tensor]:
            - new_betas: The normalized log-learning rates.
            - new_alphas: The normalized actual learning rates.
    """
    temp_alphas = torch.exp(temp_betas)

    # Calculate effective step size: sum(alpha_i * x_i^2)
    # NOTE: In multi-output mode, Autostep normalization is applied independently per output unit (per row of the weight matrix), equivalent to running separate scalar predictors sharing the same input features.
    effective_step_size = torch.sum(temp_alphas * (inputs**2), dim=-1, keepdim=True)

    # Cap at minimum 1.0 (if it's less than 1.0, m = 1.0 and no scaling occurs)
    m = torch.clamp(effective_step_size, min=1.0)

    # Scale alphas down by M, and shift betas down by log(M)
    new_alphas = temp_alphas / m
    new_betas = temp_betas - torch.log(m)

    return new_betas, new_alphas


def compute_autostep_rates(
    betas: torch.Tensor,
    h: torch.Tensor,
    v: torch.Tensor,
    inputs: torch.Tensor,
    error: torch.Tensor,
    meta_lr: float = 0.01,
    tau: float = 10000.0,
    min_beta: float = -10.0,
    max_beta_change: float = 2.0,
):
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
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
            - new_betas: The normalized log-learning rates.
            - new_h: The updated memory trace.
            - new_v: The updated running maximums.
            - new_alphas: The normalized actual learning rates.
    """
    assert (
        betas.shape == inputs.shape
    ), f"Shape mismatch: betas({betas.shape}), inputs({inputs.shape})"

    if inputs.dim() > 1:
        assert (
            error.dim() == inputs.dim() and error.shape[-1] == 1
        ), f"Batched error must have shape [..., 1], got {error.shape}"

    # 1. Calculate base IDBD meta-gradient
    meta_grad = error * inputs * h
    abs_meta_grad = torch.abs(meta_grad)
    alphas = torch.exp(betas)

    # 2. Apply Autostep Idea 1: Unit Normalization
    new_v = compute_autostep_v_normalizer(v, abs_meta_grad, alphas, inputs, tau)
    normalized_meta_grad = torch.where(
        new_v != 0, meta_grad / new_v, torch.zeros_like(meta_grad)
    )

    # 3. Propose new betas
    delta_beta = torch.clamp(
        meta_lr * normalized_meta_grad, min=-max_beta_change, max=max_beta_change
    )
    temp_betas = torch.clamp(betas + delta_beta, min=min_beta)

    # 4. Apply Autostep Idea 2: Overshoot Prevention
    new_betas, new_alphas = compute_autostep_m_cap(temp_betas, inputs)

    # 5. Update trace using the safe alphas
    # NOTE: the autostep paper pseudocode does not do a positive only clamp for 1.0 - new_alphas * (inputs**2) like IDBD does. As shown in Table 1. They correctly have the clamp in the IDBD pseudocode but not their Autostep pseudocode so it is assumed to be intentional.
    new_h = h * (1.0 - new_alphas * (inputs**2)) + new_alphas * error * inputs

    return new_betas, new_h, new_v, new_alphas
