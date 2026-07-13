"""
Stream AC(λ) — streaming actor-critic on Pendulum-v1.

Algorithm 7 from Elsayed, Vasan & Mahmood (2024):
"Streaming Deep Reinforcement Learning Finally Works" (arXiv:2410.14606).

The actor is a Gaussian policy (mean + learnable log_std).  The critic is a
state-value network.  Both use LayerNorm, SparseInit, eligibility traces,
ObGD (with separate κ for actor/critic), online observation normalisation,
and reward scaling.  Entropy regularisation follows Appendix E.

Key components (shared stream-x toolkit)
----------------------------------------
- LayerNorm MLP (affine=False) with SparseInit (sparsity=0.9).
- Online observation normalisation (Welford, via Gym wrapper).
- Reward scaling via discounted trace (Welford, via Gym wrapper, Algorithm 5).
- Accumulating eligibility traces for both actor and critic.
- ObGD step-size controller (κ_π=3, κ_v=2).
- Entropy regularisation: τ·sign(δ)·H(π|s) added to the policy trace.
"""

import math

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import wandb

from envs.wrappers.normalization import (
    WelfordNormalizeObservation,
    WelfordNormalizeReward,
)
from functional.initialization import set_seed, sparse_init_weight_
from functional.optimizer import obgd_td_update_
from functional.utils import to_tensor
from functional.visualization import compute_explained_variance

# ---------------------------------------------------------------------------
# Constants  (single hyper-parameter set from the paper)
# ---------------------------------------------------------------------------
GAMMA = 0.99
LAMBDA = 0.8
ALPHA = 1.0
KAPPA_ACTOR = 3.0
KAPPA_CRITIC = 2.0
TAU_ENTROPY = 0.01
SPARSITY = 0.9
HIDDEN_SIZE = 256
MAX_STEPS = 200_000
SEED = 42
LOG_INTERVAL = 100

set_seed(SEED)
device = torch.device("cpu")


# ---------------------------------------------------------------------------
# Networks
# ---------------------------------------------------------------------------
def _lecun_uniform_(tensor: torch.Tensor) -> None:
    fan_in = nn.init._calculate_fan_in_and_fan_out(tensor)[0]
    bound = 1.0 / math.sqrt(fan_in)
    nn.init.uniform_(tensor, -bound, bound)


class LayerNormMLP(nn.Module):
    """Shared backbone: Linear → LayerNorm(affine=False) → ReLU (x2)."""

    def __init__(self, input_dim: int, hidden_dim: int):
        super().__init__()
        self.l1 = nn.Linear(input_dim, hidden_dim)
        self.ln1 = nn.LayerNorm(hidden_dim, elementwise_affine=False)
        self.l2 = nn.Linear(hidden_dim, hidden_dim)
        self.ln2 = nn.LayerNorm(hidden_dim, elementwise_affine=False)

        self._sparse_init_()

    def _sparse_init_(self):
        for name, param in self.named_parameters():
            if param.dim() >= 2:
                _lecun_uniform_(param)
                sparse_init_weight_(param, SPARSITY)
            elif param.dim() == 1:
                nn.init.zeros_(param)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.ln1(self.l1(x)))
        x = F.relu(self.ln2(self.l2(x)))
        return x


class GaussianActor(nn.Module):
    """
    Gaussian policy for continuous control.

    Outputs mean via a linear head from the backbone, then scales to the
    action range with tanh.  Log_std is a learnable parameter (state-independent,
    standard convention for single-task RL).
    """

    def __init__(self, backbone: nn.Module, action_dim: int, action_scale: float):
        super().__init__()
        self.backbone = backbone
        self.mean_head = nn.Linear(HIDDEN_SIZE, action_dim)
        self.log_std = nn.Parameter(torch.zeros(action_dim))
        self.action_scale = action_scale

        nn.init.zeros_(self.mean_head.weight)
        nn.init.zeros_(self.mean_head.bias)

    def forward(self, x: torch.Tensor) -> torch.distributions.Normal:
        features = self.backbone(x)
        mean = self.mean_head(features)
        mean = torch.tanh(mean) * self.action_scale
        std = F.softplus(self.log_std).exp()
        std = std.clamp(min=1e-4)
        return torch.distributions.Normal(mean, std)


class CriticNet(nn.Module):
    """State-value network: backbone → Linear(1)."""

    def __init__(self, backbone: nn.Module):
        super().__init__()
        self.backbone = backbone
        self.value_head = nn.Linear(HIDDEN_SIZE, 1)
        nn.init.zeros_(self.value_head.weight)
        nn.init.zeros_(self.value_head.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.backbone(x)
        return self.value_head(features)


# ---------------------------------------------------------------------------
#  Initialise
# ---------------------------------------------------------------------------
env = gym.make("Pendulum-v1")
env = gym.wrappers.RecordEpisodeStatistics(env)
env = WelfordNormalizeObservation(env, device=device)
env = WelfordNormalizeReward(env, gamma=GAMMA, device=device)
obs_shape = env.observation_space.shape
action_dim = env.action_space.shape[0]
action_scale = float(env.action_space.high[0])

# Shared backbone for actor and critic (separate instances)
actor_backbone = LayerNormMLP(obs_shape[0], HIDDEN_SIZE).to(device)
critic_backbone = LayerNormMLP(obs_shape[0], HIDDEN_SIZE).to(device)

actor = GaussianActor(actor_backbone, action_dim, action_scale).to(device)
critic = CriticNet(critic_backbone).to(device)

# Eligibility traces for actor and critic params
actor_traces = [torch.zeros_like(p, device=device) for p in actor.parameters()]
critic_traces = [torch.zeros_like(p, device=device) for p in critic.parameters()]

# Buffer for computing explained variance over a window
value_buffer = []
return_buffer = []
EXPLAINED_VAR_WINDOW = 1000

rng_key = torch.Generator(device=device)
rng_key.manual_seed(SEED)

obs, info = env.reset(seed=SEED)

wandb.init(project="stream-ac-lambda-pendulum")
wandb.define_metric("*", step_metric="global_step")
global_step = 0


# ---------------------------------------------------------------------------
#  Main loop
# ---------------------------------------------------------------------------
for step in range(MAX_STEPS):
    global_step += 1

    obs_t = to_tensor(obs, device=device).unsqueeze(0)  # [1, D], already normalised

    # ---- act ---------------------------------------------------------------
    with torch.inference_mode():
        dist = actor(obs_t)
        action = dist.sample()
        log_prob = dist.log_prob(action).sum(dim=-1)
        action_np = (
            action.squeeze(0)
            .cpu()
            .numpy()
            .clip(env.action_space.low, env.action_space.high)
        )

    # ---- environment step (obs & reward pre-normalised by wrappers) --------
    next_obs, reward, terminated, truncated, info = env.step(action_np)
    done = terminated or truncated

    next_obs_t = to_tensor(next_obs, device=device).unsqueeze(0)  # already normalised
    reward_t = to_tensor(reward, device=device)  # already scaled

    # ---- forward passes (with grad for current state) ----------------------
    v_current = critic(obs_t).squeeze(0)

    with torch.no_grad():
        v_next = critic(next_obs_t).squeeze(0)

    # ---- TD error ---------------------------------------------------------
    if terminated:
        v_next = torch.tensor(0.0, device=device)
    delta = reward_t + GAMMA * v_next - v_current

    # ---- policy gradient trace --------------------------------------------
    dist_grad = actor(obs_t)
    log_prob_grad = dist_grad.log_prob(torch.as_tensor(action, device=device)).sum(
        dim=-1
    )
    entropy = dist_grad.entropy().sum(dim=-1)

    sign_delta = delta.sign().detach()
    policy_objective = log_prob_grad + TAU_ENTROPY * sign_delta * entropy

    actor.zero_grad()
    policy_objective.backward()

    for i, p in enumerate(actor.parameters()):
        if p.grad is not None:
            actor_traces[i] = GAMMA * LAMBDA * actor_traces[i] + p.grad.detach()

    # ---- value function trace ---------------------------------------------
    critic.zero_grad()
    v_current.backward()

    for i, p in enumerate(critic.parameters()):
        if p.grad is not None:
            critic_traces[i] = GAMMA * LAMBDA * critic_traces[i] + p.grad.detach()

    # ---- ObGD updates -----------------------------------------------------
    for i, p in enumerate(critic.parameters()):
        obgd_td_update_(
            theta=p,
            error=delta.detach(),
            trace=critic_traces[i],
            lr=ALPHA,
            scaling_factor=KAPPA_CRITIC,
        )

    for i, p in enumerate(actor.parameters()):
        obgd_td_update_(
            theta=p,
            error=delta.detach(),
            trace=actor_traces[i],
            lr=ALPHA,
            scaling_factor=KAPPA_ACTOR,
        )

    # ---- episode logging --------------------------------------------------
    if done:
        if "episode" in info:
            wandb.log(
                {
                    "episode_return": info["episode"]["r"][0],
                    "episode_length": info["episode"]["l"][0],
                    "global_step": global_step,
                },
                step=global_step,
            )
        obs, info = env.reset()
        for i in range(len(actor_traces)):
            actor_traces[i].zero_()
        for i in range(len(critic_traces)):
            critic_traces[i].zero_()
    else:
        obs = next_obs

    # ---- periodic logging (A2C-style + stream-specific) -------------------
    if step % LOG_INTERVAL == 0:
        returns_val = reward_t + GAMMA * v_next
        advantage = delta

        value_buffer.append(v_current.item())
        return_buffer.append(returns_val.item())
        if len(value_buffer) > EXPLAINED_VAR_WINDOW:
            value_buffer.pop(0)
            return_buffer.pop(0)

        explained_var = compute_explained_variance(
            np.array(return_buffer), np.array(value_buffer)
        )

        wandb.log(
            {
                "learning_rate": ALPHA,
                "loss/total": delta.abs().item(),
                "loss/critic": (delta**2).item(),
                "value/mean": v_current.item(),
                "value/return_mean": returns_val.item(),
                "value/explained_variance": explained_var,
                "advantages/mean": advantage.item(),
                "advantages/std": delta.abs().item(),
                "entropy": entropy.item(),
                "log_prob": log_prob.item(),
                "global_step": global_step,
            },
            step=global_step,
        )

wandb.finish()
