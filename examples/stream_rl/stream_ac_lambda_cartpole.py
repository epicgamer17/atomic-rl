"""
Stream AC(λ) — streaming actor-critic on CartPole-v1.

Algorithm 7 from Elsayed, Vasan & Mahmood (2024):
"Streaming Deep Reinforcement Learning Finally Works" (arXiv:2410.14606).

NOTE: We chose to use AdaptiveObGD over the standard ObGD. The reference code
(github.com/mohmdelsayed/streaming-drl) uses plain ObGD for all released stream
algorithms, and the paper's AC setup (Appendix F.1) uses plain ObGD as well.

NOTE (reference vs. paper): We intentionally match the authors' released code
(github.com/mohmdelsayed/streaming-drl) rather than the paper algorithms — a conscious
and intentional decision.
  - HIDDEN_SIZE matches the reference (128).
  - Reward scaling matches the reference `SampleMeanStd` centered variance via
    `WelfordNormalizeReward` (see envs/wrappers/normalization.py); this differs from
    paper Algorithm 5's mean-zero second moment.
  - Intentional divergences from the reference: we use AdaptiveObGD (the reference
    stream_ac_continuous.py uses plain ObGD, alpha=1, kappa=2), and we do not add the
    reference's AddTimeInfo observation wrapper (which appends the episode time to the
    state).
"""

import math
import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import wandb

from atomic_rl.action_selection import sample_distribution
from atomic_rl.initialization import set_seed, lecun_uniform_, make_sparse_init
from atomic_rl.optimizer import AdaptiveObGD
from atomic_rl.td import compute_v_td_target
from atomic_rl.traces import update_accumulating_traces
from atomic_rl.utils import (
    to_tensor,
    to_numpy_action,
    update_welford_stats,
)
from atomic_rl.metrics import compute_explained_variance
from envs.wrappers.normalization import (
    WelfordNormalizeObservation,
    WelfordNormalizeReward,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
GAMMA = 0.99
LAMBDA = 0.8
ALPHA = 1.0
KAPPA_ACTOR = 3.0
KAPPA_CRITIC = 2.0
TAU_ENTROPY = 0.01
SPARSITY = 0.9
HIDDEN_SIZE = 128  # matches the reference implementation
MAX_STEPS = 200_000
SEED = 42
LOG_INTERVAL = 100
EXPLAINED_VAR_WINDOW = 1000

set_seed(SEED)
device = torch.device("cpu")


# ---------------------------------------------------------------------------
# Networks
# ---------------------------------------------------------------------------
class LayerNormMLP(nn.Module):
    # Reference: https://github.com/mohmdelsayed/streaming-drl/blob/main/src/layer.py
    #   The authors' LayerNormMLP (hidden_size=128) with sparse init is in layer.py.
    def __init__(self, input_dim: int, hidden_dim: int):
        super().__init__()
        self.l1 = nn.Linear(input_dim, hidden_dim)
        self.ln1 = nn.LayerNorm(hidden_dim, elementwise_affine=False)
        self.l2 = nn.Linear(hidden_dim, hidden_dim)
        self.ln2 = nn.LayerNorm(hidden_dim, elementwise_affine=False)

        # Configure and apply sparse initialization
        sparse_init = make_sparse_init(lecun_uniform_, SPARSITY)
        for param in self.parameters():
            sparse_init(param)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # LayerNorm added before LeakyReLU activation
        x = F.leaky_relu(self.ln1(self.l1(x)))
        x = F.leaky_relu(self.ln2(self.l2(x)))
        return x


class CategoricalActor(nn.Module):
    def __init__(self, feature_net: nn.Module, num_actions: int):
        super().__init__()
        self.feature_net = feature_net
        self.policy_head = nn.Linear(HIDDEN_SIZE, num_actions)

        # Configure and apply sparse initialization
        sparse_init = make_sparse_init(lecun_uniform_, SPARSITY)
        for param in self.policy_head.parameters():
            sparse_init(param)

    def forward(self, x: torch.Tensor) -> torch.distributions.Categorical:
        features = self.feature_net(x)
        logits = self.policy_head(features)
        return torch.distributions.Categorical(logits=logits)


class CriticNet(nn.Module):
    def __init__(self, feature_net: nn.Module):
        super().__init__()
        self.feature_net = feature_net
        self.value_head = nn.Linear(HIDDEN_SIZE, 1)

        # Configure and apply sparse initialization
        sparse_init = make_sparse_init(lecun_uniform_, SPARSITY)
        for param in self.value_head.parameters():
            sparse_init(param)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.feature_net(x)
        return self.value_head(features)


# ---------------------------------------------------------------------------
# Initialise State & Tracking Abstractions
# ---------------------------------------------------------------------------
env = gym.make("CartPole-v1")
env = gym.wrappers.RecordEpisodeStatistics(env)
env = WelfordNormalizeObservation(env, device=device)
env = WelfordNormalizeReward(env, gamma=GAMMA, device=device)

obs_shape = env.observation_space.shape
num_actions = env.action_space.n

actor = CategoricalActor(LayerNormMLP(obs_shape[0], HIDDEN_SIZE), num_actions).to(
    device
)
critic = CriticNet(LayerNormMLP(obs_shape[0], HIDDEN_SIZE)).to(device)

actor_optimizer = AdaptiveObGD(actor.parameters(), lr=ALPHA, scaling_factor=KAPPA_ACTOR)
critic_optimizer = AdaptiveObGD(
    critic.parameters(), lr=ALPHA, scaling_factor=KAPPA_CRITIC
)

actor_traces = {p: torch.zeros_like(p, device=device) for p in actor.parameters()}
critic_traces = {p: torch.zeros_like(p, device=device) for p in critic.parameters()}

value_buffer = []
return_buffer = []

obs, info = env.reset(seed=SEED)
wandb.init(project="stream-ac-lambda-cartpole")
wandb.define_metric("*", step_metric="global_step")
global_step = 0

# ---------------------------------------------------------------------------
# Main Algorithmic Loop (The Imperative Shell)
# ---------------------------------------------------------------------------
for step in range(MAX_STEPS):
    global_step += 1
    obs_t = to_tensor(obs, device=device).unsqueeze(0)

    # ---- 1. Action Selection Phase ----------------------------------------
    with torch.inference_mode():
        dist = actor(obs_t)
        # Wrap discrete sampling via canonical action selection utilities
        action, act_info = sample_distribution(dist, explore=True)
        log_prob = act_info["log_prob"].sum(dim=-1)
        action_np = to_numpy_action(action)

    action = action.clone()

    # ---- 2. Environment Step ----------------------------------------------
    next_obs, reward, terminated, truncated, info = env.step(int(action_np.item()))
    done = terminated or truncated

    next_obs_t = to_tensor(next_obs, device=device).unsqueeze(0)
    reward_t = to_tensor(reward, device=device)

    # ---- 3. Unified Evaluator Pass ----------------------------------------
    v_current = critic(obs_t).view(-1)

    with torch.no_grad():
        v_next = critic(next_obs_t).view(-1)
        rewards_b = torch.tensor([reward], dtype=torch.float32, device=device)
        terminated_b = torch.tensor([terminated], dtype=torch.bool, device=device)
        gamma_b = torch.tensor([GAMMA], dtype=torch.float32, device=device)

        td_target = compute_v_td_target(
            next_values=v_next,
            rewards=rewards_b,
            terminated=terminated_b,
            gamma=gamma_b,
        )

    delta = td_target - v_current

    # ---- 4. Complete Actor Gradient Evaluation ---------------------------
    dist_grad = actor(obs_t)
    log_prob_grad = dist_grad.log_prob(action.detach()).sum(dim=-1)
    entropy = dist_grad.entropy().sum(dim=-1)

    # Apply entropy regularized objective vector (Appendix E)
    sign_delta = torch.sign(delta).detach()
    policy_objective = log_prob_grad + TAU_ENTROPY * sign_delta * entropy

    actor.zero_grad(set_to_none=True)
    policy_objective.backward()

    # ---- 5. Complete Critic Gradient Evaluation --------------------------
    critic.zero_grad(set_to_none=True)
    v_current.backward()

    # ---- 6. Trace Accumulation (Shared Canonical Core) --------------------
    with torch.no_grad():
        terminated_batch = torch.tensor(
            [terminated], dtype=torch.float32, device=device
        )

        # Accumulate Actor Traces
        for p in actor.parameters():
            if p.grad is not None:
                batched_trace = actor_traces[p].unsqueeze(0)
                batched_grad = p.grad.unsqueeze(0)

                updated_trace = update_accumulating_traces(
                    traces=batched_trace,
                    gradients=batched_grad,
                    gamma=GAMMA,
                    lam=LAMBDA,
                    terminated=terminated_batch,
                )
                actor_traces[p] = updated_trace.squeeze(0)

        # Accumulate Critic Traces
        for p in critic.parameters():
            if p.grad is not None:
                batched_trace = critic_traces[p].unsqueeze(0)
                batched_grad = p.grad.unsqueeze(0)

                updated_trace = update_accumulating_traces(
                    traces=batched_trace,
                    gradients=batched_grad,
                    gamma=GAMMA,
                    lam=LAMBDA,
                    terminated=terminated_batch,
                )
                critic_traces[p] = updated_trace.squeeze(0)

    # ---- 7. In-Place Weight Step Alignment (ObGD Optimizer) ---------------
    critic_optimizer.td_step(error=delta.squeeze(), traces=critic_traces)
    actor_optimizer.td_step(error=delta.squeeze(), traces=actor_traces)

    # ---- 8. Logging & Boundary Life-cycle Processing ----------------------
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
        # Explicit reset tracking states for truncation anomalies
        for t in actor_traces.values():
            t.zero_()
        for t in critic_traces.values():
            t.zero_()
    else:
        obs = next_obs

    # ---- 9. Metric Logging Window Evaluation -------------------------------
    if step % LOG_INTERVAL == 0:
        returns_val = td_target

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
                "loss/total": torch.abs(delta).item(),
                "loss/critic": (delta**2).item(),
                "value/mean": v_current.item(),
                "value/return_mean": returns_val.item(),
                "value/explained_variance": explained_var,
                "advantages/mean": delta.item(),
                "entropy": entropy.item(),
                "log_prob": log_prob.item(),
                "global_step": global_step,
            },
            step=global_step,
        )

wandb.finish()
