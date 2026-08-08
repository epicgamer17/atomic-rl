"""
Stream Q(λ) — off-policy streaming Q-learning on CartPole-v1.

Algorithm 8 from Elsayed, Vasan & Mahmood (2024):
"Streaming Deep Reinforcement Learning Finally Works" (arXiv:2410.14606).

Key Details:
-----------
- Leverages custom Welford Gym wrappers for inline real-time scaling.
- Follows Watkins's Q(λ): eligibility traces are reset to zero BEFORE taking
  an environment step if a non-greedy action is selected.
- Integrates the canonical atomic_rl.traces core engine.

NOTE: We chose to use AdaptiveObGD over the standard ObGD. The reference code
(github.com/mohmdelsayed/streaming-drl) uses plain ObGD for all released stream
algorithms, and the paper's DQN setup (Appendix F.1) uses plain ObGD as well.

NOTE (reference vs. paper): We intentionally match the authors' released code
(github.com/mohmdelsayed/streaming-drl) rather than the paper algorithms — a conscious
and intentional decision.
  - HIDDEN_SIZE matches the reference (128).
  - Reward scaling matches the reference `SampleMeanStd` centered variance via
    `WelfordNormalizeReward` (see envs/wrappers/normalization.py); this differs from
    paper Algorithm 5's mean-zero second moment.
  - Intentional divergence from the reference: we use AdaptiveObGD (the reference
    stream_dqn.py uses plain ObGD, alpha=1, kappa=2).
"""

import math
import gymnasium as gym
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import wandb

from atomic_rl.action_selection import (
    with_epsilon_greedy,
    argmax_selector,
    gather_q_values,
)
from atomic_rl.initialization import (
    set_seed,
    lecun_uniform_,
    make_sparse_init,
)
from atomic_rl.optimizer import AdaptiveObGD
from atomic_rl.traces import compute_accumulating_traces
from atomic_rl.utils import (
    to_tensor,
    to_numpy_action,
    compute_welford_stats,
)
from atomic_rl.schedules import get_linear_schedule
from atomic_rl.td import compute_q_td_target

from envs.wrappers.normalization import (
    WelfordNormalizeObservation,
    WelfordNormalizeReward,
)

# ---------------------------------------------------------------------------
# Hyperparameters & Constants
# ---------------------------------------------------------------------------
GAMMA = 0.99
LAMBDA = 0.8
ALPHA = 1.0
KAPPA_CRITIC = 2.0
SPARSITY = 0.9
HIDDEN_SIZE = 128  # matches the reference implementation
MAX_STEPS = 200_000
SEED = 42
LOG_INTERVAL = 100

EPS_START = 1.0
EPS_END = 0.01
EPS_DECAY_FRAMES = 50000

set_seed(SEED)
device = torch.device("cpu")


# ---------------------------------------------------------------------------
# TODO/NOTE: figure out if we want to make a network component for these in networks/
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


class LayerNormQNet(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, num_actions: int):
        super().__init__()
        self.feature_net = LayerNormMLP(input_dim, hidden_dim)
        self.q_head = nn.Linear(hidden_dim, num_actions)

        # Configure and apply sparse initialization
        sparse_init = make_sparse_init(lecun_uniform_, SPARSITY)
        for param in self.q_head.parameters():
            sparse_init(param)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.feature_net(x)
        return self.q_head(features)


# ---------------------------------------------------------------------------
# Initialization
# ---------------------------------------------------------------------------
env = gym.make("CartPole-v1")
env = gym.wrappers.RecordEpisodeStatistics(env)

env = WelfordNormalizeObservation(env, device=device)
env = WelfordNormalizeReward(env, gamma=GAMMA, device=device)

obs_shape = env.observation_space.shape
num_actions = env.action_space.n

model = LayerNormQNet(obs_shape[0], HIDDEN_SIZE, num_actions).to(device)
optimizer = AdaptiveObGD(model.parameters(), lr=ALPHA, scaling_factor=KAPPA_CRITIC)
traces = {p: torch.zeros_like(p, device=device) for p in model.parameters()}

action_selector = with_epsilon_greedy(argmax_selector)
rng_key = torch.Generator(device=device).manual_seed(SEED)

obs, info = env.reset(seed=SEED)
wandb.init(project="stream-q-lambda-cartpole")

# ---------------------------------------------------------------------------
# Main Algorithmic Loop
# ---------------------------------------------------------------------------
for step in range(MAX_STEPS):
    obs_t = to_tensor(obs, device=device).unsqueeze(0)
    current_epsilon = get_linear_schedule(step, EPS_START, EPS_END, EPS_DECAY_FRAMES)

    # ---- 1. Policy Forward Pass & Action Selection ------------------------
    with torch.inference_mode():
        q_vals = model(obs_t)
        action, act_info = action_selector(
            q_vals, current_epsilon, num_actions, rng_key
        )
        rng_key = act_info["generator"]
        action_np = to_numpy_action(action)

    # Bring the inference tensor back to a normal autograd-safe layout
    action = action.clone()

    # ---- 2. Evaluate Greediness BEFORE Env Step (Watkins Alignment) -------
    # TODO: should we use the argmax_selector here instead? maybe this is unecessarily complicated, but maybe its also more principled as in theory whatever our internal, non random, on policy, selector is is fine.
    greedy_action_tensor = q_vals.argmax(dim=-1)
    is_greedy = torch.eq(action.squeeze(-1), greedy_action_tensor).item()

    # TODO: should this be a helper function?
    if not is_greedy:
        for t in traces.values():
            t.zero_()

    # ---- 3. Environment Step ----------------------------------------------
    next_obs, reward, terminated, truncated, info = env.step(int(action_np.item()))
    done = terminated or truncated

    next_obs_t = to_tensor(next_obs, device=device).unsqueeze(0)
    reward_t = to_tensor(reward, device=device)

    # ---- 4. Complete Mathematical Target Calculation ----------------------
    q_current_all = model(obs_t)

    # Extract online values cleanly via tensor-native gathering
    q_current_action = gather_q_values(q_current_all, action)

    with torch.no_grad():
        q_next_all = model(next_obs_t)

        # Q-learning bootstraps using the greedy next action profile
        next_actions, _ = argmax_selector(q_next_all)

        # Rule: "If a function expects a batch dimension, it must receive a batch dimension."
        # Package single environment scalars into explicit 1D batch arrays [B=1]
        rewards_batch = torch.tensor([reward], dtype=torch.float32, device=device)
        terminated_batch = torch.tensor([terminated], dtype=torch.bool, device=device)
        gamma_batch = torch.tensor([GAMMA], dtype=torch.float32, device=device)

        # Delegate target generation entirely to your canonical functional layer
        # TODO: should we make compute_q_td_target also work with scalars, and add an assert that if one is a scalar the other musnt have a batch dim/be scalars?
        td_target = compute_q_td_target(
            next_q_values=q_next_all,
            next_actions=next_actions.squeeze(
                -1
            ),  # Reshape [1, 1] -> [1] to meet contract
            rewards=rewards_batch,
            terminated=terminated_batch,
            gamma=gamma_batch,
        )

    # Both values are explicit 1D tensors of layout [1], yielding a clean [1] delta
    delta = td_target - q_current_action

    # ---- 5. Differentiate and Accumulate Traces Functional Core ----------
    model.zero_grad(set_to_none=True)
    # Explicitly remove the singular batch dimension [1] -> [] scalar for autograd
    q_current_action.squeeze(0).backward()

    with torch.no_grad():
        terminated_batch = torch.tensor(
            [terminated], dtype=torch.float32, device=device
        )

        for p in model.parameters():
            if p.grad is not None:
                batched_trace = traces[p].unsqueeze(0)
                batched_grad = p.grad.unsqueeze(0)

                updated_batched_trace = compute_accumulating_traces(
                    traces=batched_trace,
                    gradients=batched_grad,
                    gamma=GAMMA,
                    lam=LAMBDA,
                    terminated=terminated_batch,
                )
                traces[p] = updated_batched_trace.squeeze(0)

    # ---- 6. In-Place Weight Step Alignment (ObGD Core Execution) ----------
    optimizer.td_step(error=delta.squeeze(), traces=traces)

    # ---- 7. Logging Metrics -----------------------------------------------
    if step % LOG_INTERVAL == 0:
        wandb.log(
            {
                "loss": torch.abs(delta).item(),
                "epsilon": current_epsilon,
                "q_values/mean": q_current_all.mean().item(),
                "q_values/max": q_current_all.max().item(),
                "rewards/scaled": reward_t.item(),
                "delta": delta.item(),
            },
            step=step,
        )

    # ---- 8. Structural Boundary Management --------------------------------
    if done:
        if "episode" in info:
            wandb.log(
                {
                    "episode_return": info["episode"]["r"][0],
                    "episode_length": info["episode"]["l"][0],
                },
                step=step,
            )

        obs, info = env.reset()
        for t in traces.values():
            t.zero_()
    else:
        obs = next_obs

wandb.finish()
