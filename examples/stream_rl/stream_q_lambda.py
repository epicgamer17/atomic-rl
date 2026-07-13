"""
Stream Q(λ) — off-policy streaming Q-learning on CartPole-v1.

Algorithm 8 from Elsayed, Vasan & Mahmood (2024):
"Streaming Deep Reinforcement Learning Finally Works" (arXiv:2410.14606).

Uses Watkins's Q(λ): eligibility traces are reset to zero whenever a
non-greedy action is taken.  No replay buffer, no target network.

Key components (shared stream-x toolkit)
----------------------------------------
- LayerNorm MLP (affine=False) with SparseInit (sparsity=0.9).
- Online observation normalisation (Welford, via Gym wrapper).
- Reward scaling via discounted trace (Welford, via Gym wrapper, Algorithm 5).
- Accumulating eligibility traces (γλ decay).
- ObGD step-size controller (κ=2).
"""

import math

import gymnasium as gym
import torch
import torch.nn as nn
import torch.nn.functional as F
import wandb

from envs.wrappers.normalization import (
    WelfordNormalizeObservation,
    WelfordNormalizeReward,
)
from functional.action_selection import with_epsilon_greedy, argmax_selector
from functional.initialization import set_seed, sparse_init_weight_
from functional.optimizer import obgd_td_update_
from functional.utils import to_tensor, to_numpy_action

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
GAMMA = 0.99
LAMBDA = 0.8
ALPHA = 1.0
KAPPA_CRITIC = 2.0
SPARSITY = 0.9
HIDDEN_SIZE = 256
MAX_STEPS = 200_000
EPSILON = 0.1
SEED = 42
LOG_INTERVAL = 100

set_seed(SEED)
device = torch.device("cpu")


# ---------------------------------------------------------------------------
# Network
# ---------------------------------------------------------------------------
class LayerNormQNet(nn.Module):
    """MLP: Linear → LayerNorm(affine=False) → ReLU (x2) → Linear(num_actions)."""

    def __init__(self, input_dim: int, hidden_dim: int, num_actions: int):
        super().__init__()
        self.l1 = nn.Linear(input_dim, hidden_dim)
        self.ln1 = nn.LayerNorm(hidden_dim, elementwise_affine=False)
        self.l2 = nn.Linear(hidden_dim, hidden_dim)
        self.ln2 = nn.LayerNorm(hidden_dim, elementwise_affine=False)
        self.l3 = nn.Linear(hidden_dim, num_actions)

        self._sparse_init_()

    def _sparse_init_(self):
        def lecun_uniform_(t):
            fan_in = nn.init._calculate_fan_in_and_fan_out(t)[0]
            bound = 1.0 / math.sqrt(fan_in)
            nn.init.uniform_(t, -bound, bound)

        for name, param in self.named_parameters():
            if param.dim() >= 2:
                lecun_uniform_(param)
                sparse_init_weight_(param, SPARSITY)
            elif param.dim() == 1:
                nn.init.zeros_(param)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.ln1(self.l1(x)))
        x = F.relu(self.ln2(self.l2(x)))
        x = self.l3(x)
        return x


# ---------------------------------------------------------------------------
#  Initialise
# ---------------------------------------------------------------------------
env = gym.make("CartPole-v1")
env = gym.wrappers.RecordEpisodeStatistics(env)
env = WelfordNormalizeObservation(env, device=device)
env = WelfordNormalizeReward(env, gamma=GAMMA, device=device)
obs_shape = env.observation_space.shape
num_actions = env.action_space.n

model = LayerNormQNet(obs_shape[0], HIDDEN_SIZE, num_actions).to(device)

traces = [torch.zeros_like(p, device=device) for p in model.parameters()]

action_selector = with_epsilon_greedy(argmax_selector)
rng_key = torch.Generator(device=device)
rng_key.manual_seed(SEED)

obs, info = env.reset(seed=SEED)

wandb.init(project="stream-q-lambda-cartpole")

# ---------------------------------------------------------------------------
#  Main loop
# ---------------------------------------------------------------------------
for step in range(MAX_STEPS):
    obs_t = to_tensor(obs, device=device).unsqueeze(0)  # [1, D], already normalised

    # ---- act ---------------------------------------------------------------
    with torch.inference_mode():
        q_vals = model(obs_t)
        action, act_info = action_selector(
            predictions=q_vals,
            epsilon=EPSILON,
            num_actions=num_actions,
            generator=rng_key,
        )
        rng_key = act_info["generator"]
        action_np = to_numpy_action(action)

    # ---- environment step (obs & reward pre-normalised by wrappers) --------
    next_obs, reward, terminated, truncated, info = env.step(int(action_np.item()))
    done = terminated or truncated

    next_obs_t = to_tensor(next_obs, device=device).unsqueeze(0)  # already normalised
    reward_t = to_tensor(reward, device=device)  # already scaled

    # ---- forward passes ----------------------------------------------------
    q_current = model(obs_t).squeeze(0)

    with torch.no_grad():
        q_next = model(next_obs_t).squeeze(0)

    greedy_action = q_current.argmax().item()
    action_taken = action.item()
    is_greedy = action_taken == greedy_action

    # ---- TD error  (Watkins's Q(λ): max over next actions) -----------------
    v_next = q_next.max()
    if terminated:
        v_next = torch.tensor(0.0, device=device)
    delta = reward_t + GAMMA * v_next - q_current[action_taken]

    # ---- Watkins's Q(λ): reset traces on non-greedy action -----------------
    if not is_greedy:
        for i in range(len(traces)):
            traces[i].zero_()

    # ---- 5. Differentiate and Accumulate Traces Functional style ----------
    model.zero_grad(set_to_none=True)
    q_current_action.backward()

    with torch.no_grad():
        # Create a explicit 1D batch tensor for the streaming step [B=1]
        terminated_batch = torch.tensor(
            [terminated], dtype=torch.float32, device=device
        )

        for p in model.parameters():
            if p.grad is not None:
                batched_trace = traces[p].unsqueeze(0)
                batched_grad = p.grad.unsqueeze(0)
                traces[p] = update_accumulating_traces(
                    traces=batched_trace,
                    gradients=batched_grad,
                    gamma=GAMMA,
                    lam=LAMBDA,
                    terminated=terminated_batch,
                ).squeeze(0)

    # ---- ObGD update -------------------------------------------------------
    for i, p in enumerate(model.parameters()):
        obgd_td_update_(
            theta=p,
            error=delta.detach(),
            trace=traces[i],
            lr=ALPHA,
            scaling_factor=KAPPA_CRITIC,
        )

    # ---- logging (standard DQN-style + stream-specific) --------------------
    if step % LOG_INTERVAL == 0:
        wandb.log(
            {
                "loss": delta.abs().item(),
                "epsilon": EPSILON,
                "q_values/mean": q_current.mean().item(),
                "q_values/min": q_current.min().item(),
                "q_values/max": q_current.max().item(),
                "td_targets/mean": v_next.item(),
                "rewards/mean": reward_t.item(),
                "delta": delta.item(),
            },
            step=step,
        )

    # ---- handle episode end ------------------------------------------------
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
        for i in range(len(traces)):
            traces[i].zero_()
    else:
        obs = next_obs

wandb.finish()
