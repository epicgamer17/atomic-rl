"""
Stream TD(λ) — value prediction on the ETTm2 electricity transformer dataset.

Reproduces Experiment 4.5 / Algorithm 9 from
Elsayed, Vasan & Mahmood (2024): "Streaming Deep Reinforcement Learning Finally Works"
(arXiv:2410.14606).

Instead of logging internal algorithm signals (δ, v̂, etc.), this example logs the
actual prediction quality: the unscaled value-function prediction V̂(s) against
the λ-return target G^λ over two held-out segments of the time series (early and
late training).  A comparison baseline (vanilla MLP + Adam + same traces) runs
alongside to show the "stream barrier."

Key design decisions
--------------------
- **Stream model**: LayerNorm MLP (affine=False) + SparseInit + ObGD + eligibility traces.
- **Vanilla model**: standard MLP (no LayerNorm) + standard init + Adam + eligibility traces.
- Both share the same online observation normalisation and reward scaling.
- λ-return evaluation segments are computed offline (no learning) every N steps.
- Figure saved to ``stream_td_lambda_comparison.png``.
"""

import math
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import wandb

from envs.streams.ett import make_ettm2_stream
from functional.initialization import set_seed, sparse_init_weight_
from functional.optimizer import obgd_td_update_
from functional.utils import (
    normalize_features,
    update_welford_stats,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
GAMMA = 0.99
LAMBDA = 0.8
ALPHA = 1.0
KAPPA = 2.0
SPARSITY = 0.9
HIDDEN_SIZE = 256
TOTAL_STEPS = 69_680
LOG_INTERVAL = 200
EVAL_INTERVAL = 10_000
SEED = 42
OBS_EPS = 1e-8
REWARD_EPS = 1e-8

# λ-return evaluation segments
EVAL_WINDOW = 4_000
EVAL_START = 2_000  # early segment start
EVAL_END = TOTAL_STEPS - EVAL_WINDOW - 2_000  # late segment start

set_seed(SEED)
device = torch.device("cpu")
os.makedirs("figures", exist_ok=True)


# ---------------------------------------------------------------------------
# Networks
# ---------------------------------------------------------------------------
class LayerNormValueNet(nn.Module):
    """MLP: Linear → LayerNorm(affine=False) → ReLU (x2) → Linear(1)."""

    def __init__(self, input_dim: int, hidden_dim: int):
        super().__init__()
        self.l1 = nn.Linear(input_dim, hidden_dim)
        self.ln1 = nn.LayerNorm(hidden_dim, elementwise_affine=False)
        self.l2 = nn.Linear(hidden_dim, hidden_dim)
        self.ln2 = nn.LayerNorm(hidden_dim, elementwise_affine=False)
        self.l3 = nn.Linear(hidden_dim, 1)
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


class PlainValueNet(nn.Module):
    """Standard MLP (no LayerNorm, standard init) for the vanilla baseline."""

    def __init__(self, input_dim: int, hidden_dim: int):
        super().__init__()
        self.l1 = nn.Linear(input_dim, hidden_dim)
        self.l2 = nn.Linear(hidden_dim, hidden_dim)
        self.l3 = nn.Linear(hidden_dim, 1)
        self._init_()

    def _init_(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.l1(x))
        x = F.relu(self.l2(x))
        x = self.l3(x)
        return x


# ---------------------------------------------------------------------------
# λ-return evaluator
# ---------------------------------------------------------------------------
@torch.no_grad()
def compute_lambda_returns(
    obs_seq: torch.Tensor,  # [T, D]
    cumulant_seq: torch.Tensor,  # [T]
    model: nn.Module,
    gamma: float,
    lam: float,
    reward_std: float,
    obs_mean: torch.Tensor,
    obs_m2: torch.Tensor,
    obs_count: torch.Tensor,
) -> tuple:  # (lam_returns [T], values [T])
    """
    Compute λ-returns G^λ_t over a held-out sequence.

    The λ-return is defined as the geometric mixture of all n-step returns:
        G^λ_t = r_{t+1} + γ[(1-λ)v̂(s_{t+1}) + λ·G^λ_{t+1}]

    with G^λ_T = v̂(s_T) at the terminal step T.
    """
    seq_len = obs_seq.shape[0]
    values = torch.zeros(seq_len, device=device)

    for t in range(seq_len):
        normalized = normalize_features(
            obs_seq[t].unsqueeze(0), obs_mean, obs_m2, obs_count, OBS_EPS
        ).squeeze(0)
        values[t] = model(normalized.unsqueeze(0)).squeeze(0)

    # Scale values back to original units
    values = values * reward_std

    # Backward pass: λ-return
    lam_returns = torch.zeros(seq_len, device=device)
    G = values[-1].clone()
    lam_returns[-1] = G

    for t in range(seq_len - 2, -1, -1):
        r = cumulant_seq[t + 1]
        v_next = values[t + 1]
        G = r + gamma * ((1.0 - lam) * v_next + lam * G)
        lam_returns[t] = G

    return lam_returns, values


# ---------------------------------------------------------------------------
#  Initialise
# ---------------------------------------------------------------------------
stream_model = LayerNormValueNet(input_dim=7, hidden_dim=HIDDEN_SIZE).to(device)
vanilla_model = PlainValueNet(input_dim=7, hidden_dim=HIDDEN_SIZE).to(device)

stream_traces = [torch.zeros_like(p, device=device) for p in stream_model.parameters()]
vanilla_traces = [
    torch.zeros_like(p, device=device) for p in vanilla_model.parameters()
]

vanilla_optimizer = torch.optim.Adam(vanilla_model.parameters(), lr=1e-3)

# Online observation normalisation (multi-dim Welford via batch-dim convention)
stream_obs_mean = torch.zeros(7, device=device)
stream_obs_m2 = torch.zeros(7, device=device)
stream_obs_count = torch.tensor(0.0, device=device)

# Reward scaling (Algorithm 5): discounted trace + Welford
stream_u = torch.tensor(0.0, device=device)
stream_u_mean = torch.tensor(0.0, device=device)
stream_u_m2 = torch.tensor(0.0, device=device)
stream_u_count = torch.tensor(0.0, device=device)

stream_stream = make_ettm2_stream(gamma=GAMMA, device=device)

wandb.init(project="stream-td-lambda-ett")

# Evaluation: separate normaliser stats for offline λ-return eval
eval_obs_mean = torch.zeros(7, device=device)
eval_obs_m2 = torch.zeros(7, device=device)
eval_obs_count = torch.tensor(0.0, device=device)


# ---------------------------------------------------------------------------
#  λ-return evaluation + figure
# ---------------------------------------------------------------------------
def _run_lambda_eval(
    s_model: nn.Module,
    v_model: nn.Module,
    obs_mean: torch.Tensor,
    obs_m2: torch.Tensor,
    obs_count: torch.Tensor,
    r_std: torch.Tensor,
    current_step: int,
):
    with torch.no_grad():
        from envs.streams.ett import _ensure_data_downloaded, load_ettm2_array

        path = _ensure_data_downloaded()
        data = load_ettm2_array(path)

        segments = {
            "Early training": (EVAL_START, EVAL_START + EVAL_WINDOW),
            "Late training": (EVAL_END, EVAL_END + EVAL_WINDOW),
        }

        fig, axes = plt.subplots(1, 2, figsize=(14, 4.5), sharey=True)
        reward_std_val = r_std.item() if r_std.numel() == 1 else 1.0

        for ax, (title, (start, end)) in zip(axes, segments.items()):
            T = end - start
            obs_seq = torch.as_tensor(
                data[start:end, :], dtype=torch.float32, device=device
            )
            cum_seq = torch.as_tensor(
                data[start:end, 6], dtype=torch.float32, device=device
            )

            lam_ret_s, vals_s = compute_lambda_returns(
                obs_seq,
                cum_seq,
                s_model,
                GAMMA,
                LAMBDA,
                reward_std_val,
                obs_mean,
                obs_m2,
                obs_count,
            )
            lam_ret_v, vals_v = compute_lambda_returns(
                obs_seq,
                cum_seq,
                v_model,
                GAMMA,
                LAMBDA,
                reward_std_val,
                obs_mean,
                obs_m2,
                obs_count,
            )

            x = np.arange(T)
            ax.plot(
                x,
                lam_ret_s.cpu().numpy(),
                label="λ-return (target)",
                color="black",
                linewidth=1.2,
            )
            ax.plot(
                x,
                vals_s.cpu().numpy(),
                label="Stream TD(λ)",
                color="#1f77b4",
                linewidth=0.8,
                alpha=0.9,
            )
            ax.plot(
                x,
                vals_v.cpu().numpy(),
                label="Vanilla TD(λ)",
                color="#ff7f0e",
                linewidth=0.8,
                alpha=0.9,
                linestyle="--",
            )

            ax.set_title(title)
            ax.set_xlabel("Time step")
            if ax == axes[0]:
                ax.set_ylabel("Oil temperature / prediction")

            mse_s = F.mse_loss(vals_s, lam_ret_s).item()
            mse_v = F.mse_loss(vals_v, lam_ret_v).item()
            wandb.log(
                {
                    f"lambda_mse/stream_{title.replace(' ', '_').lower()}": mse_s,
                    f"lambda_mse/vanilla_{title.replace(' ', '_').lower()}": mse_v,
                },
                step=current_step,
            )

        axes[0].legend(fontsize=9)
        fig.suptitle(
            f"Stream TD(λ) vs. Vanilla TD(λ) — step {current_step}", fontsize=12
        )
        fig.tight_layout(rect=(0, 0, 1, 0.96))
        fig.savefig("figures/stream_td_lambda_comparison.png", dpi=150)
        wandb.log(
            {
                "comparison_figure": wandb.Image(
                    "figures/stream_td_lambda_comparison.png"
                )
            },
            step=current_step,
        )
        plt.close(fig)


# Run initial eval (model at init, before any training)
unbiased_var = stream_u_m2 / torch.clamp(stream_u_count - 1.0, min=1.0)
reward_std = torch.sqrt(unbiased_var + REWARD_EPS)
_run_lambda_eval(
    stream_model,
    vanilla_model,
    eval_obs_mean,
    eval_obs_m2,
    eval_obs_count,
    reward_std,
    0,
)

# ---------------------------------------------------------------------------
#  Main loop
# ---------------------------------------------------------------------------
obs, cumulant = next(stream_stream)

obs_t = obs.unsqueeze(0)  # [1, 7]
stream_obs_mean, stream_obs_m2, stream_obs_count = update_welford_stats(
    stream_obs_mean, stream_obs_m2, stream_obs_count, obs_t
)
obs_n_stream = normalize_features(
    obs_t, stream_obs_mean, stream_obs_m2, stream_obs_count, OBS_EPS
).squeeze(0)
obs_n_vanilla = obs_n_stream.clone()

# Eval normaliser tracks its own separate stats
eval_obs_mean, eval_obs_m2, eval_obs_count = update_welford_stats(
    eval_obs_mean, eval_obs_m2, eval_obs_count, obs_t
)

for step in range(1, TOTAL_STEPS):
    next_obs, cumulant = next(stream_stream)

    # ---- normalise observations -------------------------------------------
    next_obs_t = next_obs.unsqueeze(0)
    stream_obs_mean, stream_obs_m2, stream_obs_count = update_welford_stats(
        stream_obs_mean, stream_obs_m2, stream_obs_count, next_obs_t
    )
    next_obs_n_stream = normalize_features(
        next_obs_t, stream_obs_mean, stream_obs_m2, stream_obs_count, OBS_EPS
    ).squeeze(0)
    next_obs_n_vanilla = next_obs_n_stream.clone()

    eval_obs_mean, eval_obs_m2, eval_obs_count = update_welford_stats(
        eval_obs_mean, eval_obs_m2, eval_obs_count, next_obs_t
    )

    # ---- scale reward (Algorithm 5) ---------------------------------------
    stream_u = GAMMA * stream_u + cumulant
    stream_u_mean, stream_u_m2, stream_u_count = update_welford_stats(
        stream_u_mean, stream_u_m2, stream_u_count, stream_u.unsqueeze(0)
    )
    unbiased_var = stream_u_m2 / torch.clamp(stream_u_count - 1.0, min=1.0)
    reward_std = torch.sqrt(unbiased_var + REWARD_EPS)
    stream_scaled_r = cumulant / reward_std
    vanilla_scaled_r = cumulant / reward_std

    # ---- forward passes ---------------------------------------------------

    # Stream model
    v_stream = stream_model(obs_n_stream.unsqueeze(0)).squeeze(0)
    with torch.no_grad():
        v_next_stream = stream_model(next_obs_n_stream.unsqueeze(0)).squeeze(0)

    # Vanilla model
    v_vanilla = vanilla_model(obs_n_vanilla.unsqueeze(0)).squeeze(0)
    with torch.no_grad():
        v_next_vanilla = vanilla_model(next_obs_n_vanilla.unsqueeze(0)).squeeze(0)

    # ---- TD errors --------------------------------------------------------
    delta_stream = stream_scaled_r + GAMMA * v_next_stream - v_stream
    delta_vanilla = vanilla_scaled_r + GAMMA * v_next_vanilla - v_vanilla

    # ---- Stream: eligibility trace + ObGD update --------------------------
    stream_model.zero_grad()
    v_stream.backward()
    for i, p in enumerate(stream_model.parameters()):
        if p.grad is not None:
            stream_traces[i] = GAMMA * LAMBDA * stream_traces[i] + p.grad.detach()
    for i, p in enumerate(stream_model.parameters()):
        obgd_td_update_(
            theta=p,
            error=delta_stream.detach(),
            trace=stream_traces[i],
            lr=ALPHA,
            scaling_factor=KAPPA,
        )

    # ---- Vanilla: eligibility trace + Adam update -------------------------
    vanilla_model.zero_grad()
    v_vanilla.backward()
    for i, p in enumerate(vanilla_model.parameters()):
        if p.grad is not None:
            vanilla_traces[i] = GAMMA * LAMBDA * vanilla_traces[i] + p.grad.detach()
    for i, p in enumerate(vanilla_model.parameters()):
        if vanilla_traces[i] is not None:
            p.grad = vanilla_traces[i] * delta_vanilla.detach()
    vanilla_optimizer.step()

    # ---- log scaled values ------------------------------------------------
    if step % LOG_INTERVAL == 0:
        wandb.log(
            {
                "step": step,
                "v_stream_scaled": v_stream.item(),
                "v_vanilla_scaled": v_vanilla.item(),
                "delta_stream": delta_stream.item(),
                "delta_vanilla": delta_vanilla.item(),
                "reward_std": reward_std.item(),
            },
            step=step,
        )

    # ---- λ-return evaluation ----------------------------------------------
    if step % EVAL_INTERVAL == 0 or step == TOTAL_STEPS - 1:
        _run_lambda_eval(
            stream_model,
            vanilla_model,
            eval_obs_mean,
            eval_obs_m2,
            eval_obs_count,
            reward_std,
            step,
        )

    # ---- shift to next step -----------------------------------------------
    obs_n_stream = next_obs_n_stream
    obs_n_vanilla = next_obs_n_vanilla

    # ---- progress ---------------------------------------------------------
    if step % (EVAL_INTERVAL // 2) == 0:
        print(
            f"[{step:6d}/{TOTAL_STEPS}]  σ_R={reward_std.item():.3f}  "
            f"v̂_stream={v_stream.item():.4f}  v̂_vanilla={v_vanilla.item():.4f}  "
            f"δ_stream={delta_stream.item():.3f}"
        )


print("Finished — figure saved to figures/stream_td_lambda_comparison.png")
wandb.finish()
