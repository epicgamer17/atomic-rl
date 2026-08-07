"""
Stream TD(λ) — streaming temporal-difference learning on the ETTm2 dataset.

Recreates the prediction results of Elsayed, Vasan & Mahmood (2024):
"Streaming Deep Reinforcement Learning Finally Works" (arXiv:2410.14606).

NOTE: We chose to use AdaptiveObGD over the standard ObGD for the weight update.
The paper's ETT setup (Appendix F.1) and the reference code use plain ObGD
(alpha=1, kappa=2); we deliberately keep AdaptiveObGD here so the update is
scale-invariant via its EMA second moment.

TODO (reference vs. paper):
  - The reference implementation (github.com/mohmdelsayed/streaming-drl) uses plain
    ObGD in stream_td.py; we match the paper's plain-ObGD ETT experiment only in the
    hyperparameters (ALPHA=1, KAPPA=2) while using AdaptiveObGD for the update.
  - The reference ETT environment min-max normalizes the cumulant (observation traces)
    to [0, 1] before scaling the reward (see envs/streams/ett.py); the paper defines
    the trace without that normalization and we follow the paper.
  - Reference uses HIDDEN_SIZE=128 (we do too here); the other stream examples use 256.
"""

import math
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import wandb

from tqdm import tqdm
from functional.initialization import set_seed, lecun_uniform_, make_sparse_init
from functional.optimizer import AdaptiveObGD, apply_gradients
from functional.td import compute_v_td_target
from functional.traces import update_accumulating_traces
from functional.utils import to_tensor, update_welford_stats
from envs.streams.ett import make_ettm2_stream

# ---------------------------------------------------------------------------
# Constants & Hyperparameters (Appendix F.1)
# ---------------------------------------------------------------------------
GAMMA = 0.99
LAMBDA = 0.8
ALPHA = 1.0  # Step size for Stream TD (ObGD)
KAPPA = 2.0  # Scaling factor κ for Stream TD (ObGD)
SPARSITY = 0.9  # Sparsity ratio s for Stream TD (SparseInit)
HIDDEN_SIZE = 128  # Hidden units for 128x128 network
NUM_RUNS = 5  # 5 independent runs for confidence intervals
SEED_START = 42


# ---------------------------------------------------------------------------
# Networks
# ---------------------------------------------------------------------------
class LayerNormMLP(nn.Module):
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


class StreamTDNet(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int):
        super().__init__()
        self.feature_net = LayerNormMLP(input_dim, hidden_dim)
        self.value_head = nn.Linear(hidden_dim, 1)

        # Configure and apply sparse initialization
        sparse_init = make_sparse_init(lecun_uniform_, SPARSITY)
        for param in self.value_head.parameters():
            sparse_init(param)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.feature_net(x)
        return self.value_head(features)


class ClassicMLP(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int):
        super().__init__()
        self.l1 = nn.Linear(input_dim, hidden_dim)
        self.l2 = nn.Linear(hidden_dim, hidden_dim)
        # Default PyTorch initialization used for classic MLP

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.leaky_relu(self.l1(x))
        return F.leaky_relu(self.l2(x))


class ValueNet(nn.Module):
    def __init__(self, backbone: nn.Module, hidden_dim: int):
        super().__init__()
        self.backbone = backbone
        self.value_head = nn.Linear(hidden_dim, 1)
        nn.init.zeros_(self.value_head.weight)
        nn.init.zeros_(self.value_head.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.value_head(self.backbone(x))


# ---------------------------------------------------------------------------
# Training Logic
# ---------------------------------------------------------------------------
def run_full_pass(device: torch.device):
    print("\n--- Loading Full ETTm2 Stream ---")

    # Collect all observations and cumulants
    obs_list = []
    cum_list = []
    for obs, cum in make_ettm2_stream(start=0, device=device):
        obs_list.append(obs)
        cum_list.append(cum)

    T = len(obs_list)
    print(f"Loaded {T} data points.")

    # Move to GPU/device tensors for speed
    obs_tensor = torch.stack(obs_list)
    cum_tensor = torch.stack(cum_list)

    # Arrays to store training history for post-run returns & plotting
    # Shape: [NUM_RUNS, T - 1]
    stream_preds = np.zeros((NUM_RUNS, T - 1))
    stream_scaled_rewards = np.zeros((NUM_RUNS, T - 1))
    stream_sigmas = np.zeros((NUM_RUNS, T - 1))

    classic_preds = np.zeros((NUM_RUNS, T - 1))

    print(f"Running {NUM_RUNS} independent trials over the entire dataset...")
    for run_idx in tqdm(range(NUM_RUNS), desc="Trials"):
        seed = SEED_START + run_idx
        set_seed(seed)

        # Pre-allocate device tensors to avoid host-device sync in step loop
        stream_preds_dev = torch.zeros(T - 1, device=device)
        stream_scaled_rewards_dev = torch.zeros(T - 1, device=device)
        stream_sigmas_dev = torch.zeros(T - 1, device=device)
        classic_preds_dev = torch.zeros(T - 1, device=device)

        # Initialize Models
        stream_model = StreamTDNet(7, HIDDEN_SIZE).to(device)
        classic_model = ValueNet(ClassicMLP(7, HIDDEN_SIZE), HIDDEN_SIZE).to(device)

        # Optimizers
        stream_optimizer = AdaptiveObGD(
            stream_model.parameters(), lr=ALPHA, scaling_factor=KAPPA
        )
        # Classic TD Optimizer: Adam (Appendix F.1)
        classic_optimizer = torch.optim.Adam(
            classic_model.parameters(),
            lr=3e-4,
            betas=(0.9, 0.999),
            eps=1e-4,
        )

        # Initialize Traces
        stream_traces = {
            p: torch.zeros_like(p, device=device) for p in stream_model.parameters()
        }
        classic_traces = {
            p: torch.zeros_like(p, device=device) for p in classic_model.parameters()
        }

        # Welford Normalization Stats for Stream TD
        obs_mean = torch.zeros(7, device=device)
        obs_sq_diff = torch.ones(7, device=device)
        obs_var = torch.ones(7, device=device)
        obs_count = torch.tensor(0.0, device=device)

        rew_u = torch.tensor(0.0, device=device)
        rew_sq_diff = torch.tensor(1.0, device=device)
        rew_var = torch.tensor(1.0, device=device)
        rew_count = torch.tensor(0.0, device=device)

        # Initial observation normalization
        obs_0 = obs_tensor[0]
        obs_mean, obs_sq_diff, obs_var, obs_count = update_welford_stats(
            obs_mean, obs_sq_diff, obs_count, obs_0.unsqueeze(0)
        )
        norm_obs = (obs_0 - obs_mean) / torch.sqrt(obs_var + 1e-8)

        # Run through the sequence step-by-step
        for t in tqdm(
            range(T - 1), desc=f"Trial {run_idx + 1}/{NUM_RUNS}", leave=False
        ):
            obs = obs_tensor[t]
            next_obs = obs_tensor[t + 1]
            reward = cum_tensor[t + 1]
            terminated = t == T - 2

            # --- Stream TD(0.8) Update Step ---
            # 1. Normalize next observation
            obs_mean, obs_sq_diff, obs_var, obs_count = update_welford_stats(
                obs_mean, obs_sq_diff, obs_count, next_obs.unsqueeze(0)
            )
            norm_next_obs = (next_obs - obs_mean) / torch.sqrt(obs_var + 1e-8)

            # 2. Scale reward via discounted Welford trace
            term_val = 1.0 if terminated else 0.0
            rew_u = GAMMA * (1.0 - term_val) * rew_u + reward
            _, rew_sq_diff, rew_var, rew_count = update_welford_stats(
                torch.zeros_like(rew_u),
                rew_sq_diff,
                rew_count,
                rew_u.unsqueeze(0),
            )
            scaled_reward = reward / torch.sqrt(rew_var + 1e-8)
            std_reward = torch.sqrt(rew_var + 1e-8)

            # Record standard deviation and scaled reward
            stream_sigmas_dev[t] = std_reward
            stream_scaled_rewards_dev[t] = scaled_reward

            # 3. Predict V(S) and V(S')
            v_current = stream_model(norm_obs).squeeze(0)
            with torch.no_grad():
                if terminated:
                    v_next = torch.tensor(0.0, device=device)
                else:
                    v_next = stream_model(norm_next_obs).squeeze(0)
                td_target = scaled_reward + GAMMA * v_next

            delta = td_target - v_current

            # Store the normalized prediction
            stream_preds_dev[t] = v_current

            # 4. Stream TD Backprop & Trace Accumulation
            stream_model.zero_grad(set_to_none=True)
            v_current.backward()

            with torch.no_grad():
                terminated_t = torch.tensor(
                    [terminated], dtype=torch.float32, device=device
                )
                for p in stream_model.parameters():
                    if p.grad is not None:
                        batched_trace = stream_traces[p].unsqueeze(0)
                        batched_grad = p.grad.unsqueeze(0)
                        updated_trace = update_accumulating_traces(
                            traces=batched_trace,
                            gradients=batched_grad,
                            gamma=GAMMA,
                            lam=LAMBDA,
                            terminated=terminated_t,
                        )
                        stream_traces[p] = updated_trace.squeeze(0)

            # 5. In-Place Weight Step Alignment via ObGD
            stream_optimizer.td_step(error=delta, traces=stream_traces)

            # --- Classic TD(0.8) Update Step ---
            # 1. Predict V(S) and V(S')
            v_current_classic = classic_model(obs).squeeze(0)
            with torch.no_grad():
                if terminated:
                    v_next_classic = torch.tensor(0.0, device=device)
                else:
                    v_next_classic = classic_model(next_obs).squeeze(0)
                td_target_classic = reward + GAMMA * v_next_classic

            delta_classic = td_target_classic - v_current_classic
            classic_preds_dev[t] = v_current_classic

            # 2. Classic TD Backprop & Trace Accumulation
            classic_optimizer.zero_grad(set_to_none=True)
            v_current_classic.backward()

            with torch.no_grad():
                for p in classic_model.parameters():
                    if p.grad is not None:
                        batched_trace = classic_traces[p].unsqueeze(0)
                        batched_grad = p.grad.unsqueeze(0)
                        updated_trace = update_accumulating_traces(
                            traces=batched_trace,
                            gradients=batched_grad,
                            gamma=GAMMA,
                            lam=LAMBDA,
                            terminated=terminated_t,
                        )
                        classic_traces[p] = updated_trace.squeeze(0)

            # 3. Apply gradients manually to match TD(λ) update direction
            for p in classic_model.parameters():
                if p.grad is not None:
                    p.grad.copy_(-delta_classic.detach() * classic_traces[p])

            # 4. Adam optimizer step
            classic_optimizer.step()

            # Next state setup
            norm_obs = norm_next_obs

        # Copy to CPU arrays
        stream_preds[run_idx] = stream_preds_dev.detach().cpu().numpy()
        stream_scaled_rewards[run_idx] = (
            stream_scaled_rewards_dev.detach().cpu().numpy()
        )
        stream_sigmas[run_idx] = stream_sigmas_dev.detach().cpu().numpy()
        classic_preds[run_idx] = classic_preds_dev.detach().cpu().numpy()

        if (run_idx + 1) % 5 == 0:
            print(f"Finished run {run_idx + 1}/{NUM_RUNS}")

    # Compute raw predictions V_t * sigma_t for Stream TD
    stream_preds_raw = stream_preds * stream_sigmas

    # Compute the true returns G_scaled of the scaled rewards
    print("Computing true returns of the scaled rewards backward...")
    stream_returns_raw = np.zeros_like(stream_preds_raw)
    for r in range(NUM_RUNS):
        curr_ret = 0.0
        for t in reversed(range(T - 1)):
            curr_ret = stream_scaled_rewards[r, t] + GAMMA * curr_ret
            stream_returns_raw[r, t] = curr_ret * stream_sigmas[r, t]

    # Classic TD True Returns: standard raw returns of raw cumulants
    classic_returns_raw = np.zeros(T - 1)
    curr_ret = 0.0
    for t in reversed(range(T - 1)):
        curr_ret = cum_list[t + 1].item() + GAMMA * curr_ret
        classic_returns_raw[t] = curr_ret

    return (
        stream_preds_raw,
        stream_returns_raw,
        classic_preds,
        classic_returns_raw,
        np.array(cum_list[:-1]),
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--wandb", action="store_true", help="Enable weights and biases logging"
    )
    args = parser.parse_args()

    # Determine device: use GPU if available and requested, otherwise cpu
    device = torch.device("cpu")
    print(f"Running on device: {device}")

    # 1. Run the single pass training over the entire dataset
    (
        stream_preds_raw,
        stream_returns_raw,
        classic_preds,
        classic_returns_raw,
        raw_temps,
    ) = run_full_pass(device)

    # Compute global temperature min and max for normalization
    min_temp = np.min(raw_temps)
    max_temp = np.max(raw_temps)
    temp_range = max_temp - min_temp
    print(
        f"Global Temperature Stats - Min: {min_temp:.2f}, Max: {max_temp:.2f}, Range: {temp_range:.2f}"
    )

    # Set up plots (we will recreate the two subplots of Figure 10)
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))

    windows = [
        ("Start of the time series", 0, 5000, axes[0], "ett_prediction_start.png"),
        ("End of the time series", 63000, 68000, axes[1], "ett_prediction_end.png"),
    ]

    if args.wandb:
        wandb.init(project="stream-td-lambda-ett")

    for name, start_idx, stop_idx, ax, save_filename in windows:
        # Slice predictions and returns
        stream_preds_slice = stream_preds_raw[:, start_idx:stop_idx]
        stream_rets_slice = stream_returns_raw[:, start_idx:stop_idx]

        classic_preds_slice = classic_preds[:, start_idx:stop_idx]
        # For classic TD, the true return is constant across runs
        classic_rets_slice = classic_returns_raw[start_idx:stop_idx]

        # Scale predictions and returns by (1 - gamma) to match the temperature scale
        # and then normalize using global min/max temperature range to percentage [0, 100]
        norm_stream_preds = (
            (stream_preds_slice * (1.0 - GAMMA) - min_temp) / temp_range * 100.0
        )
        norm_stream_rets = (
            (stream_rets_slice * (1.0 - GAMMA) - min_temp) / temp_range * 100.0
        )

        norm_classic_preds = (
            (classic_preds_slice * (1.0 - GAMMA) - min_temp) / temp_range * 100.0
        )
        norm_classic_rets = (
            (classic_rets_slice * (1.0 - GAMMA) - min_temp) / temp_range * 100.0
        )

        # Compute mean and 90% confidence intervals (1.645 * SEM)
        stream_preds_mean = np.mean(norm_stream_preds, axis=0)
        stream_preds_sem = np.std(norm_stream_preds, axis=0, ddof=1) / math.sqrt(
            NUM_RUNS
        )
        stream_preds_ci = 1.645 * stream_preds_sem

        stream_rets_mean = np.mean(norm_stream_rets, axis=0)

        classic_preds_mean = np.mean(norm_classic_preds, axis=0)
        classic_preds_sem = np.std(norm_classic_preds, axis=0, ddof=1) / math.sqrt(
            NUM_RUNS
        )
        classic_preds_ci = 1.645 * classic_preds_sem

        time_steps = np.arange(start_idx, stop_idx)

        # Plot True Return
        # For Stream TD, the true return curve actually starts at 0 due to scaled reward transient
        ax.plot(
            time_steps,
            stream_rets_mean,
            label="True Return (Stream TD scale)",
            color="#800080",
            alpha=0.8,
            linewidth=1.5,
        )

        # Plot Stream TD Prediction
        ax.plot(
            time_steps,
            stream_preds_mean,
            label="Stream TD(0.8) + ObGD",
            color="#008080",
            alpha=0.9,
            linewidth=1.5,
        )
        ax.fill_between(
            time_steps,
            stream_preds_mean - stream_preds_ci,
            stream_preds_mean + stream_preds_ci,
            color="#008080",
            alpha=0.2,
        )

        # Plot Classic TD Prediction
        ax.plot(
            time_steps,
            classic_preds_mean,
            label="Classic TD(0.8) + Adam",
            color="#FF6347",
            alpha=0.9,
            linewidth=1.5,
        )
        ax.fill_between(
            time_steps,
            classic_preds_mean - classic_preds_ci,
            classic_preds_mean + classic_preds_ci,
            color="#FF6347",
            alpha=0.2,
        )

        ax.set_title(name, fontsize=14, fontweight="bold")
        ax.set_xlabel("Time Step", fontsize=12)
        ax.set_ylabel("Normalized Oil Temp.", fontsize=12)
        ax.legend(loc="upper right", fontsize=10)
        ax.grid(True, linestyle="--", alpha=0.5)

        # Set specific y-limits and x-limits to match the paper plots exactly
        if "Start" in name:
            ax.set_ylim(0, 100)
            ax.set_xlim(start_idx, stop_idx)
        else:
            ax.set_ylim(40, 80)
            ax.set_xlim(start_idx, stop_idx)

        # Save individual plot
        plt.figure(figsize=(8, 6))
        plt.plot(
            time_steps,
            stream_rets_mean,
            label="True Return (Stream TD scale)",
            color="#800080",
            alpha=0.8,
        )
        plt.plot(
            time_steps,
            stream_preds_mean,
            label="Stream TD(0.8)",
            color="#008080",
            alpha=0.9,
        )
        plt.fill_between(
            time_steps,
            stream_preds_mean - stream_preds_ci,
            stream_preds_mean + stream_preds_ci,
            color="#008080",
            alpha=0.2,
        )
        plt.plot(
            time_steps,
            classic_preds_mean,
            label="Classic TD(0.8)",
            color="#FF6347",
            alpha=0.9,
        )
        plt.fill_between(
            time_steps,
            classic_preds_mean - classic_preds_ci,
            classic_preds_mean + classic_preds_ci,
            color="#FF6347",
            alpha=0.2,
        )
        plt.title(name, fontsize=14, fontweight="bold")
        plt.xlabel("Time Step", fontsize=12)
        plt.ylabel("Normalized Oil Temp.", fontsize=12)
        plt.legend(loc="upper right")
        plt.grid(True, linestyle="--", alpha=0.5)
        if "Start" in name:
            plt.ylim(0, 100)
            plt.xlim(start_idx, stop_idx)
        else:
            plt.ylim(40, 80)
            plt.xlim(start_idx, stop_idx)
        plt.savefig(save_filename, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"Saved plot: {save_filename}")

    # Save combined plot
    fig.tight_layout()
    combined_filename = "ett_prediction_comparison.png"
    fig.savefig(combined_filename, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved combined plot: {combined_filename}")

    if args.wandb:
        wandb.log({"ett_prediction_comparison": wandb.Image(combined_filename)})
        wandb.finish()


if __name__ == "__main__":
    main()
