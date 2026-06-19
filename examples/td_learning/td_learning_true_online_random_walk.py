"""
Reproduction of Sutton (1988): "Learning to Predict by the Methods of Temporal Differences"
Experiment: The Random Walk (Section 3).

This example uses True Online TD(lambda) instead of the standard semi-gradient accumulating trace TD(lambda).
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import random
from typing import List, Tuple

from envs.mdp.random_walk import RandomWalkEnv
from functional.td import true_online_td_update
from functional.traces import update_true_online_traces

# --- CONSTANTS ---
NUM_NON_TERMINAL_STATES = 5
START_STATE = 2  # State D (0:B, 1:C, 2:D, 3:E, 4:F)
INITIAL_WEIGHT = 0.5
TRUE_VALUES = torch.tensor([1 / 6, 2 / 6, 3 / 6, 4 / 6, 5 / 6])

# Experiment parameters for Figure 3 (Online)
LAMBDAS_ONLINE = [0.0, 0.1, 0.3, 0.5, 0.7, 0.9, 1.0]
ALPHAS_ONLINE = np.linspace(0, 0.6, 13)

NUM_TRAINING_SETS = 100
EPISODES_PER_SET = 10
SEED = 42

# Set random seeds for reproducibility
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)


def true_online_random_walk_episode(
    weights: torch.Tensor, alpha: float, lam: float
) -> torch.Tensor:
    """
    Runs a single random walk episode and updates weights online using True Online TD(lambda).
    """
    env = RandomWalkEnv(num_states=NUM_NON_TERMINAL_STATES, start_state=START_STATE)
    phi_t = env.reset()
    traces = torch.zeros_like(weights)

    # Explicitly calculate the initial v_old using the starting weights
    v_old = torch.dot(weights, phi_t)

    with torch.inference_mode():
        while True:
            # 1. Take a step in the environment
            phi_next, reward, terminated = env.step()

            # 2. Update True Online Eligibility Trace
            # update_true_online_traces expects [batch, features]
            traces = update_true_online_traces(
                traces=traces.unsqueeze(0),
                features=phi_t.unsqueeze(0),
                alpha=alpha,
                gamma=1.0,
                lam=lam,
                terminated=torch.tensor([False]),  # Only clear trace after episode
            ).squeeze(0)

            # 3. Compute and Apply True Online TD Update
            # Notice how v_old goes in, and the new v_next comes out to be used in the next step
            weights, v_old = true_online_td_update(
                features=phi_t,
                reward=reward,
                next_features=phi_next,
                v_old=v_old,
                gamma=1.0,
                weights=weights,
                alpha=alpha,
                trace=traces,
                terminated=terminated,
            )

            if terminated:
                break

            phi_t = phi_next

    return weights


def run_online_experiment():
    """
    Runs the online experiment (Experiment 2) to replicate Figure 3
    and Figure 5 from Sutton (1988).
    """
    results = {lam: [] for lam in LAMBDAS_ONLINE}

    print(f"Running Experiment 2 (True Online Updates - Figure 3/5)...")

    for lam in LAMBDAS_ONLINE:
        print(f"Testing lambda = {lam}")
        for alpha in ALPHAS_ONLINE:
            total_rms_error = 0.0

            for _ in range(NUM_TRAINING_SETS):
                weights = torch.full((NUM_NON_TERMINAL_STATES,), INITIAL_WEIGHT)
                for _ in range(EPISODES_PER_SET):
                    weights = true_online_random_walk_episode(weights, alpha, lam)

                rms_error = torch.sqrt(torch.mean((weights - TRUE_VALUES) ** 2)).item()
                total_rms_error += rms_error

            avg_rms_error = total_rms_error / NUM_TRAINING_SETS
            results[lam].append(avg_rms_error)

    plot_results(
        results,
        ALPHAS_ONLINE,
        "Replication of Sutton (1988) Figure 3 (True Online TD)",
        "true_online_random_walk_fig3.png",
        xlabel=r"$\alpha$",
    )


def plot_results(
    results: dict, x_values: np.ndarray, title: str, filename: str, xlabel: str
):
    """
    General plotting function for Figure 3.
    """
    plt.figure(figsize=(10, 6))

    for lam, errors in results.items():
        valid_indices = [i for i, error in enumerate(errors) if error < 0.7]
        plt.plot(
            [x_values[i] for i in valid_indices],
            [errors[i] for i in valid_indices],
            marker="o",
            label=r"$\lambda$ = " + str(lam),
        )

    plt.xlabel(xlabel)
    plt.ylabel("ERROR (RMS)")
    plt.title(title)
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.7)

    plot_path = f"/Users/jonathanlamontange-kratz/Documents/GitHub/rl-stuff/{filename}"
    plt.savefig(plot_path)
    print(f"Plot saved to {plot_path}")
    plt.show()


if __name__ == "__main__":
    run_online_experiment()
