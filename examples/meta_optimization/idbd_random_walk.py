"""
Reproduction of Sutton (1992b): "Gain Adaptation Beats Least Squares?"
Comparison of seven algorithms: LMS, NLMS, IDBD, K1, K2, LS (RLS), and Kalman.

Task:
- n = 20 features
- inputs phi_i ~ N(0, 1)
- observation noise variance R = 1
- drift covariance Q: first 5 diagonal entries are 1.0, rest are 0.0.
- 30,000 steps total (20,000 burn-in, 10,000 evaluation)
- Seed reset at the beginning of each algorithm/parameter run.
"""

import torch
import math
import os
import matplotlib.pyplot as plt
from tqdm import tqdm
from typing import Tuple

# TODO: find better ranges for params and U shaped curves.

from functional.meta_optimization import (
    compute_idbd_update,
    compute_k1_update,
    compute_k2_update,
)
from envs.streams.random_walk import make_random_walk_tracking_task


from functional.utils import set_seed


def run_tracking_experiment(
    algorithm: str,
    param: float,
    num_steps: int = 30_000,
    burn_in: int = 20_000,
    num_features: int = 20,
    r_true: float = 1.0,
):
    """
    Runs a single tracking experiment for a specific algorithm and its free parameter.
    """
    set_seed(42)  # Reset seed for each run
    task_generator = make_random_walk_tracking_task(
        num_features=num_features, num_relevant=5, obs_noise_var=r_true, drift_var=1.0
    )

    weights = torch.zeros(num_features)

    # State for IDBD/K1/K2
    initial_beta = math.log(1.0 / num_features)
    betas = torch.full((num_features,), initial_beta)
    h = torch.zeros(num_features)

    # State for LS (RLS)
    # Parameter 'param' is the initialization P(0) = param * I
    p_mat = torch.eye(num_features) * param
    k_p_mat = torch.eye(num_features) * 1.0

    squared_errors = []

    for step in range(num_steps):
        inputs, target = next(task_generator)

        pred = torch.dot(weights, inputs)
        error = target - pred

        if torch.isnan(error) or torch.isinf(error):
            return 25.0  # Divergence penalty

        if step >= burn_in:
            squared_errors.append(error.item() ** 2)

        # Base Algorithmic Updates
        if algorithm == "LMS":
            weights = weights + param * error * inputs

        elif algorithm == "NLMS":
            # Standard NLMS update
            norm = torch.dot(inputs, inputs) + r_true
            weights = weights + (param * error * inputs) / norm

        elif algorithm == "IDBD":
            weights, betas, h, _ = compute_idbd_update(
                weights.unsqueeze(0),
                betas.unsqueeze(0),
                h.unsqueeze(0),
                inputs.unsqueeze(0),
                error.view(1, 1),
                meta_lr=param,
            )
            weights, betas, h = weights.squeeze(0), betas.squeeze(0), h.squeeze(0)

        elif algorithm == "K1":
            weights, betas, h, _ = compute_k1_update(
                weights.unsqueeze(0),
                betas.unsqueeze(0),
                h.unsqueeze(0),
                inputs.unsqueeze(0),
                error.view(1, 1),
                meta_lr=param,
                r_hat=r_true,
            )
            weights, betas, h = weights.squeeze(0), betas.squeeze(0), h.squeeze(0)

        elif algorithm == "K2":
            weights, betas, _ = compute_k2_update(
                weights.unsqueeze(0),
                betas.unsqueeze(0),
                inputs.unsqueeze(0),
                error.view(1, 1),
                meta_lr=param,
                r_hat=r_true,
            )
            weights, betas = weights.squeeze(0), betas.squeeze(0)

        elif algorithm == "LS":
            # Paper Eq 6 & 7: Least Squares with covariance modification
            # Q_hat for LS is lambda * I (param * I)
            p_phi = torch.mv(p_mat, inputs)
            denom = r_true + torch.dot(inputs, p_phi)
            k_gain = p_phi / denom

            weights = weights + k_gain * error

            # P(t+1) = P(t) - P(t) phi phi^T P(t) / denom + Q_hat
            q_hat = param * torch.eye(num_features)
            p_mat = p_mat - torch.ger(p_phi, p_phi) / denom + q_hat

        elif algorithm == "Kalman":
            # Paper Eq 6 & 7 using the True Q matrix
            # Q_hat for Kalman is (1-rho)*Q + rho*I
            q_true = torch.zeros(num_features)
            q_true[0:5] = 1.0  # We know only the first 5 features drift
            q_true_mat = torch.diag(q_true)

            q_hat = (1.0 - param) * q_true_mat + param * torch.eye(num_features)

            p_phi = torch.mv(k_p_mat, inputs)
            denom = r_true + torch.dot(inputs, p_phi)
            k_gain = p_phi / denom

            weights = weights + k_gain * error

            k_p_mat = k_p_mat - torch.ger(p_phi, p_phi) / denom + q_hat

    return math.sqrt(sum(squared_errors) / len(squared_errors))


def main():
    """Main execution block to reproduce Sutton (1992b) Figure 2."""
    # Define parameter ranges based on user provided constraints
    configs = {
        "LMS": {
            "params": [0.001, 0.002, 0.005, 0.01, 0.02, 0.03, 0.05, 0.07],
            "color": "black",
            "marker": "s",
            "label": "LMS",
        },
        "NLMS": {
            "params": [0.01, 0.05, 0.1, 0.2, 0.5, 1.0, 1.5, 2.0],
            "color": "gray",
            "marker": "o",
            "label": "NLMS",
        },
        "IDBD": {
            "params": [0.0001, 0.0002, 0.0003, 0.0005, 0.0006, 0.0008, 0.0009, 0.001],
            "color": "blue",
            "marker": "^",
            "label": "IDBD",
        },
        "K1": {
            "params": [0.0001, 0.0002, 0.0003, 0.0005, 0.0006, 0.0008, 0.0009, 0.001],
            "color": "green",
            "marker": "v",
            "label": "K1",
        },
        "K2": {
            "params": [0.001, 0.005, 0.01, 0.015, 0.02, 0.025, 0.03],
            "color": "red",
            "marker": "d",
            "label": "K2",
        },
        "LS": {
            "params": [0.01, 0.1, 1.0, 10.0, 50.0, 100.0],
            "color": "purple",
            "marker": "x",
            "label": "LS",
        },
        "Kalman": {
            "params": [0.01, 0.05, 0.1, 0.2, 0.5, 1.0, 1.5, 2.0],
            "color": "orange",
            "marker": "*",
            "label": "Kalman",
        },
    }

    results = {algo: [] for algo in configs.keys()}

    print("Reproducing Sutton (1992b) Experiment...")
    for algo, config in configs.items():
        print(f"Evaluating {algo}...")
        for param in tqdm(config["params"]):
            rmse = run_tracking_experiment(algorithm=algo, param=param)
            results[algo].append(rmse)
            print(f"{algo} {param}: {rmse}")

    plt.figure(figsize=(10, 6))
    for algo, config in configs.items():
        p_min = min(config["params"])
        p_max = max(config["params"])
        normalized_params = [
            (p - p_min) / (p_max - p_min) if p_max > p_min else 0.5
            for p in config["params"]
        ]

        plt.plot(
            normalized_params,
            results[algo],
            label=config["label"],
            color=config["color"],
            marker=config["marker"],
            alpha=0.8,
        )

    plt.title("Reproduction of Sutton 1992b: Asymptotic RMSE Comparison")
    plt.xlabel("Rescaled Free Parameter (min to max)")
    plt.ylabel("Asymptotic RMSE")
    plt.ylim(0, 25)
    plt.grid(True, alpha=0.3)
    plt.legend()

    script_dir = os.path.dirname(os.path.abspath(__file__))
    save_path = os.path.join(script_dir, "idbd_random_walk_reproduction.png")

    plt.savefig(save_path)
    print(f"Plot saved to {save_path}")


if __name__ == "__main__":
    main()
