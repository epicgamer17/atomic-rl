"""
Reproduction of Autostep (2012) Figure 1: "Tuning-Free Step-Size Adaptation"

This script tests IDBD vs Autostep on two variants of the non-stationary tracking problem:
1. Normal variance (drift variance = 1.0)
2. High variance (drift variance = 100.0)

The y-axis plots MSE relative to Standard LMS (step-size = 0.1/n = 0.005)
to exactly match Figure 1 of the Mahmood et al. (2012) paper.

Autostep should maintain a stable relative MSE well below 1.0 across a vast
range of meta_lr values, whereas IDBD will heavily depend on tuning meta_lr
depending on the task's variance scale.
"""

import torch
import math
import os
import matplotlib.pyplot as plt
from tqdm import tqdm

from functional.meta_optimization import (
    compute_idbd_update,
    compute_autostep_idbd_update,
)
from envs.streams.random_walk import make_random_walk_tracking_task
from functional.utils import set_seed


def evaluate_algorithm(
    algo: str, param: float, drift_var: float, num_steps=30_000, burn_in=20_000
):
    set_seed(42)
    task_gen = make_random_walk_tracking_task(
        num_features=20, num_relevant=5, obs_noise_var=1.0, drift_var=drift_var
    )

    weights = torch.zeros(20)

    # Initialization based on paper heuristics:
    # Autostep relies on the M-cap so it initializes at a confident 0.1
    # IDBD must be initialized safely at 0.1/n to prevent immediate divergence
    if algo == "Autostep":
        betas = torch.full((20,), math.log(0.1))
    else:
        betas = torch.full((20,), math.log(0.1 / 20))

    h = torch.zeros(20)
    v = torch.zeros(20)  # Only used for autostep

    squared_errors = []

    for step in range(num_steps):
        inputs, target = next(task_gen)

        pred = torch.dot(weights, inputs)
        error = target - pred

        if torch.isnan(error) or torch.isinf(error):
            return float("inf")  # Divergence penalty

        if step >= burn_in:
            squared_errors.append(error.item() ** 2)

        # Algorithmic Updates
        if algo == "LMS":
            weights = weights + param * error * inputs
        else:
            b_w = weights.unsqueeze(0)
            b_b = betas.unsqueeze(0)
            b_h = h.unsqueeze(0)
            b_x = inputs.unsqueeze(0)
            b_e = error.view(1, 1)

            if algo == "IDBD":
                weights, betas, h, _ = compute_idbd_update(
                    b_w, b_b, b_h, b_x, b_e, meta_lr=param
                )
                weights, betas, h = weights.squeeze(0), betas.squeeze(0), h.squeeze(0)

            elif algo == "Autostep":
                weights, betas, h, v, _ = compute_autostep_idbd_update(
                    b_w, b_b, b_h, v.unsqueeze(0), b_x, b_e, meta_lr=param, tau=10000.0
                )
                weights, betas, h, v = (
                    weights.squeeze(0),
                    betas.squeeze(0),
                    h.squeeze(0),
                    v.squeeze(0),
                )

    return sum(squared_errors) / len(squared_errors)


def main():
    # Wider range to capture the U-shape and divergence boundaries
    meta_lrs = [1e-8, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1.0]
    drift_variants = {"Normal (Q=1)": 1.0, "High Variance (Q=100)": 100.0}

    results = {"IDBD": {}, "Autostep": {}}

    for variant_name, drift_var in drift_variants.items():
        print(f"\nEvaluating Task: {variant_name}")

        # 1. Compute Standard LMS Baseline
        # The standard LMS heuristic step size is 0.1 / n
        baseline_mse = evaluate_algorithm("LMS", 0.1 / 20.0, drift_var)
        print(f"Standard LMS Baseline MSE: {baseline_mse:.4f}")

        results["IDBD"][variant_name] = []
        results["Autostep"][variant_name] = []

        for lr in tqdm(meta_lrs):
            idbd_mse = evaluate_algorithm("IDBD", lr, drift_var)
            autostep_mse = evaluate_algorithm("Autostep", lr, drift_var)

            # 2. Store relative MSE
            results["IDBD"][variant_name].append(idbd_mse / baseline_mse)
            results["Autostep"][variant_name].append(autostep_mse / baseline_mse)
            print(f"Autostep {variant_name} {lr}: {autostep_mse:.4f}")
            print(f"IDBD {variant_name} {lr}: {idbd_mse:.4f}")

    # Plotting
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    for i, (variant_name, _) in enumerate(drift_variants.items()):
        # Draw the standard LMS baseline at y=1.0
        axes[i].axhline(
            y=1.0, color="gray", linestyle="--", label="Standard LMS", alpha=0.7
        )

        axes[i].plot(
            meta_lrs,
            results["IDBD"][variant_name],
            label="IDBD",
            color="blue",
            marker="^",
        )
        axes[i].plot(
            meta_lrs,
            results["Autostep"][variant_name],
            label="Autostep",
            color="green",
            marker="s",
        )

        axes[i].set_xscale("log")
        axes[i].set_title(f"Task: {variant_name}")
        axes[i].set_xlabel("Meta-Learning Rate (\u03bc)")
        axes[i].set_ylabel("MSE Relative to Standard LMS")

        # Match paper's aesthetic by clipping Y-axis
        axes[i].set_ylim(0, 1.1)
        axes[i].grid(True, alpha=0.3)
        axes[i].legend(loc="lower left")

    plt.suptitle(
        "Autostep vs IDBD: Robustness to Meta-LR and Variance (Mahmood et al. 2012)"
    )

    script_dir = os.path.dirname(os.path.abspath(__file__))
    save_path = os.path.join(script_dir, "autostep_robustness.png")
    plt.savefig(save_path)
    print(f"\nPlot saved to {save_path}")


if __name__ == "__main__":
    main()
