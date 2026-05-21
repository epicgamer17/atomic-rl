"""
Reproduction of Selective Weight Reinitialization (Hernandez-Garcia et al. 2025).
Task: Permuted MNIST (Section 3 & Appendix C).

The base network (Standard SGD) will lose plasticity, suffering from a drop
in average online accuracy, exploding weight magnitudes, vanishing gradients,
and a collapse in stable rank.
SWR prevents this entirely.

TODO: is this true (AI generated text below)
Comparison Note: This implementation follows the SWR paper (3 hidden layers of 100 units,
SGD optimizer, batch size 30). This differs significantly from cbp_permuted_mnist.py,
which uses a wider architecture (2 hidden layers of 2000 units), the Adam optimizer,
and a larger batch size of 64.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import math
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

from envs.streams.permuted_mnist import make_permuted_mnist_stream
from functional.plasticity import (
    apply_selective_weight_reinitialization,
    init_cbp_state,
    apply_continual_backprop,
)
from functional.metrics import (
    compute_dead_units_proportion,
    compute_average_weight_magnitude,
    compute_average_gradient_magnitude,
    compute_stable_rank,
)
from functional.visualization import (
    plot_plasticity_correlates,
    plot_continual_learning_performance,
)
from functional.utils import gnt_init_wrapper

# Standardized initialization for GnT methods (SWR/CBP): Weights use Kaiming Uniform, Biases use Zero
init_weights_kaiming = gnt_init_wrapper(
    lambda t: nn.init.kaiming_uniform_(t, a=math.sqrt(5))
)


def run_permuted_mnist(mode: str, num_tasks: int = 50):
    torch.manual_seed(42)

    # Paper Architecture: 3 hidden layers of 100 units, ReLU activations
    model = nn.Sequential(
        nn.Linear(784, 100),
        nn.ReLU(),
        nn.Linear(100, 100),
        nn.ReLU(),
        nn.Linear(100, 100),
        nn.ReLU(),
        nn.Linear(100, 10),
    )

    # CBP Setup
    cbp_states = {}
    layer_pairs = []
    if mode == "cbp":
        layer_pairs = [
            (model[0], model[2]),
            (model[2], model[4]),
            (model[4], model[6]),
        ]
        cbp_states = {layer: init_cbp_state(layer) for layer, _ in layer_pairs}

    # Isolate parameters for SWR (paper applies it to all parameters including biases)
    hidden_params = list(model.parameters())

    # Paper uses standard SGD with step size 0.05
    optimizer = optim.SGD(model.parameters(), lr=0.05)
    loss_fn = nn.CrossEntropyLoss()

    # SWR Hyperparameters
    reinit_frequency = 2048  # tau = 2^11
    reinit_factor_k = 1e-5

    # CBP Hyperparameters (Rho tuned to match SWR reinit frequency approx)
    # SWR reinit ~1/2048 parameters per step. Rho = 1/2048 is approx 5e-4
    rho = 1e-4
    eta = 0.99
    maturity = 100

    task_stream = make_permuted_mnist_stream(batch_size=30)
    metrics = {
        "Accuracy": [],
        "Dead Units": [],
        "Weight Mag": [],
        "Grad Mag": [],
        "Stable Rank": [],
    }

    global_step = 0
    desc = f"Training {mode.upper()}"

    for task_idx in tqdm(range(num_tasks), desc=desc):
        dataloader = next(task_stream)

        correct = 0
        total = 0

        for step, (x, target) in enumerate(dataloader):
            # 1. Forward Pass (Capture hidden states for metrics)
            h1 = model[0](x)
            a1 = model[1](h1)

            h2 = model[2](a1)
            a2 = model[3](h2)

            h3 = model[4](a2)
            a3 = model[5](h3)

            pred = model[6](a3)

            # Measure "Online Accuracy" (Accuracy before updating on the batch)
            _, predicted = torch.max(pred.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()

            # 2. Backward Pass
            loss = loss_fn(pred, target)
            optimizer.zero_grad()
            loss.backward()

            # 3. Apply Plasticity Method
            if mode == "swr" and (global_step + 1) % reinit_frequency == 0:
                apply_selective_weight_reinitialization(
                    parameters=hidden_params,
                    optimizer=optimizer,
                    init_fn=init_weights_kaiming,
                    k=reinit_factor_k,
                    utility_type="gradient",
                    prune_type="threshold",
                )

            optimizer.step()

            if mode == "cbp":
                apply_continual_backprop(
                    layer_pairs=layer_pairs,
                    activations=[a1, a2, a3],
                    cbp_states=cbp_states,
                    optimizer=optimizer,
                    init_fn=init_weights_kaiming,
                    eta=eta,
                    replacement_rate=rho,
                    maturity_threshold=maturity,
                )

            global_step += 1

        # --- End of Task Metric Tracking ---
        metrics["Accuracy"].append((correct / total) * 100.0)  # Percentage
        metrics["Dead Units"].append(compute_dead_units_proportion(a1))
        metrics["Weight Mag"].append(compute_average_weight_magnitude(hidden_params))
        metrics["Grad Mag"].append(compute_average_gradient_magnitude(hidden_params))
        metrics["Stable Rank"].append(compute_stable_rank(a3))
        print("Accuracy: ", metrics["Accuracy"][-1])
        print("Dead Units: ", metrics["Dead Units"][-1])
        print("Weight Mag: ", metrics["Weight Mag"][-1])
        print("Grad Mag: ", metrics["Grad Mag"][-1])
        print("Stable Rank: ", metrics["Stable Rank"][-1])
        print("\n")

    return metrics


def main():
    # Note: The paper runs 1,000 tasks. We default to 50 for a quick local test.
    # Change num_tasks=1000 to perfectly replicate the paper's full runtime.
    num_tasks = 100

    print("Running Base System (Standard SGD)...")
    base_metrics = run_permuted_mnist(mode="base", num_tasks=num_tasks)

    print("\nRunning Selective Weight Reinitialization (SWR)...")
    swr_metrics = run_permuted_mnist(mode="swr", num_tasks=num_tasks)

    print("\nRunning Continual Backpropagation (CBP)...")
    cbp_metrics = run_permuted_mnist(mode="cbp", num_tasks=num_tasks)

    # 1. Plot Figure 1: Average Online Accuracy
    plot_continual_learning_performance(
        results_dict={
            "Base System": base_metrics["Accuracy"],
            "SWR": swr_metrics["Accuracy"],
            "CBP": cbp_metrics["Accuracy"],
        },
        title="Permuted MNIST: Average Online Accuracy (SWR Paper Fig 1)",
        xlabel="Permutation Number (Task)",
        ylabel="Accuracy (%)",
        save_path="swr_permuted_mnist_accuracy.png",
    )

    # 2. Plot Figure 10: Plasticity Correlates
    # Remove Accuracy from dicts so the correlates plotter only gets the 4 physical metrics
    del base_metrics["Accuracy"]
    del swr_metrics["Accuracy"]
    del cbp_metrics["Accuracy"]

    plot_plasticity_correlates(
        metrics_dict={
            "Base System": base_metrics,
            "SWR": swr_metrics,
            "CBP": cbp_metrics,
        },
        save_path="swr_permuted_mnist_correlates.png",
    )


if __name__ == "__main__":
    main()
