"""
Reproduction of Sutton (1992a) Figure 4: "Adapting Bias by Gradient Descent"
Demonstrates IDBD's ability to perform automatic feature selection in an online setting.

Task:
- 20 features, only 5 are relevant.
- Target drifts by flipping signs occasionally.
- Over 250,000 steps, IDBD increases the learning rate of relevant features
  while decaying the learning rate of irrelevant features to zero.
"""

import torch
import math
import os
import numpy as np
from tqdm import tqdm

from functional.meta_optimization import compute_idbd_rates
from functional.visualization import plot_learning_rate_traces
from envs.streams.drifting_concept import make_drifting_concept_task


def main():
    num_features = 20
    num_relevant = 5
    num_steps = 250_000
    meta_lr = 0.001
    init_lr = 0.05

    # 1. Initialize Task Generator
    task_gen = make_drifting_concept_task(
        num_features=num_features,
        num_relevant=num_relevant,
        flip_interval=20,
        obs_noise_var=0.0,
    )

    # 2. Initialize IDBD State
    weights = torch.zeros(num_features)
    betas = torch.full((num_features,), math.log(init_lr))
    h = torch.zeros(num_features)

    # 3. Logging Buffer
    # We log every 100 steps to save memory/plotting time
    log_interval = 100
    alphas_history = np.zeros((num_steps // log_interval, num_features))

    print(f"Running IDBD Feature Selection over {num_steps} steps...")

    for step in tqdm(range(num_steps)):
        inputs, target = next(task_gen)

        pred = torch.dot(weights, inputs)
        error = target - pred

        # Compute IDBD (Adding batch dimension via unsqueeze)
        betas, h, alphas = compute_idbd_rates(
            betas.unsqueeze(0),
            h.unsqueeze(0),
            inputs.unsqueeze(0),
            error.view(1, 1),
            meta_lr=meta_lr,
        )
        betas, h, alphas = betas.squeeze(0), h.squeeze(0), alphas.squeeze(0)
        weights = weights + alphas * error * inputs

        if step % log_interval == 0:
            alphas_history[step // log_interval] = alphas.squeeze(0).numpy()

    # 4. Visualize Results
    script_dir = os.path.dirname(os.path.abspath(__file__))
    save_path = os.path.join(script_dir, "idbd_feature_selection.png")

    plot_learning_rate_traces(
        alphas_history=alphas_history,
        num_relevant=num_relevant,
        title="IDBD Feature Selection (Sutton 1992a Fig 4)",
        xlabel=f"Time Steps (x{log_interval})",
        ylabel="Learning Rate (\u03b1)",
        save_path=save_path,
    )


if __name__ == "__main__":
    torch.manual_seed(42)
    main()
