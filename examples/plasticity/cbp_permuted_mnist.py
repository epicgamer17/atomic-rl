"""
Example: CBP vs Standard Adam on Online Permuted MNIST
"""

# TODO: improve and make plots and stuff like SWR example
from functional.initialization import gnt_init_wrapper
import torch
import torch.nn as nn
import torch.nn.functional as F
from functional.plasticity import (
    init_cbp_state,
    apply_continual_backprop,
)
from functional.metrics import (
    compute_dead_units_proportion,
    compute_average_weight_magnitude,
    compute_average_gradient_magnitude,
)
from functional.visualization import (
    plot_plasticity_correlates,
    plot_continual_learning_performance,
)
from envs.streams.permuted_mnist import make_permuted_mnist_stream

HIDDEN, RHO, ETA, MATURITY, LR = 2000, 1e-4, 0.99, 1000, 0.01


def make_mnist_net():
    return nn.Sequential(
        nn.Linear(784, HIDDEN),
        nn.ReLU(),
        nn.Linear(HIDDEN, HIDDEN),
        nn.ReLU(),
        nn.Linear(HIDDEN, 10),
    )


model_std = make_mnist_net()
model_cbp = make_mnist_net()

opt_std = torch.optim.Adam(model_std.parameters(), lr=LR)
opt_cbp = torch.optim.Adam(model_cbp.parameters(), lr=LR)
init_fn = gnt_init_wrapper(nn.init.kaiming_uniform_)

cbp_states = {
    model_cbp[0]: init_cbp_state(model_cbp[0]),
    model_cbp[2]: init_cbp_state(model_cbp[2]),
}
layer_pairs = [(model_cbp[0], model_cbp[2]), (model_cbp[2], model_cbp[4])]

stream = make_permuted_mnist_stream(batch_size=64)

metrics_std = {
    "Accuracy": [],
    "Dead Units": [],
    "Weight Mag": [],
    "Grad Mag": [],
}
metrics_cbp = {
    "Accuracy": [],
    "Dead Units": [],
    "Weight Mag": [],
    "Grad Mag": [],
}

for task_id, loader in enumerate(stream):
    total_acc_std, total_acc_cbp, count = 0, 0, 0

    for batch_idx, (data, target) in enumerate(loader):
        # Standard Update
        h1_std = model_std[0](data)
        a1_std = F.relu(h1_std)
        h2_std = model_std[2](a1_std)
        a2_std = F.relu(h2_std)
        out_std = model_std[4](a2_std)

        loss_std = F.cross_entropy(out_std, target)
        opt_std.zero_grad()
        loss_std.backward()
        opt_std.step()

        # CBP Update
        h1 = model_cbp[0](data)
        a1 = F.relu(h1)
        h2 = model_cbp[2](a1)
        a2 = F.relu(h2)
        out_cbp = model_cbp[4](a2)

        loss_cbp = F.cross_entropy(out_cbp, target)
        opt_cbp.zero_grad()
        loss_cbp.backward()
        opt_cbp.step()

        apply_continual_backprop(
            layer_pairs, [a1, a2], cbp_states, opt_cbp, init_fn, ETA, MATURITY, RHO
        )

        total_acc_std += (out_std.argmax(1) == target).float().mean().item()
        total_acc_cbp += (out_cbp.argmax(1) == target).float().mean().item()
        count += 1

    # End of Task Metrics
    metrics_std["Accuracy"].append(total_acc_std / count)
    metrics_std["Dead Units"].append(compute_dead_units_proportion(a1_std))
    metrics_std["Weight Mag"].append(
        compute_average_weight_magnitude(list(model_std.parameters()))
    )
    metrics_std["Grad Mag"].append(
        compute_average_gradient_magnitude(list(model_std.parameters()))
    )

    metrics_cbp["Accuracy"].append(total_acc_cbp / count)
    metrics_cbp["Dead Units"].append(compute_dead_units_proportion(a1))
    metrics_cbp["Weight Mag"].append(
        compute_average_weight_magnitude(list(model_cbp.parameters()))
    )
    metrics_cbp["Grad Mag"].append(
        compute_average_gradient_magnitude(list(model_cbp.parameters()))
    )

    print(
        f"Task {task_id:2d} | Std Acc: {metrics_std['Accuracy'][-1]:.4f} (Dead: {metrics_std['Dead Units'][-1]:.2f}) | "
        f"CBP Acc: {metrics_cbp['Accuracy'][-1]:.4f} (Dead: {metrics_cbp['Dead Units'][-1]:.2f})"
    )
    if task_id >= 20:
        break

# --- Plotting ---
plot_continual_learning_performance(
    results_dict={
        "Standard Adam": metrics_std["Accuracy"],
        "CBP": metrics_cbp["Accuracy"],
    },
    title="Permuted MNIST: Online Accuracy",
    save_path="cbp_mnist_accuracy.png",
)

# Remove Accuracy for correlates plot
acc_std = metrics_std.pop("Accuracy")
acc_cbp = metrics_cbp.pop("Accuracy")

plot_plasticity_correlates(
    metrics_dict={"Standard Adam": metrics_std, "CBP": metrics_cbp},
    save_path="cbp_mnist_correlates.png",
)
