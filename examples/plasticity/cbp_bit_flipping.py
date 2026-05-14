"""
Example: CBP vs Adam on Slowly-Changing Regression (Bit-Flipping Problem)

The learning network has a single hidden layer with 5 units, while the target network
is more complex with 100 hidden units. Because the input distribution changes over
time (bit flips) and the target is more complex than the learner, the best
approximation continually changes, requiring the learner to track it.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from functional.plasticity import (
    init_cbp_state,
    apply_continual_backprop,
    compute_dead_units_proportion,
    compute_average_weight_magnitude,
    compute_average_gradient_magnitude,
)
from functional.visualization import (
    plot_plasticity_correlates,
    plot_continual_learning_performance,
)
from functional.utils import gnt_init_wrapper
from envs.streams.bit_flipping import make_bit_flipping_stream

# TODO: improve and make plots and stuff like SWR example
# TODO: MSE plots seem slightly different than the paper.
# Hyperparameters
M, F_BITS, T_FLIP = 20, 15, 10000
RHO, ETA, MATURITY = 1e-4, 0.99, 100
LR = 0.01


# Setup two identical architectures
def make_net():
    # TODO: is this one hidden layer or two hidden layers? should be one hidden layer of size 5.
    return nn.Sequential(
        nn.Linear(M, 5),
        nn.ReLU(),
        nn.Linear(5, 5),
        nn.ReLU(),
        nn.Linear(5, 1),
    )


model_adam = make_net()
model_cbp = make_net()

opt_adam = torch.optim.Adam(model_adam.parameters(), lr=LR)
opt_cbp = torch.optim.Adam(model_cbp.parameters(), lr=LR)

init_fn = gnt_init_wrapper(nn.init.kaiming_uniform_)
cbp_states = {model_cbp[0]: init_cbp_state(model_cbp[0])}
stream = make_bit_flipping_stream(m=M, f=F_BITS, t_flip=T_FLIP, target_hidden_size=100)

# Linear Tracker Ensemble (vectorized for step-size sweep)
LRS_LINEAR = torch.tensor([0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1.0])
linear_weights = torch.zeros((len(LRS_LINEAR), M))
linear_biases = torch.zeros(len(LRS_LINEAR))
linear_block_losses = torch.zeros(len(LRS_LINEAR))
linear_history = [[] for _ in range(len(LRS_LINEAR))]

metrics_adam = {
    "Loss": [],
    "Dead Units": [],
    "Weight Mag": [],
    "Grad Mag": [],
}
metrics_cbp = {
    "Loss": [],
    "Dead Units": [],
    "Weight Mag": [],
    "Grad Mag": [],
}

current_loss_adam = 0
current_loss_cbp = 0
count = 0
# Track which units fire at least once in a block (hidden layer has 5 units)
has_fired_adam = torch.zeros(5, dtype=torch.bool)
has_fired_cbp = torch.zeros(5, dtype=torch.bool)

print("Step | Adam Loss | CBP Loss | Improvement")
for step, (x, y) in enumerate(stream):
    # --- Standard Adam Step ---
    h1_adam = model_adam[0](x)
    a1_adam = F.relu(h1_adam)
    has_fired_adam |= (a1_adam > 0).squeeze()
    pred_adam = model_adam[2](a1_adam)
    loss_adam = F.mse_loss(pred_adam, y)
    opt_adam.zero_grad()
    loss_adam.backward()
    opt_adam.step()

    # --- CBP Step ---
    h1_cbp = model_cbp[0](x)
    a1_cbp = F.relu(h1_cbp)
    has_fired_cbp |= (a1_cbp > 0).squeeze()
    pred_cbp = model_cbp[2](a1_cbp)
    loss_cbp = F.mse_loss(pred_cbp, y)

    opt_cbp.zero_grad()
    loss_cbp.backward()
    opt_cbp.step()

    apply_continual_backprop(
        layer_pairs=[(model_cbp[0], model_cbp[2])],
        activations=[a1_cbp.unsqueeze(0)],
        cbp_states=cbp_states,
        optimizer=opt_cbp,
        init_fn=init_fn,
        eta=ETA,
        replacement_rate=RHO,
        maturity_threshold=MATURITY,
    )

    # --- Linear Tracker Step (Vectorized) ---
    with torch.no_grad():
        # pred: [num_lrs]
        l_preds = F.linear(x, linear_weights, linear_biases)
        l_errors = l_preds - y.squeeze()
        l_losses = l_errors**2
        linear_block_losses += l_losses

        # Manual SGD update: grad = 2 * error * x
        # Use 1.0 * error * x as standard in some RL contexts, but 2.0 is true MSE grad.
        # Most papers use 2.0. We'll use 2.0.
        grad_w = 2.0 * l_errors.unsqueeze(1) * x.unsqueeze(0)
        grad_b = 2.0 * l_errors

        linear_weights -= LRS_LINEAR.unsqueeze(1) * grad_w
        linear_biases -= LRS_LINEAR * grad_b

    current_loss_adam += loss_adam.item()
    current_loss_cbp += loss_cbp.item()
    count += 1

    if (step + 1) % T_FLIP == 0:
        # End of "Task" (one set of bit flips)
        metrics_adam["Loss"].append(current_loss_adam / count)
        metrics_adam["Dead Units"].append((~has_fired_adam).float().mean().item())
        metrics_adam["Weight Mag"].append(
            compute_average_weight_magnitude(list(model_adam.parameters()))
        )
        metrics_adam["Grad Mag"].append(
            compute_average_gradient_magnitude(list(model_adam.parameters()))
        )

        metrics_cbp["Loss"].append(current_loss_cbp / count)
        metrics_cbp["Dead Units"].append((~has_fired_cbp).float().mean().item())
        metrics_cbp["Weight Mag"].append(
            compute_average_weight_magnitude(list(model_cbp.parameters()))
        )
        metrics_cbp["Grad Mag"].append(
            compute_average_gradient_magnitude(list(model_cbp.parameters()))
        )

        # Record Linear losses
        for i in range(len(LRS_LINEAR)):
            linear_history[i].append(linear_block_losses[i].item() / count)
        linear_block_losses.zero_()

        print(
            f"{step+1:5d} | {metrics_adam['Loss'][-1]:.4f}    | {metrics_cbp['Loss'][-1]:.4f}   | "
            f"Dead Std: {metrics_adam['Dead Units'][-1]:.2f}, Dead CBP: {metrics_cbp['Dead Units'][-1]:.2f}"
        )

        # Reset for next block
        current_loss_adam = 0
        current_loss_cbp = 0
        count = 0
        has_fired_adam.fill_(False)
        has_fired_cbp.fill_(False)

    if step >= 250_000:
        break

# Find the best linear tracker (least total error)
total_linear_losses = [sum(h) for h in linear_history]
best_idx = total_linear_losses.index(min(total_linear_losses))
best_linear_lr = LRS_LINEAR[best_idx].item()
best_linear_losses = linear_history[best_idx]

print(f"Best Linear Tracker LR: {best_linear_lr}")

# --- Plotting ---
plot_continual_learning_performance(
    results_dict={
        "Standard Adam": metrics_adam["Loss"],
        "CBP": metrics_cbp["Loss"],
        f"Linear Tracker (lr={best_linear_lr})": best_linear_losses,
    },
    title="Bit Flipping: MSE Loss",
    ylabel="MSE Loss",
    save_path="cbp_bit_flipping_loss.png",
)

# Remove Loss for correlates plot
metrics_adam.pop("Loss")
metrics_cbp.pop("Loss")

plot_plasticity_correlates(
    metrics_dict={"Standard Adam": metrics_adam, "CBP": metrics_cbp},
    save_path="cbp_bit_flipping_correlates.png",
)
