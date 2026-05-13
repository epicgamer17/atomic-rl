"""
Example: Online Permuted MNIST
Based on: Dohare et al. (2024), Section 4.2
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from functional.plasticity import (
    init_cbp_state,
    apply_continual_backprop,
)
from functional.utils import gnt_init_wrapper
from envs.streams.permuted_mnist import make_permuted_mnist_stream

# 1. Hyperparameters (Paper Section 4.2)
# Using 2000 units per layer as in the ImageNet/MNIST scaling experiments
HIDDEN_SIZE = 2000
ETA = 0.99
RHO = 1e-4
MATURITY = 1000
LR = 0.001

# 2. Setup Architecture
model = nn.Sequential(
    nn.Linear(784, HIDDEN_SIZE),
    nn.ReLU(),
    nn.Linear(HIDDEN_SIZE, HIDDEN_SIZE),
    nn.ReLU(),
    nn.Linear(HIDDEN_SIZE, 10),
)
optimizer = torch.optim.Adam(model.parameters(), lr=LR)
init_fn = gnt_init_wrapper(nn.init.kaiming_uniform_)

# Track states for all hidden layers
cbp_states = {model[0]: init_cbp_state(model[0]), model[2]: init_cbp_state(model[2])}
layer_pairs = [(model[0], model[2]), (model[2], model[4])]

mnist_stream = make_permuted_mnist_stream(batch_size=32)

# 3. Continual Learning Loop
for task_id, loader in enumerate(mnist_stream):
    print(f"\n--- Starting Task {task_id} (New Permutation) ---")

    for batch_idx, (data, target) in enumerate(loader):
        # Forward pass & Capture activations
        h1 = model[0](data)
        a1 = F.relu(h1)
        h2 = model[2](a1)
        a2 = F.relu(h2)
        logits = model[4](a2)

        loss = F.cross_entropy(logits, target)

        # Optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # CBP Generate-and-Test Step
        apply_continual_backprop(
            layer_pairs=layer_pairs,
            activations=[a1, a2],
            cbp_states=cbp_states,
            optimizer=optimizer,
            init_fn=init_fn,
            eta=ETA,
            maturity_threshold=MATURITY,
            replacement_rate=RHO,
        )

        if batch_idx % 500 == 0:
            acc = (logits.argmax(dim=1) == target).float().mean()
            print(f"Batch {batch_idx} | Loss: {loss.item():.4f} | Acc: {acc:.2%}")

    if task_id >= 10:
        break  # Run for 10 permutations to see stability
