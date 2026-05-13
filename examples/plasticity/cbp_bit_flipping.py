"""
Example: Slowly-Changing Regression (Bit-Flipping)
Based on: Dohare et al. (2024), Section 4.1
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from functional.plasticity import (
    init_cbp_state,
    apply_continual_backprop,
)
from functional.utils import ema_update, gnt_init_wrapper
from envs.streams.bit_flipping import make_bit_flipping_stream

# 1. Hyperparameters (Paper Section 4.1)
M, F_BITS, T_FLIP = 20, 15, 10000
ETA = 0.99
RHO = 1e-4
MATURITY = 100
STEP_SIZE = 0.01

# 2. Network & State
# The learner is a small bottleneck network (5 units) tracking a 100-unit target
model = nn.Sequential(nn.Linear(M, 5), nn.ReLU(), nn.Linear(5, 1))
optimizer = torch.optim.Adam(model.parameters(), lr=STEP_SIZE)
init_fn = gnt_init_wrapper(nn.init.kaiming_uniform_)

cbp_states = {model[0]: init_cbp_state(model[0])}
stream = make_bit_flipping_stream(m=M, f=F_BITS, t_flip=T_FLIP)

# 3. Training Loop
for step, (x, y) in enumerate(stream):
    # Forward
    h1 = model[0](x)
    a1 = F.relu(h1)
    pred = model[2](a1)
    loss = F.mse_loss(pred, y)

    # Backward
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # CBP Orchestration
    apply_continual_backprop(
        layer_pairs=[(model[0], model[2])],
        activations=[a1.unsqueeze(0)],  # Stream yields [M], CBP expects [B, M]
        cbp_states=cbp_states,
        optimizer=optimizer,
        init_fn=init_fn,
        eta=ETA,
        replacement_rate=RHO,
        maturity_threshold=MATURITY,
    )

    if step % 1000 == 0:
        print(f"Step {step} | Loss: {loss.item():.4f}")
