import torch
from typing import Iterator, Tuple


def make_bit_flipping_stream(
    m: int = 20,
    f: int = 15,
    t_flip: int = 10_000,
    target_hidden_size: int = 100,
    beta: float = 0.7,
    seed: int = 42,
) -> Iterator[Tuple[torch.Tensor, torch.Tensor]]:
    """
    Generates an infinite stream for the Bit-Flipping problem (Dohare et al., CBP).

    The input is an m-dimensional binary vector. The first `f` bits are "flipping bits"
    which remain constant except for one randomly chosen bit that flips every `t_flip` steps.
    The remaining `m - f` bits are random at every step.

    The target function is a wider, fixed 2-layer network using Linear Threshold Units (LTUs).

    Args:
        m: Total number of input bits.
        f: Number of flipping bits.
        t_flip: Time-steps between flips of a single flipping bit.
        target_hidden_size: Number of hidden units in the target network (complexity control).
        beta: Threshold hyperparameter for the LTU activation.
        seed: Random seed for reproducibility.

    Yields:
        Tuple of (input_tensor, target_tensor)
        - input_tensor: Shape [m], values in {0, 1}
        - target_tensor: Shape [1], continuous regression target
    """
    # 1. State Initialization
    rng = torch.Generator()
    if seed is not None:
        rng.manual_seed(seed)

    # Initial flipping bits sampled from {0, 1}
    flipping_bits = torch.randint(0, 2, (f,), generator=rng).float()

    # 2. Target Network Initialization
    # Weights uniformly sampled from {-1, 1}. We sample {0, 1} and map to {-1, 1}
    # Note: size is (target_hidden_size, m + 1) to account for the bias term (x_0 = 1)
    v_weights = (
        torch.randint(0, 2, (target_hidden_size, m + 1), generator=rng).float() * 2.0
        - 1.0
    )

    # Calculate LTU Thresholds (theta)
    # S_i is the number of input weights with value -1 for hidden unit i
    s_counts = (v_weights == -1.0).sum(dim=1).float()
    theta = (m + 1) * beta - s_counts

    # Output layer weights (Continuous, normal distribution scaled by fan-in)
    w_weights = torch.randn(target_hidden_size + 1, generator=rng) / (
        target_hidden_size**0.5
    )

    step = 0
    while True:
        # 3. Input Generation & Shifting
        if step > 0 and step % t_flip == 0:
            # Flip exactly one random bit among the first f bits
            flip_idx = torch.randint(0, f, (1,), generator=rng).item()
            flipping_bits[flip_idx] = 1.0 - flipping_bits[flip_idx]

        # The remaining m - f bits are purely random every step
        random_bits = torch.randint(0, 2, (m - f,), generator=rng).float()

        # Construct the raw input x_t (Size: m)
        x_out = torch.cat([flipping_bits, random_bits])

        # 4. Target Network Forward Pass
        # Add bias term (1.0) to match the sum from i=0 to m
        x_t_biased = torch.cat([x_out, torch.ones(1)])

        # Linear projection: sum(v * x)
        pre_activation = torch.mv(v_weights, x_t_biased)

        # LTU Activation: 1 if pre_activation > theta else 0
        h_t = (pre_activation > theta).float()

        # Output projection (with hidden bias = 1.0)
        h_t_biased = torch.cat([h_t, torch.ones(1)])
        y_t = torch.dot(w_weights, h_t_biased)

        yield x_out, y_t.unsqueeze(0)

        step += 1
