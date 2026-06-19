import random
from typing import Callable, List

import numpy as np
import torch
import torch.nn as nn
from tensordict import TensorDict


def set_seed(seed: int) -> None:
    """
    Sets random seeds for reproducibility across random, numpy, and torch.

    Args:
        seed: The integer seed to use.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        # Ensure deterministic behavior for some ops if needed,
        # though this can slow down performance.
        # torch.backends.cudnn.deterministic = True
        # torch.backends.cudnn.benchmark = False

    if torch.backends.mps.is_available():
        torch.mps.manual_seed(seed)


def _allocate_tensordict(
    shapes: dict, batch_size: List[int], device: str = "cpu"
) -> TensorDict:
    """Allocates a zeroed TensorDict of any arbitrary geometry."""
    data = TensorDict({}, batch_size=batch_size, device=device)
    for key, shape in shapes.items():
        dtype = torch.long if "action" in key else torch.float32
        data.set(key, torch.zeros((*batch_size, *shape), dtype=dtype, device=device))
    return data


# TODO: should i rename it to make_gnt_init?
def gnt_init_wrapper(
    init_fn: Callable[[torch.Tensor], None],
) -> Callable[[torch.Tensor], None]:
    """
    Standardizes initialization for Generate and Test methods (CBP/SWR).

    Wraps/Returns a new init fn to be used for GnT. If it's a weight matrix (2D+),
    it uses the provided init_fn. If it's a bias vector (1D), it uses zeros.

    Args:
        init_fn: The base initialization function to use for weight matrices.

    Returns:
        Callable[[torch.Tensor], None]: A wrapped initialization function that
            dispatches to init_fn for weights and zeros_ for biases.
    """

    def wrapped_init(tensor: torch.Tensor) -> None:
        if tensor.dim() >= 2:
            init_fn(tensor)
        elif tensor.dim() == 1:
            nn.init.zeros_(tensor)

    return wrapped_init


def layer_init(
    layer: nn.Module, std: float = np.sqrt(2), bias_const: float = 0.0
) -> nn.Module:
    """
    Orthogonal initialization of weights and constant initialization of biases.
    Standard for PPO and other policy gradient methods in the CleanRL style.

    Args:
        layer (nn.Module): The layer to initialize.
        std (float): The scaling factor (gain) for orthogonal initialization.
        bias_const (float): The constant value to initialize the bias with.

    Returns:
        nn.Module: The initialized layer.
    """
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


def sparse_init_weight_(tensor: torch.Tensor, sparsity: float) -> None:
    """
    Applies sparsity to a weight matrix based on the Stream RL pseudocode.
    Zeroes out a `sparsity` fraction of the input connections (fan_in) for ALL output neurons.

    Args:
        tensor: The weight tensor to modify in-place (assumes [out_features, in_features]).
        sparsity: The fraction of fan_in to set to 0.
    """
    with torch.no_grad():
        if tensor.dim() < 2:
            fan_in = tensor.size(0)
        else:
            fan_in = torch.nn.init._calculate_fan_in_and_fan_out(tensor)[0]

        n = int(sparsity * fan_in)

        if n == 0:
            return

        # Permutation set P of size fan in
        P = torch.randperm(fan_in, device=tensor.device)

        # Index set I of size n (subset of P)
        I = P[:n]

        # Wi,j <- 0, \forall i \in I, \forall j
        if tensor.dim() == 2:
            tensor[:, I] = 0.0
        elif tensor.dim() > 2:
            view = tensor.view(tensor.size(0), -1)
            view[:, I] = 0.0


def make_sparse_init(
    init_fn: Callable[[torch.Tensor], None], sparsity: float
) -> Callable[[torch.Tensor], None]:
    """
    Returns a configured initialization function.

    This factory allows you to inject your base initializer (Xavier, Kaiming, etc.)
    and the sparsity level, returning a standard interface that can be
    passed to model.apply().
    """

    def initialized_sparse_tensor(tensor: torch.Tensor) -> None:
        if tensor.dim() >= 2:
            # 1. Apply base initialization
            init_fn(tensor)
            # 2. Apply sparsity (the specific SparseInit logic)
            sparse_init_weight_(tensor, sparsity)
        elif tensor.dim() == 1:
            # 3. Handle bias explicitly
            # TODO: should bias also be "sparse init"?
            nn.init.zeros_(tensor)

    return initialized_sparse_tensor
