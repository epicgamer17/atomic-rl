import random
import math
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


# TODO: unify our custom initializers. some operate on layers others on modules. Probably should for the most part be modules. However some initializations only apply to certain layers (output layers). could we do this with a higher order function?
def make_gnt_init(
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


def lecun_uniform_(tensor: torch.Tensor) -> None:
    """
    Applies LeCun uniform initialization to the weight tensor.

    Reference: https://github.com/mohmdelsayed/streaming-drl/blob/main/src/sparse_init.py
        The authors' SparseMLP layers use a LeCun-style init + sparse mask; see their
        `sparse_init.py` (and `layer.py`) for the canonical implementation.
    """
    fan_in = nn.init._calculate_fan_in_and_fan_out(tensor)[0]
    bound = 1.0 / math.sqrt(fan_in)
    nn.init.uniform_(tensor, -bound, bound)


def sparse_init_weight_(tensor: torch.Tensor, sparsity: float) -> None:
    """
    Applies sparsity to a weight matrix based on the Stream RL pseudocode.
    Zeroes out a `sparsity` fraction of incoming connections (fan_in) independently for EACH output neuron.
    From the Stream RL Paper (Section 4). LeCun initialization is recommended.

    Reference: https://github.com/mohmdelsayed/streaming-drl/blob/main/src/sparse_init.py
        The authors' `sparse_init.py` implements the exact SparseInit scheme used in
        the paper (see also `layer.py` for the SparseMLP that applies it).

    Args:
        tensor: The weight tensor to modify in-place (assumes [out_features, in_features]).
        sparsity: The fraction of fan_in to set to 0 per neuron.
    """
    with torch.no_grad():
        if tensor.dim() < 2:
            return

        out_features = tensor.size(0)
        in_features = (
            tensor.size(1)
            if tensor.dim() == 2
            else tensor.view(tensor.size(0), -1).size(1)
        )

        n = int(sparsity * in_features)
        if n == 0:
            return

        view = tensor if tensor.dim() == 2 else tensor.view(out_features, -1)

        # For each output neuron (row), sample n random incoming weight indices (columns) to set to 0.0
        mask_indices = (
            torch.rand(out_features, in_features, device=tensor.device)
            .topk(n, dim=1)
            .indices
        )

        view.scatter_(1, mask_indices, 0.0)


def make_sparse_init(
    init_fn: Callable[[torch.Tensor], None], sparsity: float
) -> Callable[[torch.Tensor], None]:
    """
    Returns a configured initialization function.

    This factory allows you to inject your base initializer (Xavier, Kaiming, etc.)
    and the sparsity level, returning a standard interface that can be
    passed to model.apply().

    From the Stream RL Paper. LeCun initialization is recommended.

    Reference: https://github.com/mohmdelsayed/streaming-drl/blob/main/src/sparse_init.py
        See the authors' `sparse_init.py` for the exact sparse-initialization routine.
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
