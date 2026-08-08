"""
ETTm2 streaming prediction environment (Zhou et al. 2021).

Yields (obs, target_cumulant) pairs where:
  - obs is a 7-D tensor (6 load features + OT at t-1) with EMA memory traces
  - target_cumulant is the oil temperature at the current timestep

EMA memory traces: S_t = beta * S_{t-1} + (1 - beta) * O_t   (beta=0.999)
GVF with gamma ~ 0.99 gives a ~100-step / 25-hour prediction horizon.

TODO (reference vs. paper):
  - The reference ETT environment (github.com/mohmdelsayed/streaming-drl, stream_td.py)
    min-max normalizes the cumulant to [0, 1] before applying ScaleReward. We follow
    the paper and yield the raw cumulant. Optionally match the reference for parity.
  - The reference ObservationTraces wrapper applies a bias-corrected EMA
    (mean / (1 - beta^count)) when yielding the trace. The paper (Section 4.5) defines
    the plain EMA trace S_t = beta * S_{t-1} + (1 - beta) * O_t, which we follow here.

Data source: https://github.com/zhouhaoyi/ETDataset
"""

import os
import urllib.request
import zipfile
from typing import Iterator, Tuple

import numpy as np
import torch

ETTM2_URL = "https://github.com/zhouhaoyi/ETDataset/raw/main/ETT-small/ETTm2.csv"


def _default_data_dir() -> str:
    """Return a user cache dir for downloaded data (never inside the package)."""
    base = os.environ.get("XDG_CACHE_HOME") or os.path.join(
        os.path.expanduser("~"), ".cache"
    )
    return os.path.join(base, "atomic-rl", "env_data")


DATA_DIR = _default_data_dir()
LOCAL_PATH = os.path.join(DATA_DIR, "ETTm2.csv")

NUM_FEATURES = 7  # HUFL, HULL, MUFL, MULL, LUFL, LULL, OT
BETA = 0.999  # memory trace decay


def _ensure_data_downloaded() -> str:
    """Download ETTm2.csv if it does not exist locally.  Returns the local path."""
    os.makedirs(DATA_DIR, exist_ok=True)
    if not os.path.isfile(LOCAL_PATH):
        print(f"Downloading ETTm2.csv from {ETTM2_URL} ...")
        urllib.request.urlretrieve(ETTM2_URL, LOCAL_PATH)
        print("Done.")
    return LOCAL_PATH


def load_ettm2_array(path: str) -> np.ndarray:
    """
    Load ETTm2 CSV and return a float64 NumPy array of shape [T, 7].

    Columns: HUFL, HULL, MUFL, MULL, LUFL, LULL, OT.
    The first column (date) is discarded.
    """
    raw = np.loadtxt(path, delimiter=",", dtype=str, skiprows=1)
    data = raw[:, 1:].astype(np.float64)  # [T, 7]
    return data


def make_ettm2_stream(
    *,
    gamma: float = 0.99,
    beta: float = BETA,
    start: int = 0,
    stop: int | None = None,
    device: torch.device = torch.device("cpu"),
) -> Iterator[Tuple[torch.Tensor, torch.Tensor]]:
    """
    Yield (obs, cumulant) pairs from the ETTm2 dataset.

    Each observation is a 7-D tensor of EMA memory traces.  The cumulant is the
    raw OT (oil temperature) at the current step, which the agent should learn
    to predict via a GVF with discount ``gamma``.

    Parameters
    ----------
    gamma:
        Discount factor (determines prediction horizon).  Not used in the
        yielded values, but documented here since the caller needs it.
    beta:
        Memory trace decay factor.
    start:
        Index into the time series to begin yielding from.
    stop:
        Index to stop before (exclusive).  Defaults to the full series length.
    device:
        Device for the output tensors.
    """
    path = _ensure_data_downloaded()
    data = load_ettm2_array(path)  # [T, 7]

    if stop is None:
        stop = data.shape[0]

    # EMA memory traces (initialised to zero = first observation is just (1-beta)*O)
    traces = np.zeros(NUM_FEATURES, dtype=np.float64)

    for t in range(start, stop):
        obs_t = data[t]  # [7]
        # Update EMA: S_t = beta * S_{t-1} + (1 - beta) * O_t
        traces = beta * traces + (1.0 - beta) * obs_t

        # Cumulant = current oil temperature (column 6)
        cumulant = obs_t[6]

        yield (
            torch.as_tensor(traces, dtype=torch.float32, device=device),
            torch.as_tensor(cumulant, dtype=torch.float32, device=device),
        )
