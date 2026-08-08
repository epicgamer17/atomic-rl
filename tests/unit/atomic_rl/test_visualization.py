import os
import pytest
import torch
import numpy as np
import math
import matplotlib.pyplot as plt
from unittest.mock import MagicMock, patch

from atomic_rl.visualization import (
    create_wandb_lr_plot,
    plot_learning_rate_traces,
    plot_continual_learning_performance,
    plot_plasticity_correlates,
    compute_explained_variance,
    log_distributional_metrics,
)

pytestmark = pytest.mark.unit


def test_compute_explained_variance():
    y_true = np.array([1.0, 2.0, 3.0, 4.0, 5.0])

    # 1. Perfect prediction
    y_pred_perfect = y_true.copy()
    assert compute_explained_variance(y_true, y_pred_perfect) == 1.0

    # 2. Constant mean prediction (explained variance = 0)
    y_pred_mean = np.full_like(y_true, np.mean(y_true))
    # var(y_true - mean) = var(y_true). 1 - 1 = 0
    assert math.isclose(
        compute_explained_variance(y_true, y_pred_mean), 0.0, abs_tol=1e-7
    )

    # 3. Bad prediction
    y_pred_bad = y_true * -1
    assert compute_explained_variance(y_true, y_pred_bad) < 0.0

    # 4. Zero variance case
    y_true_zero = np.array([1.0, 1.0, 1.0])
    assert np.isnan(compute_explained_variance(y_true_zero, y_true_zero))


@patch("wandb.Table")
@patch("wandb.plot.line")
def test_log_distributional_metrics(mock_line, mock_table):
    support = torch.tensor([0.0, 1.0, 2.0])
    # Batch size 1, 3 atoms. Logits make atom 1 certain.
    info_dict = {"predictions": torch.tensor([[0.0, 100.0, 0.0]])}
    step = 10

    # 1. Test basic metrics without chart
    metrics = log_distributional_metrics(info_dict, support, step, log_chart=False)

    # Expected Q = sum([0, 1, 0] * [0, 1, 2]) = 1.0
    assert math.isclose(metrics["metrics/expected_q_value"], 1.0, rel_tol=1e-5)
    assert "charts/distribution_curve" not in metrics

    # 2. Test with chart logging
    metrics_with_chart = log_distributional_metrics(
        info_dict, support, step, log_chart=True
    )

    assert mock_table.called
    assert mock_line.called
    assert "charts/distribution_curve" in metrics_with_chart


# ==========================================
# Tests for W&B Telemetry Payloads
# ==========================================


def test_create_wandb_lr_plot():
    """Verify standard mapping layout of the payload returned for learning rates."""
    payload = create_wandb_lr_plot(step=50, relevant_lr=0.25, irrelevant_lr=0.02)

    assert payload == {
        "step": 50,
        "metrics/relevant_lr_mean": 0.25,
        "metrics/irrelevant_lr_mean": 0.02,
    }


# ==========================================
# Tests for Matplotlib Graphic Renderers
# ==========================================


def test_plot_learning_rate_traces(tmp_path):
    """Verify that learning rate trends are processed and exported correctly."""
    # 10 execution steps, 4 individual features
    alphas_history = np.ones((10, 4))
    alphas_history[:, :2] = 0.5  # Relevant partitions
    alphas_history[:, 2:] = 0.05  # Irrelevant partitions

    save_file = tmp_path / "lr_traces.png"

    fig = plot_learning_rate_traces(
        alphas_history=alphas_history, num_relevant=2, save_path=str(save_file)
    )

    assert isinstance(fig, plt.Figure)
    assert os.path.exists(save_file)

    # Close explicitly to avoid background memory accumulation leaks
    plt.close(fig)


def test_plot_continual_learning_performance(tmp_path):
    """Verify parsing and plot structure for cross-task comparison metrics."""
    results_dict = {
        "Base Engine": [0.85, 0.70, 0.55],
        "SWR Enhanced": [0.85, 0.84, 0.86],
    }
    save_file = tmp_path / "continual_learning_perf.png"

    fig = plot_continual_learning_performance(
        results_dict=results_dict, save_path=str(save_file)
    )

    assert isinstance(fig, plt.Figure)
    assert os.path.exists(save_file)

    plt.close(fig)


def test_plot_plasticity_correlates(tmp_path):
    """Verify grid generation side-effects, including feature name layout filtering checks."""
    metrics_dict = {
        "Base System": {
            "Dead Units": [0.4, 0.5, 0.6],
            "Weight Magnitude": [1.2, 1.5, 1.9],
        },
        "SWR": {"Dead Units": [0.1, 0.1, 0.1], "Weight Magnitude": [0.8, 0.8, 0.8]},
    }
    save_file = tmp_path / "plasticity_correlates.png"

    # Runs the plotting code entirely (internal context calls plt.close() at termination)
    plot_plasticity_correlates(metrics_dict=metrics_dict, save_path=str(save_file))

    assert os.path.exists(save_file)
