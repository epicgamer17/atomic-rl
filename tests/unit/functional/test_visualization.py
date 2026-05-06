import pytest
import torch
import numpy as np
import math
from unittest.mock import MagicMock, patch
from functional.visualization import compute_explained_variance, log_distributional_metrics

pytestmark = pytest.mark.unit


def test_compute_explained_variance():
    y_true = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    
    # 1. Perfect prediction
    y_pred_perfect = y_true.copy()
    assert compute_explained_variance(y_true, y_pred_perfect) == 1.0
    
    # 2. Constant mean prediction (explained variance = 0)
    y_pred_mean = np.full_like(y_true, np.mean(y_true))
    # var(y_true - mean) = var(y_true). 1 - 1 = 0
    assert math.isclose(compute_explained_variance(y_true, y_pred_mean), 0.0, abs_tol=1e-7)
    
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
    info_dict = {
        "predictions": torch.tensor([[0.0, 100.0, 0.0]])
    }
    step = 10
    
    # 1. Test basic metrics without chart
    metrics = log_distributional_metrics(info_dict, support, step, log_chart=False)
    
    # Expected Q = sum([0, 1, 0] * [0, 1, 2]) = 1.0
    assert math.isclose(metrics["metrics/expected_q_value"], 1.0, rel_tol=1e-5)
    assert "charts/distribution_curve" not in metrics
    
    # 2. Test with chart logging
    metrics_with_chart = log_distributional_metrics(info_dict, support, step, log_chart=True)
    
    assert mock_table.called
    assert mock_line.called
    assert "charts/distribution_curve" in metrics_with_chart
