import torch
import wandb
import numpy as np


def compute_explained_variance(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Computes the explained variance of a regression problem.
    Formula: 1 - Var(y_true - y_pred) / Var(y_true)

    Args:
        y_true (np.ndarray): The ground truth returns.
        y_pred (np.ndarray): The predicted values.

    Returns:
        float: The explained variance. 1.0 is perfect prediction, 0.0 or less means it is bad.
    """
    var_y = np.var(y_true)
    if var_y == 0:
        return np.nan
    return 1 - np.var(y_true - y_pred) / var_y


def log_distributional_metrics(
    info_dict: dict, support: torch.Tensor, step: int, log_chart: bool = False
) -> dict:
    """
    Generate readable line charts and log expected Q-values for distributional RL.

    Args:
        info_dict (dict): The info dictionary from compute_q_td_loss.
        support (torch.Tensor): The atom support tensor.
        step (int): The current training step.
        log_chart (bool): Whether to generate the heavy W&B line chart.

    Returns:
        dict: A dictionary of W&B plots and metrics.
    """
    metrics = {}
    if "predictions" in info_dict:
        # 1. Calculate probabilities [Batch, Atoms]
        # predictions are logits [Batch, Atoms]
        probs = torch.softmax(info_dict["predictions"], dim=-1)

        # 2. Calculate the Expected Q-value (Mean of the distribution)
        # E[Q] = sum(prob * support)
        expected_q_batch = (probs * support.to(probs.device)).sum(dim=-1)
        metrics["metrics/expected_q_value"] = expected_q_batch.mean().detach()

        if log_chart:
            # 3. Average probabilities over the batch for the distribution curve
            mean_probs = probs.mean(dim=0).detach().cpu().numpy()
            support_np = support.cpu().numpy()

            # 4. Create a Line Plot instead of a Bar Chart for readability
            data = [[s, p] for s, p in zip(support_np, mean_probs)]
            table = wandb.Table(data=data, columns=["Support", "Probability"])
            metrics["charts/distribution_curve"] = wandb.plot.line(
                table,
                "Support",
                "Probability",
                title=f"Atom Distribution (Step {step})",
            )

    return metrics
