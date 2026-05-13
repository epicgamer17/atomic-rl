import torch
import wandb
import numpy as np
import matplotlib.pyplot as plt
from typing import Optional


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


# TODO: refine so we dont have a mix of matplot stuff AND wandb stuff. how can we stop this
def plot_learning_rate_traces(
    alphas_history: np.ndarray,
    num_relevant: int,
    title: str = "Learning Rates Over Time",
    xlabel: str = "Time Steps",
    ylabel: str = "Learning Rate (alpha)",
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Plots the average learning rates for relevant vs. irrelevant features over time.
    Excellent for visualizing IDBD's automatic feature selection.

    Args:
        alphas_history (np.ndarray): Array of shape [Steps, Features] containing learning rates.
        num_relevant (int): The first N features are considered relevant.
        title (str): Chart title.
        save_path (str, optional): If provided, saves the figure to this path.

    Returns:
        plt.Figure: The generated matplotlib figure.
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    steps = np.arange(alphas_history.shape[0])

    # Calculate means
    relevant_alphas = alphas_history[:, :num_relevant].mean(axis=1)
    irrelevant_alphas = alphas_history[:, num_relevant:].mean(axis=1)

    ax.plot(
        steps, relevant_alphas, label="Relevant Features", color="black", linewidth=1.5
    )
    ax.plot(
        steps,
        irrelevant_alphas,
        label="Irrelevant Features",
        color="gray",
        linewidth=1.5,
    )

    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.3)
    ax.legend()

    if save_path:
        plt.savefig(save_path, bbox_inches="tight")
        print(f"Plot saved to {save_path}")

    return fig


def create_wandb_lr_plot(step: int, relevant_lr: float, irrelevant_lr: float) -> dict:
    """
    Creates a simple payload for wandb logging of learning rates.
    """
    return {
        "step": step,
        "metrics/relevant_lr_mean": relevant_lr,
        "metrics/irrelevant_lr_mean": irrelevant_lr,
    }
