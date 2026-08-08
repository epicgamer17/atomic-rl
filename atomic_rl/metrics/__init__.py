from .metrics import (
    compute_average_gradient_magnitude,
    compute_average_weight_magnitude,
    compute_dead_units_proportion,
    compute_explained_variance,
    compute_stable_rank,
)
from .visualization import (
    create_wandb_lr_plot,
    log_distributional_metrics,
    plot_continual_learning_performance,
    plot_learning_rate_traces,
    plot_plasticity_correlates,
)
