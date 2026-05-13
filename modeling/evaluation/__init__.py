"""evaluation/ - Unified metric and evaluation package."""

from .metrics import (
    compute_horizon_metrics,
    compute_aggregate_diagnostics,
    last_value_baseline,
    mean_window_baseline,
    mae, rmse, mape, smape, r2,
    prediction_interval,
    relative_rmse, relative_mae,
    residual_pearson_lag1, ljung_box_pvalue,
)
from .evaluate import evaluate_predictions

__all__ = [
    "compute_horizon_metrics", "compute_aggregate_diagnostics",
    "last_value_baseline", "mean_window_baseline",
    "evaluate_predictions",
    "mae", "rmse", "mape", "smape", "r2",
    "prediction_interval", "relative_rmse", "relative_mae",
    "residual_pearson_lag1", "ljung_box_pvalue",
]
