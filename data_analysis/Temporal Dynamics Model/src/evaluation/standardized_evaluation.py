#!/usr/bin/env python3
"""
Standardized Time Series Model Evaluation Framework
==================================================

Extended with:
- Residual diagnostics (relative RMSE, relative MAE)
- Autocorrelation checks (lag-1 Pearson r, Ljung–Box test p-values)

Ensures consistent evaluation, diagnostics, and reporting across all models.
"""

import os
import json
import numpy as np
import pandas as pd
from typing import Dict, Tuple, Any
from datetime import datetime
import warnings
warnings.filterwarnings("ignore")

# Optional: Ljung–Box
try:
    import statsmodels.api as sm
    _HAS_SM = True
except Exception:
    _HAS_SM = False

from scipy.stats import pearsonr

# JSON serialization helper
def convert_to_json_serializable(obj):
    """Convert numpy types to JSON-serializable Python types."""
    if isinstance(obj, dict):
        return {key: convert_to_json_serializable(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_to_json_serializable(item) for item in obj]
    elif isinstance(obj, (np.integer, np.int64, np.int32)):
        return int(obj)
    elif isinstance(obj, (np.floating, np.float64, np.float32)):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    else:
        return obj


# Standard target columns for SatCom time series analysis
STANDARD_TARGET_COLUMNS = [
    "throughput_mbps",
    "packet_loss_percent",
    "jitter_ms",
    "rtt_mean",
    "sinr_db",
    "elevation_deg",
    "azimuth_deg",
]


class StandardizedMetrics:
    """Standardized metrics + new diagnostics for time series forecasting."""

    @staticmethod
    def _to_np(x):
        return np.asarray(x).reshape(-1)

    @staticmethod
    def calculate_rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        return np.sqrt(np.mean((y_true - y_pred) ** 2))

    @staticmethod
    def calculate_mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        return np.mean(np.abs(y_true - y_pred))

    @staticmethod
    def calculate_mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        mask = y_true != 0
        if np.sum(mask) == 0:
            return 0.0
        return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100

    @staticmethod
    def calculate_r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        ss_res = np.sum((y_true - y_pred) ** 2)
        ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
        if ss_tot == 0:
            return 0.0
        return 1 - (ss_res / ss_tot)

    @staticmethod
    def _persistence_baseline(y):
        y = StandardizedMetrics._to_np(y)
        if y.size == 0:
            return y
        yhat = np.empty_like(y)
        yhat[0] = y[0]
        yhat[1:] = y[:-1]
        return yhat

    @staticmethod
    def residual_pearson_lag1(residuals):
        r = StandardizedMetrics._to_np(residuals)
        if r.size < 2:
            return float("nan"), float("nan")
        r_t, r_tm1 = r[1:], r[:-1]
        m = np.isfinite(r_t) & np.isfinite(r_tm1)
        if m.sum() < 2:
            return float("nan"), float("nan")
        return pearsonr(r_t[m], r_tm1[m])

    @staticmethod
    def ljung_box_pvalue(residuals, lags: int = 10):
        if not _HAS_SM:
            return float("nan")
        r = StandardizedMetrics._to_np(residuals)
        if np.sum(np.isfinite(r)) < lags + 3:
            return float("nan")
        try:
            lb = sm.stats.acorr_ljungbox(r, lags=[lags], return_df=True)
            return float(lb["lb_pvalue"].iloc[0])
        except Exception:
            return float("nan")

    @classmethod
    def calculate_all_metrics(
        cls, y_true_dict: Dict[str, np.ndarray], y_pred_dict: Dict[str, np.ndarray]
    ) -> Dict[str, Dict[str, float]]:
        """Calculate metrics + diagnostics per target."""
        metrics = {}
        for target in STANDARD_TARGET_COLUMNS:
            if target not in y_true_dict or target not in y_pred_dict:
                metrics[target] = {m: np.nan for m in [
                    "rmse","mae","mape","r2","samples",
                    "relative_rmse","relative_mae",
                    "resid_pearson_lag1","resid_pearson_lag1_p",
                    "resid_ljung_p_lag10"]}
                continue

            y_true, y_pred = cls._to_np(y_true_dict[target]), cls._to_np(y_pred_dict[target])
            if len(y_true) == 0 or len(y_true) != len(y_pred):
                metrics[target] = {m: np.nan for m in [
                    "rmse","mae","mape","r2","samples",
                    "relative_rmse","relative_mae",
                    "resid_pearson_lag1","resid_pearson_lag1_p",
                    "resid_ljung_p_lag10"]}
                continue

            mask = np.isfinite(y_true) & np.isfinite(y_pred)
            if mask.sum() == 0:
                metrics[target] = {m: np.nan for m in [
                    "rmse","mae","mape","r2","samples",
                    "relative_rmse","relative_mae",
                    "resid_pearson_lag1","resid_pearson_lag1_p",
                    "resid_ljung_p_lag10"]}
                continue

            y_true, y_pred = y_true[mask], y_pred[mask]

            rmse = cls.calculate_rmse(y_true, y_pred)
            mae = cls.calculate_mae(y_true, y_pred)
            mape = cls.calculate_mape(y_true, y_pred)
            r2 = cls.calculate_r2(y_true, y_pred)

            # relative errors vs persistence
            naive_pred = cls._persistence_baseline(y_true)
            naive_rmse = cls.calculate_rmse(y_true, naive_pred)
            naive_mae = cls.calculate_mae(y_true, naive_pred)
            rel_rmse = rmse / (naive_rmse + 1e-12) if np.isfinite(naive_rmse) else np.nan
            rel_mae = mae / (naive_mae + 1e-12) if np.isfinite(naive_mae) else np.nan

            residuals = y_true - y_pred
            r1, r1p = cls.residual_pearson_lag1(residuals)
            lbp = cls.ljung_box_pvalue(residuals, lags=10)

            metrics[target] = {
                "rmse": rmse,
                "mae": mae,
                "mape": mape,
                "r2": r2,
                "samples": len(y_true),
                "relative_rmse": rel_rmse,
                "relative_mae": rel_mae,
                "resid_pearson_lag1": r1,
                "resid_pearson_lag1_p": r1p,
                "resid_ljung_p_lag10": lbp,
            }
        return metrics


class StandardizedModelEvaluator:
    """Evaluator that now also attaches residual diagnostics."""

    def __init__(self, split_method="sample_based",
                 data_dir="processed_data", results_dir="comprehensive_results"):
        self.split_method = split_method
        self.data_dir = data_dir
        self.results_dir = results_dir
        os.makedirs(results_dir, exist_ok=True)
        self.train_data, self.test_data = self._load_train_test_data()

    def _load_train_test_data(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        data_path = (f"data/{self.data_dir}" if not self.data_dir.startswith("/") else self.data_dir)
        train_file = f"{data_path}/combined_DL_train.csv"
        test_file = f"{data_path}/combined_DL_test.csv"
        if os.path.exists(train_file) and os.path.exists(test_file):
            return pd.read_csv(train_file), pd.read_csv(test_file)
        raise FileNotFoundError(f"Missing pre-split data in {data_path}")

    def get_train_test_data(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        return self.train_data.copy(), self.test_data.copy()

    def evaluate_model_predictions(
        self, model_name: str, y_true_dict: Dict[str, np.ndarray], y_pred_dict: Dict[str, np.ndarray],
        training_metadata: Dict[str, Any] = None,
    ) -> Dict[str, Any]:
        metrics = StandardizedMetrics.calculate_all_metrics(y_true_dict, y_pred_dict)

        # summary over valid targets
        valid = {t:m for t,m in metrics.items() if not np.isnan(m["r2"])}
        if valid:
            summary = {
                "avg_r2": np.mean([m["r2"] for m in valid.values()]),
                "avg_rmse": np.mean([m["rmse"] for m in valid.values()]),
                "avg_mae": np.mean([m["mae"] for m in valid.values()]),
                "avg_mape": np.mean([m["mape"] for m in valid.values()]),
                "avg_relative_rmse": np.mean([m["relative_rmse"] for m in valid.values()]),
                "avg_relative_mae": np.mean([m["relative_mae"] for m in valid.values()]),
                "avg_resid_pearson_lag1": np.mean([m["resid_pearson_lag1"] for m in valid.values()]),
                "avg_resid_ljung_p_lag10": np.mean([m["resid_ljung_p_lag10"] for m in valid.values()]),
                "total_samples": sum([m["samples"] for m in valid.values()]),
                "valid_targets": len(valid),
            }
        else:
            summary = {k: 0.0 for k in [
                "avg_r2","avg_rmse","avg_mae","avg_mape",
                "avg_relative_rmse","avg_relative_mae",
                "avg_resid_pearson_lag1","avg_resid_ljung_p_lag10",
                "total_samples","valid_targets"]}

        results = {
            "model_name": model_name,
            "evaluation_timestamp": datetime.now().isoformat(),
            "metrics": metrics,
            "summary": summary,
            "training_metadata": training_metadata or {},
            "evaluation_info": {
                "split_method": self.split_method,
                "target_columns": STANDARD_TARGET_COLUMNS,
                "train_samples": len(self.train_data),
                "test_samples": len(self.test_data),
                "data_dir": self.data_dir,
            },
        }

        # save json
        results_file = f"{self.results_dir}/{model_name.lower()}_evaluation_results.json"
        with open(results_file, "w") as f:
            json.dump(convert_to_json_serializable(results), f, indent=2)

        print(f"\n=== {model_name.upper()} EVALUATION COMPLETE ===")
        print(f"Performance Summary:")
        print(f"  - Average R²:              {summary['avg_r2']:.4f}")
        print(f"  - Average RMSE:            {summary['avg_rmse']:.4f}")
        print(f"  - Average MAE:             {summary['avg_mae']:.4f}")
        print(f"  - Average MAPE:            {summary['avg_mape']:.2f}%")
        print(f"  - Avg Relative RMSE:       {summary['avg_relative_rmse']:.3f}")
        print(f"  - Avg Relative MAE:        {summary['avg_relative_mae']:.3f}")
        print(f"  - Avg Resid. Pearson lag1: {summary['avg_resid_pearson_lag1']:.3f}")
        print(f"  - Avg Ljung–Box p (lag10): {summary['avg_resid_ljung_p_lag10']:.3f}")
        print(f"  - Valid targets:           {summary['valid_targets']}/{len(STANDARD_TARGET_COLUMNS)}")
        print(f"  - Total samples:           {summary['total_samples']:,}")
        print(f"  - Results saved:           {results_file}")
        print("=" * 60)
        return results