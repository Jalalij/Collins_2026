#!/usr/bin/env python3
"""
Standardized Data Preparation for All Models

This module ensures all models use exactly the same data preprocessing
to enable fair comparison.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from typing import Tuple, Dict, Any
from ..evaluation.standardized_evaluation import STANDARD_TARGET_COLUMNS


def prepare_standardized_data(
    train_data: pd.DataFrame,
    test_data: pd.DataFrame,
    train_samples: int = 50000,
    test_samples: int = 10000,
) -> Dict[str, Any]:
    """
    Prepare standardized data for all models with consistent preprocessing.

    Returns:
        Dictionary containing:
        - train_raw: Raw training data
        - test_raw: Raw testing data
        - train_scaled: MinMax scaled training data
        - test_scaled: MinMax scaled testing data
        - scaler: Fitted scaler object
        - metadata: Information about the data
    """

    # Sort by timestamp to ensure chronological order
    if "timestamp" in train_data.columns:
        train_sorted = train_data.sort_values("timestamp")
        test_sorted = test_data.sort_values("timestamp")
    else:
        train_sorted = train_data
        test_sorted = test_data

    # Take chronological subsets
    train_subset = train_sorted.head(train_samples)
    test_subset = test_sorted.head(test_samples)

    # Standardized preprocessing: forward fill then zero fill
    # Get available target columns only
    available_targets = [
        col for col in STANDARD_TARGET_COLUMNS if col in train_subset.columns
    ]

    if not available_targets:
        raise ValueError("No standard target columns found in data")

    train_clean = train_subset[available_targets].ffill().fillna(0)
    test_clean = test_subset[available_targets].ffill().fillna(0)

    # Convert to numpy arrays
    train_raw = train_clean.values
    test_raw = test_clean.values

    # Collect metadata
    metadata = {
        "train_samples": len(train_raw),
        "test_samples": len(test_raw),
        "features": STANDARD_TARGET_COLUMNS,
        "train_time_range": (
            (train_subset["timestamp"].min(), train_subset["timestamp"].max())
            if "timestamp" in train_subset.columns
            else None
        ),
        "test_time_range": (
            (test_subset["timestamp"].min(), test_subset["timestamp"].max())
            if "timestamp" in test_subset.columns
            else None
        ),
        "preprocessing": "ffill_then_zero_fill + MinMaxScaler",
        "scaling_range": "[0, 1]",
    }

    return {
        "train_raw": train_raw,
        "test_raw": test_raw,
        "metadata": metadata,
    }


def create_sequences(
    data: np.ndarray, sequence_length: int = 10
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Create sequences for time series models (LSTM, Transformer).

    Args:
        data: 2D array (samples, features)
        sequence_length: Length of input sequences

    Returns:
        X: Input sequences (samples, sequence_length, features)
        y: Target values (samples, features)
    """
    X, y = [], []
    for i in range(len(data) - sequence_length):
        X.append(data[i : i + sequence_length])
        y.append(data[i + sequence_length])
    return np.array(X), np.array(y)


def evaluate_predictions_standardized(
    y_true: np.ndarray, y_pred: np.ndarray, target_names: list = None
) -> Dict[str, float]:
    """
    Standardized evaluation metrics for all models.
    """
    if target_names is None:
        target_names = STANDARD_TARGET_COLUMNS

    # Calculate metrics for each target
    results = {}
    for i, target in enumerate(target_names):
        if i < y_true.shape[1] and i < y_pred.shape[1]:
            y_t = y_true[:, i]
            y_p = y_pred[:, i]

            # Remove any NaN/Inf values
            mask = np.isfinite(y_t) & np.isfinite(y_p)
            if np.sum(mask) > 0:
                y_t_clean = y_t[mask]
                y_p_clean = y_p[mask]

                # Calculate metrics
                mse = np.mean((y_t_clean - y_p_clean) ** 2)
                rmse = np.sqrt(mse)
                mae = np.mean(np.abs(y_t_clean - y_p_clean))

                # MAPE (avoid division by zero)
                mape_mask = y_t_clean != 0
                if np.sum(mape_mask) > 0:
                    mape = (
                        np.mean(
                            np.abs(
                                (y_t_clean[mape_mask] - y_p_clean[mape_mask])
                                / y_t_clean[mape_mask]
                            )
                        )
                        * 100
                    )
                else:
                    mape = 0

                # R²
                ss_res = np.sum((y_t_clean - y_p_clean) ** 2)
                ss_tot = np.sum((y_t_clean - np.mean(y_t_clean)) ** 2)
                r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0

                results[target] = {
                    "rmse": rmse,
                    "mae": mae,
                    "mape": mape,
                    "r2": r2,
                    "samples": len(y_t_clean),
                }
            else:
                results[target] = {
                    "rmse": np.nan,
                    "mae": np.nan,
                    "mape": np.nan,
                    "r2": np.nan,
                    "samples": 0,
                }

    # Calculate averages
    valid_results = {k: v for k, v in results.items() if not np.isnan(v["r2"])}
    if valid_results:
        avg_metrics = {
            "avg_rmse": np.mean([v["rmse"] for v in valid_results.values()]),
            "avg_mae": np.mean([v["mae"] for v in valid_results.values()]),
            "avg_mape": np.mean([v["mape"] for v in valid_results.values()]),
            "avg_r2": np.mean([v["r2"] for v in valid_results.values()]),
            "total_samples": sum([v["samples"] for v in valid_results.values()]),
            "valid_targets": len(valid_results),
        }
    else:
        avg_metrics = {
            "avg_rmse": 0,
            "avg_mae": 0,
            "avg_mape": 0,
            "avg_r2": 0,
            "total_samples": 0,
            "valid_targets": 0,
        }

    results["summary"] = avg_metrics
    return results
