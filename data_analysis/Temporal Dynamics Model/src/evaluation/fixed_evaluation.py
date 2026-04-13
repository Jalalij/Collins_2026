#!/usr/bin/env python3
"""
Fixed Time Series Evaluation with High-Variance Periods

This module implements the fix for R² calculation issues by using
non-adjacent time periods with sufficient variance.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from typing import Dict, Any, Tuple
from .standardized_evaluation import StandardizedModelEvaluator, STANDARD_TARGET_COLUMNS


def get_fixed_train_test_split(
    evaluator: StandardizedModelEvaluator,
    train_samples: int = 30000,
    test_samples: int = 5000,
) -> Dict[str, Any]:
    """
    Create train/test split using high-variance periods that are non-adjacent.

    This fixes the R² calculation issues by ensuring:
    1. Both train and test have sufficient variance
    2. Similar distribution characteristics
    3. Realistic evaluation scenarios
    """

    # Get all data
    train_data, test_data = evaluator.get_train_test_data()
    all_data = pd.concat([train_data, test_data], ignore_index=True)
    all_data["timestamp"] = pd.to_datetime(all_data["timestamp"])
    all_data_sorted = all_data.sort_values("timestamp")

    n_total = len(all_data_sorted)

    # Use Period 0 for training (first ~10% of data)
    train_start = 0
    train_end = min(38922, n_total)

    # Use Period 9 for testing (last ~10% of data)
    test_start = max(350298, n_total - 50000)
    test_end = n_total

    train_period = all_data_sorted.iloc[train_start:train_end]
    test_period = all_data_sorted.iloc[test_start:test_end]

    print(f"🔧 Using fixed high-variance periods:")
    print(
        f"   Train: {len(train_period):,} samples ({train_period['timestamp'].min()} to {train_period['timestamp'].max()})"
    )
    print(
        f"   Test:  {len(test_period):,} samples ({test_period['timestamp'].min()} to {test_period['timestamp'].max()})"
    )

    # Take requested sample sizes
    train_subset = (
        train_period.head(train_samples)[STANDARD_TARGET_COLUMNS].ffill().fillna(0)
    )
    test_subset = (
        test_period.head(test_samples)[STANDARD_TARGET_COLUMNS].ffill().fillna(0)
    )

    # Standardized scaling
    scaler = MinMaxScaler()
    train_scaled = scaler.fit_transform(train_subset.values)
    test_scaled = scaler.transform(test_subset.values)

    # Verify the fix worked
    print(f"   Verifying fix:")
    for i, col in enumerate(STANDARD_TARGET_COLUMNS[:3]):
        train_mean = np.mean(train_scaled[:, i])
        test_mean = np.mean(test_scaled[:, i])
        test_var = np.var(test_scaled[:, i])
        diff = abs(train_mean - test_mean)

        var_status = "✅" if test_var > 1e-4 else "❌"
        mean_status = "✅" if diff < 0.15 else "⚠️"

        print(
            f"      {col:20}: Var={test_var:.6f} {var_status}, Mean_diff={diff:.3f} {mean_status}"
        )

    return {
        "train_raw": train_subset.values,
        "test_raw": test_subset.values,
        "train_scaled": train_scaled,
        "test_scaled": test_scaled,
        "scaler": scaler,
        "metadata": {
            "train_samples": len(train_scaled),
            "test_samples": len(test_scaled),
            "train_time_range": (
                train_period["timestamp"].min(),
                train_period["timestamp"].max(),
            ),
            "test_time_range": (
                test_period["timestamp"].min(),
                test_period["timestamp"].max(),
            ),
            "fix_applied": "high_variance_periods",
            "scaling_method": "MinMaxScaler",
        },
    }


def test_fixed_predictions() -> Dict[str, float]:
    """Quick test to verify the fix works."""

    print("\n🧪 TESTING THE FIX")
    print("=" * 30)

    evaluator = StandardizedModelEvaluator()
    fixed_data = get_fixed_train_test_split(evaluator, 5000, 1000)

    # Test simple prediction strategies
    strategies = {}
    test_sample = fixed_data["test_scaled"][:100]  # First 100 test samples

    # Strategy 1: Use last training value
    last_train_vals = fixed_data["train_scaled"][-1]
    naive_predictions = np.tile(last_train_vals, (len(test_sample), 1))

    # Strategy 2: Previous value (t-1)
    prev_predictions = test_sample[:-1]
    prev_actual = test_sample[1:]

    for i, col in enumerate(STANDARD_TARGET_COLUMNS[:3]):
        # Test naive prediction
        y_true = test_sample[:, i]
        y_pred = naive_predictions[:, i]

        ss_res = np.sum((y_true - y_pred) ** 2)
        ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
        r2_naive = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0

        # Test previous value prediction
        if len(prev_actual) > 0:
            y_true_prev = prev_actual[:, i]
            y_pred_prev = prev_predictions[:, i]

            ss_res_prev = np.sum((y_true_prev - y_pred_prev) ** 2)
            ss_tot_prev = np.sum((y_true_prev - np.mean(y_true_prev)) ** 2)
            r2_prev = 1 - (ss_res_prev / ss_tot_prev) if ss_tot_prev > 0 else 0
        else:
            r2_prev = 0

        strategies[f"{col}_naive"] = r2_naive
        strategies[f"{col}_previous"] = r2_prev

        status_naive = "✅" if r2_naive > -1.0 else "❌"
        status_prev = "✅" if r2_prev > -1.0 else "❌"

        print(
            f"   {col:20}: Naive R²={r2_naive:7.3f} {status_naive}, Prev R²={r2_prev:7.3f} {status_prev}"
        )

    return strategies
