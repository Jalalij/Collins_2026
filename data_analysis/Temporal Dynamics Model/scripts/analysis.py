#!/usr/bin/env python3
"""
Target Variable Analysis

Using the specified targets:
- throughput_mbps
- packet_loss_percent
- jitter_ms
- rtt_mean
- sinr_db
- beam_change (0/1)
- satellite_change (0/1)

Fix training issues and show per-target R² for the variables.
"""

import pandas as pd
import numpy as np
from datetime import datetime
import json
import glob
from pathlib import Path
import warnings

warnings.filterwarnings("ignore")
import os
import sys
import argparse

from src.evaluation.standardized_evaluation import StandardizedModelEvaluator

# Import ALL working models
from src.models import (
    VARTrainer,
    LSTMTrainer,
    TransformerTrainer,
    VARIMATrainer,
    WaveNetTrainer,  
    PredictiveAutoencoderTrainer,
)
from src.models.linear_regression import LinearRegressionModel

from src.data import TARGET_VARIABLES as THE_TARGET_VARIABLES

from config import args


def load_and_split_experiments(
    csv_dir=args.data_dir, direction=args.direction, minutes=args.minutes
):

    from src.data.process_data import ExperimentDataset

    direction = direction.upper()
    files = [
        os.path.join(csv_dir, f)
        for f in os.listdir(csv_dir)
        if f.endswith(f"_{direction}.csv")
    ]
    if not files:
        raise FileNotFoundError(
            f"No CSVs found in {csv_dir} for direction '{direction}'."
        )

    train_set = ExperimentDataset(
        files,
        mode="train",
        direction=direction,
        window_size=1,
        minutes=minutes,
        target_hz=args.target_hz,
    )
    test_set = ExperimentDataset(
        files,
        mode="test",
        direction=direction,
        window_size=1,
        minutes=minutes,
        target_hz=args.target_hz,
    )

    return train_set, test_set


def calculate_target_r2(y_true, y_pred):
    """Calculate R² for a single target."""
    try:
        y_true = np.array(y_true).flatten()
        y_pred = np.array(y_pred).flatten()

        # Remove NaN values
        mask = ~(np.isnan(y_true) | np.isnan(y_pred))
        y_true_clean = y_true[mask]
        y_pred_clean = y_pred[mask]

        if len(y_true_clean) < 2:
            return -999

        ss_res = np.sum((y_true_clean - y_pred_clean) ** 2)
        ss_tot = np.sum((y_true_clean - np.mean(y_true_clean)) ** 2)

        if ss_tot == 0:
            return 0 if ss_res == 0 else -999

        r2 = 1 - (ss_res / ss_tot)
        return float(r2)
    except:
        return -999


def extract_the_target_metrics(results, y_true_dict, y_pred_dict):
    """Extract R² for the target variables from model results."""
    per_target_r2 = {}

    if results and isinstance(results, dict) and "metrics" in results:
        # Try to extract from detailed results
        metrics = results["metrics"]
        for target in THE_TARGET_VARIABLES:
            if target in metrics:
                per_target_r2[target] = metrics[target].get("r2", -999)
            else:
                per_target_r2[target] = -999
    elif y_true_dict and y_pred_dict:
        # Calculate manually from predictions
        for target in THE_TARGET_VARIABLES:
            if target in y_true_dict and target in y_pred_dict:
                print(y_true_dict[target], y_pred_dict[target])
                r2 = calculate_target_r2(y_true_dict[target], y_pred_dict[target])
                per_target_r2[target] = r2
            else:
                per_target_r2[target] = -999
    else:
        # No valid data
        for target in THE_TARGET_VARIABLES:
            per_target_r2[target] = -999

    return per_target_r2


def run_correct_target_analysis():
    """Run analysis with The target variables."""
    print("TARGET VARIABLE ANALYSIS")
    print("=" * 60)
    print(f"Using THE targets: {THE_TARGET_VARIABLES}")
    print(f"Started: {datetime.now()}")

    # Load data
    train_data, test_data = load_and_split_experiments()

    if args.smoke_test == True:
        models_to_test = [
            ("VAR", VARTrainer),
        ]
    else:
        models_to_test = [
            ("LSTM", LSTMTrainer),
            ("Transformer", TransformerTrainer),
            # ("WaveNet", WaveNetTrainer),
            ("VAR", VARTrainer),
        ]
    all_model_results = {}

    print(f"\\n TESTING MODELS WITH THE TARGET VARIABLES")
    print("=" * 60)

    print(f" Training Models: {[m[0] for m in models_to_test]}")

    for i, (model_name, ModelTrainer) in enumerate(models_to_test, 1):
        print(f"\\n{i}️⃣ {model_name}")

        evaluator = StandardizedModelEvaluator()
        results, y_true_dict, y_pred_dict = ModelTrainer.train_and_evaluate(
            train_data, test_data, evaluator, "/tmp"
        )

        per_target_r2 = extract_the_target_metrics(results, y_true_dict, y_pred_dict)
        all_model_results[model_name] = {"y_true": y_true_dict, "y_pred": y_pred_dict}

    # Save results
    results_data = {
        "timestamp": datetime.now().isoformat(),
        "analysis_type": "CORRECT_TARGET_VARIABLE_ANALYSIS",
        "the_target_variables": THE_TARGET_VARIABLES,
        "per_target_results": all_model_results,
    }

    save_dir = Path("results") / f"{args.direction}_hz{args.target_hz}"
    save_dir.mkdir(parents=True, exist_ok=True)
    save_dir = save_dir / "results.csv"

    frames = []
    for model_name, res in all_model_results.items():
        y_true = res["y_true"]
        y_pred = res["y_pred"]
        df = pd.DataFrame(
            {f"{model_name}_true_{t}": y_true[t] for t in y_true}
            | {f"{model_name}_pred_{t}": y_pred[t] for t in y_pred}
        )
        frames.append(df)

    df = pd.concat(frames, axis=1)

    df.to_csv(save_dir, index=False)
    print(f"Results saved to {save_dir} and results.csv")

    return all_model_results


if __name__ == "__main__":

    results = run_correct_target_analysis()
