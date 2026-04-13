#!/usr/bin/env python3
"""
VAR (Vector Autoregression) Model Implementation – Dataset-Driven & Safe

- Matches the trainer interface used by LSTM/Transformer trainers:
  train_and_evaluate(train_set, test_set, evaluator, results_dir, direction="UL")
- Consumes ExperimentDataset directly (scaling handled there)
- Forecasts the whole test horizon at once (no per-step Python loop)
"""

import numpy as np
import pandas as pd
from datetime import datetime
from typing import Dict, Any, Tuple
import warnings
import time

warnings.filterwarnings("ignore")

from statsmodels.tsa.api import VAR
from ..evaluation.standardized_evaluation import STANDARD_TARGET_COLUMNS


class VARTrainer:
    """VAR model trainer (statsmodels) that consumes ExperimentDataset like neural trainers."""

    @staticmethod
    def _stack_targets(dataset) -> np.ndarray:
        """
        Stack all target vectors from the dataset in order into a 2D array [N, F].
        Assumes __getitem__ returns (X, y) with y shape [F].
        """
        # Avoid materializing huge Python lists if dataset supports fast tensor stack
        ys = []
        for i in range(len(dataset)):
            _, y = dataset[i]  # y: torch.Tensor [F]
            ys.append(y.detach().cpu().numpy())
        return np.vstack(ys)  # [N, F]

    @staticmethod
    def _targets_list(F: int) -> list[str]:
        if len(STANDARD_TARGET_COLUMNS) >= F:
            return STANDARD_TARGET_COLUMNS[:F]
        return [f"var_{i}" for i in range(F)]

    @staticmethod
    def train_and_evaluate(
        train_set,
        test_set,
        evaluator,
        results_dir: str,
        direction: str = "UL",
        max_lags: int = 10,
    ) -> Tuple[Dict[str, Any], Dict[str, np.ndarray], Dict[str, np.ndarray]]:
        """
        Train and evaluate a VAR model using targets from ExperimentDataset.

        - train_set / test_set: ExperimentDataset (already scaled consistently)
        - We build the VAR on the stacked training targets [N,F]
        - We forecast the entire test horizon in one call to avoid per-step loops
        """
        print("\n" + "=" * 20 + " VAR MODEL (DATASET-DRIVEN) " + "=" * 20)

        try:
            # ----- Stack targets from datasets -----
            y_train = VARTrainer._stack_targets(train_set)  # [Ntr, F] (scaled)
            y_test = VARTrainer._stack_targets(test_set)  # [Nte, F] (scaled)

            if y_train.ndim != 2 or y_test.ndim != 2:
                raise ValueError(
                    f"Expected 2D arrays; got train {y_train.shape}, test {y_test.shape}"
                )

            Ntr, F = y_train.shape
            Nte, F2 = y_test.shape
            if F != F2:
                raise ValueError(f"Train/Test feature mismatch: {F} vs {F2}")

            targets = VARTrainer._targets_list(F)
            print(
                f"Training VAR on {Ntr:,} timesteps, {F} variables; forecasting {Nte:,} steps"
            )

            # ----- Fit VAR -----
            start_time = datetime.now()
            train_df = pd.DataFrame(y_train, columns=targets)

            training_time = 0
            testing_time = 0

            t0 = time.time()
            model = VAR(train_df)
            var_res = model.fit(maxlags=max_lags, ic=None)
            training_time += time.time() - t0
            
            print(
                f"VAR fit complete in {training_time:.2f}s (selected lag order: {var_res.k_ar})"
            )

            # ----- Forecast the whole test horizon in one call -----
            # initial "context" = last k_ar rows of training data
            k = var_res.k_ar
            if Ntr <= k:
                raise ValueError(
                    f"Training series too short for lag order {k}: Ntr={Ntr}"
                )
            context = y_train[-k:, :]  # [k, F]
            # statsmodels can forecast the next Nte steps in one go:
            
            t0 = time.time()
            y_pred_scaled = var_res.forecast(y=context, steps=Nte)  # [Nte, F]
            testing_time += time.time() - t0

            # ----- Inverse transform back to original units -----
            # test_set.inverse_transform expects torch tensors; convert both
            import torch

            y_pred_orig = train_set.inverse_transform(
                torch.tensor(y_pred_scaled, dtype=torch.float32)
            )
            y_true_orig = train_set.inverse_transform(
                torch.tensor(y_test, dtype=torch.float32)
            )
            y_pred = y_pred_orig.cpu().numpy()
            y_true = y_true_orig.cpu().numpy()

            # ----- Package outputs -----
            y_true_dict = {targets[i]: y_true[:, i] for i in range(F)}
            y_pred_dict = {targets[i]: y_pred[:, i] for i in range(F)}

            # ----- Metadata & evaluation -----
            training_metadata = {
                "model_type": "VAR",
                "direction": direction,
                "training_time_sec": training_time,
                "testing_time_sec": testing_time,
                "train_len": int(Ntr),
                "test_len": int(Nte),
                "lag_order": int(var_res.k_ar),
                "num_features": int(F),
                "max_lags_arg": int(max_lags),
            }

            results = evaluator.evaluate_model_predictions(
                "VAR", y_true_dict, y_pred_dict, training_metadata
            )
            return results, y_true_dict, y_pred_dict, training_metadata

        except Exception as e:
            print(f"[ERROR] VAR failed: {e}")
            import traceback

            traceback.print_exc()
            return None, {}, {}, None
