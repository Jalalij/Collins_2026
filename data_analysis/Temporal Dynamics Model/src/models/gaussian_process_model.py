"""
Gaussian Process Model Implementation
"""

import os
import sys
import numpy as np
import pandas as pd
from datetime import datetime
from typing import Dict, List, Tuple, Any
import warnings

warnings.filterwarnings("ignore")

from ..evaluation.standardized_evaluation import STANDARD_TARGET_COLUMNS


class GaussianProcessTrainer:
    """Gaussian Process trainer compatible with unified framework."""

    @staticmethod
    def train_and_evaluate(
        train_set,
        test_set,
        results_dir,
        config,
        val_set=None,
    ) -> dict:

        print("\n" + "=" * 20 + " GAUSSIAN PROCESS MODEL " + "=" * 20)

        try:
            import numpy as np
            from sklearn.gaussian_process import GaussianProcessRegressor
            from sklearn.gaussian_process.kernels import RBF, WhiteKernel
            from sklearn.multioutput import MultiOutputRegressor

            # -----------------------------
            # CONFIG
            # -----------------------------
            train_limit = config.get("train_limit", 1000)
            n_jobs = config.get("n_jobs", 1)

            # -----------------------------
            # EXTRACT DATA (FRAMEWORK FIX)
            # -----------------------------
            def extract(ds):
                X, y = [], []
                for i in range(len(ds)):
                    xi, yi = ds[i][:2]   # SAFE: ignores meta if present
                    X.append(xi)
                    y.append(yi)
                return np.array(X), np.array(y)

            X_train, y_train = extract(train_set)
            X_test, y_test = extract(test_set)

            # -----------------------------
            # LIMIT SIZE (CRITICAL FOR GP)
            # -----------------------------
            if len(X_train) > train_limit:
                idx = np.random.choice(len(X_train), train_limit, replace=False)
                X_train = X_train[idx]
                y_train = y_train[idx]

            print(f"GP: {len(X_train)} train / {len(X_test)} test samples")

            # -----------------------------
            # KERNEL
            # -----------------------------
            kernel = RBF(length_scale=1.0) + WhiteKernel(noise_level=0.1)

            gp = GaussianProcessRegressor(
                kernel=kernel,
                normalize_y=True,
                n_restarts_optimizer=0,
            )

            model = MultiOutputRegressor(gp, n_jobs=n_jobs)

            # -----------------------------
            # TRAIN
            # -----------------------------
            model.fit(X_train, y_train)

            # -----------------------------
            # PREDICT
            # -----------------------------
            y_pred = model.predict(X_test)

            # -----------------------------
            # FORMAT OUTPUT
            # -----------------------------
            y_true_dict = {}
            y_pred_dict = {}

            for i in range(y_test.shape[1]):
                key = f"target_{i}"
                y_true_dict[key] = y_test[:, i]
                y_pred_dict[key] = y_pred[:, i]

            metadata = {
                "model": "GaussianProcess",
                "train_samples": len(X_train),
                "test_samples": len(X_test),
                "kernel": str(kernel),
            }

            return {
                "model_name": "gaussian_process",
                "y_true_dict": y_true_dict,
                "y_pred_dict": y_pred_dict,
                "metadata": metadata,
            }

        except Exception as e:
            import traceback
            print(f"[ERROR] GP failed: {e}")
            traceback.print_exc()
            return None