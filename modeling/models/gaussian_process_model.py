"""
models/gaussian_process_model.py - Gaussian Process regressor (sklearn).
Computationally expensive — uses small sample counts by default.
"""

from __future__ import annotations

import time
import numpy as np
import torch
from torch.utils.data import DataLoader

from .base import BaseTrainer


def _unwrap(ds):
    """Unwrap ConcatDataset to first constituent for metadata/inverse_transform."""
    if hasattr(ds, "datasets"):
        return ds.datasets[0]
    return ds


class GaussianProcessTrainer:

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

            # -------------------------
            # CONFIG
            # -------------------------
            train_limit = config.get("train_limit", 1000)

            # -------------------------
            # SAFE EXTRACTION + FLATTEN (IMPORTANT FIX)
            # -------------------------
            def extract(ds):
                X, y = [], []

                for i in range(len(ds)):
                    xi, yi = ds[i][:2]

                    xi = np.array(xi)
                    yi = np.array(yi)

                    # 🔴 CRITICAL FIX: flatten time dimension
                    # (T, F) -> (F * T) or last timestep
                    if xi.ndim == 2:
                        xi = xi.reshape(-1)   # simplest safe fix

                    X.append(xi)
                    y.append(yi)

                return np.array(X), np.array(y)

            X_train, y_train = extract(train_set)
            X_test, y_test = extract(test_set)

            # -------------------------
            # FORCE 2D SAFETY CHECK
            # -------------------------
            if X_train.ndim == 3:
                X_train = X_train.reshape(X_train.shape[0], -1)

            if X_test.ndim == 3:
                X_test = X_test.reshape(X_test.shape[0], -1)

            # -------------------------
            # LIMIT SIZE (GP SAFE ZONE)
            # -------------------------
            if len(X_train) > train_limit:
                idx = np.random.choice(len(X_train), train_limit, replace=False)
                X_train = X_train[idx]
                y_train = y_train[idx]

            print(f"GP: {len(X_train)} train / {len(X_test)} test samples")

            # -------------------------
            # MODEL
            # -------------------------
            kernel = RBF() + WhiteKernel()

            gp = GaussianProcessRegressor(
                kernel=kernel,
                normalize_y=True,
                n_restarts_optimizer=0,
            )

            model = MultiOutputRegressor(gp, n_jobs=1)

            # -------------------------
            # TRAIN
            # -------------------------
            model.fit(X_train, y_train)

            # -------------------------
            # PREDICT
            # -------------------------
            y_pred = model.predict(X_test)

            # -------------------------
            # FORMAT OUTPUT
            # -------------------------
            y_true_dict = {}
            y_pred_dict = {}

            for i in range(y_test.shape[1]):
                key = f"target_{i}"
                y_true_dict[key] = y_test[:, i]
                y_pred_dict[key] = y_pred[:, i]

            return {
                "model_name": "gaussian_process",
                "y_true_dict": y_true_dict,
                "y_pred_dict": y_pred_dict,
                "metadata": {
                    "model": "GaussianProcess",
                    "train_samples": len(X_train),
                    "test_samples": len(X_test),
                },
            }

        except Exception as e:
            import traceback
            print(f"[ERROR] GP failed: {e}")
            traceback.print_exc()
            return None