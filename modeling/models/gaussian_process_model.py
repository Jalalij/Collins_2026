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


class GaussianProcessTrainer(BaseTrainer):

    @staticmethod
    def train_and_evaluate(
        train_set,
        test_set,
        results_dir: str,
        config: dict,
        val_set=None,
    ) -> dict:
        from sklearn.gaussian_process import GaussianProcessRegressor
        from sklearn.gaussian_process.kernels import (
            RBF, WhiteKernel, Matern, RationalQuadratic,
        )
        from sklearn.multioutput import MultiOutputRegressor

        print("\n" + "=" * 20 + " GAUSSIAN PROCESS MODEL " + "=" * 20)

        train_samples = config.get("train_samples", 1000)
        test_samples  = config.get("test_samples",  500)

        base_train = _unwrap(train_set)

        def stack(ds, max_n: int):
            """
            Collect up to max_n samples from an ExperimentDataset.
            Yields (X [N, F], y [N, n_out]) using the last timestep
            of each window as the feature vector.
            """
            xs, ys = [], []
            loader = DataLoader(ds, batch_size=256, shuffle=False)
            for xb, yb, _, _ in loader:
                # xb is always (B, T, F) from ExperimentDataset
                xs.append(xb[:, -1, :].numpy())
                ys.append(yb.numpy())
                if sum(len(a) for a in xs) >= max_n:
                    break
            X = np.vstack(xs)[:max_n]
            y = np.vstack(ys)[:max_n]
            return X.astype(np.float32), y.astype(np.float32)

        X_train, y_train = stack(train_set, train_samples)
        X_test,  y_test  = stack(test_set,  test_samples)
        n_out = y_train.shape[1]

        print(f"GP: {len(X_train):,} train / {len(X_test):,} test samples, "
              f"{X_train.shape[1]} features, {n_out} targets")

        lsb    = (1e-2, 1e2)
        kernel = (
            RBF(length_scale=1.0,        length_scale_bounds=lsb)
            + Matern(length_scale=1.0,   nu=2.5, length_scale_bounds=lsb)
            + RationalQuadratic(length_scale=1.0, alpha=1.0,
                                length_scale_bounds=lsb)
            + WhiteKernel(noise_level=0.1, noise_level_bounds=(1e-10, 1e1))
        )
        gp = GaussianProcessRegressor(
            kernel=kernel, n_restarts_optimizer=2,
            alpha=1e-6, normalize_y=True, copy_X_train=False,
        )
        multi_gp = MultiOutputRegressor(gp, n_jobs=-1)

        print("Training GP...")
        t0 = time.time()
        multi_gp.fit(X_train, y_train)
        train_time = time.time() - t0
        print(f"GP training complete in {train_time:.2f}s")

        y_pred = multi_gp.predict(X_test).astype(np.float32)

        # Inverse transform via the base dataset
        def inv(arr: np.ndarray) -> np.ndarray:
            return (base_train
                    .inverse_transform(torch.tensor(arr, dtype=torch.float32))
                    .cpu().numpy())

        y_pred_orig = inv(y_pred)
        y_true_orig = inv(y_test)
        targets     = [f"var_{i}" for i in range(n_out)]

        return {
            "model_name":  "GaussianProcess",
            "y_true_dict": {targets[i]: y_true_orig[:, i] for i in range(n_out)},
            "y_pred_dict": {targets[i]: y_pred_orig[:, i] for i in range(n_out)},
            "metadata": {
                "model_type":        "GaussianProcess",
                "train_samples":     len(X_train),
                "test_samples":      len(X_test),
                "training_time_sec": train_time,
            },
        }