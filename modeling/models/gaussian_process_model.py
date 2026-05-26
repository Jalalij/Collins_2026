"""
models/gaussian_process_model.py - Gaussian Process regressor (sklearn).

Mode: MISO — Multiple Input, Single Output.
  - Input:  (B, T, F) window flattened to (B, T*F)
  - Output: (B, output_len) — primary target across all horizon steps

Computationally expensive O(n³) — keep train_limit small (default: 1000).

Trainer contract
----------------
train_and_evaluate(cfg: DataConfig, results_dir: str, config: dict) -> dict
"""

from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import DataLoader

from .base import BaseTrainer

try:
    from ..dataset import DataConfig, build_dataloaders
except ImportError:
    from dataset import DataConfig, build_dataloaders


def _unwrap(ds):
    if hasattr(ds, "datasets"):
        return _unwrap(ds.datasets[0])
    return ds


class GaussianProcessTrainer(BaseTrainer):

    @staticmethod
    def train_and_evaluate(
        cfg: DataConfig,
        results_dir: str,
        config: dict,
    ) -> dict:
        """
        MISO contract
        -------------
        Dataset (io_mode=MISO) yields:
            feat : (B, T, F)
            hour : (B, T)
            dow  : (B, T)
            tgt  : (B, output_len)   ← primary target, full horizon

        The feature window is flattened to (B, T*F) per sample.
        One GP is fit per horizon step.
        """
        print("\n" + "=" * 20 + " GAUSSIAN PROCESS MODEL (MISO) " + "=" * 20)

        try:
            from sklearn.gaussian_process import GaussianProcessRegressor
            from sklearn.gaussian_process.kernels import RBF, WhiteKernel
        except ImportError:
            raise ImportError("scikit-learn is required for GaussianProcessTrainer.")

        train_limit = config.get("train_limit", 1000)

        # ------------------------------------------------------------------ data
        train_loader, _, test_loader, _, _, _ = build_dataloaders(cfg)
        if test_loader is None:
            raise ValueError("No test windows available for GP evaluation.")

        base_ds    = _unwrap(train_loader.dataset)
        output_len = cfg.output_len

        # Verify MISO target shape from first batch
        feat_b, _, _, tgt_b = next(iter(train_loader))
        assert tgt_b.dim() == 2 and tgt_b.shape[1] == output_len, (
            f"Expected MISO target (B, {output_len}), got {tuple(tgt_b.shape)}. "
            f"Ensure DataConfig.io_mode='MISO'."
        )
        T = feat_b.shape[1]
        F = feat_b.shape[2]
        input_dim = T * F

        # ------------------------------------------------------------------ collect arrays
        def _collect(loader: torch.utils.data.DataLoader):
            X_list, y_list = [], []
            for feat, hour, dow, tgt in loader:
                # Flatten (B, T, F) → (B, T*F)
                X_list.append(feat.view(feat.size(0), -1).numpy())
                y_list.append(tgt.numpy())           # (B, output_len)
            return (np.concatenate(X_list, axis=0),
                    np.concatenate(y_list, axis=0))

        X_train, y_train = _collect(train_loader)
        X_test,  y_test  = _collect(test_loader)

        # Sub-sample training set — GP is O(n³)
        if len(X_train) > train_limit:
            idx     = np.random.choice(len(X_train), train_limit, replace=False)
            X_train = X_train[idx]
            y_train = y_train[idx]

        print(
            f"  GP (MISO): {len(X_train)} train / {len(X_test)} test samples | "
            f"input_dim={input_dim} | output_len={output_len} | "
            f"target={cfg.primary_target}"
        )

        # ------------------------------------------------------------------ fit one GP per horizon step
        kernel = RBF() + WhiteKernel()
        y_pred = np.zeros_like(y_test)   # (N, output_len)

        for h in range(output_len):
            gp = GaussianProcessRegressor(
                kernel=kernel,
                normalize_y=True,
                n_restarts_optimizer=1,
            )
            gp.fit(X_train, y_train[:, h])
            y_pred[:, h] = gp.predict(X_test)

        # ------------------------------------------------------------------ denormalise
        # Wrap in tensor for inverse_transform (single-target MISO: target_idx=0)
        preds_t  = torch.from_numpy(y_pred.astype(np.float32))
        true_t   = torch.from_numpy(y_test.astype(np.float32))
        preds_dn = base_ds.inverse_transform(preds_t,  target_idx=0).numpy()
        true_dn  = base_ds.inverse_transform(true_t,   target_idx=0).numpy()

        # ------------------------------------------------------------------ output dicts
        y_true_dict: dict[str, np.ndarray] = {}
        y_pred_dict: dict[str, np.ndarray] = {}
        for h in range(output_len):
            key = f"t+{h+1}s"
            y_true_dict[key] = true_dn[:,  h]
            y_pred_dict[key] = preds_dn[:, h]

        return {
            "model_name":  "GaussianProcess",
            "y_true_dict": y_true_dict,
            "y_pred_dict": y_pred_dict,
            "metadata": {
                "model_type":     "GaussianProcess",
                "strategy":       "MISO",
                "output_len":     output_len,
                "primary_target": cfg.primary_target,
                "train_samples":  len(X_train),
                "test_samples":   len(X_test),
                "input_dim":      input_dim,
            },
        }