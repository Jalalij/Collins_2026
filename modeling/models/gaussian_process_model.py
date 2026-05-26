"""
models/gaussian_process_model.py - Gaussian Process regressor (sklearn).

Mode: MISO — Multiple Input, Single Output (single-step ahead prediction).
  - Input:  (T, F) window flattened to (T*F,) feature vector per sample
  - Output: (O,) — target values at t+1 only (one horizon step)

Computationally expensive — uses small sample counts by default.
"""

from __future__ import annotations

import numpy as np


class GaussianProcessTrainer:

    @staticmethod
    def train_and_evaluate(
        train_set,
        test_set,
        results_dir,
        config,
        val_set=None,
    ) -> dict:

        print("\n" + "=" * 20 + " GAUSSIAN PROCESS MODEL (MISO) " + "=" * 20)

        try:
            from sklearn.gaussian_process import GaussianProcessRegressor
            from sklearn.gaussian_process.kernels import RBF, WhiteKernel

            # ----------------------------------------------------------------
            # CONFIG
            # ----------------------------------------------------------------
            train_limit = config.get("train_limit", 1000)

            # ----------------------------------------------------------------
            # DATA EXTRACTION
            # MISO contract:
            #   xi shape: (T, F)  — sequence window with T timesteps, F features
            #   yi shape: (O,)    — single-step target (t+1), O output channels
            #
            # If your dataset returns yi as (H, O) multi-horizon targets,
            # we take only the first horizon step: yi[0].
            # ----------------------------------------------------------------
            def extract(ds):
                X, y = [], []

                for i in range(len(ds)):
                    sample = ds[i]
                    xi = np.array(sample[0])   # features
                    yi = np.array(sample[1])   # targets

                    # --- Input: enforce 2D (T, F), then flatten to 1D ---
                    if xi.ndim == 1:
                        # Already flat — accept as-is (single timestep or
                        # caller already flattened)
                        pass
                    elif xi.ndim == 2:
                        # (T, F) -> (T*F,)  — correct MISO flattening
                        xi = xi.reshape(-1)
                    elif xi.ndim == 3:
                        # Unexpected: (B, T, F) — take first item and flatten
                        xi = xi[0].reshape(-1)
                    else:
                        raise ValueError(
                            f"Unexpected xi ndim={xi.ndim} at index {i}"
                        )

                    # --- Target: enforce 1D (O,) — single step only ---
                    if yi.ndim == 1:
                        # (O,) — already correct single-step format
                        pass
                    elif yi.ndim == 2:
                        # (H, O) multi-horizon — take only t+1 (first step)
                        yi = yi[0]
                    else:
                        raise ValueError(
                            f"Unexpected yi ndim={yi.ndim} at index {i}"
                        )

                    X.append(xi)
                    y.append(yi)

                return np.array(X), np.array(y)

            X_train, y_train = extract(train_set)
            X_test,  y_test  = extract(test_set)

            # Sanity checks after extraction
            assert X_train.ndim == 2, f"X_train must be 2D (N, features), got {X_train.shape}"
            assert X_test.ndim  == 2, f"X_test must be 2D (N, features), got {X_test.shape}"
            assert y_train.ndim == 2, f"y_train must be 2D (N, outputs), got {y_train.shape}"
            assert y_test.ndim  == 2, f"y_test must be 2D (N, outputs), got {y_test.shape}"

            n_outputs = y_train.shape[1]

            # ----------------------------------------------------------------
            # LIMIT TRAINING SIZE  (GP is O(n³) — keep n small)
            # ----------------------------------------------------------------
            if len(X_train) > train_limit:
                idx = np.random.choice(len(X_train), train_limit, replace=False)
                X_train = X_train[idx]
                y_train = y_train[idx]

            print(
                f"GP (MISO): {len(X_train)} train / {len(X_test)} test samples | "
                f"input_dim={X_train.shape[1]} | n_outputs={n_outputs}"
            )

            # ----------------------------------------------------------------
            # MODEL
            # RBF captures smooth correlations in the flattened feature space.
            # WhiteKernel handles observation noise — important for real data.
            #
            # One GP per output (explicit loop) instead of MultiOutputRegressor
            # so we can also retrieve predictive std if needed later.
            # ----------------------------------------------------------------
            kernel = RBF() + WhiteKernel()

            y_pred = np.zeros_like(y_test)

            for out_idx in range(n_outputs):
                gp = GaussianProcessRegressor(
                    kernel=kernel,
                    normalize_y=True,        # stabilises optimisation
                    n_restarts_optimizer=1,  # one restart — fast but not brittle
                )
                gp.fit(X_train, y_train[:, out_idx])
                y_pred[:, out_idx] = gp.predict(X_test)

            # ----------------------------------------------------------------
            # FORMAT OUTPUT  (matches framework's expected dict contract)
            # ----------------------------------------------------------------
            y_true_dict = {}
            y_pred_dict = {}

            for i in range(n_outputs):
                key = f"target_{i}"
                y_true_dict[key] = y_test[:, i]
                y_pred_dict[key] = y_pred[:, i]

            return {
                "model_name": "gaussian_process",
                "y_true_dict": y_true_dict,
                "y_pred_dict": y_pred_dict,
                "metadata": {
                    "model": "GaussianProcess",
                    "mode": "MISO",
                    "horizon": 1,
                    "train_samples": len(X_train),
                    "test_samples": len(X_test),
                    "input_dim": X_train.shape[1],
                    "n_outputs": n_outputs,
                },
            }

        except Exception as e:
            import traceback
            print(f"[ERROR] GP failed: {e}")
            traceback.print_exc()
            return None