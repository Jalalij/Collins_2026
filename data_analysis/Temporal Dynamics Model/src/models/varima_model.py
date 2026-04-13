#!/usr/bin/env python3
"""
VARIMA (VARMAX) Model Implementation – Dataset-Driven & Robust

- Matches trainer interface:
  train_and_evaluate(train_set, test_set, evaluator, results_dir, direction="UL")
- Consumes ExperimentDataset directly (scaling handled there)
- Adds adaptive jitter only for (near) constant columns
- Tries multiple solver/option combos; reduces order if needed
- Falls back to VAR when VARMAX initialization stays unstable
"""

import numpy as np
import pandas as pd
from datetime import datetime
from typing import Dict, Any, Tuple
import warnings

warnings.filterwarnings("ignore")

from statsmodels.tsa.statespace.varmax import VARMAX
from statsmodels.tsa.api import VAR
import torch

from ..evaluation.standardized_evaluation import STANDARD_TARGET_COLUMNS


class VARIMATrainer:
    """Robust VARIMA (VARMAX) trainer that consumes ExperimentDataset."""

    # ---------------- Utils ----------------

    @staticmethod
    def _stack_targets(ds) -> np.ndarray:
        ys = []
        for i in range(len(ds)):
            _, y = ds[i]  # torch.Tensor [F]
            ys.append(y.detach().cpu().numpy())
        return np.vstack(ys).astype(np.float64)  # [N, F]

    @staticmethod
    def _targets_list(F: int) -> list[str]:
        if len(STANDARD_TARGET_COLUMNS) >= F:
            return STANDARD_TARGET_COLUMNS[:F]
        return [f"var_{i}" for i in range(F)]

    @staticmethod
    def _is_near_constant(col: np.ndarray, atol: float = 1e-12) -> bool:
        # Works for scaled [0,1]; near-flat if variance tiny or allclose to 0
        if not np.isfinite(col).all():
            return False
        return (np.nanstd(col) < 1e-10) or np.allclose(col, 0.0, atol=atol)

    @staticmethod
    def _apply_adaptive_jitter(
        train: pd.DataFrame,
        test: pd.DataFrame,
        base_sigma: float = 1e-6,
        max_sigma: float = 1e-3,
    ) -> Tuple[pd.DataFrame, pd.DataFrame, list[str]]:
        """
        Add tiny Gaussian jitter ONLY to (near) constant columns in train.
        Copy the same magnitude of jitter to test to keep distributions similar.
        Escalate if still flat after base jitter (rare).
        """
        rng = np.random.default_rng(1234)
        constant_cols = []
        train_out = train.copy()
        test_out = test.copy()

        for col in train.columns:
            col_vals = train[col].values
            if VARIMATrainer._is_near_constant(col_vals):
                constant_cols.append(col)
                sigma = base_sigma
                # escalate until non-flat (upper bounded)
                for _ in range(4):
                    jitter = rng.normal(0.0, sigma, size=len(train_out))
                    train_out[col] = (train[col].values + jitter).astype(np.float64)
                    if not VARIMATrainer._is_near_constant(train_out[col].values):
                        break
                    sigma = min(sigma * 10.0, max_sigma)
                # apply similar jitter scale to test (zero-mean)
                if len(test_out) > 0:
                    test_out[col] = (
                        test[col].values + rng.normal(0.0, sigma, size=len(test_out))
                    ).astype(np.float64)
        return train_out, test_out, constant_cols

    @staticmethod
    def _clean_df(df: pd.DataFrame) -> pd.DataFrame:
        df = df.replace([np.inf, -np.inf], np.nan).astype(np.float64)
        return df.fillna(0.0)

    # ---------------- Fallback to VAR ----------------

    @staticmethod
    def _fallback_var(
        train_df: pd.DataFrame,
        test_df: pd.DataFrame,
        targets: list[str],
        max_lags: int = 10,
    ) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]:
        y_train = train_df.values
        y_test = test_df.values
        model = VAR(pd.DataFrame(y_train, columns=targets))
        var_res = model.fit(maxlags=max_lags, ic=None)
        k = var_res.k_ar
        if len(y_train) <= k:
            raise ValueError(
                f"Training series too short for VAR lag {k}: N={len(y_train)}"
            )
        ctx = y_train[-k:, :]
        y_pred = var_res.forecast(y=ctx, steps=len(y_test))
        y_true_dict = {targets[i]: y_test[:, i] for i in range(y_test.shape[1])}
        y_pred_dict = {targets[i]: y_pred[:, i] for i in range(y_pred.shape[1])}
        return y_true_dict, y_pred_dict

    # ---------------- Main ----------------

    @staticmethod
    def train_and_evaluate(
        train_set,
        test_set,
        evaluator,
        results_dir: str,
        direction: str = "UL",
        order: Tuple[int, int] = (1, 0),  # (p, q)
        integration_order: int = 0,  # d  (we don't difference here; dataset already windowed/scaled)
        maxiter: int = 500,
        max_lags_fallback_var: int = 10,
    ) -> Tuple[Dict[str, Any], Dict[str, np.ndarray], Dict[str, np.ndarray]]:
        """
        Train VARIMA (VARMAX) on ExperimentDataset targets.

        If the state-space model remains unstable after retries, falls back to VAR.
        """
        print("\n" + "=" * 20 + " VARIMA MODEL (DATASET-DRIVEN & ROBUST) " + "=" * 20)

        try:
            # ------- Stack scaled targets -------
            y_train = VARIMATrainer._stack_targets(train_set)  # [Ntr, F], scaled
            y_test = VARIMATrainer._stack_targets(test_set)  # [Nte, F], scaled
            if y_train.ndim != 2 or y_test.ndim != 2:
                raise ValueError(
                    f"Expected 2D arrays; got train {y_train.shape}, test {y_test.shape}"
                )
            Ntr, F = y_train.shape
            Nte, F2 = y_test.shape
            if F != F2:
                raise ValueError(f"Train/Test target mismatch: {F} vs {F2}")
            targets = VARIMATrainer._targets_list(F)

            train_df = pd.DataFrame(y_train, columns=targets)
            test_df = pd.DataFrame(y_test, columns=targets)

            # ------- Clean & jitter near-constant -------
            train_df = VARIMATrainer._clean_df(train_df)
            test_df = VARIMATrainer._clean_df(test_df)

            train_df, test_df, const_cols = VARIMATrainer._apply_adaptive_jitter(
                train_df, test_df
            )
            if const_cols:
                print(
                    f"   Applied adaptive jitter to near-constant columns: {const_cols}"
                )

            # ------- Try a few robust fit configurations -------
            # We do NOT do differencing here (d=integration_order) because the dataset already
            # uses windowed next-step prediction on scaled values.
            configs = [
                dict(
                    enforce_stationarity=False,
                    enforce_invertibility=False,
                    trend="c",
                    method="bfgs",
                ),
                dict(
                    enforce_stationarity=False,
                    enforce_invertibility=False,
                    trend="c",
                    method="lbfgs",
                ),
                dict(
                    enforce_stationarity=False,
                    enforce_invertibility=False,
                    trend="c",
                    method="powell",
                ),
                dict(
                    enforce_stationarity=False,
                    enforce_invertibility=False,
                    trend="nc",
                    method="bfgs",
                ),
            ]

            varima_res = None
            used_cfg = None
            start_time = datetime.now()

            for cfg in configs:
                try:
                    model = VARMAX(
                        train_df,
                        order=order,
                        trend=cfg.get("trend", "c"),
                        enforce_stationarity=cfg.get("enforce_stationarity", False),
                        enforce_invertibility=cfg.get("enforce_invertibility", False),
                    )
                    varima_res = model.fit(
                        disp=False, maxiter=maxiter, method=cfg.get("method", "lbfgs")
                    )
                    used_cfg = cfg
                    break
                except Exception as e:
                    # escalate jitter slightly and try again with next cfg
                    train_df, test_df, _ = VARIMATrainer._apply_adaptive_jitter(
                        train_df, test_df, base_sigma=1e-5, max_sigma=1e-3
                    )
                    continue

            if varima_res is None:
                # Last resort: reduce order and one more try
                try:
                    print("   Reducing order to (1,0) and retrying...")
                    model = VARMAX(
                        train_df,
                        order=(1, 0),
                        trend="c",
                        enforce_stationarity=False,
                        enforce_invertibility=False,
                    )
                    varima_res = model.fit(disp=False, maxiter=maxiter, method="powell")
                    used_cfg = {"trend": "c", "method": "powell"}
                except Exception:
                    # Hard fallback to VAR
                    print("   VARIMA still unstable; falling back to VAR.")
                    y_true_dict, y_pred_dict = VARIMATrainer._fallback_var(
                        train_df, test_df, targets, max_lags=max_lags_fallback_var
                    )
                    # inverse transform back to original units
                    y_pred_orig = train_set.inverse_transform(
                        torch.tensor(
                            np.column_stack([y_pred_dict[t] for t in targets]),
                            dtype=torch.float32,
                        )
                    )
                    y_true_orig = train_set.inverse_transform(
                        torch.tensor(
                            np.column_stack([y_true_dict[t] for t in targets]),
                            dtype=torch.float32,
                        )
                    )
                    y_pred = y_pred_orig.cpu().numpy()
                    y_true = y_true_orig.cpu().numpy()
                    y_true_dict = {
                        targets[i]: y_true[:, i] for i in range(y_true.shape[1])
                    }
                    y_pred_dict = {
                        targets[i]: y_pred[:, i] for i in range(y_pred.shape[1])
                    }

                    meta = {
                        "model_type": "VARIMA->VAR_FALLBACK",
                        "direction": direction,
                        "training_time_sec": (
                            datetime.now() - start_time
                        ).total_seconds(),
                        "train_len": int(Ntr),
                        "test_len": int(Nte),
                        "order": (1, 0),
                        "fallback": True,
                        "num_features": int(F),
                    }
                    results = evaluator.evaluate_model_predictions(
                        "VARIMA_FALLBACK_VAR", y_true_dict, y_pred_dict, meta
                    )
                    return results, y_true_dict, y_pred_dict

            training_time = (datetime.now() - start_time).total_seconds()
            print(f"VARIMA fit complete in {training_time:.2f}s with cfg={used_cfg}")

            # ------- Forecast entire test horizon at once -------
            y_pred_scaled = varima_res.forecast(steps=len(test_df)).values  # [Nte, F]
            y_true_scaled = test_df.values

            # ------- Inverse transform back to original units -------
            y_pred_orig = train_set.inverse_transform(
                torch.tensor(y_pred_scaled, dtype=torch.float32)
            )
            y_true_orig = train_set.inverse_transform(
                torch.tensor(y_true_scaled, dtype=torch.float32)
            )
            y_pred = y_pred_orig.cpu().numpy()
            y_true = y_true_orig.cpu().numpy()

            # Package dicts
            y_true_dict = {targets[i]: y_true[:, i] for i in range(y_true.shape[1])}
            y_pred_dict = {targets[i]: y_pred[:, i] for i in range(y_pred.shape[1])}

            meta = {
                "model_type": "VARIMA",
                "direction": direction,
                "training_time_sec": training_time,
                "train_len": int(Ntr),
                "test_len": int(Nte),
                "order": order,
                "integration_order": integration_order,
                "num_features": int(F),
                "used_cfg": used_cfg,
            }
            results = evaluator.evaluate_model_predictions(
                "VARIMA", y_true_dict, y_pred_dict, meta
            )
            return results, y_true_dict, y_pred_dict

        except Exception as e:
            print(f"[ERROR] VARIMA failed: {e}")
            import traceback

            traceback.print_exc()
            return None, {}, {}
