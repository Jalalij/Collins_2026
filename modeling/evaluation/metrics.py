"""
evaluation/metrics.py - Full metric suite.

Per-horizon (applied per target for multi-target):
    MAE, RMSE, MAPE, sMAPE, R², PICP-95, MIS

Aggregate diagnostics:
    Relative RMSE/MAE vs persistence baseline
    Residual Pearson lag-1, Ljung-Box p-value (lag 10)
"""

from __future__ import annotations

import numpy as np
import pandas as pd

try:
    import statsmodels.api as sm
    _HAS_SM = True
except Exception:
    _HAS_SM = False

try:
    from scipy.stats import pearsonr as _pearsonr
    _HAS_SCIPY = True
except Exception:
    _HAS_SCIPY = False


# ---------------------------------------------------------------------------
# Per-horizon metrics  [N, H] arrays
# ---------------------------------------------------------------------------

def mae(pred, target):
    return np.abs(pred - target).mean(axis=0)

def rmse(pred, target):
    return np.sqrt(((pred - target) ** 2).mean(axis=0))

def mape(pred, target, eps=1.0):
    pred   = np.asarray(pred);   pred   = pred.reshape(-1, 1)   if pred.ndim   == 1 else pred
    target = np.asarray(target); target = target.reshape(-1, 1) if target.ndim == 1 else target
    mask = np.abs(target) >= eps
    # Use a safe denominator everywhere so NumPy never divides by zero,
    # then overwrite excluded positions with NaN before nanmean.
    safe_denom = np.where(mask, np.abs(target), 1.0)   # 1.0 is a dummy; result overwritten
    pct_err    = np.abs(pred - target) / safe_denom * 100
    pct_err    = np.where(mask, pct_err, np.nan)
    return np.nanmean(pct_err, axis=0)

def smape(pred, target, eps=1e-8):
    denom = (np.abs(pred) + np.abs(target)) / 2 + eps
    return (np.abs(pred - target) / denom).mean(axis=0) * 100

def r2(pred, target):
    ss_res = ((target - pred) ** 2).sum(axis=0)
    ss_tot = ((target - target.mean(axis=0)) ** 2).sum(axis=0)
    return 1 - ss_res / (ss_tot + 1e-8)

def prediction_interval(pred, target, alpha=0.05):
    residuals = target - pred
    lo      = np.quantile(residuals, alpha / 2,     axis=0)
    hi      = np.quantile(residuals, 1 - alpha / 2, axis=0)
    lower   = pred + lo;  upper = pred + hi
    covered = ((target >= lower) & (target <= upper)).mean(axis=0)
    width   = upper - lower
    # MIS: average of (width + penalties) per sample — do NOT pre-average width
    # before adding penalty, that double-counts the averaging.
    mis = (width + (2/alpha) * np.maximum(lower - target, 0)
                 + (2/alpha) * np.maximum(target - upper, 0)).mean(axis=0)
    return covered, mis


# ---------------------------------------------------------------------------
# Aggregate / diagnostic metrics  [N] flat arrays
# ---------------------------------------------------------------------------

def _persistence(y):
    # yhat[i] = y[i-1].  We have no lag-1 for the first sample, so we drop
    # it from both arrays in the callers (relative_rmse / relative_mae) rather
    # than assigning zero error to it, which flatters the baseline.
    return y[:-1]          # shape (N-1,); callers must align y[1:]

def relative_rmse(y_true, y_pred):
    m = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
    # persistence baseline: yhat[i] = y[i-1]; compare on y[1:] vs y[:-1]
    n = float(np.sqrt(np.mean((y_true[1:] - _persistence(y_true)) ** 2)))
    return m / (n + 1e-12)

def relative_mae(y_true, y_pred):
    m = float(np.mean(np.abs(y_true - y_pred)))
    # persistence baseline: yhat[i] = y[i-1]; compare on y[1:] vs y[:-1]
    n = float(np.mean(np.abs(y_true[1:] - _persistence(y_true))))
    return m / (n + 1e-12)

def residual_pearson_lag1(residuals):
    r = np.asarray(residuals).ravel()
    if r.size < 2:
        return float("nan"), float("nan")
    mask = np.isfinite(r[1:]) & np.isfinite(r[:-1])
    if mask.sum() < 2:
        return float("nan"), float("nan")
    if _HAS_SCIPY:
        return tuple(float(v) for v in _pearsonr(r[1:][mask], r[:-1][mask]))
    c = float(np.corrcoef(r[1:][mask], r[:-1][mask])[0, 1])
    return c, float("nan")

def ljung_box_pvalue(residuals, lags=10):
    if not _HAS_SM:
        return float("nan")
    r = np.asarray(residuals).ravel()
    if np.sum(np.isfinite(r)) < lags + 3:
        return float("nan")
    try:
        lb = sm.stats.acorr_ljungbox(r, lags=[lags], return_df=True)
        return float(lb["lb_pvalue"].iloc[0])
    except Exception:
        return float("nan")


# ---------------------------------------------------------------------------
# Convenience builders
# ---------------------------------------------------------------------------

def compute_horizon_metrics(pred, target, label="model", mape_eps=1.0):
    """
    pred, target : (N, H)  denormalised values.
    Returns tidy DataFrame with one row per horizon.
    """
    H        = pred.shape[1]
    horizons = [f"t+{h+1}s" for h in range(H)]
    picp, mis = prediction_interval(pred, target)
    return pd.DataFrame({
        "horizon":   horizons,
        "model":     label,
        "MAE":       mae(pred, target),
        "RMSE":      rmse(pred, target),
        "MAPE_pct":  mape(pred, target, eps=mape_eps),
        "sMAPE_pct": smape(pred, target),
        "R2":        r2(pred, target),
        "PICP_95":   picp,
        "MIS":       mis,
    })


def compute_aggregate_diagnostics(pred_flat, target_flat):
    # Guarantee 1-D — callers may pass (N,1) which would broadcast to (N,N)
    pred_flat   = np.asarray(pred_flat).ravel()
    target_flat = np.asarray(target_flat).ravel()
    residuals = target_flat - pred_flat
    r1, r1p   = residual_pearson_lag1(residuals)
    return {
        "relative_rmse":        relative_rmse(target_flat, pred_flat),
        "relative_mae":         relative_mae(target_flat, pred_flat),
        "resid_pearson_lag1":   r1,
        "resid_pearson_lag1_p": r1p,
        "resid_ljung_p_lag10":  ljung_box_pvalue(residuals, lags=10),
    }


# ---------------------------------------------------------------------------
# Baselines
# ---------------------------------------------------------------------------

def last_value_baseline(last_inputs, output_len):
    return np.tile(last_inputs[:, -1:], (1, output_len))

def mean_window_baseline(last_inputs, output_len):
    return np.tile(last_inputs.mean(axis=1, keepdims=True), (1, output_len))