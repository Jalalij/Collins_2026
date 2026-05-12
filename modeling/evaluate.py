"""
evaluate.py - Comprehensive evaluation of a trained checkpoint on the test set.

Metrics computed
----------------
Per-horizon (t+1s ... t+15s) AND aggregate:
  MAE    - Mean Absolute Error           (Mbps)
  RMSE   - Root Mean Squared Error       (Mbps)
  MAPE   - Mean Absolute Percentage Error (%)   [skips near-zero targets]
  sMAPE  - Symmetric MAPE                (%)
  R2     - Coefficient of Determination
  PICP   - Prediction Interval Coverage  (95% empirical)
  MIS    - Mean Interval Score           (sharpness + coverage penalty)

Baselines compared automatically:
  Last-value  - predict the last observed value for all future steps
  Last-mean   - predict the mean of the input window for all future steps

Outputs (all written to --out_dir):
  metrics_summary.csv    - per-horizon + aggregate table
  baseline_comparison.csv
  predictions.npz        - raw preds & targets (normalised and denormalised)
  horizon_metrics.png    - per-horizon MAE / RMSE line plot
  scatter.png            - pred vs actual scatter (aggregate)
  attention_mean.png     - mean attention heatmap across test windows

Usage
-----
    python evaluate.py \
        --checkpoint checkpoints/best_model.pt \
        --data_root  data/starlink \
        --mode       dl \
        --out_dir    eval_results/dl
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")   # headless
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from dataset import (
    ALL_FEATURES, DEFAULT_FEATURES, MODE_SUFFIX, TARGET_COL,
    DataConfig, build_dataloaders,
)
from model import ThroughputSeq2Seq


# ---------------------------------------------------------------------------
# Metric functions  (all operate on numpy arrays: pred/target shape (N, H))
# ---------------------------------------------------------------------------

def mae(pred: np.ndarray, target: np.ndarray) -> np.ndarray:
    """Mean Absolute Error per horizon step -> (H,)"""
    return np.abs(pred - target).mean(axis=0)

def rmse(pred: np.ndarray, target: np.ndarray) -> np.ndarray:
    """Root Mean Squared Error per horizon step -> (H,)"""
    return np.sqrt(((pred - target) ** 2).mean(axis=0))

def mape(pred: np.ndarray, target: np.ndarray, eps: float = 1.0) -> np.ndarray:
    """
    Mean Absolute Percentage Error per horizon step -> (H,).
    Windows where |target| < eps are excluded to avoid division by near-zero.
    """
    mask = np.abs(target) >= eps
    result = np.full(pred.shape[1], np.nan)
    for h in range(pred.shape[1]):
        m = mask[:, h]
        if m.sum() > 0:
            result[h] = (np.abs(pred[m, h] - target[m, h]) / np.abs(target[m, h])).mean() * 100
    return result

def smape(pred: np.ndarray, target: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """Symmetric MAPE per horizon step -> (H,)"""
    denom = (np.abs(pred) + np.abs(target)) / 2 + eps
    return (np.abs(pred - target) / denom).mean(axis=0) * 100

def r2(pred: np.ndarray, target: np.ndarray) -> np.ndarray:
    """R² per horizon step -> (H,)"""
    ss_res = ((target - pred) ** 2).sum(axis=0)
    ss_tot = ((target - target.mean(axis=0)) ** 2).sum(axis=0)
    return 1 - ss_res / (ss_tot + 1e-8)

def prediction_interval(
    pred: np.ndarray, target: np.ndarray, alpha: float = 0.05
) -> tuple[np.ndarray, np.ndarray]:
    """
    Empirical (1-alpha) prediction interval per horizon using residual quantiles.
    Returns:
        picp : (H,)  coverage probability (should be ~0.95)
        mis  : (H,)  Mean Interval Score  (lower = sharper + well-calibrated)
    """
    residuals = target - pred                                  # (N, H)
    lo = np.quantile(residuals, alpha / 2,     axis=0)        # (H,)
    hi = np.quantile(residuals, 1 - alpha / 2, axis=0)        # (H,)
    lower = pred + lo
    upper = pred + hi
    covered = ((target >= lower) & (target <= upper)).mean(axis=0)
    width   = upper - lower
    penalty = (
        (2 / alpha) * np.maximum(lower - target, 0) +
        (2 / alpha) * np.maximum(target - upper, 0)
    ).mean(axis=0)
    mis = width.mean(axis=0) + penalty
    return covered, mis

def compute_all_metrics(
    pred: np.ndarray, target: np.ndarray, label: str = "model"
) -> pd.DataFrame:
    """Compute all metrics per horizon, return tidy DataFrame."""
    H = pred.shape[1]
    horizons = [f"t+{h+1}s" for h in range(H)]

    picp, mis = prediction_interval(pred, target)

    df = pd.DataFrame({
        "horizon":  horizons,
        "model":    label,
        "MAE":      mae(pred, target),
        "RMSE":     rmse(pred, target),
        "MAPE_pct": mape(pred, target),
        "sMAPE_pct":smape(pred, target),
        "R2":       r2(pred, target),
        "PICP_95":  picp,
        "MIS":      mis,
    })
    return df


# ---------------------------------------------------------------------------
# Baselines
# ---------------------------------------------------------------------------

def last_value_baseline(features_seq: np.ndarray, output_len: int) -> np.ndarray:
    """
    Predict the last observed TARGET value for every future step.
    features_seq shape: (N, input_len)  — target column only
    Returns: (N, output_len)
    """
    last = features_seq[:, -1:].repeat(output_len, axis=1) if features_seq.ndim == 2 \
        else np.tile(features_seq[:, -1:], (1, output_len))
    return last

def mean_window_baseline(features_seq: np.ndarray, output_len: int) -> np.ndarray:
    """
    Predict the mean of the input window for every future step.
    features_seq: (N, input_len)
    Returns: (N, output_len)
    """
    mean = features_seq.mean(axis=1, keepdims=True)
    return np.tile(mean, (1, output_len))


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------

@torch.no_grad()
def run_inference(
    model: ThroughputSeq2Seq,
    loader: DataLoader,
    device: torch.device,
    target_stats: dict | None,
    normalise: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Returns
    -------
    preds_norm   : (N, H)  normalised predictions
    targets_norm : (N, H)  normalised ground truth
    preds_mbps   : (N, H)  denormalised predictions  (Mbps)
    targets_mbps : (N, H)  denormalised ground truth (Mbps)
    attn_maps    : (N, H, T_enc)  attention weights
    last_inputs  : (N, input_len)  last feature dim (used for baselines)
    """
    model.eval()
    all_preds, all_targets, all_attns, all_last = [], [], [], []

    for feat, hour, dow, target in loader:
        feat   = feat.to(device)
        hour   = hour.to(device)
        dow    = dow.to(device)
        target = target.to(device)

        preds, attn = model(feat, hour, dow)

        all_preds.append(preds.cpu().numpy())
        all_targets.append(target.cpu().numpy())
        all_attns.append(attn.cpu().numpy())
        # last column of input features (index 0 = first feature) for baseline
        all_last.append(feat[:, :, 0].cpu().numpy())

    preds_norm   = np.concatenate(all_preds,   axis=0)
    targets_norm = np.concatenate(all_targets, axis=0)
    attn_maps    = np.concatenate(all_attns,   axis=0)
    last_inputs  = np.concatenate(all_last,    axis=0)

    def denorm(x):
        if target_stats is None or normalise not in ("zscore", "minmax"):
            return x
        if normalise == "zscore":
            return x * target_stats["std"] + target_stats["mean"]
        return x * (target_stats["max"] - target_stats["min"]) + target_stats["min"]

    preds_mbps   = denorm(preds_norm)
    targets_mbps = denorm(targets_norm)

    return preds_norm, targets_norm, preds_mbps, targets_mbps, attn_maps, last_inputs


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_horizon_metrics(metrics_df: pd.DataFrame, out_dir: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    for model_label, grp in metrics_df.groupby("model"):
        axes[0].plot(grp["horizon"], grp["MAE"],  marker="o", label=model_label)
        axes[1].plot(grp["horizon"], grp["RMSE"], marker="o", label=model_label)

    for ax, title in zip(axes, ["MAE (Mbps)", "RMSE (Mbps)"]):
        ax.set_title(title)
        ax.set_xlabel("Horizon")
        ax.tick_params(axis="x", rotation=45)
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(out_dir / "horizon_metrics.png", dpi=150)
    plt.close()


def plot_scatter(preds: np.ndarray, targets: np.ndarray, out_dir: Path) -> None:
    # Flatten all horizons
    p = preds.ravel()
    t = targets.ravel()
    # Subsample if huge
    if len(p) > 50_000:
        idx = np.random.choice(len(p), 50_000, replace=False)
        p, t = p[idx], t[idx]

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(t, p, alpha=0.15, s=4, color="steelblue")
    lims = [min(t.min(), p.min()), max(t.max(), p.max())]
    ax.plot(lims, lims, "r--", linewidth=1, label="perfect")
    ax.set_xlabel("Actual throughput (Mbps)")
    ax.set_ylabel("Predicted throughput (Mbps)")
    ax.set_title("Predicted vs Actual (all horizons)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_dir / "scatter.png", dpi=150)
    plt.close()


def plot_attention(attn_maps: np.ndarray, out_dir: Path, input_len: int = 30) -> None:
    # attn_maps: (N, H, T_enc) — average over samples
    mean_attn = attn_maps.mean(axis=0)   # (H, T_enc)
    H, T = mean_attn.shape

    fig, ax = plt.subplots(figsize=(10, 4))
    im = ax.imshow(mean_attn, aspect="auto", cmap="viridis", origin="lower")
    ax.set_xlabel("Encoder timestep (t-N ... t)")
    ax.set_ylabel("Decoder step (prediction horizon)")
    ax.set_title("Mean attention weights (test set)")
    ax.set_xticks(range(0, T, max(1, T // 10)))
    ax.set_xticklabels([f"t-{T-i}" for i in range(0, T, max(1, T // 10))])
    ax.set_yticks(range(H))
    ax.set_yticklabels([f"t+{i+1}s" for i in range(H)])
    plt.colorbar(im, ax=ax)
    plt.tight_layout()
    plt.savefig(out_dir / "attention_mean.png", dpi=150)
    plt.close()


def plot_sample_forecast(
    preds: np.ndarray, targets: np.ndarray, out_dir: Path, n_samples: int = 6
) -> None:
    """Plot a few individual forecast windows vs ground truth."""
    indices = np.linspace(0, len(preds) - 1, n_samples, dtype=int)
    fig, axes = plt.subplots(2, 3, figsize=(14, 7))
    axes = axes.ravel()
    H = preds.shape[1]
    x = np.arange(1, H + 1)

    for i, idx in enumerate(indices):
        axes[i].plot(x, targets[idx], "k-o", markersize=4, label="actual")
        axes[i].plot(x, preds[idx],   "b--s", markersize=4, label="predicted")
        axes[i].set_title(f"Window #{idx}")
        axes[i].set_xlabel("Horizon (s)")
        axes[i].set_ylabel("Throughput (Mbps)")
        axes[i].legend(fontsize=8)
        axes[i].grid(True, alpha=0.3)

    plt.suptitle("Sample forecasts — predicted vs actual")
    plt.tight_layout()
    plt.savefig(out_dir / "sample_forecasts.png", dpi=150)
    plt.close()


# ---------------------------------------------------------------------------
# Main evaluation routine
# ---------------------------------------------------------------------------

def evaluate(
    checkpoint: str,
    data_root: str,
    mode: str,
    features: list[str] | None,
    out_dir: str,
    input_len: int = 30,
    output_len: int = 15,
    hidden_size: int = 128,
    embed_dim: int = 16,
    normalise: str = "zscore",
    batch_size: int = 512,
    seed: int = 42,
    mape_eps: float = 1.0,
) -> pd.DataFrame:
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ----------------------------------------------------------------- data
    cfg = DataConfig(
        data_root  = data_root,
        mode       = mode,
        target     = mode,
        features   = features,
        input_len  = input_len,
        output_len = output_len,
        normalise  = normalise,
        batch_size = batch_size,
        seed       = seed,
    )
    _, _, test_loader, feature_stats, target_stats, feature_cols = build_dataloaders(cfg)

    if test_loader is None:
        raise ValueError("No test sessions available — check train_frac/val_frac or data.")

    print(f"\nEvaluating on {len(test_loader.dataset):,} test windows")

    # ---------------------------------------------------------------- model
    model = ThroughputSeq2Seq(
        num_features = len(feature_cols),
        embed_dim    = embed_dim,
        hidden_size  = hidden_size,
        input_len    = input_len,
        output_len   = output_len,
    )
    model.load_state_dict(torch.load(checkpoint, map_location=device))
    model = model.to(device)
    print(f"Loaded: {checkpoint}")

    # -------------------------------------------------------------- inference
    (preds_norm, targets_norm,
     preds_mbps, targets_mbps,
     attn_maps, last_inputs) = run_inference(
        model, test_loader, device, target_stats, normalise
    )

    # Save raw arrays for offline analysis
    np.savez_compressed(
        out_path / "predictions.npz",
        preds_norm=preds_norm, targets_norm=targets_norm,
        preds_mbps=preds_mbps, targets_mbps=targets_mbps,
        attn_maps=attn_maps,
    )

    # ---------------------------------------------------------------- baselines
    # Denormalise last_inputs (first feature) for last-value baseline
    def denorm_tgt(x):
        if target_stats is None:
            return x
        if normalise == "zscore":
            return x * target_stats["std"] + target_stats["mean"]
        if normalise == "minmax":
            return x * (target_stats["max"] - target_stats["min"]) + target_stats["min"]
        return x

    last_inputs_mbps = denorm_tgt(last_inputs)
    bl_last  = last_value_baseline(last_inputs_mbps, output_len)
    bl_mean  = mean_window_baseline(last_inputs_mbps, output_len)

    # ------------------------------------------------------------ metrics
    print("\nComputing metrics...")
    model_metrics    = compute_all_metrics(preds_mbps,  targets_mbps, "Seq2Seq-GRU")
    last_val_metrics = compute_all_metrics(bl_last,     targets_mbps, "Last-value")
    mean_win_metrics = compute_all_metrics(bl_mean,     targets_mbps, "Mean-window")

    all_metrics = pd.concat([model_metrics, last_val_metrics, mean_win_metrics],
                             ignore_index=True)

    # Aggregate row (mean over horizons)
    agg_rows = []
    for lbl, grp in all_metrics.groupby("model"):
        row = grp[["MAE","RMSE","MAPE_pct","sMAPE_pct","R2","PICP_95","MIS"]].mean()
        row["horizon"] = "AGGREGATE"
        row["model"]   = lbl
        agg_rows.append(row)
    agg_df = pd.DataFrame(agg_rows)
    full_metrics = pd.concat([all_metrics, agg_df], ignore_index=True)

    # Save metrics
    full_metrics.to_csv(out_path / "metrics_summary.csv", index=False, float_format="%.4f")

    # Separate baseline comparison table
    baseline_pivot = (
        full_metrics[full_metrics["horizon"] == "AGGREGATE"]
        .set_index("model")[["MAE","RMSE","MAPE_pct","sMAPE_pct","R2"]]
    )
    baseline_pivot.to_csv(out_path / "baseline_comparison.csv", float_format="%.4f")

    # ------------------------------------------------------------ print summary
    print("\n" + "="*60)
    print("AGGREGATE METRICS (mean over all horizons, original Mbps scale)")
    print("="*60)
    print(baseline_pivot.to_string())
    print("="*60)

    # ---------------------------------------------------------------- plots
    print("\nGenerating plots...")
    plot_horizon_metrics(all_metrics, out_path)
    plot_scatter(preds_mbps, targets_mbps, out_path)
    plot_attention(attn_maps, out_path, input_len)
    plot_sample_forecast(preds_mbps, targets_mbps, out_path)
    print(f"Plots saved to {out_path}/")

    return full_metrics


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="Evaluate a trained Throughput Seq2Seq model")
    p.add_argument("--checkpoint",  required=True, help="Path to best_model.pt")
    p.add_argument("--data_root",   default="data/starlink")
    p.add_argument("--mode",        default="dl", choices=list(MODE_SUFFIX))
    p.add_argument("--features",    nargs="+", default=None, choices=ALL_FEATURES,
                   metavar="FEATURE")
    p.add_argument("--out_dir",     default="eval_results")
    p.add_argument("--input_len",   type=int,   default=30)
    p.add_argument("--output_len",  type=int,   default=15)
    p.add_argument("--hidden_size", type=int,   default=128)
    p.add_argument("--embed_dim",   type=int,   default=16)
    p.add_argument("--normalise",   default="zscore", choices=["zscore","minmax","none"])
    p.add_argument("--batch_size",  type=int,   default=512)
    p.add_argument("--seed",        type=int,   default=42)
    p.add_argument("--mape_eps",    type=float, default=1.0,
                   help="Min absolute target value to include in MAPE (Mbps). "
                        "Excludes near-zero rows from the percentage calculation.")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    evaluate(
        checkpoint  = args.checkpoint,
        data_root   = args.data_root,
        mode        = args.mode,
        features    = args.features,
        out_dir     = args.out_dir,
        input_len   = args.input_len,
        output_len  = args.output_len,
        hidden_size = args.hidden_size,
        embed_dim   = args.embed_dim,
        normalise   = args.normalise if args.normalise != "none" else None,
        batch_size  = args.batch_size,
        seed        = args.seed,
        mape_eps    = args.mape_eps,
    )
