"""
evaluation/evaluate.py - Unified evaluation for any trainer's output dict.

Called by run.py after each model's train_and_evaluate(). Also usable standalone.

Input:
    result dict from a trainer:
        {
            "model_name"  : str,
            "pred_mode"   : str,          # "MIMO" | "MISO" | "SISO"  (default: "MIMO")
            "y_true_dict" : {key: np.ndarray shape (N,)},
            "y_pred_dict" : {key: np.ndarray shape (N,)},
            "metadata"    : dict,
        }

    Keys in y_true_dict can be:
        "t+1s", "t+2s", ...               -> single-target Seq2Seq
        "dl_t+1s", "rtt_t+2s", ...        -> multi-target Seq2Seq
        "var_0", "var_1", ...             -> legacy next-step trainers

    pred_mode controls which plots are produced:
        MIMO  -> per-key metrics + scatter + horizon_metrics.png
        MISO  -> per-key metrics + scatter  (no horizon plot)
        SISO  -> per-key metrics + scatter  (no horizon plot)

Output (returned + written to out_dir):
    metrics_summary.csv     per-key + AGGREGATE rows
    diagnostics.csv         residual diagnostic metrics
    plots/scatter.png
    plots/horizon_metrics.png   (MIMO only, if horizon keys detected)
"""

from __future__ import annotations

import re
from pathlib import Path
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

try:
    from .metrics import (
        compute_horizon_metrics, compute_aggregate_diagnostics,
        mae, rmse, r2, mape, smape, prediction_interval,
    )
except ImportError:
    from evaluation.metrics import (
        compute_horizon_metrics, compute_aggregate_diagnostics,
        mae, rmse, r2, mape, smape, prediction_interval,
    )


# ---------------------------------------------------------------------------
# Key parsing helpers
# ---------------------------------------------------------------------------

_HORIZON_RE = re.compile(r"t\+(\d+)s$")

def _is_horizon_key(k: str) -> bool:
    return bool(_HORIZON_RE.search(k))

def _target_prefix(k: str) -> str:
    """'dl_t+1s' -> 'dl',  't+1s' -> '',  'var_0' -> 'var'"""
    m = _HORIZON_RE.search(k)
    if m:
        prefix = k[:m.start()].rstrip("_")
        return prefix
    return k.rsplit("_", 1)[0] if "_" in k else k

def _horizon_step(k: str) -> int:
    m = _HORIZON_RE.search(k)
    return int(m.group(1)) if m else 0


# ---------------------------------------------------------------------------
# Core evaluation
# ---------------------------------------------------------------------------

def evaluate_predictions(
    result:    dict,
    out_dir:   str,
    mape_eps:  float = 1.0,
    make_plots: bool = True,
) -> pd.DataFrame:
    """
    Compute all metrics from a trainer result dict.

    Returns a DataFrame with one row per key + one AGGREGATE row.
    Writes CSV and plots to out_dir.
    """
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    model_name  = result.get("model_name", "Model")
    pred_mode   = result.get("pred_mode", "MIMO")   # "MIMO" | "MISO" | "SISO"
    y_true_dict = result["y_true_dict"]
    y_pred_dict = result["y_pred_dict"]
    keys        = list(y_true_dict.keys())

    # ------------------------------------------------------------------ metrics
    rows = []
    for k in keys:
        # .ravel() first — if trainer returns (N,1), reshape(-1,1) would give
        # (N,1,1) and subsequent broadcasting blows up to (N,N).
        yt = np.asarray(y_true_dict[k], dtype=np.float32).ravel()
        yp = np.asarray(y_pred_dict[k], dtype=np.float32).ravel()
        # Wrap in (N,1) for horizon functions
        yt2, yp2 = yt.reshape(-1, 1), yp.reshape(-1, 1)
        picp, mis = prediction_interval(yp2, yt2)
        rows.append({
            "key":       k,
            "model":     model_name,
            "pred_mode": pred_mode,
            "MAE":       float(mae(yp2, yt2)[0]),
            "RMSE":      float(rmse(yp2, yt2)[0]),
            "MAPE_pct":  float(mape(yp2, yt2, eps=mape_eps)[0]),
            "sMAPE_pct": float(smape(yp2, yt2)[0]),
            "R2":        float(r2(yp2, yt2)[0]),
            "PICP_95":   float(picp[0]),
            "MIS":       float(mis[0]),
        })

    df = pd.DataFrame(rows)

    # Aggregate row — sample-weighted so multi-target keys with unequal N
    # don't get equal weight. Store per-key N alongside the rows.
    key_n = {k: int(np.asarray(y_true_dict[k]).ravel().size) for k in keys}
    total_n = sum(key_n.values())
    num_cols = ["MAE", "RMSE", "MAPE_pct", "sMAPE_pct", "R2", "PICP_95", "MIS"]
    agg: dict = {}
    for col in num_cols:
        col_vals = df.set_index("key")[col]
        agg[col] = float(sum(
            col_vals[k] * key_n[k] for k in keys if np.isfinite(col_vals[k])
        ) / max(sum(key_n[k] for k in keys if np.isfinite(col_vals[k])), 1))
    agg["key"] = "AGGREGATE"; agg["model"] = model_name; agg["pred_mode"] = pred_mode
    df = pd.concat([df, pd.DataFrame([agg])], ignore_index=True)
    df.to_csv(out_path / "metrics_summary.csv", index=False, float_format="%.4f")

    # ----------------------------------------------------------- diagnostics
    # .ravel() is critical: if any trainer returns (N,1) arrays instead of
    # (N,), concatenation produces (total, 1) and the subsequent subtraction
    # in compute_aggregate_diagnostics broadcasts to (total, total) — 560 GiB.
    all_yt = np.concatenate([np.asarray(y_true_dict[k]).ravel() for k in keys])
    all_yp = np.concatenate([np.asarray(y_pred_dict[k]).ravel() for k in keys])
    diag   = compute_aggregate_diagnostics(all_yp, all_yt)
    diag["model"] = model_name
    diag_df = pd.DataFrame([diag]).set_index("model")
    diag_df.to_csv(out_path / "diagnostics.csv", float_format="%.4f")

    # ------------------------------------------------------------------ plots
    if make_plots:
        plots_dir = out_path / "plots"
        plots_dir.mkdir(exist_ok=True)
        _scatter(all_yp, all_yt, model_name, plots_dir)

        # Horizon plot only makes sense for MIMO models that produce a sequence
        # of predictions at multiple future steps.  MISO and SISO models output
        # a single step, so the plot would be a single point and is skipped.
        if pred_mode == "MIMO" and any(_is_horizon_key(k) for k in keys):
            _horizon_plot(y_true_dict, y_pred_dict, model_name, plots_dir)

    # ------------------------------------------------------------------ print
    print(f"\n  [{model_name}] ({pred_mode}) Aggregate metrics:")
    for col in num_cols:
        print(f"    {col:<14}: {agg[col]:.4f}")
    diag_show = {k: v for k, v in diag.items() if k != "model"}
    for k, v in diag_show.items():
        if isinstance(v, float):
            print(f"    {k:<30}: {v:.4f}")

    return df


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------

def _scatter(y_pred, y_true, label, out_dir):
    p, t = y_pred.ravel(), y_true.ravel()
    if len(p) > 50_000:
        rng = np.random.default_rng(seed=0)   # fixed seed — plot is reproducible
        idx = rng.choice(len(p), 50_000, replace=False)
        p, t = p[idx], t[idx]
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(t, p, alpha=0.15, s=4, color="steelblue")
    lims = [min(t.min(), p.min()), max(t.max(), p.max())]
    ax.plot(lims, lims, "r--", lw=1, label="perfect")
    ax.set_xlabel("Actual"); ax.set_ylabel("Predicted")
    ax.set_title(f"{label} — Predicted vs Actual")
    ax.legend(); ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_dir / "scatter.png", dpi=150)
    plt.close()


def _horizon_plot(y_true_dict, y_pred_dict, label, out_dir):
    # Group keys by target prefix
    groups: dict[str, list[str]] = defaultdict(list)
    for k in y_true_dict:
        if _is_horizon_key(k):
            groups[_target_prefix(k)].append(k)

    for group, ks in groups.items():
        ks_sorted = sorted(ks, key=_horizon_step)
        horizons  = [k for k in ks_sorted]
        # .ravel() prevents (N,1) - (N,1) broadcasting to (N,N) — same issue
        # as in compute_aggregate_diagnostics; values must be strictly 1-D.
        mae_vals  = [float(np.mean(np.abs(
                        np.asarray(y_pred_dict[k]).ravel() -
                        np.asarray(y_true_dict[k]).ravel()))) for k in ks_sorted]
        rmse_vals = [float(np.sqrt(np.mean((
                        np.asarray(y_pred_dict[k]).ravel() -
                        np.asarray(y_true_dict[k]).ravel())**2))) for k in ks_sorted]

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
        ax1.plot(horizons, mae_vals,  marker="o"); ax1.set_title("MAE per horizon")
        ax2.plot(horizons, rmse_vals, marker="o"); ax2.set_title("RMSE per horizon")
        for ax in (ax1, ax2):
            ax.tick_params(axis="x", rotation=45); ax.grid(True, alpha=0.3)
        name = f"{label}" + (f" [{group}]" if group else "")
        plt.suptitle(name); plt.tight_layout()
        fname = f"horizon_{group}.png" if group else "horizon_metrics.png"
        plt.savefig(out_dir / fname, dpi=150); plt.close()


# ---------------------------------------------------------------------------
# Comparison across saved result CSVs
# ---------------------------------------------------------------------------

def compare_saved_runs(run_dirs: list[str], out_dir: str) -> pd.DataFrame:
    """
    Load metrics_summary.csv from multiple run directories and build a
    side-by-side comparison table.

    Returns a DataFrame indexed by model name with AGGREGATE metrics.
    Writes comparison.csv and comparison.png to out_dir.
    """
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    records = []
    for d in run_dirs:
        csv_path = Path(d) / "metrics_summary.csv"
        if not csv_path.exists():
            print(f"  [warn] no metrics_summary.csv in {d}, skipping")
            continue
        df = pd.read_csv(csv_path)
        agg = df[df["key"] == "AGGREGATE"]
        if agg.empty:
            print(f"  [warn] no AGGREGATE row in {csv_path}, skipping")
            continue
        row = agg.iloc[0].to_dict()
        row["run_dir"] = str(d)
        records.append(row)

    if not records:
        print("No valid run directories found.")
        return pd.DataFrame()

    comp_df = pd.DataFrame(records)
    # Guard: duplicate model names would produce a non-unique index and silently
    # corrupt downstream .loc lookups.  Suffix duplicates so every row is unique.
    seen: dict[str, int] = {}
    deduped = []
    for name in comp_df["model"]:
        if name in seen:
            seen[name] += 1
            deduped.append(f"{name} ({seen[name]})")
            print(f"  [warn] duplicate model name '{name}' — renamed to '{deduped[-1]}'")
        else:
            seen[name] = 0
            deduped.append(name)
    comp_df["model"] = deduped
    comp_df = comp_df.set_index("model")
    comp_df.drop(columns=["key", "pred_mode"], errors="ignore", inplace=True)
    comp_df.to_csv(out_path / "comparison.csv", float_format="%.4f")
    print(f"Comparison saved to {out_path / 'comparison.csv'}")

    _comparison_plot(comp_df, out_path)
    return comp_df


def _comparison_plot(df: pd.DataFrame, out_dir: Path) -> None:
    plot_cols = [c for c in ["MAE", "RMSE", "R2", "relative_rmse"] if c in df.columns]
    if not plot_cols:
        return
    n   = len(plot_cols)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, max(4, len(df) * 0.6 + 1)))
    if n == 1:
        axes = [axes]
    for ax, col in zip(axes, plot_cols):
        vals   = df[col].values.astype(float)
        models = df.index.tolist()
        order  = np.argsort(vals) if col != "R2" else np.argsort(vals)[::-1]
        sv, sm = vals[order], [models[i] for i in order]
        colors = ["#2ecc71" if i == 0 else "#3498db" for i in range(len(sm))]
        bars   = ax.barh(sm, sv, color=colors)
        ax.set_title(col); ax.invert_yaxis(); ax.grid(True, axis="x", alpha=0.3)
        if col == "relative_rmse":
            ax.axvline(1.0, color="black", ls="--", lw=1, label="persistence")
            ax.legend(fontsize=8)
        for bar, val in zip(bars, sv):
            ax.text(bar.get_width() * 1.01, bar.get_y() + bar.get_height() / 2,
                    f"{val:.3f}", va="center", fontsize=8)
    plt.suptitle("Model comparison", fontsize=14)
    plt.tight_layout()
    plt.savefig(out_dir / "comparison.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Comparison plot  -> {out_dir / 'comparison.png'}")