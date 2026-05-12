"""
ablation.py - Ablation study for the Throughput Seq2Seq model.

Supports two types of ablation, selectable via --study:

1. feature  (default)
   Trains and evaluates one model per feature subset.
   Built-in strategies:
     "leave_one_out"  - remove one feature at a time from DEFAULT_FEATURES
     "add_one"        - start from best single feature, add one at a time
     "groups"         - remove entire semantic groups (network / geometry / weather / dish)
     "custom"         - use --feature_sets to supply explicit subsets (see below)

2. architecture
   Trains models with different hyperparameters to measure sensitivity:
     hidden_size  : [64, 128, 256]
     input_len    : [10, 20, 30, 60]
     output_len   : [5, 10, 15]
     embed_dim    : [0, 8, 16, 32]   (0 = no periodic embedding)
     num_layers   : [1, 2, 3]         (requires a small model.py change, stubbed here)

All results are aggregated into a single CSV and a bar chart per metric.

Usage
-----
# Leave-one-out feature ablation on DL:
    python ablation.py --mode dl --data_root data/starlink \
        --study feature --strategy leave_one_out \
        --checkpoint_dir checkpoints/ablation \
        --out_dir ablation_results/dl_leave_one_out

# Group ablation:
    python ablation.py --mode dl --data_root data/starlink \
        --study feature --strategy groups

# Custom explicit feature subsets (each subset separated by |):
    python ablation.py --mode dl --data_root data/starlink \
        --study feature --strategy custom \
        --feature_sets "iperf_jitter_ms,dish_snr" \
                       "iperf_jitter_ms,dish_snr,distance_km"

# Architecture ablation over hidden sizes:
    python ablation.py --mode dl --data_root data/starlink \
        --study architecture --arch_param hidden_size

# Skip retraining if checkpoint already exists (useful for resuming):
    python ablation.py ... --skip_existing
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import shutil
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from dataset import (
    ALL_FEATURES, DEFAULT_FEATURES, MODE_SUFFIX, DataConfig, build_dataloaders,
)
from model  import ThroughputSeq2Seq
from train  import train as run_training
from evaluate import evaluate as run_evaluate


# ---------------------------------------------------------------------------
# Semantic feature groups for group ablation
# ---------------------------------------------------------------------------
FEATURE_GROUPS: dict[str, list[str]] = {
    "network_perf": [
        "iperf_jitter_ms", "iperf_loss_pct",
        "ping_tunnel_rtt_ms_mean", "ping_internet_rtt_ms_mean", "dish_pop_ping_ms",
    ],
    "satellite_geometry": [
        "distance_km", "sat_lat", "sat_lon", "sat_alt_km",
    ],
    "dish_state": [
        "dish_snr", "dish_gps_sats", "dish_boresight_azimuth_deg",
        "dish_boresight_elevation_deg", "dish_tilt_angle_deg", "dish_obstruction_fraction",
    ],
    "weather": [
        "weather_temp_f", "weather_humidity", "weather_pressure",
        "weather_wind_speed", "weather_wind_dir", "weather_rain_rate",
    ],
    "satellite_meta": [
        "satellite_dtc", "satellite_handoff", "satellite_age_s",
    ],
}

# Architecture sweep grid
ARCH_GRIDS: dict[str, list[Any]] = {
    "hidden_size": [64, 128, 256],
    "input_len":   [10, 20, 30, 60],
    "output_len":  [5, 10, 15],
    "embed_dim":   [0, 8, 16, 32],
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def safe_name(s: str) -> str:
    """Make a string filesystem-safe."""
    return s.replace(" ", "_").replace(",", "-").replace("/", "-")[:80]


def build_feature_sets(
    strategy: str,
    base_features: list[str],
    custom_sets: list[str] | None,
) -> list[tuple[str, list[str]]]:
    """
    Returns a list of (label, feature_list) tuples.
    """
    sets: list[tuple[str, list[str]]] = []

    if strategy == "leave_one_out":
        # Full set first as baseline
        sets.append(("ALL_FEATURES", list(base_features)))
        for feat in base_features:
            subset = [f for f in base_features if f != feat]
            sets.append((f"drop_{feat}", subset))

    elif strategy == "add_one":
        # Incrementally add features one by one (order = DEFAULT_FEATURES order)
        for i in range(1, len(base_features) + 1):
            subset = base_features[:i]
            sets.append((f"top{i}_{subset[-1]}", list(subset)))

    elif strategy == "groups":
        # Full set first
        sets.append(("ALL_FEATURES", list(base_features)))
        for group_name, group_feats in FEATURE_GROUPS.items():
            # Only remove features that are actually in base_features
            to_remove = set(group_feats) & set(base_features)
            if not to_remove:
                continue
            subset = [f for f in base_features if f not in to_remove]
            if len(subset) == 0:
                continue
            sets.append((f"drop_group_{group_name}", subset))

    elif strategy == "custom":
        if not custom_sets:
            raise ValueError("--strategy custom requires --feature_sets")
        for raw in custom_sets:
            feats = [f.strip() for f in raw.split(",") if f.strip()]
            bad = [f for f in feats if f not in ALL_FEATURES]
            if bad:
                raise ValueError(f"Unknown features in custom set: {bad}")
            sets.append((safe_name(",".join(feats)), feats))

    else:
        raise ValueError(f"Unknown feature strategy: {strategy}")

    return sets


# ---------------------------------------------------------------------------
# Single training + evaluation run
# ---------------------------------------------------------------------------

def run_one(
    label: str,
    data_root: str,
    mode: str,
    features: list[str],
    out_dir: Path,
    checkpoint_dir: Path,
    epochs: int,
    batch_size: int,
    hidden_size: int,
    embed_dim: int,
    input_len: int,
    output_len: int,
    normalise: str,
    seed: int,
    skip_existing: bool,
) -> dict:
    """Train one model variant and evaluate it. Returns aggregate metric dict."""

    ckpt_path   = checkpoint_dir / safe_name(label) / "best_model.pt"
    eval_subdir = out_dir / safe_name(label)

    print(f"\n{'='*60}")
    print(f"  Variant : {label}")
    print(f"  Features: {len(features)}  {features}")
    print(f"  hidden={hidden_size}  embed={embed_dim}  "
          f"in={input_len}  out={output_len}")
    print(f"{'='*60}")

    # ---- Train ----
    if skip_existing and ckpt_path.exists():
        print(f"  [skip] checkpoint exists: {ckpt_path}")
    else:
        ckpt_path.parent.mkdir(parents=True, exist_ok=True)
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
        train_loader, val_loader, test_loader, _, _, _ = build_dataloaders(cfg)

        model = ThroughputSeq2Seq(
            num_features = len(features),
            embed_dim    = embed_dim,
            hidden_size  = hidden_size,
            input_len    = input_len,
            output_len   = output_len,
        )
        run_training(
            model,
            train_loader, val_loader, test_loader,
            num_epochs = epochs,
            save_dir   = str(ckpt_path.parent),
        )

    # ---- Evaluate ----
    metrics_df = run_evaluate(
        checkpoint  = str(ckpt_path),
        data_root   = data_root,
        mode        = mode,
        features    = features,
        out_dir     = str(eval_subdir),
        input_len   = input_len,
        output_len  = output_len,
        hidden_size = hidden_size,
        embed_dim   = embed_dim,
        normalise   = normalise,
        batch_size  = batch_size,
        seed        = seed,
    )

    # Pull aggregate row
    agg = metrics_df[metrics_df["horizon"] == "AGGREGATE"].iloc[0]
    result = {
        "label":       label,
        "n_features":  len(features),
        "features":    ",".join(features),
        "hidden_size": hidden_size,
        "embed_dim":   embed_dim,
        "input_len":   input_len,
        "output_len":  output_len,
        "MAE":         float(agg["MAE"]),
        "RMSE":        float(agg["RMSE"]),
        "MAPE_pct":    float(agg["MAPE_pct"]),
        "R2":          float(agg["R2"]),
    }
    return result


# ---------------------------------------------------------------------------
# Plotting ablation results
# ---------------------------------------------------------------------------

def plot_ablation(results_df: pd.DataFrame, out_dir: Path, study: str) -> None:
    metrics = ["MAE", "RMSE", "MAPE_pct", "R2"]
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.ravel()

    for ax, metric in zip(axes, metrics):
        vals   = results_df[metric].values
        labels = results_df["label"].values
        colors = ["steelblue" if i > 0 else "tomato" for i in range(len(labels))]
        bars   = ax.barh(labels, vals, color=colors)
        ax.set_title(metric)
        ax.set_xlabel(metric)
        ax.invert_yaxis()
        ax.grid(True, axis="x", alpha=0.3)
        # Annotate bars
        for bar, val in zip(bars, vals):
            ax.text(bar.get_width() * 1.01, bar.get_y() + bar.get_height() / 2,
                    f"{val:.3f}", va="center", fontsize=7)

    title = "Feature ablation" if study == "feature" else "Architecture ablation"
    plt.suptitle(title, fontsize=14)
    plt.tight_layout()
    plt.savefig(out_dir / "ablation_summary.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved ablation chart → {out_dir / 'ablation_summary.png'}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser(description="Ablation study for Throughput Seq2Seq")

    # Data
    p.add_argument("--data_root",       default="data/starlink")
    p.add_argument("--mode",            default="dl", choices=list(MODE_SUFFIX))
    p.add_argument("--normalise",       default="zscore", choices=["zscore","minmax","none"])
    p.add_argument("--seed",            type=int, default=42)

    # Study type
    p.add_argument("--study",           default="feature",
                   choices=["feature", "architecture"],
                   help="'feature' ablates input features; 'architecture' sweeps hyperparams")

    # Feature ablation options
    p.add_argument("--strategy",        default="leave_one_out",
                   choices=["leave_one_out", "add_one", "groups", "custom"],
                   help="Feature selection strategy (used when --study=feature)")
    p.add_argument("--base_features",   nargs="+", default=None, choices=ALL_FEATURES,
                   metavar="FEATURE",
                   help="Base feature set to ablate from. Default: DEFAULT_FEATURES (11)")
    p.add_argument("--feature_sets",    nargs="+", default=None,
                   metavar="FEAT,FEAT,...",
                   help="For --strategy custom: comma-separated feature sets, one per argument")

    # Architecture ablation options
    p.add_argument("--arch_param",      default="hidden_size",
                   choices=list(ARCH_GRIDS),
                   help="Which architectural parameter to sweep (--study=architecture)")

    # Shared training config
    p.add_argument("--epochs",          type=int, default=30,
                   help="Epochs per variant (lower than full training is fine for ablation)")
    p.add_argument("--batch_size",      type=int, default=512)
    p.add_argument("--input_len",       type=int, default=30)
    p.add_argument("--output_len",      type=int, default=15)
    p.add_argument("--hidden_size",     type=int, default=128)
    p.add_argument("--embed_dim",       type=int, default=16)

    # Output
    p.add_argument("--checkpoint_dir",  default="checkpoints/ablation")
    p.add_argument("--out_dir",         default="ablation_results")
    p.add_argument("--skip_existing",   action="store_true",
                   help="Skip training if a checkpoint already exists for this variant")

    args = p.parse_args()

    out_dir  = Path(args.out_dir)
    ckpt_dir = Path(args.checkpoint_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    normalise = args.normalise if args.normalise != "none" else None

    base_features = args.base_features if args.base_features else list(DEFAULT_FEATURES)

    # ---------------------------------------------------------------- build variants
    variants: list[tuple[str, dict]] = []   # (label, overrides)

    if args.study == "feature":
        feature_sets = build_feature_sets(args.strategy, base_features, args.feature_sets)
        for label, feats in feature_sets:
            variants.append((label, {"features": feats,
                                     "hidden_size": args.hidden_size,
                                     "embed_dim":   args.embed_dim,
                                     "input_len":   args.input_len,
                                     "output_len":  args.output_len}))

    elif args.study == "architecture":
        param  = args.arch_param
        values = ARCH_GRIDS[param]
        for val in values:
            label = f"{param}={val}"
            overrides = {
                "features":    base_features,
                "hidden_size": args.hidden_size,
                "embed_dim":   args.embed_dim,
                "input_len":   args.input_len,
                "output_len":  args.output_len,
            }
            overrides[param] = val
            # embed_dim=0 means no periodic embedding — handled gracefully by model
            if param == "embed_dim" and val == 0:
                overrides["embed_dim"] = 0
            variants.append((label, overrides))

    print(f"\nAblation study : {args.study}")
    print(f"Total variants : {len(variants)}")

    # ---------------------------------------------------------------- run all variants
    all_results: list[dict] = []
    for label, overrides in variants:
        try:
            result = run_one(
                label          = label,
                data_root      = args.data_root,
                mode           = args.mode,
                features       = overrides["features"],
                out_dir        = out_dir,
                checkpoint_dir = ckpt_dir,
                epochs         = args.epochs,
                batch_size     = args.batch_size,
                hidden_size    = overrides["hidden_size"],
                embed_dim      = overrides["embed_dim"],
                input_len      = overrides["input_len"],
                output_len     = overrides["output_len"],
                normalise      = normalise,
                seed           = args.seed,
                skip_existing  = args.skip_existing,
            )
            all_results.append(result)
        except Exception as exc:
            print(f"  [ERROR] variant '{label}' failed: {exc}")
            all_results.append({"label": label, "MAE": np.nan, "RMSE": np.nan,
                                 "MAPE_pct": np.nan, "R2": np.nan})

    # ---------------------------------------------------------------- summarise
    results_df = pd.DataFrame(all_results)
    results_path = out_dir / "ablation_results.csv"
    results_df.to_csv(results_path, index=False, float_format="%.4f")
    print(f"\nResults saved → {results_path}")

    print("\n" + "="*60)
    print("ABLATION SUMMARY (aggregate MAE / RMSE / R2)")
    print("="*60)
    print(results_df[["label","MAE","RMSE","MAPE_pct","R2"]].to_string(index=False))

    plot_ablation(results_df, out_dir, args.study)

    # Save config for reproducibility
    with open(out_dir / "ablation_config.json", "w") as f:
        json.dump(vars(args), f, indent=2)

    print(f"\nAll outputs in: {out_dir}/")


if __name__ == "__main__":
    main()
