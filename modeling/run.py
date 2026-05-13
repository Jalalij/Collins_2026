"""
run.py - Unified entry point for the Throughput Prediction Framework.

Modes
-----
  train       Train one or more models, save checkpoints + metrics.
  evaluate    Evaluate a saved checkpoint (single model).
  compare     Compare metrics across multiple saved run directories.
  ablation    Feature or architecture ablation (delegates to evaluation/ablation.py).

Quick examples
--------------
  # Train Seq2Seq (default) with default 11 features predicting DL throughput:
      python run.py train --mode dl --data_root data/starlink

  # Train predicting two targets simultaneously:
      python run.py train --mode dl --data_root data/starlink \\
          --targets dish_dl_throughput_mbps ping_internet_rtt_ms_mean

  # Train multiple models and compare:
      python run.py train --mode dl --data_root data/starlink \\
          --models seq2seq lstm transformer

  # Train with all models:
      python run.py train --mode dl --data_root data/starlink --models all

  # Evaluate a saved checkpoint:
      python run.py evaluate \\
          --checkpoint results/seq2seq/metadata.json \\
          --data_root data/starlink --mode dl

  # Compare multiple existing result directories:
      python run.py compare \\
          --run_dirs results/seq2seq results/lstm results/transformer \\
          --out_dir results/comparison

  # Feature ablation:
      python run.py ablation --mode dl --data_root data/starlink \\
          --study feature --strategy leave_one_out

  # Architecture sweep:
      python run.py ablation --mode dl --data_root data/starlink \\
          --study architecture --arch_param hidden_size

  # List all registered models:
      python run.py --list_models

  # List all available features:
      python run.py --list_features

Model keys
----------
  seq2seq, lstm, transformer, gaussian_process, autoencoder

Feature selection (--features flag)
------------------------------------
  default                  11-feature default set
  all                      all 24 available features
  <name> <name> ...        explicit column names
  --feature_group <group>  one semantic group
  --feature_groups g1 g2   union of multiple groups

Target selection (--targets)
-----------------------------
  Shorthand keys:  dl, ul, iperf, rtt, snr
  Or full column:  dish_dl_throughput_mbps, ping_internet_rtt_ms_mean, ...
  Multiple targets: --targets dl rtt

Output layout
-------------
  <out_dir>/
      <model_key>/
          metadata.json          model config + checkpoint path
          metrics_summary.csv    per-key + AGGREGATE metrics
          diagnostics.csv        residual diagnostics
          plots/
              scatter.png
              horizon_metrics.png
      comparison.csv             cross-model aggregate table  (if >1 model)
      comparison.png
      run_config.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from dataset import (
    ALL_FEATURES, ALL_TARGETS, DEFAULT_FEATURES, TARGET_COL,
    DataConfig, build_dataloaders, build_experiment_datasets, MODE_SUFFIX,
)
from model  import ThroughputSeq2Seq
from models import ModelRegistry
from models.base import _register_builtins   # noqa: ensure registry populated
from evaluation.evaluate import evaluate_predictions, compare_saved_runs


# ---------------------------------------------------------------------------
# Semantic feature groups
# ---------------------------------------------------------------------------

FEATURE_GROUPS: dict[str, list[str]] = {
    "network_perf":       ["iperf_jitter_ms", "iperf_loss_pct",
                           "ping_tunnel_rtt_ms_mean", "ping_internet_rtt_ms_mean", "dish_pop_ping_ms"],
    "satellite_geometry": ["distance_km", "sat_lat", "sat_lon", "sat_alt_km"],
    "dish_state":         ["dish_snr", "dish_gps_sats", "dish_boresight_azimuth_deg",
                           "dish_boresight_elevation_deg", "dish_tilt_angle_deg", "dish_obstruction_fraction"],
    "weather":            ["weather_temp_f", "weather_humidity", "weather_pressure",
                           "weather_wind_speed", "weather_wind_dir", "weather_rain_rate"],
    "satellite_meta":     ["satellite_dtc", "satellite_handoff", "satellite_age_s"],
}


# ---------------------------------------------------------------------------
# Argument resolution helpers
# ---------------------------------------------------------------------------

def resolve_features(args) -> list[str]:
    if getattr(args, "feature_group", None) or getattr(args, "feature_groups", None):
        groups = list(getattr(args, "feature_groups", None) or [])
        if getattr(args, "feature_group", None):
            groups.append(args.feature_group)
        combined: list[str] = []
        for g in groups:
            if g not in FEATURE_GROUPS:
                raise ValueError(f"Unknown feature group '{g}'. Available: {list(FEATURE_GROUPS)}")
            combined += [f for f in FEATURE_GROUPS[g] if f not in combined]
        return [f for f in combined if f in ALL_FEATURES]

    raw = getattr(args, "features", None) or ["default"]
    if raw == ["all"]:
        return list(ALL_FEATURES)
    if raw == ["default"]:
        return list(DEFAULT_FEATURES)
    bad = [f for f in raw if f not in ALL_FEATURES]
    if bad:
        raise ValueError(f"Unknown features: {bad}. Run --list_features to see options.")
    return raw


def resolve_targets(args) -> list[str]:
    raw = getattr(args, "targets", None)
    if not raw:
        # Default: primary target for the chosen mode
        return [TARGET_COL[args.mode]]
    resolved = [TARGET_COL.get(t, t) for t in raw]
    bad = [t for t in resolved if t not in ALL_TARGETS]
    if bad:
        raise ValueError(
            f"Unknown target column(s): {bad}\n"
            f"Available: {ALL_TARGETS}\n"
            f"Shorthand keys: {list(TARGET_COL)}"
        )
    return resolved


def resolve_models(args) -> list[str]:
    raw = getattr(args, "models", None) or ["seq2seq"]
    if raw == ["all"]:
        return ModelRegistry.keys()
    bad = [m for m in raw if m not in ModelRegistry]
    if bad:
        raise ValueError(f"Unknown model(s): {bad}. Available: {ModelRegistry.keys()}")
    return raw


# ---------------------------------------------------------------------------
# Checkpoint re-evaluation helper (used by ablation.py too)
# ---------------------------------------------------------------------------

def _eval_from_checkpoint(
    ckpt_path, data_root, mode, targets, features,
    input_len, output_len, hidden_size, embed_dim,
    normalise, batch_size, seed,
) -> dict:
    """Load a saved checkpoint and collect predictions for evaluation."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg = DataConfig(
        data_root=data_root, mode=mode, targets=targets,
        features=features, input_len=input_len, output_len=output_len,
        normalise=normalise, batch_size=batch_size, seed=seed,
    )
    _, _, test_loader, _, target_stats, feature_cols = build_dataloaders(cfg)
    if test_loader is None:
        raise ValueError("No test windows available.")

    model = ThroughputSeq2Seq(
        num_features=len(feature_cols), embed_dim=embed_dim,
        hidden_size=hidden_size, input_len=input_len,
        output_len=output_len, num_targets=len(targets),
    )
    model.load_state_dict(torch.load(ckpt_path, map_location=device, weights_only=True))
    model = model.to(device).eval()

    all_preds, all_targets_list = [], []
    with torch.no_grad():
        for feat, hour, dow, tgt in test_loader:
            preds, _ = model(feat.to(device), hour.to(device), dow.to(device))
            all_preds.append(preds.cpu())
            all_targets_list.append(tgt.cpu())

    preds_t   = torch.cat(all_preds,          dim=0)
    targets_t = torch.cat(all_targets_list,   dim=0)
    num_targets = len(targets)

    def denorm(arr, col):
        stats = target_stats.get(col, {})
        if not stats:
            return arr.numpy()
        if normalise == "zscore":
            return (arr.float() * stats["std"] + stats["mean"]).numpy()
        return (arr.float() * (stats["max"] - stats["min"]) + stats["min"]).numpy()

    def _short(col):
        return col.replace("dish_","").replace("_throughput_mbps","").replace("_ms_mean","_ms")

    y_true_dict, y_pred_dict = {}, {}
    if num_targets == 1:
        col = targets[0]
        short = _short(col)
        p_np = denorm(preds_t,   col)
        t_np = denorm(targets_t, col)
        H = p_np.shape[1]
        for h in range(H):
            key = f"t+{h+1}s" if num_targets == 1 else f"{short}_t+{h+1}s"
            y_true_dict[key] = t_np[:, h]
            y_pred_dict[key] = p_np[:, h]
    else:
        H = preds_t.shape[1]
        for ti, col in enumerate(targets):
            short = _short(col)
            p_np = denorm(preds_t[:, :, ti],   col)
            t_np = denorm(targets_t[:, :, ti], col)
            for h in range(H):
                key = f"{short}_t+{h+1}s"
                y_true_dict[key] = t_np[:, h]
                y_pred_dict[key] = p_np[:, h]

    return {
        "model_name":  "Seq2Seq-GRU",
        "y_true_dict": y_true_dict,
        "y_pred_dict": y_pred_dict,
        "metadata": {"checkpoint": ckpt_path, "targets": targets, "features": feature_cols},
    }


# ---------------------------------------------------------------------------
# Per-model train + evaluate
# ---------------------------------------------------------------------------

def run_one_model(model_key, feature_cols, targets, args, out_dir) -> dict | None:
    entry      = ModelRegistry.get(model_key)
    trainer    = entry["cls"]
    def_config = entry["default_config"].copy()

    for key in ("hidden_size", "embed_dim", "dropout", "batch_size"):
        val = getattr(args, key, None)
        if val is not None:
            def_config[key] = val
    def_config["epochs"]     = args.epochs
    def_config["lr"]         = args.lr
    def_config["input_len"]  = args.input_len
    def_config["output_len"] = args.output_len

    normalise = args.normalise if args.normalise != "none" else None
    model_out = out_dir / model_key
    model_out.mkdir(parents=True, exist_ok=True)

    cfg = DataConfig(
        data_root  = args.data_root,
        mode       = args.mode,
        targets    = targets,
        features   = feature_cols,
        input_len  = args.input_len,
        output_len = args.output_len,
        normalise  = normalise,
        batch_size = def_config.get("batch_size", 512),
        train_frac = args.train_frac,
        val_frac   = args.val_frac,
        seed       = args.seed,
    )

    try:
        if model_key == "seq2seq":
            train_loader, val_loader, test_loader, _, _, _ = build_dataloaders(cfg)
            train_set = train_loader.dataset
            test_set  = test_loader.dataset if test_loader else train_loader.dataset
        else:
            train_set, test_set, _, _, _ = build_experiment_datasets(cfg)

        result = trainer.train_and_evaluate(
            train_set  = train_set,
            test_set   = test_set,
            results_dir= str(model_out),
            config     = def_config,
        )
    except Exception as exc:
        print(f"\n  [ERROR] {model_key} failed: {exc}")
        import traceback; traceback.print_exc()
        return None

    # Evaluate predictions
    metrics_df = evaluate_predictions(result, str(model_out))

    # Save metadata
    meta = result.get("metadata", {})
    meta["model_key"]   = model_key
    meta["targets"]     = targets
    meta["features"]    = feature_cols
    meta["out_dir"]     = str(model_out)
    with open(model_out / "metadata.json", "w") as f:
        json.dump(meta, f, indent=2, default=str)

    agg = metrics_df[metrics_df["key"] == "AGGREGATE"]
    if agg.empty:
        return None
    summary = agg.iloc[0].to_dict()
    summary["model"]     = result.get("model_name", model_key)
    summary["model_key"] = model_key
    return summary


# ---------------------------------------------------------------------------
# Subcommand handlers
# ---------------------------------------------------------------------------

def cmd_train(args):
    try:
        feature_cols = resolve_features(args)
        targets      = resolve_targets(args)
        model_keys   = resolve_models(args)
    except ValueError as e:
        print(f"[ERROR] {e}"); sys.exit(1)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 65)
    print("THROUGHPUT PREDICTION FRAMEWORK — TRAIN")
    print("=" * 65)
    print(f"  Data root : {args.data_root}")
    print(f"  Mode      : {args.mode}")
    print(f"  Targets   : {targets}")
    print(f"  Features  : {len(feature_cols)}  {feature_cols}")
    print(f"  Models    : {model_keys}")
    print(f"  Out dir   : {out_dir}")
    print("=" * 65)

    with open(out_dir / "run_config.json", "w") as f:
        cfg_dict = vars(args).copy()
        cfg_dict["resolved_features"] = feature_cols
        cfg_dict["resolved_targets"]  = targets
        cfg_dict["resolved_models"]   = model_keys
        json.dump(cfg_dict, f, indent=2, default=str)

    summaries = []
    for model_key in model_keys:
        print(f"\n{'#'*65}\n  MODEL: {model_key.upper()}\n{'#'*65}")
        result = run_one_model(model_key, feature_cols, targets, args, out_dir)
        if result is not None:
            summaries.append(result)

    if len(summaries) > 1:
        print(f"\n{'='*65}\nCROSS-MODEL COMPARISON\n{'='*65}")
        comp_df = pd.DataFrame(summaries).set_index("model")
        disp = [c for c in ["MAE","RMSE","R2","PICP_95"] if c in comp_df.columns]
        print(comp_df[disp].to_string(float_format="%.4f"))
        compare_saved_runs([str(Path(args.out_dir) / m["model_key"]) for m in summaries],
                           out_dir=args.out_dir)

    print(f"\nAll results in: {out_dir}/")


def cmd_evaluate(args):
    """Re-evaluate an existing checkpoint."""
    meta_path = Path(args.checkpoint)
    if meta_path.suffix == ".json":
        with open(meta_path) as f:
            meta = json.load(f)
        ckpt = meta.get("checkpoint", "")
        targets      = meta.get("targets", [TARGET_COL[args.mode]])
        features     = meta.get("features", list(DEFAULT_FEATURES))
        hidden_size  = meta.get("hidden_size", 128)
        embed_dim    = meta.get("embed_dim", 16)
    else:
        ckpt         = str(meta_path)
        targets      = resolve_targets(args)
        features     = resolve_features(args)
        hidden_size  = args.hidden_size
        embed_dim    = args.embed_dim

    normalise = args.normalise if args.normalise != "none" else None
    result = _eval_from_checkpoint(
        ckpt_path=ckpt, data_root=args.data_root, mode=args.mode,
        targets=targets, features=features,
        input_len=args.input_len, output_len=args.output_len,
        hidden_size=hidden_size, embed_dim=embed_dim,
        normalise=normalise, batch_size=args.batch_size, seed=args.seed,
    )
    evaluate_predictions(result, args.out_dir)
    print(f"\nEvaluation results -> {args.out_dir}/")


def cmd_compare(args):
    """Load and compare metrics from multiple saved run dirs."""
    df = compare_saved_runs(args.run_dirs, args.out_dir)
    if not df.empty:
        print(df.to_string(float_format="%.4f"))


def cmd_ablation(args):
    """Delegate to evaluation/ablation.py main() passing through args."""
    # Rebuild argv and re-run ablation main
    abl_argv = ["ablation"]
    for k, v in vars(args).items():
        if k in ("subcommand",):
            continue
        if v is None or v is False:
            continue
        flag = f"--{k}"
        if isinstance(v, bool):
            abl_argv.append(flag)
        elif isinstance(v, list):
            abl_argv.extend([flag] + [str(x) for x in v])
        else:
            abl_argv.extend([flag, str(v)])

    sys.argv = abl_argv
    from evaluation.ablation import main as ablation_main
    ablation_main()


# ---------------------------------------------------------------------------
# Argument parser
# ---------------------------------------------------------------------------

def _add_data_args(p):
    p.add_argument("--data_root",  default="data/starlink")
    p.add_argument("--mode",       default="dl", choices=list(MODE_SUFFIX))
    p.add_argument("--targets",    nargs="+", default=None, metavar="TARGET",
                   help="Target column(s) to predict. Shorthand: dl, ul, rtt, snr. "
                        "Full names also accepted. Default: primary target for --mode.")
    p.add_argument("--input_len",  type=int, default=30)
    p.add_argument("--output_len", type=int, default=15)
    p.add_argument("--normalise",  default="zscore", choices=["zscore","minmax","none"])
    p.add_argument("--train_frac", type=float, default=0.70)
    p.add_argument("--val_frac",   type=float, default=0.15)
    p.add_argument("--batch_size", type=int,   default=None)
    p.add_argument("--seed",       type=int,   default=42)

def _add_feature_args(p):
    fg = p.add_mutually_exclusive_group()
    fg.add_argument("--features", nargs="+", default=["default"], metavar="FEATURE",
                    help="'default', 'all', or explicit column names.")
    fg.add_argument("--feature_group", default=None, choices=list(FEATURE_GROUPS))
    fg.add_argument("--feature_groups", nargs="+", default=None, choices=list(FEATURE_GROUPS))

def _add_model_hparam_args(p):
    p.add_argument("--hidden_size", type=int,   default=128)
    p.add_argument("--embed_dim",   type=int,   default=16)
    p.add_argument("--dropout",     type=float, default=None)
    p.add_argument("--epochs",      type=int,   default=100)
    p.add_argument("--lr",          type=float, default=1e-3)


def build_parser():
    parser = argparse.ArgumentParser(
        description="Throughput Prediction Framework",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument("--list_models",   action="store_true",
                        help="Print all registered models and exit.")
    parser.add_argument("--list_features", action="store_true",
                        help="Print all available features and groups, then exit.")
    parser.add_argument("--list_targets",  action="store_true",
                        help="Print all available target columns, then exit.")

    subs = parser.add_subparsers(dest="subcommand")

    # ---- train ----
    tr = subs.add_parser("train", help="Train one or more models.")
    tr.add_argument("--models", nargs="+", default=["seq2seq"], metavar="MODEL",
                    help="Model keys (or 'all'). Available: seq2seq, lstm, transformer, "
                         "gaussian_process, autoencoder")
    _add_data_args(tr)
    _add_feature_args(tr)
    _add_model_hparam_args(tr)
    tr.add_argument("--out_dir", default="results")

    # ---- evaluate ----
    ev = subs.add_parser("evaluate", help="Evaluate a saved checkpoint.")
    ev.add_argument("--checkpoint", required=True,
                    help="Path to metadata.json or best_model.pt")
    _add_data_args(ev)
    _add_feature_args(ev)
    ev.add_argument("--hidden_size", type=int,   default=128)
    ev.add_argument("--embed_dim",   type=int,   default=16)
    ev.add_argument("--out_dir",     default="eval_results")

    # ---- compare ----
    cp = subs.add_parser("compare",
                         help="Compare metrics across multiple saved run directories.")
    cp.add_argument("--run_dirs", nargs="+", required=True,
                    help="Paths to run output directories (each must have metrics_summary.csv).")
    cp.add_argument("--out_dir", default="results/comparison")

    # ---- ablation ----
    ab = subs.add_parser("ablation", help="Feature or architecture ablation study.")
    ab.add_argument("--data_root",      default="data/starlink")
    ab.add_argument("--mode",           default="dl", choices=list(MODE_SUFFIX))
    ab.add_argument("--targets",        nargs="+", default=None)
    ab.add_argument("--normalise",      default="zscore", choices=["zscore","minmax","none"])
    ab.add_argument("--seed",           type=int, default=42)
    ab.add_argument("--study",          default="feature", choices=["feature","architecture"])
    ab.add_argument("--strategy",       default="leave_one_out",
                    choices=["leave_one_out","add_one","groups","custom"])
    ab.add_argument("--base_features",  nargs="+", default=None, choices=ALL_FEATURES)
    ab.add_argument("--feature_sets",   nargs="+", default=None)
    ab.add_argument("--arch_param",     default="hidden_size", choices=["hidden_size","input_len","output_len","embed_dim"])
    ab.add_argument("--epochs",         type=int, default=30)
    ab.add_argument("--batch_size",     type=int, default=512)
    ab.add_argument("--input_len",      type=int, default=30)
    ab.add_argument("--output_len",     type=int, default=15)
    ab.add_argument("--hidden_size",    type=int, default=128)
    ab.add_argument("--embed_dim",      type=int, default=16)
    ab.add_argument("--checkpoint_dir", default="checkpoints/ablation")
    ab.add_argument("--out_dir",        default="ablation_results")
    ab.add_argument("--skip_existing",  action="store_true")

    return parser


# ---------------------------------------------------------------------------
# Listing helpers
# ---------------------------------------------------------------------------

def print_models():
    print("\nRegistered models:")
    print(f"  {'Key':<22} Description")
    print("  " + "-" * 60)
    for entry in ModelRegistry.list_models():
        print(f"  {entry['key']:<22} {entry['description']}")
    print("\nUsage: python run.py train --models seq2seq lstm ...")

def print_features():
    print("\nAll available features (24):")
    for i, f in enumerate(ALL_FEATURES, 1):
        print(f"  {i:2d}. {f}")
    print("\nDefault 11-feature set:")
    for f in DEFAULT_FEATURES:
        print(f"  • {f}")
    print("\nSemantic groups:")
    for g, feats in FEATURE_GROUPS.items():
        print(f"  {g}: {feats}")

def print_targets():
    print("\nAvailable prediction targets:")
    print(f"  {'Shorthand':<10} Full column name")
    print("  " + "-" * 50)
    for short, col in TARGET_COL.items():
        print(f"  {short:<10} {col}")
    print("\nAny column in ALL_TARGETS can also be used by full name:")
    for col in ALL_TARGETS:
        if col not in TARGET_COL.values():
            print(f"  {col}")
    print("\nUsage: python run.py train --targets dl rtt")
    print("       python run.py train --targets dish_dl_throughput_mbps ping_internet_rtt_ms_mean")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = build_parser()
    args   = parser.parse_args()

    if args.list_models:
        print_models(); sys.exit(0)
    if args.list_features:
        print_features(); sys.exit(0)
    if getattr(args, "list_targets", False):
        print_targets(); sys.exit(0)

    if args.subcommand is None:
        parser.print_help()
        sys.exit(0)

    dispatch = {
        "train":    cmd_train,
        "evaluate": cmd_evaluate,
        "compare":  cmd_compare,
        "ablation": cmd_ablation,
    }
    dispatch[args.subcommand](args)


if __name__ == "__main__":
    main()
