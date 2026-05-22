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
  # Train Seq2Seq on Starlink throughput data (DL default target):
      python run.py train --provider starlink --experiment_type throughput_dl

  # Train predicting two targets simultaneously:
      python run.py train --provider starlink --experiment_type throughput_dl \\
          --targets dish_dl_throughput_mbps ping_tunnel_rtt_ms

  # Train multiple models:
      python run.py train --provider starlink --experiment_type throughput_dl \\
          --models seq2seq lstm transformer

  # Train all models:
      python run.py train --provider starlink --experiment_type throughput_dl --models all

  # OneWeb downlink, default target:
      python run.py train --provider oneweb --experiment_type dl

  # Quectel latency+loss+throughput, two targets:
      python run.py train --provider quectel --experiment_type latency_loss_throughput_dl \\
          --targets iperf sinr

  # Evaluate a saved checkpoint:
      python run.py evaluate \\
          --checkpoint results/seq2seq/metadata.json \\
          --provider starlink --experiment_type throughput_dl

  # Compare multiple existing result directories:
      python run.py compare \\
          --run_dirs results/seq2seq results/lstm results/transformer \\
          --out_dir results/comparison

  # Feature ablation:
      python run.py ablation \\
          --provider starlink --experiment_type throughput_dl \\
          --study feature --strategy leave_one_out

  # Architecture sweep:
      python run.py ablation \\
          --provider starlink --experiment_type throughput_dl \\
          --study architecture --arch_param hidden_size

  # List all registered models:
      python run.py --list_models

  # List all available features (per provider):
      python run.py --list_features --provider starlink
      python run.py --list_features --provider oneweb
      python run.py --list_features --provider quectel

  # List all available targets (per provider):
      python run.py --list_targets --provider starlink

Model keys
----------
  seq2seq, lstm, transformer, gaussian_process, autoencoder

Feature selection (--features flag)
------------------------------------
  default                  Provider-specific default feature set
  all                      All features for this provider
  <name> <name> ...        Explicit column names
  --feature_group <group>  One semantic group (Starlink only)
  --feature_groups g1 g2   Union of multiple groups (Starlink only)

Target selection (--targets)
-----------------------------
  Shorthand keys vary by provider — run --list_targets --provider <p> to see.
  Multiple targets: --targets dl rtt

Output layout
-------------
  <out_dir>/
      <model_key>/
          metadata.json
          metrics_summary.csv
          diagnostics.csv
          plots/
              scatter.png
              horizon_metrics.png
      comparison.csv
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

from models import ModelRegistry
from models.base import _register_builtins

from evaluation.evaluate import evaluate_predictions, compare_saved_runs

from dataset import (
    # Provider constants
    PROVIDERS, PROVIDER_STARLINK, PROVIDER_ONEWEB, PROVIDER_QUECTEL,
    VALID_EXPERIMENT_TYPES,

    # Per-provider feature / target registries
    ALL_FEATURES,
    DEFAULT_FEATURES,
    ALL_TARGETS,
    TARGET_ALIASES,
    DEFAULT_TARGET,

    # ADD THESE
    ALL_FEATURES_BY_EXP,
    DEFAULT_FEATURES_BY_EXP,

    # Feature groups + flat name list
    FEATURE_GROUPS, ALL_FEATURE_NAMES,

    # Config + builders
    DataConfig, build_dataloaders, build_experiment_datasets,
)

# ---------------------------------------------------------------------------
# Legacy experiment compatibility
# ---------------------------------------------------------------------------

LEGACY_EXPERIMENT_MAP = {
    "throughput": "throughput_dl",
    "latency_loss_throughput": "latency_loss_throughput_dl",
}


# ---------------------------------------------------------------------------
# Argument resolution helpers
# ---------------------------------------------------------------------------

def resolve_features(args) -> list[str]:
    provider = getattr(args, "provider", PROVIDER_STARLINK)
    exp_type = getattr(args, "experiment_type", "throughput_dl")

    exp_type = LEGACY_EXPERIMENT_MAP.get(exp_type, exp_type)

    if getattr(args, "feature_group", None) or getattr(args, "feature_groups", None):
        groups = list(getattr(args, "feature_groups", None) or [])
        if getattr(args, "feature_group", None):
            groups.append(args.feature_group)

        combined: list[str] = []
        for g in groups:
            if g not in FEATURE_GROUPS:
                raise ValueError(
                    f"Unknown feature group '{g}'. Available: {list(FEATURE_GROUPS)}"
                )
            combined += [f for f in FEATURE_GROUPS[g] if f not in combined]

        exp_features = ALL_FEATURES_BY_EXP[(provider, exp_type)]
        return [f for f in combined if f in exp_features]

    raw = getattr(args, "features", None) or ["default"]

    if raw == ["all"]:
        return list(ALL_FEATURES_BY_EXP[(provider, exp_type)])

    if raw == ["default"]:
        return list(DEFAULT_FEATURES_BY_EXP[(provider, exp_type)])

    exp_features = ALL_FEATURES_BY_EXP[(provider, exp_type)]

    bad = [f for f in raw if f not in exp_features]
    if bad:
        raise ValueError(
            f"Unknown features for provider '{provider}' "
            f"experiment_type '{exp_type}': {bad}\n"
            f"Run:\n"
            f"  python run.py --list_features --provider {provider}"
        )

    return raw


def resolve_targets(args) -> list[str]:
    provider = getattr(args, "provider", PROVIDER_STARLINK)

    exp_type = getattr(args, "experiment_type", "throughput_dl")
    exp_type = LEGACY_EXPERIMENT_MAP.get(exp_type, exp_type)

    aliases     = TARGET_ALIASES[provider]
    all_targets = ALL_TARGETS[provider]

    raw = getattr(args, "targets", None)

    if not raw:
        return [DEFAULT_TARGET[(provider, exp_type)]]

    resolved = [aliases.get(t, t) for t in raw]

    bad = [t for t in resolved if t not in all_targets]

    if bad:
        raise ValueError(
            f"Unknown target column(s) for provider '{provider}': {bad}\n"
            f"Available: {all_targets}\n"
            f"Shorthand keys: {list(aliases)}"
        )

    return resolved


def resolve_models(args) -> list[str]:
    raw = getattr(args, "models", None) or ["seq2seq"]
    if raw == ["all"]:
        return ModelRegistry.keys()
    bad = [m for m in raw if m not in ModelRegistry]
    if bad:
        raise ValueError(
            f"Unknown model(s): {bad}. Available: {ModelRegistry.keys()}"
        )
    return raw


# ---------------------------------------------------------------------------
# Checkpoint re-evaluation helper (used by ablation.py too)
# ---------------------------------------------------------------------------

def _eval_from_checkpoint(
    ckpt_path,
    data_root, provider, experiment_type,
    targets, features,
    input_len, output_len,
    hidden_size, embed_dim,
    normalise, batch_size, seed,
) -> dict:
    """Load a saved checkpoint and collect predictions for evaluation."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg = DataConfig(
        data_root       = data_root,
        provider        = provider,
        experiment_type = experiment_type,
        targets         = targets,
        features        = features,
        input_len       = input_len,
        output_len      = output_len,
        normalise       = normalise,
        batch_size      = batch_size,
        seed            = seed,
    )
    _, _, test_loader, _, target_stats, feature_cols = build_dataloaders(cfg)
    if test_loader is None:
        raise ValueError("No test windows available.")

    model = ThroughputSeq2Seq(
        num_features = len(feature_cols),
        embed_dim    = embed_dim,
        hidden_size  = hidden_size,
        input_len    = input_len,
        output_len   = output_len,
        num_targets  = len(targets),
    )
    model.load_state_dict(
        torch.load(ckpt_path, map_location=device, weights_only=True)
    )
    model = model.to(device).eval()

    all_preds, all_targets_list = [], []
    with torch.no_grad():
        for feat, hour, dow, tgt in test_loader:
            preds, _ = model(feat.to(device), hour.to(device), dow.to(device))
            all_preds.append(preds.cpu())
            all_targets_list.append(tgt.cpu())

    preds_t   = torch.cat(all_preds,        dim=0)
    targets_t = torch.cat(all_targets_list, dim=0)

    def denorm(arr, col):
        stats = target_stats.get(col, {})
        if not stats:
            return arr.numpy()
        if normalise == "zscore":
            return (arr.float() * stats["std"] + stats["mean"]).numpy()
        return (arr.float() * (stats["max"] - stats["min"]) + stats["min"]).numpy()

    def _short(col: str) -> str:
        return (col.replace("dish_", "")
                   .replace("_throughput_mbps", "")
                   .replace("_ms_mean", "_ms")
                   .replace("_mbps", ""))

    y_true_dict, y_pred_dict = {}, {}
    num_targets = len(targets)

    if num_targets == 1:
        col  = targets[0]
        p_np = denorm(preds_t,   col)
        t_np = denorm(targets_t, col)
        H    = p_np.shape[1]
        for h in range(H):
            key = f"t+{h+1}s"
            y_true_dict[key] = t_np[:, h]
            y_pred_dict[key] = p_np[:, h]
    else:
        H = preds_t.shape[1]
        for ti, col in enumerate(targets):
            short = _short(col)
            p_np  = denorm(preds_t[:, :, ti],   col)
            t_np  = denorm(targets_t[:, :, ti], col)
            for h in range(H):
                key = f"{short}_t+{h+1}s"
                y_true_dict[key] = t_np[:, h]
                y_pred_dict[key] = p_np[:, h]

    return {
        "model_name":  "Seq2Seq-GRU",
        "y_true_dict": y_true_dict,
        "y_pred_dict": y_pred_dict,
        "metadata": {
            "checkpoint": ckpt_path,
            "targets":    targets,
            "features":   feature_cols,
        },
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
        data_root       = args.data_root,
        provider        = args.provider,
        experiment_type = args.experiment_type,
        targets         = targets,
        features        = feature_cols,
        input_len       = args.input_len,
        output_len      = args.output_len,
        normalise       = normalise,
        batch_size      = def_config.get("batch_size", 512),
        train_frac      = args.train_frac,
        val_frac        = args.val_frac,
        seed            = args.seed,
    )

    # Models that consume DataConfig directly (next-step predictors).
    # These call build_dataloaders() internally and set output_len=1 themselves.
    CFG_NATIVE_MODELS = {"lstm", "transformer"}

    try:
        if model_key in CFG_NATIVE_MODELS:
            # New interface: trainer.train_and_evaluate(cfg, results_dir, config)
            result = trainer.train_and_evaluate(
                cfg         = cfg,
                results_dir = str(model_out),
                config      = def_config,
            )
        else:
            # Legacy interface: trainer.train_and_evaluate(train_set, test_set, ...)
            # seq2seq uses build_dataloaders directly; other legacy models use
            # ExperimentDataset via build_experiment_datasets.
            if model_key == "seq2seq":
                train_loader, val_loader, test_loader, _, _, _ = build_dataloaders(cfg)
                train_set = train_loader.dataset
                val_set   = val_loader.dataset   if val_loader   else None
                test_set  = test_loader.dataset  if test_loader  else train_loader.dataset
            else:
                train_set, test_set, _, _, _ = build_experiment_datasets(cfg)
                val_set = None

            result = trainer.train_and_evaluate(
                train_set   = train_set,
                val_set     = val_set,
                test_set    = test_set,
                results_dir = str(model_out),
                config      = def_config,
            )
    except Exception as exc:
        print(f"\n  [ERROR] {model_key} failed: {exc}")
        import traceback; traceback.print_exc()
        return None

    metrics_df = evaluate_predictions(result, str(model_out))

    meta = result.get("metadata", {})
    meta["model_key"]       = model_key
    meta["provider"]        = args.provider
    meta["experiment_type"] = args.experiment_type
    meta["targets"]         = targets
    meta["features"]        = feature_cols
    meta["out_dir"]         = str(model_out)
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
    print(f"  Provider        : {args.provider}")
    print(f"  Experiment type : {args.experiment_type}")
    print(f"  Data root       : {args.data_root}")
    print(f"  Targets         : {targets}")
    print(f"  Features        : {len(feature_cols)}  {feature_cols}")
    print(f"  Models          : {model_keys}")
    print(f"  Out dir         : {out_dir}")
    print("=" * 65)

    cfg_dict = vars(args).copy()
    cfg_dict["resolved_features"] = feature_cols
    cfg_dict["resolved_targets"]  = targets
    cfg_dict["resolved_models"]   = model_keys
    with open(out_dir / "run_config.json", "w") as f:
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
        disp = [c for c in ["MAE", "RMSE", "R2", "PICP_95"] if c in comp_df.columns]
        print(comp_df[disp].to_string(float_format="%.4f"))
        compare_saved_runs(
            [str(out_dir / m["model_key"]) for m in summaries],
            out_dir=args.out_dir,
        )

    print(f"\nAll results in: {out_dir}/")


# cmd_evaluate()

def cmd_evaluate(args):
    meta_path = Path(args.checkpoint)

    if meta_path.suffix == ".json":
        with open(meta_path) as f:
            meta = json.load(f)

        ckpt = meta.get("checkpoint", "")

        provider = meta.get("provider", args.provider)

        experiment_type = meta.get(
            "experiment_type",
            args.experiment_type,
        )

        experiment_type = LEGACY_EXPERIMENT_MAP.get(
            experiment_type,
            experiment_type,
        )

        targets = meta.get(
            "targets",
            [DEFAULT_TARGET[(provider, experiment_type)]],
        )

        features = meta.get(
            "features",
            list(DEFAULT_FEATURES_BY_EXP[(provider, experiment_type)]),
        )

        hidden_size = meta.get("hidden_size", args.hidden_size)
        embed_dim   = meta.get("embed_dim", args.embed_dim)

    else:
        ckpt            = str(meta_path)
        provider        = args.provider

        experiment_type = LEGACY_EXPERIMENT_MAP.get(
            args.experiment_type,
            args.experiment_type,
        )

        targets         = resolve_targets(args)
        features        = resolve_features(args)
        hidden_size     = args.hidden_size
        embed_dim       = args.embed_dim

    normalise = args.normalise if args.normalise != "none" else None

    result = _eval_from_checkpoint(
        ckpt_path       = ckpt,
        data_root       = args.data_root,
        provider        = provider,
        experiment_type = experiment_type,
        targets         = targets,
        features        = features,
        input_len       = args.input_len,
        output_len      = args.output_len,
        hidden_size     = hidden_size,
        embed_dim       = embed_dim,
        normalise       = normalise,
        batch_size      = args.batch_size,
        seed            = args.seed,
    )

    evaluate_predictions(result, args.out_dir)

    print(f"\nEvaluation results -> {args.out_dir}/")


def cmd_compare(args):
    df = compare_saved_runs(args.run_dirs, args.out_dir)
    if not df.empty:
        print(df.to_string(float_format="%.4f"))


def cmd_ablation(args):
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

def _add_provider_args(p):
    """Provider + experiment_type."""

    p.add_argument(
        "--provider",
        default=PROVIDER_STARLINK,
        choices=list(PROVIDERS),
        help="Data provider. (default: starlink)",
    )

    p.add_argument(
        "--experiment_type",
        default="throughput_dl",
        metavar="EXP_TYPE",
        help=(
            "Experiment type.\n"
            "  starlink/quectel:\n"
            "    latency\n"
            "    latency_loss_throughput_dl\n"
            "    latency_loss_throughput_ul\n"
            "    throughput_dl\n"
            "    throughput_ul\n"
            "  oneweb:\n"
            "    dl\n"
            "    ul"
        ),
    )


def _add_data_args(p):
    p.add_argument("--data_root",  default="data/lsn-data",
                   help="Root directory containing provider sub-folders.")
    p.add_argument("--targets",    nargs="+", default=None, metavar="TARGET",
                   help="Target column(s) to predict. Use shorthand keys or full "
                        "column names. Run --list_targets --provider <p> to see "
                        "options. Default: provider/experiment_type-specific.")
    p.add_argument("--input_len",  type=int,   default=30)
    p.add_argument("--output_len", type=int,   default=15)
    p.add_argument("--normalise",  default="zscore",
                   choices=["zscore", "minmax", "none"])
    p.add_argument("--train_frac", type=float, default=0.70)
    p.add_argument("--val_frac",   type=float, default=0.15)
    p.add_argument("--batch_size", type=int,   default=None)
    p.add_argument("--seed",       type=int,   default=42)


def _add_feature_args(p):
    fg = p.add_mutually_exclusive_group()
    fg.add_argument(
        "--features", nargs="+", default=["default"], metavar="FEATURE",
        help="'default', 'all', or explicit column names.",
    )
    fg.add_argument(
        "--feature_group", default=None, choices=list(FEATURE_GROUPS),
        help="One semantic feature group.",
    )
    fg.add_argument(
        "--feature_groups", nargs="+", default=None, choices=list(FEATURE_GROUPS),
        help="Union of multiple semantic feature groups.",
    )


def _add_model_hparam_args(p):
    p.add_argument("--hidden_size", type=int,   default=128)
    p.add_argument("--embed_dim",   type=int,   default=16)
    p.add_argument("--dropout",     type=float, default=None)
    p.add_argument("--epochs",      type=int,   default=100)
    p.add_argument("--lr",          type=float, default=1e-3)


def _validate_experiment_type(args):
    """Post-parse validation."""

    args.experiment_type = LEGACY_EXPERIMENT_MAP.get(
        args.experiment_type,
        args.experiment_type,
    )

    valid = VALID_EXPERIMENT_TYPES.get(args.provider, ())

    if args.experiment_type not in valid:
        print(
            f"[ERROR] experiment_type '{args.experiment_type}' "
            f"is not valid for provider '{args.provider}'.\n"
            f"  Valid choices: {list(valid)}"
        )
        sys.exit(1)


def build_parser():
    parser = argparse.ArgumentParser(
        description="Throughput Prediction Framework",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument("--list_models",   action="store_true",
                        help="Print all registered models and exit.")
    parser.add_argument("--list_features", action="store_true",
                        help="Print all features for --provider and exit.")
    parser.add_argument("--list_targets",  action="store_true",
                        help="Print all targets for --provider and exit.")
    # Allow --provider at top level for --list_* flags
    parser.add_argument("--provider", default=PROVIDER_STARLINK,
                        choices=list(PROVIDERS))

    subs = parser.add_subparsers(dest="subcommand")

    # ---- train ----
    tr = subs.add_parser("train", help="Train one or more models.")
    tr.add_argument("--models", nargs="+", default=["seq2seq"], metavar="MODEL",
                    help="Model keys (or 'all'). "
                         "Available: seq2seq, lstm, transformer, "
                         "gaussian_process, autoencoder")
    _add_provider_args(tr)
    _add_data_args(tr)
    _add_feature_args(tr)
    _add_model_hparam_args(tr)
    tr.add_argument("--out_dir", default="results")

    # ---- evaluate ----
    ev = subs.add_parser("evaluate", help="Evaluate a saved checkpoint.")
    ev.add_argument("--checkpoint", required=True,
                    help="Path to metadata.json or best_model.pt")
    _add_provider_args(ev)
    _add_data_args(ev)
    _add_feature_args(ev)
    ev.add_argument("--hidden_size", type=int, default=128)
    ev.add_argument("--embed_dim",   type=int, default=16)
    ev.add_argument("--out_dir",     default="eval_results")

    # ---- compare ----
    cp = subs.add_parser("compare",
                         help="Compare metrics across multiple saved run directories.")
    cp.add_argument("--run_dirs", nargs="+", required=True,
                    help="Paths to run output directories.")
    cp.add_argument("--out_dir", default="results/comparison")

    # ---- ablation ----
    ab = subs.add_parser("ablation", help="Feature or architecture ablation study.")
    _add_provider_args(ab)
    ab.add_argument("--data_root",      default="data/lsn-data")
    ab.add_argument("--targets",        nargs="+", default=None)
    ab.add_argument("--normalise",      default="zscore",
                    choices=["zscore", "minmax", "none"])
    ab.add_argument("--seed",           type=int, default=42)
    ab.add_argument("--study",          default="feature",
                    choices=["feature", "architecture"])
    ab.add_argument("--strategy",       default="leave_one_out",
                    choices=["leave_one_out", "add_one", "groups", "custom"])
    ab.add_argument("--base_features",  nargs="+", default=None,
                    metavar="FEATURE",
                    help="Feature names to use as ablation base.")
    ab.add_argument("--feature_sets",   nargs="+", default=None)
    ab.add_argument("--arch_param",     default="hidden_size",
                    choices=["hidden_size", "input_len", "output_len", "embed_dim"])
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


def print_features(provider: str):
    all_f  = ALL_FEATURES[provider]
    def_f  = DEFAULT_FEATURES[provider]
    groups = {g: feats for g, feats in FEATURE_GROUPS.items()
              if any(f in all_f for f in feats)}

    print(f"\nAll features for provider '{provider}' ({len(all_f)}):")
    for i, f in enumerate(all_f, 1):
        marker = " *" if f in def_f else ""
        print(f"  {i:2d}. {f}{marker}")
    print(f"\n  (* = included in default set of {len(def_f)} features)")

    if groups:
        print("\nRelevant semantic groups:")
        for g, feats in groups.items():
            relevant = [f for f in feats if f in all_f]
            if relevant:
                print(f"  {g}: {relevant}")

    print(f"\nUsage: python run.py train --provider {provider} "
          f"--experiment_type throughput_dl --features default")
    print(f"       python run.py train --provider {provider} "
          f"--experiment_type throughput_dl --features all")


def print_targets(provider: str):
    all_t   = ALL_TARGETS[provider]
    aliases = TARGET_ALIASES[provider]

    print(f"\nAvailable prediction targets for provider '{provider}':")
    print(f"  {'Shorthand':<12} Full column name")
    print("  " + "-" * 50)
    alias_map = {v: k for k, v in aliases.items()}
    for col in all_t:
        short = alias_map.get(col, "—")
        print(f"  {short:<12} {col}")

    print(f"\nUsage: python run.py train --provider {provider} "
          f"--experiment_type throughput_dl --targets <shorthand or full name>")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = build_parser()
    args   = parser.parse_args()

    # --list_* flags at the top level (no subcommand required)
    if args.list_models:
        print_models(); sys.exit(0)

    provider = getattr(args, "provider", PROVIDER_STARLINK)

    if args.list_features:
        print_features(provider); sys.exit(0)
    if getattr(args, "list_targets", False):
        print_targets(provider); sys.exit(0)

    if args.subcommand is None:
        parser.print_help()
        sys.exit(0)

    # Validate (provider, experiment_type) for subcommands that use it
    if args.subcommand in ("train", "evaluate", "ablation"):
        _validate_experiment_type(args)

    dispatch = {
        "train":    cmd_train,
        "evaluate": cmd_evaluate,
        "compare":  cmd_compare,
        "ablation": cmd_ablation,
    }
    dispatch[args.subcommand](args)


if __name__ == "__main__":
    main()