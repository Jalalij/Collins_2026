"""
run.py - Unified entry point for the Throughput Prediction Framework.

Modes
-----
  train       Train one or more models, save checkpoints + metrics.
  evaluate    Evaluate a saved checkpoint (single model).
  compare     Compare metrics across multiple saved run directories.
  ablation    Feature or architecture ablation (delegates to evaluation/ablation.py).

Prediction modes
----------------
  MIMO  (Multi-Input Multi-Output)   full (B, H, T) horizon × target tensor
        Models: seq2seq, lstm, transformer, attention, starnet
  MISO  (Multi-Input Single-Output)  (B, T) — one-step ahead, multi-target OK
        Models: gaussian_process, autoencoder, mlp
  SISO  (Single-Input Single-Output) scalar per window, single target only
        Models: arima

  The mode is stored in each ModelRegistry entry under the key ``"mode"``.
  ``DataConfig.io_mode`` controls target tensor shape in the dataset.
  ``evaluate_predictions`` detects shape and dispatches to the right metrics path.

Trainer contract
----------------
All trainers implement:
    train_and_evaluate(cfg: DataConfig, results_dir: str, config: dict) -> dict

``run_one_model`` builds a ``DataConfig`` (with the correct ``io_mode``) and
passes it directly to the trainer.  Trainers own their own dataloaders.

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
  seq2seq, lstm, transformer, gaussian_process, autoencoder, arima

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
              horizon_metrics.png   (MIMO only)
      comparison.csv
      comparison.png
      run_config.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Literal

import matplotlib
matplotlib.use("Agg")
import numpy as np
import pandas as pd
import torch

from models import ModelRegistry
from models.base import _register_builtins

# ThroughputSeq2Seq is now private to models/seq2seq.py.
# _eval_from_checkpoint imports it locally to avoid coupling run.py
# to a specific model architecture.

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

    ALL_FEATURES_BY_EXP,
    DEFAULT_FEATURES_BY_EXP,

    # Feature groups + flat name list
    FEATURE_GROUPS, ALL_FEATURE_NAMES,

    # Config + builders
    DataConfig, build_dataloaders,

    # IO mode constants
    IO_MODE_MIMO, IO_MODE_MISO, IO_MODE_SISO,
)

# Ensure all built-in trainers are registered before any lookup.
_register_builtins()

# ---------------------------------------------------------------------------
# Prediction-mode type alias
# ---------------------------------------------------------------------------

PredMode = Literal["MIMO", "MISO", "SISO"]

_VALID_MODES: frozenset[PredMode] = frozenset({"MIMO", "MISO", "SISO"})

# Map registry mode string → DataConfig io_mode constant.
_MODE_TO_IO_MODE: dict[str, str] = {
    "MIMO": IO_MODE_MIMO,
    "MISO": IO_MODE_MISO,
    "SISO": IO_MODE_SISO,
}


# ---------------------------------------------------------------------------
# Legacy experiment compatibility
# ---------------------------------------------------------------------------

LEGACY_EXPERIMENT_MAP = {
    "throughput":              "throughput_dl",
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
    raw         = getattr(args, "targets", None)

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
# Mode lookup
# ---------------------------------------------------------------------------

def get_model_mode(model_key: str) -> PredMode:
    """Return the prediction mode declared in the ModelRegistry entry."""
    entry = ModelRegistry.get(model_key)
    mode  = entry.get("mode", "MIMO")
    if mode not in _VALID_MODES:
        raise ValueError(
            f"Model '{model_key}' has unknown mode '{mode}'. "
            f"Must be one of {sorted(_VALID_MODES)}."
        )
    return mode


# ---------------------------------------------------------------------------
# Mode-aware denormalisation + dict building
# ---------------------------------------------------------------------------

def _build_pred_dicts(
    preds_t:      torch.Tensor,
    targets_t:    torch.Tensor,
    targets:      list[str],
    target_stats: dict,
    normalise:    str | None,
    mode:         PredMode,
) -> tuple[dict, dict]:
    """Convert raw tensors to {key: np.ndarray} dicts for evaluate_predictions.

    Tensor shapes on entry
    ----------------------
    MIMO : preds (B, H, T),  targets (B, H, T)
    MISO : preds (B, T),     targets (B, T)
    SISO : preds (B,),       targets (B,)

    Keys produced
    -------------
    MIMO, single target  : ``t+1s``, ``t+2s``, …
    MIMO, multi-target   : ``dl_t+1s``, ``dl_t+2s``, …
    MISO, single target  : ``t+1s``
    MISO, multi-target   : ``dl_t+1s``, ``rtt_t+1s``, …
    SISO                 : ``t+1s``
    """

    def denorm(arr: torch.Tensor, col: str) -> np.ndarray:
        stats = target_stats.get(col, {})
        if not stats or not normalise or normalise == "none":
            return arr.float().numpy()
        if normalise == "zscore":
            return (arr.float() * stats["std"] + stats["mean"]).numpy()
        return (arr.float() * (stats["max"] - stats["min"]) + stats["min"]).numpy()

    def _short(col: str) -> str:
        return (col.replace("dish_", "")
                   .replace("_throughput_mbps", "")
                   .replace("_ms_mean", "_ms")
                   .replace("_mbps", ""))

    y_true_dict: dict[str, np.ndarray] = {}
    y_pred_dict: dict[str, np.ndarray] = {}
    num_targets = len(targets)

    if mode == "SISO":
        col = targets[0]
        y_true_dict["t+1s"] = denorm(targets_t, col)
        y_pred_dict["t+1s"] = denorm(preds_t,   col)

    elif mode == "MISO":
        if num_targets == 1:
            col = targets[0]
            y_true_dict["t+1s"] = denorm(targets_t[:, 0], col)
            y_pred_dict["t+1s"] = denorm(preds_t[:, 0],   col)
        else:
            for ti, col in enumerate(targets):
                short = _short(col)
                y_true_dict[f"{short}_t+1s"] = denorm(targets_t[:, ti], col)
                y_pred_dict[f"{short}_t+1s"] = denorm(preds_t[:, ti],   col)

    else:  # MIMO
        # Normalise to 3-D in case a trainer squeezes the last dim.
        if preds_t.dim() == 2:
            preds_t   = preds_t.unsqueeze(-1)
            targets_t = targets_t.unsqueeze(-1)

        H = preds_t.shape[1]

        if num_targets == 1:
            col = targets[0]
            p_np = denorm(preds_t[:, :, 0],   col)
            t_np = denorm(targets_t[:, :, 0], col)
            for h in range(H):
                y_true_dict[f"t+{h+1}s"] = t_np[:, h]
                y_pred_dict[f"t+{h+1}s"] = p_np[:, h]
        else:
            for ti, col in enumerate(targets):
                short = _short(col)
                p_np  = denorm(preds_t[:, :, ti],   col)
                t_np  = denorm(targets_t[:, :, ti], col)
                for h in range(H):
                    y_true_dict[f"{short}_t+{h+1}s"] = t_np[:, h]
                    y_pred_dict[f"{short}_t+{h+1}s"] = p_np[:, h]

    return y_true_dict, y_pred_dict


# ---------------------------------------------------------------------------
# Checkpoint re-evaluation (evaluate subcommand + ablation)
# ---------------------------------------------------------------------------

def _eval_from_checkpoint(
    ckpt_path:       str,
    data_root:       str,
    provider:        str,
    experiment_type: str,
    targets:         list[str],
    features:        list[str],
    input_len:       int,
    output_len:      int,
    hidden_size:     int,
    embed_dim:       int,
    normalise:       str | None,
    batch_size:      int,
    seed:            int,
    mode:            PredMode = "MIMO",
) -> dict:
    """Load a saved checkpoint and collect predictions for evaluation.

    ThroughputSeq2Seq is imported locally so that run.py has no hard
    dependency on the seq2seq model architecture.
    """
    # Local import — ThroughputSeq2Seq is private to models/seq2seq.py.
    from models.seq2seq import ThroughputSeq2Seq

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
        io_mode         = _MODE_TO_IO_MODE[mode],   # DataConfig uses io_mode, not mode
    )

    _, _, test_loader, _, target_stats, feature_cols = build_dataloaders(cfg)
    if test_loader is None:
        raise ValueError("No test windows available for checkpoint evaluation.")

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

    all_preds, all_targets = [], []
    with torch.no_grad():
        for feat, hour, dow, tgt in test_loader:
            preds, _ = model(feat.to(device), hour.to(device), dow.to(device))
            all_preds.append(preds.cpu())
            all_targets.append(tgt.cpu())

    preds_t   = torch.cat(all_preds,   dim=0)
    targets_t = torch.cat(all_targets, dim=0)

    y_true_dict, y_pred_dict = _build_pred_dicts(
        preds_t, targets_t, targets, target_stats, normalise, mode,
    )

    return {
        "model_name":  "Seq2Seq-GRU",
        "y_true_dict": y_true_dict,
        "y_pred_dict": y_pred_dict,
        "pred_mode":   mode,
        "metadata": {
            "checkpoint": ckpt_path,
            "targets":    targets,
            "features":   feature_cols,
            "mode":       mode,
        },
    }


# ---------------------------------------------------------------------------
# Per-model train + evaluate
# ---------------------------------------------------------------------------

def _ensure_pred_dicts(
    result:       dict,
    targets:      list[str],
    target_stats: dict,
    normalise:    str | None,
    mode:         PredMode,
) -> dict:
    """Normalise trainer output into canonical {y_true_dict, y_pred_dict} form.

    Trainers that already populate these keys are passed through unchanged.
    Trainers that return raw ``preds`` / ``targets`` tensors are converted here.
    """
    if "y_true_dict" in result and "y_pred_dict" in result:
        return result

    preds_t   = result.get("preds")
    targets_t = result.get("targets")

    if preds_t is None or targets_t is None:
        raise ValueError(
            "Trainer result must contain either (y_true_dict, y_pred_dict) "
            "or (preds, targets) tensors."
        )

    if not isinstance(preds_t, torch.Tensor):
        preds_t   = torch.tensor(preds_t,   dtype=torch.float32)
        targets_t = torch.tensor(targets_t, dtype=torch.float32)

    y_true_dict, y_pred_dict = _build_pred_dicts(
        preds_t, targets_t, targets, target_stats, normalise, mode,
    )

    result = dict(result)
    result["y_true_dict"] = y_true_dict
    result["y_pred_dict"] = y_pred_dict
    return result


def run_one_model(
    model_key:    str,
    feature_cols: list[str],
    targets:      list[str],
    args,
    out_dir:      Path,
) -> dict | None:
    entry      = ModelRegistry.get(model_key)
    trainer    = entry["cls"]
    def_config = entry["default_config"].copy()
    mode:  PredMode = get_model_mode(model_key)

    # SISO models may only predict one target.
    if mode == "SISO" and len(targets) > 1:
        print(
            f"\n  [SKIP] {model_key} is SISO but {len(targets)} targets were "
            f"requested. Dropping to first target only: {targets[0]}"
        )
        targets = [targets[0]]

    # Merge CLI overrides into the default config.
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

    # Build DataConfig with the correct io_mode for this model.
    # Trainers own their own dataloaders — run.py just passes cfg.
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
        io_mode         = _MODE_TO_IO_MODE[mode],
    )

    try:
        result = trainer.train_and_evaluate(
            cfg         = cfg,
            results_dir = str(model_out),
            config      = def_config,
        )
    except Exception as exc:
        print(f"\n  [ERROR] {model_key} failed: {exc}")
        import traceback; traceback.print_exc()
        return None

    # Derive target_stats for _ensure_pred_dicts from a fresh dataloader
    # only if the trainer didn't already return canonical dicts.
    if "y_true_dict" not in result or "y_pred_dict" not in result:
        _, _, _, _, target_stats, _ = build_dataloaders(cfg)
        result = _ensure_pred_dicts(result, targets, target_stats, normalise, mode)

    result["pred_mode"] = mode

    metrics_df = evaluate_predictions(result, str(model_out))

    meta = result.get("metadata", {})
    meta.update({
        "model_key":       model_key,
        "provider":        args.provider,
        "experiment_type": args.experiment_type,
        "targets":         targets,
        "features":        feature_cols,
        "out_dir":         str(model_out),
        "mode":            mode,
    })
    with open(model_out / "metadata.json", "w") as f:
        json.dump(meta, f, indent=2, default=str)

    agg = metrics_df[metrics_df["key"] == "AGGREGATE"]
    if agg.empty:
        return None
    summary = agg.iloc[0].to_dict()
    summary["model"]     = result.get("model_name", model_key)
    summary["model_key"] = model_key
    summary["mode"]      = mode
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
    print(f"  Models          :")
    for mk in model_keys:
        try:
            print(f"    {mk:<22} mode={get_model_mode(mk)}")
        except Exception:
            pass
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
        print(
            f"\n{'#'*65}\n  MODEL: {model_key.upper()} "
            f"[mode={get_model_mode(model_key)}]\n{'#'*65}"
        )
        result = run_one_model(model_key, feature_cols, targets, args, out_dir)
        if result is not None:
            summaries.append(result)

    if len(summaries) > 1:
        print(f"\n{'='*65}\nCROSS-MODEL COMPARISON\n{'='*65}")
        comp_df = pd.DataFrame(summaries).set_index("model")
        disp = [c for c in ["mode", "MAE", "RMSE", "R2", "PICP_95"]
                if c in comp_df.columns]
        print(comp_df[disp].to_string(float_format="%.4f"))
        compare_saved_runs(
            [str(out_dir / m["model_key"]) for m in summaries],
            out_dir=args.out_dir,
        )

    print(f"\nAll results in: {out_dir}/")


def cmd_evaluate(args):
    meta_path = Path(args.checkpoint)

    if meta_path.suffix == ".json":
        with open(meta_path) as f:
            meta = json.load(f)

        ckpt            = meta.get("checkpoint", "")
        provider        = meta.get("provider", args.provider)
        experiment_type = LEGACY_EXPERIMENT_MAP.get(
            meta.get("experiment_type", args.experiment_type),
            meta.get("experiment_type", args.experiment_type),
        )
        targets         = meta.get("targets",
                            [DEFAULT_TARGET[(provider, experiment_type)]])
        features        = meta.get("features",
                            list(DEFAULT_FEATURES_BY_EXP[(provider, experiment_type)]))
        hidden_size     = meta.get("hidden_size", args.hidden_size)
        embed_dim       = meta.get("embed_dim",   args.embed_dim)
        mode: PredMode  = meta.get("mode", "MIMO")
    else:
        ckpt            = str(meta_path)
        provider        = args.provider
        experiment_type = LEGACY_EXPERIMENT_MAP.get(
            args.experiment_type, args.experiment_type,
        )
        targets         = resolve_targets(args)
        features        = resolve_features(args)
        hidden_size     = args.hidden_size
        embed_dim       = args.embed_dim
        mode            = "MIMO"

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
        mode            = mode,
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
    p.add_argument(
        "--provider", default=PROVIDER_STARLINK, choices=list(PROVIDERS),
        help="Data provider. (default: starlink)",
    )
    p.add_argument(
        "--experiment_type", default="throughput_dl", metavar="EXP_TYPE",
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
    p.add_argument("--data_root",  default="data/lsn-data")
    p.add_argument("--targets",    nargs="+", default=None, metavar="TARGET")
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
    )
    fg.add_argument(
        "--feature_groups", nargs="+", default=None, choices=list(FEATURE_GROUPS),
    )


def _add_model_hparam_args(p):
    p.add_argument("--hidden_size", type=int,   default=128)
    p.add_argument("--embed_dim",   type=int,   default=16)
    p.add_argument("--dropout",     type=float, default=None)
    p.add_argument("--epochs",      type=int,   default=100)
    p.add_argument("--lr",          type=float, default=1e-3)


def _validate_experiment_type(args):
    args.experiment_type = LEGACY_EXPERIMENT_MAP.get(
        args.experiment_type, args.experiment_type,
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
    parser.add_argument("--list_models",   action="store_true")
    parser.add_argument("--list_features", action="store_true")
    parser.add_argument("--list_targets",  action="store_true")
    parser.add_argument("--provider", default=PROVIDER_STARLINK,
                        choices=list(PROVIDERS))

    subs = parser.add_subparsers(dest="subcommand")

    # ---- train ----
    tr = subs.add_parser("train", help="Train one or more models.")
    tr.add_argument("--models", nargs="+", default=["seq2seq"], metavar="MODEL")
    _add_provider_args(tr)
    _add_data_args(tr)
    _add_feature_args(tr)
    _add_model_hparam_args(tr)
    tr.add_argument("--out_dir", default="results")

    # ---- evaluate ----
    ev = subs.add_parser("evaluate", help="Evaluate a saved checkpoint.")
    ev.add_argument("--checkpoint", required=True)
    _add_provider_args(ev)
    _add_data_args(ev)
    _add_feature_args(ev)
    ev.add_argument("--hidden_size", type=int, default=128)
    ev.add_argument("--embed_dim",   type=int, default=16)
    ev.add_argument("--out_dir",     default="eval_results")

    # ---- compare ----
    cp = subs.add_parser("compare")
    cp.add_argument("--run_dirs", nargs="+", required=True)
    cp.add_argument("--out_dir", default="results/comparison")

    # ---- ablation ----
    ab = subs.add_parser("ablation")
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
    ab.add_argument("--base_features",  nargs="+", default=None)
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
    print(f"  {'Key':<22} {'Mode':<6} Description")
    print("  " + "-" * 68)
    for entry in ModelRegistry.list_models():
        print(f"  {entry['key']:<22} {entry.get('mode','MIMO'):<6} {entry['description']}")
    print("\nUsage: python run.py train --models seq2seq lstm ...")


def print_features(provider: str):
    all_f  = ALL_FEATURES[provider]
    def_f  = DEFAULT_FEATURES[provider]
    groups = {g: feats for g, feats in FEATURE_GROUPS.items()
              if any(f in all_f for f in feats)}

    print(f"\nAll features for provider '{provider}' ({len(all_f)}):")
    for i, f in enumerate(all_f, 1):
        print(f"  {i:2d}. {f}{' *' if f in def_f else ''}")
    print(f"\n  (* = included in default set of {len(def_f)} features)")
    if groups:
        print("\nRelevant semantic groups:")
        for g, feats in groups.items():
            relevant = [f for f in feats if f in all_f]
            if relevant:
                print(f"  {g}: {relevant}")


def print_targets(provider: str):
    all_t   = ALL_TARGETS[provider]
    aliases = TARGET_ALIASES[provider]
    alias_map = {v: k for k, v in aliases.items()}

    print(f"\nAvailable prediction targets for provider '{provider}':")
    print(f"  {'Shorthand':<12} Full column name")
    print("  " + "-" * 50)
    for col in all_t:
        print(f"  {alias_map.get(col, '—'):<12} {col}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = build_parser()
    args   = parser.parse_args()

    if args.list_models:
        print_models(); sys.exit(0)

    provider = getattr(args, "provider", PROVIDER_STARLINK)

    if args.list_features:
        print_features(provider); sys.exit(0)
    if getattr(args, "list_targets", False):
        print_targets(provider); sys.exit(0)

    if args.subcommand is None:
        parser.print_help(); sys.exit(0)

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