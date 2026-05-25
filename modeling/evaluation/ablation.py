"""
evaluation/ablation.py - Feature and architecture ablation studies.

Usage
-----
    python evaluation/ablation.py \\
        --provider starlink --experiment_type throughput \\
        --study feature --strategy leave_one_out

    python evaluation/ablation.py \\
        --provider starlink --experiment_type throughput \\
        --study architecture --arch_param hidden_size

    python evaluation/ablation.py \\
        --provider quectel --experiment_type latency_loss_throughput \\
        --targets iperf sinr --study feature --strategy groups

    # Resume without retraining completed variants:
        --skip_existing
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dataset import (
    PROVIDERS, PROVIDER_STARLINK, VALID_EXPERIMENT_TYPES,
    ALL_FEATURES, DEFAULT_FEATURES, ALL_TARGETS, TARGET_ALIASES, DEFAULT_TARGET,
    # Feature groups + flat name list — single source of truth in dataset.py
    FEATURE_GROUPS, ALL_FEATURE_NAMES,
    DataConfig, build_dataloaders,
)
from model  import ThroughputSeq2Seq
from train  import train as run_training
from evaluation.evaluate import evaluate_predictions


ARCH_GRIDS: dict[str, list[Any]] = {
    "hidden_size": [64, 128, 256],
    "input_len":   [10, 20, 30, 60],
    "output_len":  [5, 10, 15],
    "embed_dim":   [0, 8, 16, 32],
}


def _safe_name(s: str) -> str:
    return s.replace(" ", "_").replace(",", "-").replace("/", "-")[:80]


# ---------------------------------------------------------------------------
# Feature-set generation
# ---------------------------------------------------------------------------

def build_feature_sets(
    strategy: str,
    base_features: list[str],
    provider: str,
    custom_sets: list[str] | None = None,
) -> list[tuple[str, list[str]]]:
    provider_all = ALL_FEATURES[provider]
    sets: list[tuple[str, list[str]]] = []

    if strategy == "leave_one_out":
        sets.append(("ALL_FEATURES", list(base_features)))
        for feat in base_features:
            sets.append((f"drop_{feat}", [f for f in base_features if f != feat]))

    elif strategy == "add_one":
        for i in range(1, len(base_features) + 1):
            subset = base_features[:i]
            sets.append((f"top{i}_{subset[-1]}", list(subset)))

    elif strategy == "groups":
        sets.append(("ALL_FEATURES", list(base_features)))
        for gname, gfeats in FEATURE_GROUPS.items():
            to_remove = set(gfeats) & set(base_features)
            if not to_remove:
                continue
            subset = [f for f in base_features if f not in to_remove]
            if subset:
                sets.append((f"drop_group_{gname}", subset))

    elif strategy == "custom":
        if not custom_sets:
            raise ValueError("--strategy custom requires --feature_sets")
        for raw in custom_sets:
            feats = [f.strip() for f in raw.split(",") if f.strip()]
            bad   = [f for f in feats if f not in provider_all]
            if bad:
                raise ValueError(
                    f"Unknown features for provider '{provider}': {bad}"
                )
            sets.append((_safe_name(",".join(feats)), feats))

    else:
        raise ValueError(f"Unknown strategy: {strategy}")

    return sets


# ---------------------------------------------------------------------------
# Single variant: train + evaluate
# ---------------------------------------------------------------------------

def run_one(
    label, data_root, provider, experiment_type,
    targets, features,
    out_dir, ckpt_dir,
    epochs, batch_size, hidden_size, embed_dim,
    input_len, output_len,
    normalise, seed, skip_existing,
) -> dict:
    ckpt_path   = ckpt_dir / _safe_name(label) / "best_model.pt"
    eval_subdir = out_dir  / _safe_name(label)

    print(f"\n{'='*60}")
    print(f"  Variant  : {label}")
    print(f"  Features : {len(features)}  Targets: {targets}")
    print(f"{'='*60}")

    if skip_existing and ckpt_path.exists():
        print(f"  [skip] checkpoint exists: {ckpt_path}")
    else:
        ckpt_path.parent.mkdir(parents=True, exist_ok=True)
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
        train_loader, val_loader, test_loader, _, _, _ = build_dataloaders(cfg)
        model = ThroughputSeq2Seq(
            num_features = len(features),
            embed_dim    = embed_dim,
            hidden_size  = hidden_size,
            input_len    = input_len,
            output_len   = output_len,
            num_targets  = len(targets),
        )
        history = run_training(
            model, train_loader, val_loader, test_loader,
            num_epochs=epochs, save_dir=str(ckpt_path.parent),
        )
        actual_ckpt = Path(history.get("checkpoint", ""))
        if actual_ckpt.exists() and actual_ckpt != ckpt_path:
            import shutil
            shutil.copy(actual_ckpt, ckpt_path)

    # Evaluate via run.py helper
    try:
        from run import _eval_from_checkpoint
    except ImportError as e:
        raise ImportError(
            "ablation.py requires '_eval_from_checkpoint' from run.py. "
            "Make sure run.py is on sys.path and exports that function. "
            f"Original error: {e}"
        ) from e
    try:
        result = _eval_from_checkpoint(
            ckpt_path       = str(ckpt_path),
            data_root       = data_root,
            provider        = provider,
            experiment_type = experiment_type,
            targets         = targets,
            features        = features,
            input_len       = input_len,
            output_len      = output_len,
            hidden_size     = hidden_size,
            embed_dim       = embed_dim,
            normalise       = normalise,
            batch_size      = batch_size,
            seed            = seed,
        )
        metrics_df = evaluate_predictions(result, str(eval_subdir), make_plots=False)
    except Exception as exc:
        print(f"  [warn] eval failed: {exc}")
        return {"label": label, "MAE": np.nan, "RMSE": np.nan, "R2": np.nan}

    agg = metrics_df[metrics_df["key"] == "AGGREGATE"]
    if agg.empty:
        return {"label": label, "MAE": np.nan, "RMSE": np.nan, "R2": np.nan}

    row = agg.iloc[0].to_dict()
    row["label"] = label
    return row


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_ablation(results_df: pd.DataFrame, out_dir: Path, study: str) -> None:
    metrics = [c for c in ["MAE", "RMSE", "R2", "PICP_95"] if c in results_df.columns]
    if not metrics:
        return
    n   = min(4, len(metrics))
    fig, axes = plt.subplots(1, n, figsize=(5 * n, max(4, len(results_df) * 0.6 + 1)))
    if n == 1:
        axes = [axes]
    for ax, metric in zip(axes, metrics):
        vals   = results_df[metric].values.astype(float)
        labels = results_df["label"].values
        order  = np.argsort(vals) if metric != "R2" else np.argsort(vals)[::-1]
        sv, sl = vals[order], labels[order]
        colors = ["tomato" if i == 0 else "steelblue" for i in range(len(sl))]
        bars   = ax.barh(sl, sv, color=colors)
        ax.set_title(metric); ax.invert_yaxis(); ax.grid(True, axis="x", alpha=0.3)
        for bar, val in zip(bars, sv):
            ax.text(bar.get_width() * 1.01, bar.get_y() + bar.get_height() / 2,
                    f"{val:.3f}", va="center", fontsize=7)
    study_label = "Feature" if study == "feature" else "Architecture"
    plt.suptitle(f"{study_label} ablation", fontsize=14)
    plt.tight_layout()
    plt.savefig(out_dir / "ablation_summary.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Ablation plot -> {out_dir / 'ablation_summary.png'}")


# ---------------------------------------------------------------------------
# Argument parser
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Ablation study")

    # Provider / experiment — replaces --mode
    p.add_argument("--provider",        default=PROVIDER_STARLINK,
                   choices=list(PROVIDERS))
    p.add_argument("--experiment_type", default="throughput", metavar="EXP_TYPE",
                   help=("starlink/quectel: latency | latency_loss_throughput | throughput  "
                         "oneweb: dl | ul"))

    p.add_argument("--data_root",      default="data/lsn-data")
    p.add_argument("--targets",        nargs="+", default=None, metavar="TARGET",
                   help="Shorthand keys or full column names. "
                        "Default: provider/experiment_type default target.")
    p.add_argument("--normalise",      default="zscore",
                   choices=["zscore", "minmax", "none"])
    p.add_argument("--seed",           type=int, default=42)

    # Ablation config
    p.add_argument("--study",          default="feature",
                   choices=["feature", "architecture"])
    p.add_argument("--strategy",       default="leave_one_out",
                   choices=["leave_one_out", "add_one", "groups", "custom"])
    p.add_argument("--base_features",  nargs="+", default=None, metavar="FEATURE",
                   help="Feature names to use as the ablation base set. "
                        "Default: provider default features.")
    p.add_argument("--feature_sets",   nargs="+", default=None,
                   help="Comma-separated feature lists for --strategy custom.")
    p.add_argument("--arch_param",     default="hidden_size",
                   choices=list(ARCH_GRIDS))

    # Training / model
    p.add_argument("--epochs",         type=int, default=30)
    p.add_argument("--batch_size",     type=int, default=512)
    p.add_argument("--input_len",      type=int, default=30)
    p.add_argument("--output_len",     type=int, default=15)
    p.add_argument("--hidden_size",    type=int, default=128)
    p.add_argument("--embed_dim",      type=int, default=16)

    # Output
    p.add_argument("--checkpoint_dir", default="checkpoints/ablation")
    p.add_argument("--out_dir",        default="ablation_results")
    p.add_argument("--skip_existing",  action="store_true")

    return p


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = _build_parser().parse_args()

    # Validate (provider, experiment_type)
    valid_exp = VALID_EXPERIMENT_TYPES.get(args.provider, ())
    if args.experiment_type not in valid_exp:
        print(
            f"[ERROR] experiment_type '{args.experiment_type}' is not valid for "
            f"provider '{args.provider}'.\n  Valid: {list(valid_exp)}"
        )
        sys.exit(1)

    out_dir  = Path(args.out_dir)
    ckpt_dir = Path(args.checkpoint_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    normalise     = args.normalise if args.normalise != "none" else None
    provider      = args.provider
    exp_type      = args.experiment_type

    # Resolve base features (default to provider defaults)
    provider_all  = ALL_FEATURES[provider]
    base_features = args.base_features or list(DEFAULT_FEATURES[provider])
    bad_feat = [f for f in base_features if f not in provider_all]
    if bad_feat:
        print(f"[ERROR] Unknown features for provider '{provider}': {bad_feat}")
        sys.exit(1)

    # Resolve targets
    aliases = TARGET_ALIASES[provider]
    if args.targets:
        targets = [aliases.get(t, t) for t in args.targets]
        bad_tgt = [t for t in targets if t not in ALL_TARGETS[provider]]
        if bad_tgt:
            print(f"[ERROR] Unknown targets for provider '{provider}': {bad_tgt}")
            sys.exit(1)
    else:
        targets = [DEFAULT_TARGET[(provider, exp_type)]]

    # Build variant list
    variants: list[tuple[str, dict]] = []
    if args.study == "feature":
        for label, feats in build_feature_sets(
            args.strategy, base_features, provider, args.feature_sets
        ):
            variants.append((label, {
                "features":    feats,
                "hidden_size": args.hidden_size,
                "embed_dim":   args.embed_dim,
                "input_len":   args.input_len,
                "output_len":  args.output_len,
            }))
    else:
        for val in ARCH_GRIDS[args.arch_param]:
            overrides = {
                "features":    base_features,
                "hidden_size": args.hidden_size,
                "embed_dim":   args.embed_dim,
                "input_len":   args.input_len,
                "output_len":  args.output_len,
            }
            overrides[args.arch_param] = val
            variants.append((f"{args.arch_param}={val}", overrides))

    print(f"\nAblation  : {args.study}")
    print(f"Provider  : {provider}  /  {exp_type}")
    print(f"Targets   : {targets}")
    print(f"Variants  : {len(variants)}")

    all_results: list[dict] = []
    for label, ov in variants:
        try:
            row = run_one(
                label           = label,
                data_root       = args.data_root,
                provider        = provider,
                experiment_type = exp_type,
                targets         = targets,
                features        = ov["features"],
                out_dir         = out_dir,
                ckpt_dir        = ckpt_dir,
                epochs          = args.epochs,
                batch_size      = args.batch_size,
                hidden_size     = ov["hidden_size"],
                embed_dim       = ov["embed_dim"],
                input_len       = ov["input_len"],
                output_len      = ov["output_len"],
                normalise       = normalise,
                seed            = args.seed,
                skip_existing   = args.skip_existing,
            )
            all_results.append(row)
        except Exception as exc:
            print(f"  [ERROR] '{label}': {exc}")
            all_results.append({"label": label, "MAE": np.nan,
                                 "RMSE": np.nan, "R2": np.nan})

    results_df = pd.DataFrame(all_results)
    results_df.to_csv(out_dir / "ablation_results.csv", index=False, float_format="%.4f")
    print(f"\nResults -> {out_dir / 'ablation_results.csv'}")
    print(results_df[["label", "MAE", "RMSE", "R2"]].to_string(index=False))
    plot_ablation(results_df, out_dir, args.study)
    with open(out_dir / "ablation_config.json", "w") as f:
        json.dump(vars(args), f, indent=2)


if __name__ == "__main__":
    main()