"""
inference.py - Load a trained checkpoint and predict on a single session CSV.

Usage
-----
    # Starlink throughput (provider + experiment_type read from metadata.json):
    python inference.py \\
        --checkpoint results/seq2seq/metadata.json \\
        --csv data/lsn-data/starlink/throughput/starlink_20260510_030024_throughput.csv \\
        --output_csv predictions.csv

    # Override provider/experiment_type explicitly (e.g. checkpoint is a raw .pt):
    python inference.py \\
        --checkpoint checkpoints/best_model.pt \\
        --csv data/lsn-data/quectel/latency/quectel_20260513_120012_latency.csv \\
        --provider quectel --experiment_type latency \\
        --output_csv predictions.csv

    # OneWeb downlink:
    python inference.py \\
        --checkpoint results/seq2seq/metadata.json \\
        --csv data/lsn-data/oneweb/dl/ex_20250826_060137_DL.csv \\
        --provider oneweb --experiment_type dl

Output CSV: one row per sliding window.
    Single target : columns  t+1s, t+2s, ...
    Multi-target  : columns  <target_short>_t+1s, ...
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from dataset import (
    PROVIDERS, PROVIDER_STARLINK, VALID_EXPERIMENT_TYPES,

    ALL_FEATURES,
    DEFAULT_FEATURES,
    ALL_TARGETS,
    TARGET_ALIASES,
    DEFAULT_TARGET,

    # ADD THESE
    DEFAULT_FEATURES_BY_EXP,

    DataConfig, ThroughputFileDataset, load_csv, get_device,
)
from model import ThroughputSeq2Seq


# ---------------------------------------------------------------------------
# Legacy experiment compatibility
# ---------------------------------------------------------------------------

LEGACY_EXPERIMENT_MAP = {
    "throughput": "throughput_dl",
    "latency_loss_throughput": "latency_loss_throughput_dl",
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _short(col: str) -> str:
    return (col.replace("dish_", "")
               .replace("_throughput_mbps", "")
               .replace("_ms_mean", "_ms")
               .replace("_mbps", ""))


def _resolve_targets(raw: list[str] | None,
                     provider: str,
                     exp_type: str) -> list[str]:

    exp_type = LEGACY_EXPERIMENT_MAP.get(exp_type, exp_type)

    if not raw:
        return [DEFAULT_TARGET[(provider, exp_type)]]

    aliases = TARGET_ALIASES[provider]

    return [aliases.get(t, t) for t in raw]


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------

@torch.no_grad()
def predict(
    model,
    dataset: ThroughputFileDataset,
    device,
    target_stats: dict,
    targets: list[str],
    normalise: str | None,
    batch_size: int = 256,
) -> np.ndarray | dict[str, np.ndarray]:
    model.eval()
    loader    = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    all_preds = []
    for feat, hour, dow, _ in loader:
        preds, _ = model(feat.to(device), hour.to(device), dow.to(device))
        all_preds.append(preds.cpu())

    preds_t = torch.cat(all_preds, dim=0)   # (N, H) or (N, H, T)

    def denorm(arr: torch.Tensor, col: str) -> np.ndarray:
        stats = target_stats.get(col, {})
        if not stats or normalise is None:
            return arr.numpy()
        if normalise == "zscore":
            return (arr.float() * stats["std"] + stats["mean"]).numpy()
        return (arr.float() * (stats["max"] - stats["min"]) + stats["min"]).numpy()

    if len(targets) == 1:
        return denorm(preds_t, targets[0])   # (N, H)
    else:
        return {col: denorm(preds_t[:, :, i], col)
                for i, col in enumerate(targets)}


# ---------------------------------------------------------------------------
# Argument parser
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Run predictions from a checkpoint")
    p.add_argument("--checkpoint", required=True,
                   help="Path to metadata.json or best_model.pt")
    p.add_argument("--csv",        required=True,
                   help="Path to a single session CSV file.")

    # Provider / experiment — overridden by metadata.json when available
    p.add_argument("--provider",        default=PROVIDER_STARLINK,
                   choices=list(PROVIDERS))
    p.add_argument(
        "--experiment_type",
        default="throughput_dl",
        metavar="EXP_TYPE",
        help=(
            "starlink/quectel:\n"
            "  latency\n"
            "  latency_loss_throughput_dl\n"
            "  latency_loss_throughput_ul\n"
            "  throughput_dl\n"
            "  throughput_ul\n"
            "oneweb:\n"
            "  dl\n"
            "  ul"
        ),
    )

    # Optional overrides (ignored when metadata.json supplies them)
    p.add_argument("--features",    nargs="+", default=None, metavar="FEATURE",
                   help="Feature column names. Default: provider defaults.")
    p.add_argument("--targets",     nargs="+", default=None, metavar="TARGET",
                   help="Target column name(s) or shorthand keys.")
    p.add_argument("--input_len",   type=int,   default=30)
    p.add_argument("--output_len",  type=int,   default=15)
    p.add_argument("--hidden_size", type=int,   default=128)
    p.add_argument("--embed_dim",   type=int,   default=16)
    p.add_argument("--normalise",   default="zscore",
                   choices=["zscore", "minmax", "none"])
    p.add_argument("--batch_size",  type=int,   default=256)
    p.add_argument("--output_csv",  default="predictions.csv")
    return p


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args   = _build_parser().parse_args()
    device = get_device()

    # ------------------------------------------------------------------ meta
    meta      = {}
    ckpt_path = args.checkpoint
    if args.checkpoint.endswith(".json"):
        with open(args.checkpoint) as f:
            meta = json.load(f)
        ckpt_path = meta.get("checkpoint", "")
        if not ckpt_path or not Path(ckpt_path).exists():
            siblings = list(Path(args.checkpoint).parent.rglob("best_model.pt"))
            if siblings:
                ckpt_path = str(siblings[0])
            else:
                raise FileNotFoundError(
                    f"Cannot locate best_model.pt from metadata: {args.checkpoint}"
                )

    # Provider / experiment_type — prefer metadata, fall back to CLI args
    provider = meta.get("provider", args.provider)

    experiment_type = meta.get(
        "experiment_type",
        args.experiment_type,
    )

    experiment_type = LEGACY_EXPERIMENT_MAP.get(
        experiment_type,
        experiment_type,
    )

    # Validate (provider, experiment_type)
    valid_exp = VALID_EXPERIMENT_TYPES.get(provider, ())
    if experiment_type not in valid_exp:
        raise ValueError(
            f"experiment_type '{experiment_type}' is not valid for "
            f"provider '{provider}'.  Valid: {list(valid_exp)}"
        )

    # Features / targets / hyper-params — prefer metadata, fall back to CLI
    feature_cols = (
        args.features
        or meta.get("features")
        or list(DEFAULT_FEATURES_BY_EXP[(provider, experiment_type)])
    )
    targets      = _resolve_targets(
                       args.targets or meta.get("targets"),
                       provider, experiment_type,
                   )
    hidden_size  = meta.get("hidden_size", args.hidden_size)
    embed_dim    = meta.get("embed_dim",   args.embed_dim)
    num_targets  = len(targets)
    normalise    = args.normalise if args.normalise != "none" else None

    # ----------------------------------------------------------------- model
    model = ThroughputSeq2Seq(
        num_features = len(feature_cols),
        embed_dim    = embed_dim,
        hidden_size  = hidden_size,
        input_len    = args.input_len,
        output_len   = args.output_len,
        num_targets  = num_targets,
    )
    model.load_state_dict(
        torch.load(ckpt_path, map_location=device, weights_only=True)
    )
    model = model.to(device)
    print(f"Loaded          : {ckpt_path}")
    print(f"Provider        : {provider}  /  {experiment_type}")
    print(f"Targets         : {targets}")
    print(f"Features        : {len(feature_cols)}")

    # ------------------------------------------------------------------ data
    csv_path = Path(args.csv)
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV not found: {csv_path}")

    # Build a minimal DataConfig just for load_csv (provider-aware parsing)
    cfg = DataConfig(
        provider        = provider,
        experiment_type = experiment_type,
        targets         = targets,
        features        = feature_cols,
        input_len       = args.input_len,
        output_len      = args.output_len,
        normalise       = normalise or "zscore",
    )

    df = load_csv(csv_path, cfg)
    if df is None:
        raise ValueError(f"Could not load {csv_path}")

    # Confirm targets are present
    missing = [tc for tc in targets if tc not in df.columns]
    if missing:
        raise ValueError(f"CSV missing target column(s): {missing}")

    dataset = ThroughputFileDataset(
        df, feature_cols, targets,
        args.input_len, args.output_len,
        normalise=normalise,
    )
    print(f"Windows         : {len(dataset):,}")

    if len(dataset) == 0:
        raise ValueError(
            f"No sliding windows produced — CSV may have too few rows "
            f"(need at least {args.input_len + args.output_len})."
        )

    # --------------------------------------------------------------- predict
    result = predict(
        model, dataset, device,
        dataset.target_stats, targets, normalise,
        batch_size=args.batch_size,
    )

    # --------------------------------------------------------- build output
    if isinstance(result, dict):
        frames = []
        for col, arr in result.items():
            short = _short(col)
            H     = arr.shape[1]
            frames.append(pd.DataFrame(arr, columns=[f"{short}_t+{h+1}s" for h in range(H)]))
        out_df = pd.concat(frames, axis=1)
    else:
        H      = result.shape[1]
        out_df = pd.DataFrame(result, columns=[f"t+{h+1}s" for h in range(H)])

    out_df.to_csv(args.output_csv, index=False)
    print(f"Saved           : {args.output_csv}  (shape {out_df.shape})")


if __name__ == "__main__":
    main()