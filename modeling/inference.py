"""
inference.py - Load a trained checkpoint and predict on a single session CSV.

Usage
-----
    python inference.py \\
        --checkpoint results/seq2seq/metadata.json \\
        --csv data/starlink/starlink_20260510_030024/starlink_20260510_030024__udp_dl_400.csv \\
        --mode dl \\
        --output_csv predictions.csv

    # Multi-target checkpoint (targets read from metadata.json automatically):
    python inference.py \\
        --checkpoint results/seq2seq/metadata.json \\
        --csv data/starlink/.../...csv \\
        --mode dl

Output CSV: one row per sliding window, columns t+1s … t+Ns (single target)
            or <target_short>_t+1s … (multi-target).
"""

from __future__ import annotations

import argparse
from pathlib import Path
import json

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from dataset import (
    ALL_FEATURES, DEFAULT_FEATURES, ALL_TARGETS, TARGET_COL, MODE_SUFFIX,
    ThroughputFileDataset, load_csv,
)
from model import ThroughputSeq2Seq


def _short(col: str) -> str:
    return col.replace("dish_","").replace("_throughput_mbps","").replace("_ms_mean","_ms")


@torch.no_grad()
def predict(model, dataset, device, target_stats, targets, normalise, batch_size=256):
    model.eval()
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    all_preds = []
    for feat, hour, dow, _ in loader:
        preds, _ = model(feat.to(device), hour.to(device), dow.to(device))
        all_preds.append(preds.cpu())

    preds_t = torch.cat(all_preds, dim=0)   # (N, H) or (N, H, T)

    def denorm(arr, col):
        stats = target_stats.get(col, {})
        if not stats:
            return arr.numpy()
        if normalise == "zscore":
            return (arr.float() * stats["std"] + stats["mean"]).numpy()
        return (arr.float() * (stats["max"] - stats["min"]) + stats["min"]).numpy()

    num_targets = len(targets)
    if num_targets == 1:
        return denorm(preds_t, targets[0])   # (N, H)
    else:
        parts = {}
        for i, col in enumerate(targets):
            parts[col] = denorm(preds_t[:, :, i], col)
        return parts


def parse_args():
    p = argparse.ArgumentParser(description="Run predictions from a checkpoint")
    p.add_argument("--checkpoint", required=True,
                   help="Path to metadata.json or best_model.pt")
    p.add_argument("--csv",        required=True)
    p.add_argument("--mode",       default="dl", choices=list(MODE_SUFFIX))
    p.add_argument("--features",   nargs="+", default=None, choices=ALL_FEATURES)
    p.add_argument("--targets",    nargs="+", default=None)
    p.add_argument("--input_len",   type=int,   default=30)
    p.add_argument("--output_len",  type=int,   default=15)
    p.add_argument("--hidden_size", type=int,   default=128)
    p.add_argument("--embed_dim",   type=int,   default=16)
    p.add_argument("--normalise",   default="zscore", choices=["zscore","minmax","none"])
    p.add_argument("--output_csv",  default="predictions.csv")
    return p.parse_args()


def main():
    args   = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load metadata if .json provided
    meta = {}
    ckpt_path = args.checkpoint
    if args.checkpoint.endswith(".json"):
        with open(args.checkpoint) as f:
            meta = json.load(f)
        ckpt_path = meta.get("checkpoint", "")
        if not ckpt_path or not Path(ckpt_path).exists():
            # try sibling best_model.pt
            siblings = list(Path(args.checkpoint).parent.rglob("best_model.pt"))
            if siblings:
                ckpt_path = str(siblings[0])
            else:
                raise FileNotFoundError(f"Cannot find best_model.pt from metadata: {args.checkpoint}")

    targets      = args.targets or meta.get("targets", [TARGET_COL[args.mode]])
    targets      = [TARGET_COL.get(t, t) for t in targets]
    feature_cols = args.features or meta.get("features", list(DEFAULT_FEATURES))
    hidden_size  = meta.get("hidden_size", args.hidden_size)
    embed_dim    = meta.get("embed_dim",   args.embed_dim)
    num_targets  = len(targets)
    normalise    = args.normalise if args.normalise != "none" else None

    model = ThroughputSeq2Seq(
        num_features = len(feature_cols),
        embed_dim    = embed_dim,
        hidden_size  = hidden_size,
        input_len    = args.input_len,
        output_len   = args.output_len,
        num_targets  = num_targets,
    )
    model.load_state_dict(torch.load(ckpt_path, map_location=device, weights_only=True))
    model = model.to(device)
    print(f"Loaded   : {ckpt_path}")
    print(f"Targets  : {targets}")
    print(f"Features : {len(feature_cols)}")

    csv_path = Path(args.csv)
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV not found: {csv_path}")

    df = load_csv(csv_path)
    if df is None:
        raise ValueError(f"Could not load {csv_path}")

    need = set(feature_cols + targets + ["timestamp_cst", "hour_cst", "dow_cst"])
    missing = need - set(df.columns)
    if missing:
        raise ValueError(f"CSV missing columns: {missing}")

    dataset = ThroughputFileDataset(
        df, feature_cols, targets,
        args.input_len, args.output_len,
        normalise=normalise,
    )
    print(f"Windows  : {len(dataset):,}")

    result = predict(model, dataset, device, dataset.target_stats,
                     targets, normalise or "none")

    if isinstance(result, dict):
        # Multi-target: each key is a column name, value is (N, H)
        frames = []
        for col, arr in result.items():
            short = _short(col)
            H = arr.shape[1]
            cols = [f"{short}_t+{h+1}s" for h in range(H)]
            frames.append(pd.DataFrame(arr, columns=cols))
        out_df = pd.concat(frames, axis=1)
    else:
        H    = result.shape[1]
        cols = [f"t+{h+1}s" for h in range(H)]
        out_df = pd.DataFrame(result, columns=cols)

    out_df.to_csv(args.output_csv, index=False)
    print(f"Saved    : {args.output_csv}  (shape {out_df.shape})")


if __name__ == "__main__":
    main()
