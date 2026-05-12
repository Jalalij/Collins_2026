"""
inference.py - Load a trained checkpoint and run predictions on a single CSV.

The input CSV should be one of the session files, e.g.:
    data/starlink/starlink_20260510_030024/starlink_20260510_030024__udp_dl_400.csv

Usage
-----
# DL prediction:
    python inference.py \
        --checkpoint checkpoints/best_model.pt \
        --csv  data/starlink/starlink_20260510_030024/starlink_20260510_030024__udp_dl_400.csv \
        --mode dl \
        --output_csv predictions.csv

# UL prediction with a custom feature subset:
    python inference.py \
        --checkpoint checkpoints/best_model.pt \
        --csv  data/starlink/starlink_20260510_030024/starlink_20260510_030024__udp_ul_40.csv \
        --mode ul \
        --features iperf_jitter_ms iperf_loss_pct dish_snr \
        --output_csv ul_predictions.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from dataset import (
    ALL_FEATURES,
    DEFAULT_FEATURES,
    MODE_SUFFIX,
    TARGET_COL,
    ThroughputFileDataset,
    load_csv,
)
from model import ThroughputSeq2Seq


# ---------------------------------------------------------------------------
# Prediction helper
# ---------------------------------------------------------------------------

@torch.no_grad()
def predict(
    model: ThroughputSeq2Seq,
    dataset: ThroughputFileDataset,
    device: torch.device,
    target_stats: dict | None,
    normalise: str = "zscore",
    batch_size: int = 256,
) -> np.ndarray:
    """
    Run inference over every window in dataset.

    Returns
    -------
    preds : (N, output_len) ndarray in the ORIGINAL (denormalised) scale
    """
    model.eval()
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    all_preds: list[np.ndarray] = []

    for feat, hour, dow, _ in loader:
        preds, _ = model(feat.to(device), hour.to(device), dow.to(device))
        all_preds.append(preds.cpu().numpy())

    preds_norm = np.concatenate(all_preds, axis=0)   # (N, output_len)

    # Denormalise back to Mbps
    if target_stats is not None:
        if normalise == "zscore":
            preds_norm = preds_norm * target_stats["std"] + target_stats["mean"]
        elif normalise == "minmax":
            preds_norm = preds_norm * (target_stats["max"] - target_stats["min"]) + target_stats["min"]

    return preds_norm


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="Run throughput predictions from a checkpoint")

    p.add_argument(
        "--checkpoint", required=True,
        help="Path to saved model checkpoint, e.g. checkpoints/best_model.pt",
    )
    p.add_argument(
        "--csv", required=True,
        help="Path to the session CSV to run inference on",
    )
    p.add_argument(
        "--mode", default="dl", choices=list(MODE_SUFFIX),
        help="dl -> predicts dish_dl_throughput_mbps; ul -> dish_ul_throughput_mbps",
    )
    p.add_argument(
        "--features", nargs="+", default=None, choices=ALL_FEATURES,
        metavar="FEATURE",
        help="Feature subset used when the model was trained. Default: 11-feature set.",
    )
    p.add_argument("--input_len",   type=int,   default=30)
    p.add_argument("--output_len",  type=int,   default=15)
    p.add_argument("--hidden_size", type=int,   default=128)
    p.add_argument("--embed_dim",   type=int,   default=16)
    p.add_argument("--normalise",   default="zscore", choices=["zscore", "minmax", "none"])
    p.add_argument("--output_csv",  default="predictions.csv")

    return p.parse_args()


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    feature_cols = args.features if args.features is not None else list(DEFAULT_FEATURES)
    target_col   = TARGET_COL[args.mode]
    normalise    = args.normalise if args.normalise != "none" else None

    # ---------------------------------------------------------------- model
    model = ThroughputSeq2Seq(
        num_features = len(feature_cols),
        embed_dim    = args.embed_dim,
        hidden_size  = args.hidden_size,
        input_len    = args.input_len,
        output_len   = args.output_len,
    )
    model.load_state_dict(torch.load(args.checkpoint, map_location=device))
    model = model.to(device)
    print(f"Loaded checkpoint : {args.checkpoint}")
    print(f"Mode              : {args.mode}  ->  {target_col}")
    print(f"Features ({len(feature_cols)})    : {', '.join(feature_cols)}")

    # ----------------------------------------------------------------- data
    csv_path = Path(args.csv)
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV not found: {csv_path}")

    df = load_csv(csv_path)
    if df is None:
        raise ValueError(f"Could not load {csv_path}")

    missing = (set(feature_cols) | {target_col}) - set(df.columns)
    if missing:
        raise ValueError(f"CSV is missing required columns: {missing}")

    dataset = ThroughputFileDataset(
        df, feature_cols, target_col,
        args.input_len, args.output_len,
        normalise=normalise,
    )
    print(f"Windows           : {len(dataset):,}")

    # -------------------------------------------------------------- predict
    preds = predict(model, dataset, device, dataset.target_stats, normalise or "none")
    print(f"Predictions shape : {preds.shape}  (windows x horizon)")

    # Save: one row per window, columns t+1 … t+N
    horizon_cols = [f"t+{i+1}s" for i in range(args.output_len)]
    out_df = pd.DataFrame(preds, columns=horizon_cols)
    out_df.to_csv(args.output_csv, index=False)
    print(f"Saved to          : {args.output_csv}")


if __name__ == "__main__":
    main()
