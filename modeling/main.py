"""
main.py - Entry point for training the Throughput Seq2Seq model.

Usage examples
--------------
# Train DL model (default settings):
    python main.py --mode dl --data_root data/starlink

# Train UL model:
    python main.py --mode ul --data_root data/starlink

# Custom feature subset:
    python main.py --mode dl --data_root data/starlink \
        --features iperf_jitter_ms iperf_loss_pct dish_snr distance_km

# Quick smoke-test:
    python main.py --mode dl --data_root data/starlink --epochs 10 --batch_size 64
"""

import argparse
import torch

from dataset import DataConfig, build_dataloaders, ALL_FEATURES, MODE_SUFFIX
from model   import ThroughputSeq2Seq
from train   import train


def parse_args():
    p = argparse.ArgumentParser(description="Train Throughput Seq2Seq GRU model")

    # ---- Data ----
    p.add_argument("--data_root",  default="data/starlink",
                   help="Root folder containing starlink_YYYYMMDD_HHMMSS/ session subdirs")
    p.add_argument("--mode",       default="dl", choices=list(MODE_SUFFIX),
                   help="dl -> *__udp_dl_400.csv | ul -> *__udp_ul_40.csv")
    p.add_argument("--target",     default=None, choices=["dl", "ul", "iperf"],
                   help="Column to predict. Defaults to match --mode.")
    p.add_argument("--features",   nargs="+", default=None, choices=ALL_FEATURES,
                   metavar="FEATURE",
                   help="Feature subset. Default: 11-feature set.")
    p.add_argument("--input_len",  type=int, default=30)
    p.add_argument("--output_len", type=int, default=15)
    p.add_argument("--normalise",  default="zscore", choices=["zscore", "minmax", "none"])
    p.add_argument("--train_frac", type=float, default=0.70)
    p.add_argument("--val_frac",   type=float, default=0.15)
    p.add_argument("--batch_size", type=int,   default=512)
    p.add_argument("--seed",       type=int,   default=42)

    # ---- Model ----
    p.add_argument("--embed_dim",   type=int,   default=16)
    p.add_argument("--hidden_size", type=int,   default=128)
    p.add_argument("--dropout",     type=float, default=0.3,
                   help="Dropout inside GRU layers and projection head (0=off, 0.3=default)")

    # ---- Training ----
    p.add_argument("--epochs",              type=int,   default=100,
                   help="Max epochs (early stopping may end training sooner)")
    p.add_argument("--lr",                  type=float, default=1e-3)
    p.add_argument("--lr_factor",           type=float, default=0.5,
                   help="ReduceLROnPlateau factor: LR *= lr_factor when smoothed val plateaus")
    p.add_argument("--lr_patience",         type=int,   default=8,
                   help="Epochs with no smoothed-val improvement before reducing LR")
    p.add_argument("--min_lr",              type=float, default=1e-6,
                   help="LR floor")
    p.add_argument("--early_stop_patience", type=int,   default=20,
                   help="Epochs with no smoothed-val improvement before stopping")
    p.add_argument("--ema_alpha",           type=float, default=0.3,
                   help="EMA smoothing factor for val loss (0=fully smooth, 1=no smoothing)")
    p.add_argument("--weight_decay",        type=float, default=1e-3,
                   help="AdamW weight decay (L2 regularisation). Default 1e-3 (was 1e-4)")
    p.add_argument("--save_dir",            default="checkpoints")

    return p.parse_args()


def main():
    args   = parse_args()
    target = args.target if args.target is not None else args.mode

    # ------------------------------------------------------------------ data
    cfg = DataConfig(
        data_root  = args.data_root,
        mode       = args.mode,
        target     = target,
        features   = args.features,
        input_len  = args.input_len,
        output_len = args.output_len,
        normalise  = args.normalise if args.normalise != "none" else None,
        train_frac = args.train_frac,
        val_frac   = args.val_frac,
        batch_size = args.batch_size,
        seed       = args.seed,
    )

    print("\n=== Data loading ===")
    (train_loader, val_loader, test_loader,
     feature_stats, target_stats, feature_cols) = build_dataloaders(cfg)

    print(f"\n  Features ({len(feature_cols)}):")
    for i, f in enumerate(feature_cols, 1):
        print(f"    {i:2d}. {f}")

    # ----------------------------------------------------------------- model
    model = ThroughputSeq2Seq(
        num_features = len(feature_cols),
        embed_dim    = args.embed_dim,
        hidden_size  = args.hidden_size,
        dropout      = args.dropout,
        input_len    = args.input_len,
        output_len   = args.output_len,
    )
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n  Model parameters: {n_params:,}")

    # --------------------------------------------------------------- training
    print("\n=== Training ===")
    train(
        model,
        train_loader, val_loader, test_loader,
        num_epochs          = args.epochs,
        lr                  = args.lr,
        lr_factor           = args.lr_factor,
        lr_patience         = args.lr_patience,
        min_lr              = args.min_lr,
        early_stop_patience = args.early_stop_patience,
        ema_alpha           = args.ema_alpha,
        weight_decay        = args.weight_decay,
        save_dir            = args.save_dir,
    )

    print("\nDone.")


if __name__ == "__main__":
    main()