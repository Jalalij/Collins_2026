"""
Training loop for the Throughput Seq2Seq model.

LR schedule
-----------
  ReduceLROnPlateau: halves LR when the smoothed val loss stops improving.
  Smoothing is applied before both the scheduler and early stopping decisions,
  so transient spikes from a small val set don't prematurely kill training.

Early stopping
--------------
  Triggered when the smoothed val loss hasn't improved for `early_stop_patience`
  epochs. Defaults are tuned for small session counts (3-5 val sessions).

Smoothing
---------
  A simple exponential moving average (EMA) over val loss with alpha=0.3.
  Raw val loss is still logged and printed; smoothed value drives decisions.
"""

from __future__ import annotations

import csv
import os
import time
from collections import deque
from pathlib import Path

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau

from model import ThroughputSeq2Seq


def train_one_epoch(
    model: ThroughputSeq2Seq,
    loader,
    optimizer: AdamW,
    criterion,
    device: torch.device,
    teacher_forcing_ratio: float = 0.5,
) -> float:
    model.train()
    total_loss = 0.0
    for features, hour, dow, target in loader:
        features = features.to(device)
        hour     = hour.to(device)
        dow      = dow.to(device)
        target   = target.to(device)

        optimizer.zero_grad()
        preds, _ = model(features, hour, dow, target, teacher_forcing_ratio)
        loss = criterion(preds, target)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()
        total_loss += loss.item() * features.size(0)

    return total_loss / len(loader.dataset)


@torch.no_grad()
def evaluate(
    model: ThroughputSeq2Seq,
    loader,
    criterion,
    device: torch.device,
) -> float:
    if loader is None:
        return float("nan")
    model.eval()
    total_loss = 0.0
    for features, hour, dow, target in loader:
        features = features.to(device)
        hour     = hour.to(device)
        dow      = dow.to(device)
        target   = target.to(device)
        preds, _ = model(features, hour, dow)
        total_loss += criterion(preds, target).item() * features.size(0)
    return total_loss / len(loader.dataset)


def train(
    model: ThroughputSeq2Seq,
    train_loader,
    val_loader,
    test_loader=None,
    num_epochs: int = 100,
    lr: float = 1e-3,
    weight_decay: float = 1e-3,
    # ReduceLROnPlateau settings
    lr_factor: float = 0.5,
    lr_patience: int = 8,
    min_lr: float = 1e-6,
    # Early stopping
    early_stop_patience: int = 20,
    # Val loss smoothing
    ema_alpha: float = 0.3,
    save_dir: str = "checkpoints",
    device: torch.device | None = None,
) -> dict:
    """
    Train with EMA-smoothed val loss driving LR scheduling and early stopping.

    Parameters
    ----------
    lr_factor           : LR multiplier when smoothed val plateaus (0.5 = halve)
    lr_patience         : epochs of no smoothed-val improvement before LR drops
    min_lr              : LR floor
    early_stop_patience : epochs of no smoothed-val improvement before stopping
    ema_alpha           : EMA smoothing factor for val loss (0.3 = fairly smooth)
                          higher = more reactive, lower = more stable
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nTraining on: {device}")
    model = model.to(device)

    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = ReduceLROnPlateau(
        optimizer,
        mode     = "min",
        factor   = lr_factor,
        patience = lr_patience,
        min_lr   = min_lr,
    )
    criterion = nn.MSELoss()

    from datetime import datetime
    run_name  = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir   = Path(save_dir) / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    log_path  = run_dir / "training_log.csv"
    best_path = run_dir / "best_model.pt"
    print(f"Saving run to : {run_dir}")

    history          = {"train_loss": [], "val_loss": [], "val_loss_smooth": []}
    best_smooth_val  = float("inf")
    no_improve_count = 0
    ema_val          = None   # initialised on first epoch

    with open(log_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "train_loss", "val_loss", "val_loss_smooth",
                         "lr", "elapsed_s"])

        for epoch in range(1, num_epochs + 1):
            # Teacher forcing: 0.5 → 0 over first half of training
            tf_ratio = max(0.0, 0.5 * (1 - epoch / (num_epochs / 2)))
            t0 = time.time()

            train_loss = train_one_epoch(
                model, train_loader, optimizer, criterion, device, tf_ratio
            )
            val_loss = evaluate(model, val_loader, criterion, device)
            elapsed  = time.time() - t0

            # EMA smoothing — initialise to first val loss so there's no cold-start bias
            if ema_val is None:
                ema_val = val_loss
            else:
                ema_val = ema_alpha * val_loss + (1 - ema_alpha) * ema_val

            # Scheduler and early stopping both use the smoothed value
            scheduler.step(ema_val)
            current_lr = optimizer.param_groups[0]["lr"]

            history["train_loss"].append(train_loss)
            history["val_loss"].append(val_loss)
            history["val_loss_smooth"].append(ema_val)

            writer.writerow([epoch,
                             f"{train_loss:.6f}", f"{val_loss:.6f}",
                             f"{ema_val:.6f}", f"{current_lr:.2e}",
                             f"{elapsed:.1f}"])
            f.flush()

            # Checkpoint on smoothed val improvement
            if ema_val < best_smooth_val:
                best_smooth_val = ema_val
                no_improve_count = 0
                torch.save(model.state_dict(), best_path)
                flag = f" ← best (smooth={ema_val:.4f})"
            else:
                no_improve_count += 1
                flag = f" (no improve {no_improve_count}/{early_stop_patience})"

            print(
                f"Epoch {epoch:3d}/{num_epochs} | "
                f"train={train_loss:.4f}  val={val_loss:.4f}  "
                f"smooth={ema_val:.4f}  lr={current_lr:.2e}  "
                f"[{elapsed:.1f}s]{flag}"
            )

            if no_improve_count >= early_stop_patience:
                print(f"\nEarly stopping triggered after {epoch} epochs "
                      f"(smoothed val no improvement for {early_stop_patience} epochs)")
                break

    print(f"\nBest smoothed val loss: {best_smooth_val:.4f}")
    print(f"Checkpoint saved to {best_path}")

    # Restore best weights before test evaluation
    model.load_state_dict(torch.load(best_path, map_location=device,
                                     weights_only=True))

    if test_loader is not None:
        test_loss = evaluate(model, test_loader, criterion, device)
        print(f"Test MSE loss: {test_loss:.4f}")
        history["test_loss"] = test_loss

    return history