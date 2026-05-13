"""
train.py - Training loop for ThroughputSeq2Seq.

Supports single and multi-target (num_targets > 1). Loss is mean MSE across targets.
"""

from __future__ import annotations

import csv
import os
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau

from model import ThroughputSeq2Seq


def _loss(preds, target, criterion):
    """MSE loss supporting single- and multi-target shapes."""
    if preds.shape != target.shape:
        # shape mismatch: try squeeze/unsqueeze
        if preds.dim() == target.dim() + 1:
            target = target.unsqueeze(-1).expand_as(preds)
        elif target.dim() == preds.dim() + 1:
            preds = preds.unsqueeze(-1).expand_as(target)
    return criterion(preds, target.float())


def train_one_epoch(model, loader, optimizer, criterion, device, tf_ratio=0.5):
    model.train()
    total = 0.0
    for features, hour, dow, target in loader:
        features, hour, dow, target = (
            features.to(device), hour.to(device), dow.to(device), target.to(device))
        optimizer.zero_grad()
        preds, _ = model(features, hour, dow, target, tf_ratio)
        loss = _loss(preds, target, criterion)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()
        total += loss.item() * features.size(0)
    return total / len(loader.dataset)


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    if loader is None:
        return float("nan")
    model.eval()
    total = 0.0
    for features, hour, dow, target in loader:
        features, hour, dow, target = (
            features.to(device), hour.to(device), dow.to(device), target.to(device))
        preds, _ = model(features, hour, dow)
        total += _loss(preds, target, criterion).item() * features.size(0)
    return total / len(loader.dataset)


def train(
    model: ThroughputSeq2Seq,
    train_loader,
    val_loader,
    test_loader       = None,
    num_epochs:   int   = 100,
    lr:           float = 1e-3,
    weight_decay: float = 1e-3,
    lr_factor:    float = 0.5,
    lr_patience:  int   = 8,
    min_lr:       float = 1e-6,
    early_stop_patience: int = 20,
    ema_alpha:    float = 0.3,
    save_dir:     str   = "checkpoints",
    device:       torch.device | None = None,
) -> dict:
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nTraining on: {device}")
    model = model.to(device)

    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=lr_factor,
                                  patience=lr_patience, min_lr=min_lr)
    criterion = nn.MSELoss()

    from datetime import datetime
    run_name  = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir   = Path(save_dir) / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    best_path = run_dir / "best_model.pt"
    log_path  = run_dir / "training_log.csv"
    print(f"Checkpoint dir: {run_dir}")

    history          = {"train_loss": [], "val_loss": [], "val_loss_smooth": []}
    best_smooth_val  = float("inf")
    no_improve_count = 0
    ema_val          = None

    with open(log_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "train_loss", "val_loss", "val_loss_smooth", "lr", "elapsed_s"])

        for epoch in range(1, num_epochs + 1):
            tf_ratio = max(0.0, 0.5 * (1 - epoch / max(1, num_epochs / 2)))
            t0 = time.time()
            train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device, tf_ratio)
            val_loss   = evaluate(model, val_loader, criterion, device)
            elapsed    = time.time() - t0

            ema_val = val_loss if ema_val is None else ema_alpha * val_loss + (1 - ema_alpha) * ema_val
            scheduler.step(ema_val)
            current_lr = optimizer.param_groups[0]["lr"]

            history["train_loss"].append(train_loss)
            history["val_loss"].append(val_loss)
            history["val_loss_smooth"].append(ema_val)

            writer.writerow([epoch, f"{train_loss:.6f}", f"{val_loss:.6f}",
                             f"{ema_val:.6f}", f"{current_lr:.2e}", f"{elapsed:.1f}"])
            f.flush()

            if ema_val < best_smooth_val:
                best_smooth_val  = ema_val
                no_improve_count = 0
                torch.save(model.state_dict(), best_path)
                flag = f" ← best ({ema_val:.4f})"
            else:
                no_improve_count += 1
                flag = f" (no improve {no_improve_count}/{early_stop_patience})"

            print(f"Epoch {epoch:3d}/{num_epochs} | "
                  f"train={train_loss:.4f}  val={val_loss:.4f}  "
                  f"smooth={ema_val:.4f}  lr={current_lr:.2e}  [{elapsed:.1f}s]{flag}")

            if no_improve_count >= early_stop_patience:
                print(f"\nEarly stopping after {epoch} epochs.")
                break

    print(f"\nBest val (smooth): {best_smooth_val:.4f}  ->  {best_path}")
    model.load_state_dict(torch.load(best_path, map_location=device, weights_only=True))

    if test_loader is not None:
        test_loss = evaluate(model, test_loader, criterion, device)
        print(f"Test MSE: {test_loss:.4f}")
        history["test_loss"] = test_loss

    history["checkpoint"] = str(best_path)
    return history
