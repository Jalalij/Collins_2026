"""
train.py - Training loop for throughput forecasting models.

Supports three output modes that must match the model architecture:

  MIMO  (Multi-Input Multi-Output)  – (B, output_len, num_targets)
        Models: LSTM, Transformer, Seq2Seq, Attention, StarNet
        Supports teacher-forcing for autoregressive models (Seq2Seq).

  MISO  (Multi-Input Single-Output) – (B, output_len)
        Models: Gaussian Process, Autoencoder, MLP baselines, statistical
        Output head must produce (B, output_len), NOT (B, output_len, num_targets).

  SISO  (Single-Input Single-Output) – (B, output_len)
        Models: ARIMA
        Single input feature; output shape same as MISO.

Pass `mode="mimo" | "miso" | "siso"` to `train()`.
"""

from __future__ import annotations

import csv
import time
from pathlib import Path
from typing import Literal

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau

from dataset import get_device


# ---------------------------------------------------------------------------
# Loss helpers – each asserts the exact shape contract for its mode
# ---------------------------------------------------------------------------

def _loss_mimo(preds: torch.Tensor, target: torch.Tensor, criterion: nn.Module) -> torch.Tensor:
    """
    MIMO: both tensors must be (B, output_len, num_targets).
    target is cast to float; shapes must already match.
    """
    if preds.shape != target.shape:
        raise ValueError(
            f"MIMO shape mismatch: preds {preds.shape} vs target {target.shape}. "
            "Model output head must produce (B, output_len, num_targets)."
        )
    return criterion(preds, target.float())


def _loss_miso(preds: torch.Tensor, target: torch.Tensor, criterion: nn.Module) -> torch.Tensor:
    """
    MISO: both tensors must be (B, output_len).
    Squeezes a trailing size-1 dim if the model accidentally returns (B, output_len, 1).
    """
    if preds.dim() == 3 and preds.shape[-1] == 1:
        preds = preds.squeeze(-1)  # (B, output_len, 1) -> (B, output_len)
    if preds.shape != target.shape:
        raise ValueError(
            f"MISO shape mismatch: preds {preds.shape} vs target {target.shape}. "
            "Model output head must produce (B, output_len), not (B, output_len, num_targets)."
        )
    return criterion(preds, target.float())


def _loss_siso(preds: torch.Tensor, target: torch.Tensor, criterion: nn.Module) -> torch.Tensor:
    """
    SISO: same shape contract as MISO – (B, output_len).
    ARIMA-style models predict a flat sequence of scalars.
    """
    return _loss_miso(preds, target, criterion)


# Map mode string -> loss function
_LOSS_FN = {
    "mimo": _loss_mimo,
    "miso": _loss_miso,
    "siso": _loss_siso,
}


# ---------------------------------------------------------------------------
# Forward helpers
# ---------------------------------------------------------------------------

def _forward_mimo(model, features, hour, dow, target, tf_ratio):
    """
    MIMO forward – supports optional teacher-forcing (Seq2Seq signature).
    Falls back to no-teacher-forcing call if the model doesn't accept `target`.
    Returns (preds, aux) where aux may be None.
    """
    try:
        out = model(features, hour, dow, target, tf_ratio)
    except TypeError:
        out = model(features, hour, dow)
    if isinstance(out, tuple):
        return out[0], out[1] if len(out) > 1 else None
    return out, None


def _forward_miso(model, features, hour, dow):
    """
    MISO forward – no teacher forcing. Returns (preds, aux).
    """
    out = model(features, hour, dow)
    if isinstance(out, tuple):
        preds = out[0]
    else:
        preds = out
    # Enforce (B, output_len) – squeeze trailing 1 if needed
    if preds.dim() == 3 and preds.shape[-1] == 1:
        preds = preds.squeeze(-1)
    return preds, None


def _forward_siso(model, features, hour, dow):
    """
    SISO forward – expects model to accept a single-feature input.
    Returns (preds, aux).
    """
    # SISO models receive (B, seq_len, 1); guard against accidental multi-feature tensors
    if features.dim() == 3 and features.shape[-1] != 1:
        raise ValueError(
            f"SISO model received features with {features.shape[-1]} channels. "
            "Dataset must be configured for single-feature SISO mode."
        )
    out = model(features, hour, dow)
    preds = out[0] if isinstance(out, tuple) else out
    if preds.dim() == 3 and preds.shape[-1] == 1:
        preds = preds.squeeze(-1)
    return preds, None


# ---------------------------------------------------------------------------
# Per-epoch train / evaluate
# ---------------------------------------------------------------------------

def train_one_epoch(
    model,
    loader,
    optimizer,
    criterion,
    device,
    mode: str,
    tf_ratio: float = 0.0,
) -> float:
    model.train()
    loss_fn = _LOSS_FN[mode]
    # Accumulated on-device so each batch doesn't force a GPU sync via .item().
    total = torch.zeros((), device=device)

    for batch in loader:
        features, hour, dow, target = [t.to(device) for t in batch]
        optimizer.zero_grad()

        if mode == "mimo":
            preds, _ = _forward_mimo(model, features, hour, dow, target, tf_ratio)
        elif mode == "miso":
            preds, _ = _forward_miso(model, features, hour, dow)
        else:  # siso
            preds, _ = _forward_siso(model, features, hour, dow)

        loss = loss_fn(preds, target, criterion)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()
        total += loss.detach() * features.size(0)

    return total.item() / len(loader.dataset)


@torch.no_grad()
def evaluate(
    model,
    loader,
    criterion,
    device,
    mode: str,
) -> float:
    if loader is None:
        return float("nan")
    model.eval()
    loss_fn = _LOSS_FN[mode]
    total = torch.zeros((), device=device)

    for batch in loader:
        features, hour, dow, target = [t.to(device) for t in batch]

        if mode == "mimo":
            preds, _ = _forward_mimo(model, features, hour, dow, target, tf_ratio=0.0)
        elif mode == "miso":
            preds, _ = _forward_miso(model, features, hour, dow)
        else:
            preds, _ = _forward_siso(model, features, hour, dow)

        total += loss_fn(preds, target, criterion) * features.size(0)

    return total.item() / len(loader.dataset)


# ---------------------------------------------------------------------------
# Main train() entry point
# ---------------------------------------------------------------------------

def train(
    model,
    train_loader,
    val_loader,
    test_loader=None,
    mode: Literal["mimo", "miso", "siso"] = "mimo",
    num_epochs: int = 100,
    lr: float = 1e-3,
    weight_decay: float = 1e-3,
    lr_factor: float = 0.5,
    lr_patience: int = 8,
    min_lr: float = 1e-6,
    early_stop_patience: int = 20,
    ema_alpha: float = 0.3,
    save_dir: str = "checkpoints",
    device: torch.device | None = None,
) -> dict:
    """
    Train a forecasting model.

    Parameters
    ----------
    model : nn.Module
        The model to train. Its forward pass and output shape must match `mode`.
    train_loader / val_loader / test_loader : DataLoader
        Yield (features, hour, dow, target) batches.
        Target shape depends on mode:
          mimo -> (B, output_len, num_targets)
          miso -> (B, output_len)
          siso -> (B, output_len)   [features must have 1 channel]
    mode : {"mimo", "miso", "siso"}
        Controls loss function, forward dispatch, and shape validation.
    """
    if mode not in _LOSS_FN:
        raise ValueError(f"mode must be one of {list(_LOSS_FN)}; got {mode!r}")

    if device is None:
        device = get_device()
    print(f"\nTraining on: {device}  |  mode: {mode.upper()}")
    model = model.to(device)

    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = ReduceLROnPlateau(
        optimizer, mode="min", factor=lr_factor, patience=lr_patience, min_lr=min_lr
    )
    criterion = nn.MSELoss()

    from datetime import datetime
    run_name = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = Path(save_dir) / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    best_path = run_dir / "best_model.pt"
    log_path = run_dir / "training_log.csv"
    print(f"Checkpoint dir: {run_dir}")

    history = {"train_loss": [], "val_loss": [], "val_loss_smooth": [], "mode": mode}
    best_smooth_val = float("inf")
    no_improve_count = 0
    ema_val = None

    with open(log_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "train_loss", "val_loss", "val_loss_smooth", "lr", "elapsed_s"])

        for epoch in range(1, num_epochs + 1):
            # Teacher-forcing ratio: linearly anneal from 0.5 -> 0 over first half of training.
            # Only meaningful for MIMO Seq2Seq; ignored by MISO/SISO forward functions.
            tf_ratio = max(0.0, 0.5 * (1 - epoch / max(1, num_epochs / 2))) if mode == "mimo" else 0.0

            t0 = time.time()
            train_loss = train_one_epoch(
                model, train_loader, optimizer, criterion, device, mode, tf_ratio
            )
            val_loss = evaluate(model, val_loader, criterion, device, mode)
            elapsed = time.time() - t0

            ema_val = val_loss if ema_val is None else ema_alpha * val_loss + (1 - ema_alpha) * ema_val
            scheduler.step(ema_val)
            current_lr = optimizer.param_groups[0]["lr"]

            history["train_loss"].append(train_loss)
            history["val_loss"].append(val_loss)
            history["val_loss_smooth"].append(ema_val)

            writer.writerow(
                [epoch, f"{train_loss:.6f}", f"{val_loss:.6f}",
                 f"{ema_val:.6f}", f"{current_lr:.2e}", f"{elapsed:.1f}"]
            )
            f.flush()

            if ema_val < best_smooth_val:
                best_smooth_val = ema_val
                no_improve_count = 0
                torch.save(model.state_dict(), best_path)
                flag = f" <- best ({ema_val:.4f})"
            else:
                no_improve_count += 1
                flag = f" (no improve {no_improve_count}/{early_stop_patience})"

            print(
                f"Epoch {epoch:3d}/{num_epochs} | "
                f"train={train_loss:.4f}  val={val_loss:.4f}  "
                f"smooth={ema_val:.4f}  lr={current_lr:.2e}  [{elapsed:.1f}s]{flag}"
            )

            if no_improve_count >= early_stop_patience:
                print(f"\nEarly stopping after {epoch} epochs.")
                break

    print(f"\nBest val (smooth): {best_smooth_val:.4f}  ->  {best_path}")
    model.load_state_dict(torch.load(best_path, map_location=device, weights_only=True))

    if test_loader is not None:
        test_loss = evaluate(model, test_loader, criterion, device, mode)
        print(f"Test MSE ({mode.upper()}): {test_loss:.4f}")
        history["test_loss"] = test_loss

    history["checkpoint"] = str(best_path)
    return history