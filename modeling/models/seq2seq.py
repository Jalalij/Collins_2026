"""
models/seq2seq.py - Seq2Seq GRU trainer (primary model).

Supports single and multi-target prediction.
Output keys in y_true_dict/y_pred_dict:
  - Single target:  "t+1s", "t+2s", ...
  - Multi-target:   "<target_short>_t+1s", "<target_short>_t+2s", ...
"""

from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from .base import BaseTrainer
from model import ThroughputSeq2Seq
from train import train as run_training


def _short(col: str) -> str:
    """Shorten a column name for use as a dict key prefix."""
    return (col.replace("dish_", "")
               .replace("_throughput_mbps", "")
               .replace("_ms_mean", "_ms")
               .replace("_mbps", ""))


def _unwrap(ds):
    """Unwrap ConcatDataset to its first constituent dataset."""
    if hasattr(ds, "datasets"):
        return ds.datasets[0]
    return ds


class Seq2SeqTrainer(BaseTrainer):

    @staticmethod
    def train_and_evaluate(
        train_set,
        test_set,
        results_dir: str,
        config: dict,
        val_set=None,
    ) -> dict:
        cfg         = config or {}
        hidden_size = cfg.get("hidden_size",  128)
        embed_dim   = cfg.get("embed_dim",     16)
        dropout     = cfg.get("dropout",       0.3)
        epochs      = cfg.get("epochs",       100)
        lr          = cfg.get("lr",           1e-3)
        batch_size  = cfg.get("batch_size",   512)

        # Pull metadata from the underlying ThroughputFileDataset
        base         = _unwrap(train_set)
        num_targets  = getattr(base, "num_targets",  1)
        target_cols  = getattr(base, "target_cols",  [f"var_0"])
        target_stats = getattr(base, "target_stats", {})
        normalise    = getattr(base, "normalise",    "zscore")

        train_loader = DataLoader(
            train_set, batch_size=batch_size, shuffle=True,
            pin_memory=torch.cuda.is_available(),
        )

        # If no val_set was provided, fall back to using test_set for validation
        # (better than silently discarding early-stopping signals).
        val_ds = val_set if val_set is not None else test_set
        val_loader = DataLoader(
            val_ds, batch_size=batch_size, shuffle=False,
            pin_memory=torch.cuda.is_available(),
        )

        test_loader = DataLoader(
            test_set, batch_size=batch_size, shuffle=False,
            pin_memory=torch.cuda.is_available(),
        )

        # Infer actual dims from first batch
        feat, _, _, tgt = next(iter(train_loader))
        num_features = feat.shape[-1]   # (B, T, F) -> F
        actual_H     = tgt.shape[1]     # (B, H) or (B, H, T) -> H

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model  = ThroughputSeq2Seq(
            num_features = num_features,
            embed_dim    = embed_dim,
            hidden_size  = hidden_size,
            dropout      = dropout,
            input_len    = feat.shape[1],
            output_len   = actual_H,
            num_targets  = num_targets,
        ).to(device)

        # train() signature: (model, train_loader, val_loader, test_loader=None, ...)
        history = run_training(
            model,
            train_loader,
            val_loader,
            test_loader         = test_loader,
            num_epochs          = epochs,
            lr                  = lr,
            weight_decay        = cfg.get("weight_decay",        1e-3),
            early_stop_patience = cfg.get("early_stop_patience",  20),
            ema_alpha           = cfg.get("ema_alpha",            0.3),
            save_dir            = results_dir,
        )

        ckpt_path = history.get("checkpoint", "")

        # ------------------------------------------------------------------ predictions
        model.eval()
        all_preds, all_targets = [], []
        with torch.no_grad():
            for feat, hour, dow, target in test_loader:
                preds, _ = model(feat.to(device), hour.to(device), dow.to(device))
                all_preds.append(preds.cpu())
                all_targets.append(target.cpu())

        preds_arr   = torch.cat(all_preds,   dim=0)   # (N, H) or (N, H, T)
        targets_arr = torch.cat(all_targets, dim=0)

        def denorm(arr_t: torch.Tensor, col: str) -> np.ndarray:
            stats = target_stats.get(col, {})
            if not stats:
                return arr_t.numpy()
            if normalise == "zscore":
                return (arr_t.float() * stats["std"] + stats["mean"]).numpy()
            return (arr_t.float() * (stats["max"] - stats["min"]) + stats["min"]).numpy()

        y_true_dict: dict[str, np.ndarray] = {}
        y_pred_dict: dict[str, np.ndarray] = {}

        if num_targets == 1:
            col        = target_cols[0]
            preds_np   = denorm(preds_arr,   col)   # (N, H)
            targets_np = denorm(targets_arr, col)
            for h in range(actual_H):
                key = f"t+{h+1}s"
                y_true_dict[key] = targets_np[:, h]
                y_pred_dict[key] = preds_np[:, h]
        else:
            # preds_arr: (N, H, num_targets)
            for ti, col in enumerate(target_cols):
                short      = _short(col)
                preds_np   = denorm(preds_arr[:, :, ti],   col)
                targets_np = denorm(targets_arr[:, :, ti], col)
                for h in range(actual_H):
                    key = f"{short}_t+{h+1}s"
                    y_true_dict[key] = targets_np[:, h]
                    y_pred_dict[key] = preds_np[:, h]

        return {
            "model_name":  "Seq2Seq-GRU",
            "y_true_dict": y_true_dict,
            "y_pred_dict": y_pred_dict,
            "metadata": {
                "model_type":     "Seq2Seq-GRU",
                "hidden_size":    hidden_size,
                "embed_dim":      embed_dim,
                "num_features":   num_features,
                "num_targets":    num_targets,
                "target_cols":    target_cols,
                "output_len":     actual_H,
                "epochs_trained": len(history["train_loss"]),
                "checkpoint":     ckpt_path,
            },
        }