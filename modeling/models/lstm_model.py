"""
models/lstm_model.py - 2-layer LSTM next-step forecaster.
"""

from __future__ import annotations

import time

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from .base import BaseTrainer


class SimpleLSTM(nn.Module):
    def __init__(self, input_size, hidden_size=64, num_layers=2,
                 output_size=None, dropout=0.2):
        super().__init__()
        if output_size is None:
            output_size = input_size
        self.hidden_size = hidden_size
        self.num_layers  = num_layers
        self.lstm = nn.LSTM(
            input_size, hidden_size, num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.dropout = nn.Dropout(dropout)
        self.fc      = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        B  = x.size(0)
        h0 = x.new_zeros(self.num_layers, B, self.hidden_size)
        c0 = x.new_zeros(self.num_layers, B, self.hidden_size)
        out, _ = self.lstm(x, (h0, c0))
        return self.fc(self.dropout(out[:, -1, :]))


def _unwrap(ds):
    """Unwrap ConcatDataset to first constituent for metadata access."""
    if hasattr(ds, "datasets"):
        return ds.datasets[0]
    return ds


class LSTMTrainer(BaseTrainer):

    @staticmethod
    def train_and_evaluate(
        train_set,
        test_set,
        results_dir: str,
        config: dict,
        val_set=None,
    ) -> dict:
        print("\n" + "=" * 20 + " LSTM MODEL " + "=" * 20)

        batch_size  = config.get("batch_size",  64)
        max_epochs  = config.get("epochs",      100)
        patience    = config.get("patience",    10)
        hidden_size = config.get("hidden_size", 64)
        num_layers  = config.get("num_layers",  2)
        dropout     = config.get("dropout",     0.2)
        lr          = config.get("lr",          1e-3)

        train_loader = DataLoader(train_set, batch_size=batch_size,
                                  shuffle=True,  drop_last=True)
        test_loader  = DataLoader(test_set,  batch_size=batch_size,
                                  shuffle=False, drop_last=False)

        # ExperimentDataset yields (X [B,T,F], y [B,F+num_targets])
        xb, yb = next(iter(train_loader))
        _, T, F = xb.shape          # F = num_features + 4 (sin/cos encodings)

        # Pull inverse_transform from the underlying base dataset
        base_train = _unwrap(train_set)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model  = SimpleLSTM(F, hidden_size, num_layers, F, dropout).to(device)

        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(
            model.parameters(), lr=lr,
            weight_decay=config.get("weight_decay", 1e-5),
        )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, patience=5, factor=0.5,
        )

        best_val, no_improve = float("inf"), 0
        train_losses = []
        train_time   = 0.0

        for epoch in range(max_epochs):
            model.train()
            ep_loss = 0.0
            t0 = time.time()
            for xb, yb in train_loader:
                xb, yb = xb.to(device), yb.to(device)
                optimizer.zero_grad()
                out  = model(xb)
                loss = criterion(out, yb)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                ep_loss += loss.item()
            train_time += time.time() - t0

            model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for xb, yb in test_loader:
                    out = model(xb.to(device))
                    val_loss += criterion(out, yb.to(device)).item()

            avg_train = ep_loss  / max(1, len(train_loader))
            avg_val   = val_loss / max(1, len(test_loader))
            train_losses.append(avg_train)
            scheduler.step(avg_val)

            if (epoch + 1) % 10 == 0:
                print(f"  Epoch {epoch+1:3d} | train {avg_train:.6f} | val {avg_val:.6f}")

            if avg_val < best_val:
                best_val, no_improve = avg_val, 0
            else:
                no_improve += 1
                if no_improve >= patience:
                    print(f"  Early stopping at epoch {epoch+1}")
                    break

        print(f"LSTM training complete in {train_time:.2f}s")

        # ------------------------------------------------------------------ predictions
        model.eval()
        preds_list, true_list = [], []
        with torch.no_grad():
            for xb, yb in test_loader:
                out = model(xb.to(device)).cpu()
                # inverse_transform lives on the base ThroughputFileDataset/ExperimentDataset
                preds_list.append(base_train.inverse_transform(out).numpy())
                true_list.append(base_train.inverse_transform(yb).numpy())

        y_pred = np.vstack(preds_list)
        y_true = np.vstack(true_list)
        n_out  = y_pred.shape[1]
        targets = [f"var_{i}" for i in range(n_out)]

        return {
            "model_name":  "LSTM",
            "y_true_dict": {targets[i]: y_true[:, i] for i in range(n_out)},
            "y_pred_dict": {targets[i]: y_pred[:, i] for i in range(n_out)},
            "metadata": {
                "model_type":        "LSTM",
                "hidden_size":       hidden_size,
                "num_layers":        num_layers,
                "epochs_trained":    len(train_losses),
                "training_time_sec": train_time,
                "parameters":        sum(p.numel() for p in model.parameters()),
            },
        }