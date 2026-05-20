"""
models/transformer_model.py - Encoder-only Transformer next-step forecaster.
"""

from __future__ import annotations

import time
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from .base import BaseTrainer


class SimpleTransformer(nn.Module):
    def __init__(self, input_dim, d_model=64, nhead=4, num_layers=2,
                 dim_feedforward=256, dropout=0.1, max_len=2000):
        super().__init__()
        self.input_projection = nn.Linear(input_dim, d_model)
        pe = torch.zeros(max_len, d_model)
        nn.init.normal_(pe, 0.0, 0.02)
        self.register_buffer("pe", pe)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward,
            dropout=dropout, batch_first=True,
        )
        self.encoder           = nn.TransformerEncoder(enc_layer, num_layers=num_layers)
        self.output_projection = nn.Linear(d_model, input_dim)
        self.dropout           = nn.Dropout(dropout)
        self.d_model           = d_model

    def forward(self, x):
        x = self.input_projection(x) + self.pe[:x.size(1)].unsqueeze(0)
        x = self.dropout(x)
        h = self.encoder(x)
        return self.output_projection(h[:, -1, :])


def _unwrap(ds):
    """Unwrap ConcatDataset to first constituent for metadata/inverse_transform."""
    if hasattr(ds, "datasets"):
        return ds.datasets[0]
    return ds


class TransformerTrainer(BaseTrainer):

    @staticmethod
    def train_and_evaluate(
        train_set,
        test_set,
        results_dir: str,
        config: dict,
        val_set=None,
    ) -> dict:
        print("\n" + "=" * 20 + " TRANSFORMER MODEL " + "=" * 20)

        batch_size      = config.get("batch_size",      64)
        max_epochs      = config.get("epochs",          100)
        patience        = config.get("patience",        10)
        d_model         = config.get("d_model",         64)
        nhead           = config.get("nhead",           4)
        num_layers      = config.get("num_layers",      2)
        dim_feedforward = config.get("dim_feedforward", 256)
        dropout         = config.get("dropout",         0.1)
        lr              = config.get("lr",              1e-3)
        timeout_sec     = config.get("timeout_sec",     600)

        train_loader = DataLoader(train_set, batch_size=batch_size,
                                  shuffle=True,  drop_last=True)
        test_loader  = DataLoader(test_set,  batch_size=batch_size,
                                  shuffle=False, drop_last=False)

        # ExperimentDataset always yields (B, T, F) — shape is reliable
        xb, yb = next(iter(train_loader))
        _, T, F = xb.shape

        base_train = _unwrap(train_set)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model  = SimpleTransformer(
            F, d_model, nhead, num_layers, dim_feedforward,
            dropout, max(2048, T + 1),
        ).to(device)

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
        wall_start   = time.time()

        for epoch in range(max_epochs):
            if time.time() - wall_start > timeout_sec:
                print(f"  Timeout after {epoch} epochs")
                break

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
                    val_loss += criterion(
                        model(xb.to(device)), yb.to(device)
                    ).item()

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

        print(f"Transformer training complete in {train_time:.2f}s")

        model.eval()
        preds_list, true_list = [], []
        with torch.no_grad():
            for xb, yb in test_loader:
                out = model(xb.to(device)).cpu()
                preds_list.append(base_train.inverse_transform(out).numpy())
                true_list.append(base_train.inverse_transform(yb).numpy())

        y_pred  = np.vstack(preds_list)
        y_true  = np.vstack(true_list)
        n_out   = y_pred.shape[1]
        targets = [f"var_{i}" for i in range(n_out)]

        return {
            "model_name":  "Transformer",
            "y_true_dict": {targets[i]: y_true[:, i] for i in range(n_out)},
            "y_pred_dict": {targets[i]: y_pred[:, i] for i in range(n_out)},
            "metadata": {
                "model_type":        "Transformer",
                "d_model":           d_model,
                "epochs_trained":    len(train_losses),
                "training_time_sec": train_time,
                "parameters":        sum(p.numel() for p in model.parameters()),
            },
        }