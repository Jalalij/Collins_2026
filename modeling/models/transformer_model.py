"""
models/transformer_model.py - Encoder-only Transformer for next-step target forecasting.

Consumes build_dataloaders() output directly (ThroughputFileDataset via ConcatDataset).
DataConfig should be created with output_len=1 for next-step prediction.

Batch format from ThroughputFileDataset.__getitem__:
    feat : (B, input_len, num_features)   float32  normalised
    hour : (B, input_len)                 float32  raw 0-23
    dow  : (B, input_len)                 float32  raw 0-6
    tgt  : (B, num_targets)               float32  normalised  [output_len=1, squeezed]
           (B, 1, num_targets)            float32              [output_len>1, multi-step]

The trainer fuses sin/cos time encodings into the feature tensor, giving a final
input shape of (B, input_len, num_features + 4).
"""

from __future__ import annotations

import math
import time

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from .base import BaseTrainer
try:
    from ..dataset import DataConfig, build_dataloaders          # package import
except ImportError:
    from dataset import DataConfig, build_dataloaders             # script/direct import


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

class SimpleTransformer(nn.Module):
    """
    Encoder-only Transformer that reads an input sequence (B, T, F) and
    produces a prediction (B, num_targets) from the last encoder token.
    """

    def __init__(
        self,
        input_dim: int,
        num_targets: int,
        d_model: int = 64,
        nhead: int = 4,
        num_layers: int = 2,
        dim_feedforward: int = 256,
        dropout: float = 0.1,
        max_len: int = 2048,
    ):
        super().__init__()
        self.input_projection  = nn.Linear(input_dim, d_model)

        # Learned positional encoding — initialised small so it doesn't
        # dominate at the start of training
        pe = torch.zeros(max_len, d_model)
        nn.init.normal_(pe, 0.0, 0.02)
        self.register_buffer("pe", pe)

        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
        )
        self.encoder          = nn.TransformerEncoder(enc_layer, num_layers=num_layers)
        self.output_projection = nn.Linear(d_model, num_targets)
        self.dropout          = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, F)
        x = self.input_projection(x) + self.pe[: x.size(1)].unsqueeze(0)
        x = self.dropout(x)
        h = self.encoder(x)
        return self.output_projection(h[:, -1, :])  # (B, num_targets)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _fuse_time_encodings(
    feat: torch.Tensor,
    hour: torch.Tensor,
    dow:  torch.Tensor,
) -> torch.Tensor:
    """
    Append sin/cos cyclical encodings of hour-of-day and day-of-week to feat.

    Args:
        feat : (B, T, F)
        hour : (B, T)   raw 0-23
        dow  : (B, T)   raw 0-6

    Returns:
        (B, T, F+4)
    """
    h_sin = torch.sin(2 * math.pi * hour / 24).unsqueeze(-1)
    h_cos = torch.cos(2 * math.pi * hour / 24).unsqueeze(-1)
    d_sin = torch.sin(2 * math.pi * dow  /  7).unsqueeze(-1)
    d_cos = torch.cos(2 * math.pi * dow  /  7).unsqueeze(-1)
    return torch.cat([feat, h_sin, h_cos, d_sin, d_cos], dim=-1)


def _unwrap(ds):
    """Recurse into ConcatDataset to reach the first ThroughputFileDataset."""
    if hasattr(ds, "datasets"):
        return _unwrap(ds.datasets[0])
    return ds


# ---------------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------------

class TransformerTrainer(BaseTrainer):

    @staticmethod
    def train_and_evaluate(
        cfg: DataConfig,
        results_dir: str,
        config: dict,
    ) -> dict:
        """
        Build dataloaders from cfg (with output_len forced to 1), train the
        Transformer, evaluate on the test split, and return a results dict.

        Args:
            cfg         : DataConfig — provider/experiment/feature/target settings.
                          output_len is overridden to 1 inside this method.
            results_dir : directory for any saved artefacts (not used currently).
            config      : hyper-parameter dict (batch_size, epochs, d_model, …).

        Returns:
            dict with model_name, y_true_dict, y_pred_dict, metadata.
        """
        print("\n" + "=" * 20 + " TRANSFORMER MODEL " + "=" * 20)

        # ------------------------------------------------------------------ hparams
        batch_size      = config.get("batch_size",      64)
        max_epochs      = config.get("epochs",          100)
        patience        = config.get("patience",        10)
        d_model         = config.get("d_model",         64)
        nhead           = config.get("nhead",           4)
        num_layers      = config.get("num_layers",      2)
        dim_feedforward = config.get("dim_feedforward", 256)
        dropout         = config.get("dropout",         0.1)
        lr              = config.get("lr",              1e-3)
        weight_decay    = config.get("weight_decay",    1e-5)
        timeout_sec     = config.get("timeout_sec",     600)

        # ------------------------------------------------------------------ data
        # Force output_len=1: we predict one step ahead, no seq2seq decoder needed.
        cfg.output_len = 1

        train_loader, val_loader, test_loader, _, target_stats, feature_cols = (
            build_dataloaders(cfg)
        )
        # Fall back to train loader for validation if no test split
        eval_loader = test_loader if test_loader is not None else val_loader
        if eval_loader is None:
            eval_loader = train_loader

        # Introspect shapes from one batch
        feat_b, hour_b, dow_b, tgt_b = next(iter(train_loader))
        # feat_b: (B, T, F_raw)  tgt_b: (B, num_targets) since output_len=1
        F_raw       = feat_b.shape[2]
        input_dim   = F_raw + 4          # +4 for sin/cos hour/dow
        T           = feat_b.shape[1]
        num_targets = tgt_b.shape[1] if tgt_b.dim() == 2 else 1

        print(f"  input_dim   : {input_dim}  (features={F_raw} + time_enc=4)")
        print(f"  num_targets : {num_targets}  -> {cfg.targets}")
        print(f"  seq_len     : {T}")

        # Grab the first underlying ThroughputFileDataset for inverse_transform
        base_ds = _unwrap(train_loader.dataset)

        # ------------------------------------------------------------------ model
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model  = SimpleTransformer(
            input_dim, num_targets,
            d_model, nhead, num_layers, dim_feedforward,
            dropout, max(2048, T + 1),
        ).to(device)

        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, patience=5, factor=0.5,
        )

        # ------------------------------------------------------------------ training loop
        best_val, no_improve = float("inf"), 0
        train_losses: list[float] = []
        train_time  = 0.0
        wall_start  = time.time()

        for epoch in range(max_epochs):
            if time.time() - wall_start > timeout_sec:
                print(f"  Timeout after {epoch} epochs")
                break

            # -- train --
            model.train()
            ep_loss = 0.0
            t0 = time.time()
            for feat, hour, dow, tgt in train_loader:
                x   = _fuse_time_encodings(feat, hour, dow).to(device)  # (B, T, F+4)
                tgt = tgt.to(device)
                if tgt.dim() == 1:
                    tgt = tgt.unsqueeze(-1)                              # (B, 1)

                optimizer.zero_grad()
                out  = model(x)                                          # (B, num_targets)
                loss = criterion(out, tgt)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                ep_loss += loss.item()
            train_time += time.time() - t0

            # -- validate --
            model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for feat, hour, dow, tgt in eval_loader:
                    x   = _fuse_time_encodings(feat, hour, dow).to(device)
                    tgt = tgt.to(device)
                    if tgt.dim() == 1:
                        tgt = tgt.unsqueeze(-1)
                    val_loss += criterion(model(x), tgt).item()

            avg_train = ep_loss  / max(1, len(train_loader))
            avg_val   = val_loss / max(1, len(eval_loader))
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

        print(f"  Transformer training complete in {train_time:.2f}s")

        # ------------------------------------------------------------------ evaluation
        model.eval()
        preds_list, true_list = [], []
        with torch.no_grad():
            for feat, hour, dow, tgt in eval_loader:
                x   = _fuse_time_encodings(feat, hour, dow).to(device)
                out = model(x).cpu()                                     # (B, num_targets)
                if tgt.dim() == 1:
                    tgt = tgt.unsqueeze(-1)

                # Denormalise using per-target stats on the base dataset
                preds_list.append(base_ds.inverse_transform_all(out).numpy())
                true_list.append(base_ds.inverse_transform_all(tgt.float()).numpy())

        y_pred = np.vstack(preds_list)   # (N, num_targets)
        y_true = np.vstack(true_list)    # (N, num_targets)
        n_out  = y_pred.shape[1]
        target_names = list(cfg.targets)[:n_out]

        return {
            "model_name":  "Transformer",
            "y_true_dict": {target_names[i]: y_true[:, i] for i in range(n_out)},
            "y_pred_dict": {target_names[i]: y_pred[:, i] for i in range(n_out)},
            "metadata": {
                "model_type":        "Transformer",
                "d_model":           d_model,
                "nhead":             nhead,
                "num_layers":        num_layers,
                "dim_feedforward":   dim_feedforward,
                "num_targets":       num_targets,
                "input_dim":         input_dim,
                "seq_len":           T,
                "epochs_trained":    len(train_losses),
                "best_val_loss":     best_val,
                "training_time_sec": train_time,
                "parameters":        sum(p.numel() for p in model.parameters()),
            },
        }