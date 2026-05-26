"""
models/transformer_model.py - Encoder-only Transformer for multi-horizon target forecasting.

Consumes build_dataloaders() output directly (ThroughputFileDataset via ConcatDataset).
Supports MIMO (Multiple Input, Multiple Output) — one forward pass produces predictions
for all output_len future steps simultaneously, with no autoregressive decoder.

Batch format from ThroughputFileDataset.__getitem__:
    feat : (B, input_len, num_features)          float32  normalised
    hour : (B, input_len)                        float32  raw 0-23
    dow  : (B, input_len)                        float32  raw 0-6
    tgt  : (B, num_targets)                      float32  normalised  [output_len=1, squeezed]
           (B, output_len, num_targets)           float32              [output_len>1, multi-step]

The trainer fuses sin/cos time encodings into the feature tensor, giving a final
input shape of (B, input_len, num_features + 4).

Architecture — MIMO encoder-only Transformer:
    Input projection  : Linear(F+4, d_model)
    Positional enc    : Learned (max_len, d_model), small init, broadcast-added
    Encoder           : N × TransformerEncoderLayer (MHA + FFN + LayerNorm, batch_first)
    Readout           : last sequence token h[:, -1, :]  →  (B, d_model)
    Output head       : Linear(d_model, output_len × num_targets)
    Reshape           : (B, output_len × num_targets)  →  (B, output_len, num_targets)

No causal mask is applied: the encoder attends over the entire historical context,
which is correct — all T input steps are past observations with no future leakage.
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
    produces multi-horizon predictions (B, output_len, num_targets) via a
    single MIMO linear head — no autoregressive decoding.

    The last encoder token h[:, -1, :] summarises the full history and is
    projected to output_len * num_targets values, then reshaped.
    """

    def __init__(
        self,
        input_dim: int,
        num_targets: int,
        output_len: int = 1,
        d_model: int = 64,
        nhead: int = 4,
        num_layers: int = 2,
        dim_feedforward: int = 256,
        dropout: float = 0.1,
        max_len: int = 2048,
    ):
        super().__init__()
        self.output_len  = output_len
        self.num_targets = num_targets

        self.input_projection = nn.Linear(input_dim, d_model)

        # Learned positional encoding — initialised small so it does not
        # dominate at the start of training.  Shape: (max_len, d_model).
        pe = torch.zeros(max_len, d_model)
        nn.init.normal_(pe, 0.0, 0.02)
        self.register_buffer("pe", pe)

        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,        # expects (B, T, d_model)
            norm_first=False,        # post-LN (original transformer) — stable for moderate depth
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=num_layers)

        # MIMO head: one linear layer predicts ALL future steps at once.
        # Output flattened as (output_len * num_targets,) then reshaped.
        self.output_projection = nn.Linear(d_model, output_len * num_targets)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x : (B, T, input_dim)  — fused feature + time-encoding tensor

        Returns:
            (B, output_len, num_targets)
        """
        B, T, _ = x.shape

        # Project to model dimension and add positional encoding.
        # pe[:T] is (T, d_model); unsqueeze(0) broadcasts over the batch.
        x = self.input_projection(x) + self.pe[:T].unsqueeze(0)  # (B, T, d_model)
        x = self.dropout(x)

        # No causal mask: every token attends to the full history.
        # This is correct — all T steps are observed past values.
        h = self.encoder(x)  # (B, T, d_model)

        # Readout from the final token (most temporally recent position).
        last = h[:, -1, :]   # (B, d_model)

        # Project and reshape to (B, output_len, num_targets).
        out = self.output_projection(last)                             # (B, H*M)
        return out.view(B, self.output_len, self.num_targets)          # (B, H, M)


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


def _to_3d(tgt: torch.Tensor, output_len: int) -> torch.Tensor:
    """
    Normalise target tensor to (B, output_len, num_targets) regardless of
    whether the loader squeezed the output_len dimension (output_len=1 case).

        tgt shape        →  returned shape
        (B, M)           →  (B, 1, M)    [squeezed by dataset when output_len=1]
        (B, H, M)        →  (B, H, M)    [already 3-D]
    """
    if tgt.dim() == 2:
        # Dataset squeezed the horizon dim (output_len == 1).
        return tgt.unsqueeze(1)   # (B, 1, M)
    return tgt                    # (B, H, M)


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
        Build dataloaders from cfg, train the Transformer, evaluate on the
        test split, and return a results dict.

        MIMO semantics:
            - cfg.output_len controls how many future steps are predicted.
            - The model predicts all steps in one forward pass (no decoder).
            - Results contain per-target arrays of shape (N, output_len), where
              N is the number of evaluation samples.

        Args:
            cfg         : DataConfig — provider/experiment/feature/target
                          settings.  output_len is respected as-is.
            results_dir : directory for any saved artefacts (not used currently).
            config      : hyper-parameter dict (batch_size, epochs, d_model, …).

        Returns:
            dict with model_name, y_true_dict, y_pred_dict, metadata.
        """
        print("\n" + "=" * 20 + " TRANSFORMER MODEL (MIMO) " + "=" * 20)

        # ------------------------------------------------------------------ hparams
        batch_size      = config.get("batch_size",      64)
        max_epochs      = config.get("epochs",          100)
        patience        = config.get("patience",        40)
        d_model         = config.get("d_model",         64)
        nhead           = config.get("nhead",           4)
        num_layers      = config.get("num_layers",      2)
        dim_feedforward = config.get("dim_feedforward", 256)
        dropout         = config.get("dropout",         0.1)
        lr              = config.get("lr",              1e-3)
        weight_decay    = config.get("weight_decay",    1e-5)
        timeout_sec     = config.get("timeout_sec",     600)

        # ------------------------------------------------------------------ data
        # output_len is NOT overridden here — the caller controls the horizon.
        # The Transformer supports any output_len >= 1 via its MIMO head.
        train_loader, val_loader, test_loader, _, target_stats, feature_cols = (
            build_dataloaders(cfg)
        )
        eval_loader = test_loader if test_loader is not None else val_loader
        if eval_loader is None:
            eval_loader = train_loader

        # Introspect shapes from one batch.
        feat_b, hour_b, dow_b, tgt_b = next(iter(train_loader))
        # feat_b  : (B, T, F_raw)
        # tgt_b   : (B, M)        when output_len == 1 (squeezed by dataset)
        #           (B, H, M)     when output_len >  1
        F_raw = feat_b.shape[2]
        input_dim = F_raw + 4    # +4 for sin/cos hour/dow encodings
        T = feat_b.shape[1]

        if tgt_b.dim() == 2:
            # Dataset squeezed output_len dim — single-step case.
            output_len  = 1
            num_targets = tgt_b.shape[1]
        else:
            # Multi-step: (B, output_len, num_targets)
            output_len  = tgt_b.shape[1]
            num_targets = tgt_b.shape[2]

        print(f"  input_dim   : {input_dim}  (features={F_raw} + time_enc=4)")
        print(f"  num_targets : {num_targets}  -> {cfg.targets}")
        print(f"  output_len  : {output_len}  (MIMO horizon)")
        print(f"  seq_len     : {T}")

        # Grab the first underlying ThroughputFileDataset for inverse_transform.
        base_ds = _unwrap(train_loader.dataset)

        # ------------------------------------------------------------------ model
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = SimpleTransformer(
            input_dim=input_dim,
            num_targets=num_targets,
            output_len=output_len,
            d_model=d_model,
            nhead=nhead,
            num_layers=num_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            max_len=max(2048, T + 1),
        ).to(device)

        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, patience=5, factor=0.5,
        )

        # ------------------------------------------------------------------ training loop
        best_val, no_improve = float("inf"), 0
        train_losses: list[float] = []
        train_time   = 0.0
        wall_start   = time.time()

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
                tgt = _to_3d(tgt, output_len).to(device)                # (B, H, M)

                optimizer.zero_grad()
                out  = model(x)                                          # (B, H, M)
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
                    tgt = _to_3d(tgt, output_len).to(device)
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
        # Collect (B, H, M) predictions and targets, denormalise per target.
        model.eval()
        preds_list: list[np.ndarray] = []
        true_list:  list[np.ndarray] = []

        with torch.no_grad():
            for feat, hour, dow, tgt in eval_loader:
                x   = _fuse_time_encodings(feat, hour, dow).to(device)
                out = model(x).cpu()                # (B, H, M)
                tgt = _to_3d(tgt, output_len)       # (B, H, M)

                B, H, M = out.shape

                # inverse_transform_all expects (N, M); flatten batch × horizon,
                # denormalise, then restore the horizon dimension.
                out_2d  = out.view(B * H, M)
                tgt_2d  = tgt.float().view(B * H, M)
                out_inv = base_ds.inverse_transform_all(out_2d).numpy()   # (B*H, M)
                tgt_inv = base_ds.inverse_transform_all(tgt_2d).numpy()   # (B*H, M)

                preds_list.append(out_inv.reshape(B, H, M))
                true_list.append(tgt_inv.reshape(B, H, M))

        # y_pred / y_true : (N, output_len, num_targets)
        y_pred = np.concatenate(preds_list, axis=0)
        y_true = np.concatenate(true_list,  axis=0)

        n_out        = y_pred.shape[2]
        target_names = list(cfg.targets)[:n_out]

        # Return per-target arrays of shape (N, output_len).
        # Callers expecting single-step results get (N, 1) — they can squeeze if needed.
        return {
            "model_name":  "Transformer",
            "y_true_dict": {target_names[i]: y_true[:, :, i] for i in range(n_out)},
            "y_pred_dict": {target_names[i]: y_pred[:, :, i] for i in range(n_out)},
            "metadata": {
                "model_type":        "Transformer",
                "d_model":           d_model,
                "nhead":             nhead,
                "num_layers":        num_layers,
                "dim_feedforward":   dim_feedforward,
                "num_targets":       num_targets,
                "output_len":        output_len,
                "input_dim":         input_dim,
                "seq_len":           T,
                "epochs_trained":    len(train_losses),
                "best_val_loss":     best_val,
                "training_time_sec": train_time,
                "parameters":        sum(p.numel() for p in model.parameters()),
            },
        }