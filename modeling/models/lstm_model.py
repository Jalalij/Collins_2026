"""
models/lstm_model.py - 2-layer LSTM for MIMO multi-horizon target forecasting.

Strategy: MIMO (Multiple Input, Multiple Output)
  - Input  : (B, input_len, num_features + 4)  — all features + sin/cos time encodings
  - Output : (B, output_len, num_targets)       — all targets over the full horizon

The final LSTM hidden state is projected to (output_len * num_targets) via a single
linear head, then reshaped.  No seq2seq decoder is used here; that is reserved for
the dedicated Seq2Seq model.

Batch format from ThroughputFileDataset.__getitem__:
    feat : (B, input_len, num_features)       float32  normalised
    hour : (B, input_len)                     float32  raw 0-23
    dow  : (B, input_len)                     float32  raw 0-6
    tgt  : (B, output_len, num_targets)       float32  normalised  [output_len > 1]
           (B, num_targets)                   float32  normalised  [output_len = 1, squeezed]

The trainer always reshapes tgt to (B, output_len, num_targets) internally.
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
    from ..dataset import DataConfig, build_dataloaders, get_device          # package import
except ImportError:
    from dataset import DataConfig, build_dataloaders, get_device             # script/direct import


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

class SimpleLSTM(nn.Module):
    """
    Stacked LSTM that reads (B, T, input_size) and outputs
    (B, output_len, output_size) from the final hidden state.

    The FC head projects the last hidden state to
    (output_len * output_size) and reshapes, giving a direct
    multi-horizon, multi-target prediction in one forward pass.
    """

    def __init__(
        self,
        input_size:  int,
        hidden_size: int = 64,
        num_layers:  int = 2,
        output_size: int = 1,   # num_targets
        output_len:  int = 1,   # forecast horizon
        dropout:     float = 0.2,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers  = num_layers
        self.output_len  = output_len
        self.output_size = output_size

        self.lstm = nn.LSTM(
            input_size,
            hidden_size,
            num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.dropout = nn.Dropout(dropout)
        # Projects last hidden state → (output_len * output_size)
        self.fc = nn.Linear(hidden_size, output_len * output_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x : (B, T, input_size)
        Returns:
            (B, output_len, output_size)
        """
        B  = x.size(0)
        h0 = x.new_zeros(self.num_layers, B, self.hidden_size)
        c0 = x.new_zeros(self.num_layers, B, self.hidden_size)

        _, (h_n, _) = self.lstm(x, (h0, c0))       # h_n: (num_layers, B, H)
        last_hidden  = self.dropout(h_n[-1])         # (B, H)  — top layer only
        out          = self.fc(last_hidden)           # (B, output_len * output_size)
        return out.view(B, self.output_len, self.output_size)


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


def _to_mimo(tgt: torch.Tensor, output_len: int, num_targets: int) -> torch.Tensor:
    """
    Normalise any tgt shape coming out of the dataloader into
    (B, output_len, num_targets), handling both the squeezed
    single-step case and the full multi-step case.

    Shapes handled:
        (B,)                          → (B, 1, 1)
        (B, num_targets)              → (B, 1, num_targets)   [output_len=1 squeezed]
        (B, output_len, num_targets)  → unchanged
    """
    if tgt.dim() == 1:
        tgt = tgt.unsqueeze(-1).unsqueeze(-1)                  # (B, 1, 1)
    elif tgt.dim() == 2:
        tgt = tgt.unsqueeze(1)                                 # (B, 1, num_targets)
    # tgt is now (B, output_len, num_targets)
    assert tgt.shape[1] == output_len,  \
        f"tgt horizon {tgt.shape[1]} != cfg.output_len {output_len}"
    assert tgt.shape[2] == num_targets, \
        f"tgt target count {tgt.shape[2]} != num_targets {num_targets}"
    return tgt


# ---------------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------------

class LSTMTrainer(BaseTrainer):

    @staticmethod
    def train_and_evaluate(
        cfg: DataConfig,
        results_dir: str,
        config: dict,
    ) -> dict:
        """
        Build dataloaders from cfg, train the LSTM, evaluate on the test split,
        and return a results dict.

        MIMO strategy:
          - All input features + time encodings → single encoder (LSTM)
          - Single FC head outputs the full (output_len × num_targets) horizon
          - No separate decoder; that pattern belongs to Seq2Seq

        Args:
            cfg         : DataConfig — provider/experiment/feature/target settings.
                          output_len controls the forecast horizon (≥ 1).
            results_dir : directory for any saved artefacts (not used currently).
            config      : hyper-parameter dict (batch_size, epochs, hidden_size, …).

        Returns:
            dict with model_name, y_true_dict, y_pred_dict, metadata.
        """
        print("\n" + "=" * 20 + " LSTM MODEL (MIMO) " + "=" * 20)

        # ------------------------------------------------------------------ hparams
        batch_size   = config.get("batch_size",   64)
        max_epochs   = config.get("epochs",       100)
        patience     = config.get("patience",     40)
        hidden_size  = config.get("hidden_size",  64)
        num_layers   = config.get("num_layers",   2)
        dropout      = config.get("dropout",      0.2)
        lr           = config.get("lr",           1e-3)
        weight_decay = config.get("weight_decay", 1e-5)

        # ------------------------------------------------------------------ data
        # MIMO: respect cfg.output_len; do NOT force it to 1.
        train_loader, val_loader, test_loader, _, target_stats, feature_cols = (
            build_dataloaders(cfg)
        )
        output_len  = cfg.output_len         # forecast horizon

        # Fall back to train loader for evaluation if no test/val split
        eval_loader = test_loader if test_loader is not None else val_loader
        if eval_loader is None:
            eval_loader = train_loader

        # Introspect shapes from one batch
        feat_b, hour_b, dow_b, tgt_b = next(iter(train_loader))
        F_raw       = feat_b.shape[2]                         # raw feature count
        input_size  = F_raw + 4                               # +4 for sin/cos time enc
        T           = feat_b.shape[1]                         # input sequence length

        # Determine num_targets robustly from tgt shape
        # Dataloader may yield (B, num_targets) when output_len=1 or
        # (B, output_len, num_targets) when output_len>1
        if tgt_b.dim() == 2 and output_len == 1:
            num_targets = tgt_b.shape[1]
        elif tgt_b.dim() == 3:
            num_targets = tgt_b.shape[2]
        else:
            num_targets = 1

        print(f"  input_size  : {input_size}  (features={F_raw} + time_enc=4)")
        print(f"  num_targets : {num_targets}  -> {cfg.targets}")
        print(f"  seq_len     : {T}")
        print(f"  output_len  : {output_len}   (forecast horizon)")

        # Grab the first underlying ThroughputFileDataset for inverse_transform
        base_ds = _unwrap(train_loader.dataset)

        # ------------------------------------------------------------------ model
        device = get_device()
        model  = SimpleLSTM(
            input_size  = input_size,
            hidden_size = hidden_size,
            num_layers  = num_layers,
            output_size = num_targets,
            output_len  = output_len,
            dropout     = dropout,
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

        for epoch in range(max_epochs):
            # -- train --
            model.train()
            ep_loss = 0.0
            t0 = time.time()
            for feat, hour, dow, tgt in train_loader:
                x   = _fuse_time_encodings(feat, hour, dow).to(device)  # (B, T, F+4)
                tgt = _to_mimo(tgt, output_len, num_targets).to(device)  # (B, H, num_targets)

                optimizer.zero_grad()
                out  = model(x)                             # (B, output_len, num_targets)
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
                    tgt = _to_mimo(tgt, output_len, num_targets).to(device)
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

        print(f"  LSTM training complete in {train_time:.2f}s")

        # ------------------------------------------------------------------ evaluation
        model.eval()
        preds_list, true_list = [], []
        with torch.no_grad():
            for feat, hour, dow, tgt in eval_loader:
                x   = _fuse_time_encodings(feat, hour, dow).to(device)
                out = model(x).cpu()                             # (B, output_len, num_targets)
                tgt = _to_mimo(tgt, output_len, num_targets)     # (B, output_len, num_targets)

                # Denormalise each target independently using per-target stats.
                # inverse_transform_all expects (N, num_targets); we iterate steps.
                out_denorm = np.stack(
                    [base_ds.inverse_transform_all(out[:, t, :]).numpy()
                     for t in range(output_len)],
                    axis=1,
                )  # (B, output_len, num_targets)
                tgt_denorm = np.stack(
                    [base_ds.inverse_transform_all(tgt[:, t, :].float()).numpy()
                     for t in range(output_len)],
                    axis=1,
                )  # (B, output_len, num_targets)

                preds_list.append(out_denorm)
                true_list.append(tgt_denorm)

        y_pred = np.vstack(preds_list)   # (N, output_len, num_targets)
        y_true = np.vstack(true_list)    # (N, output_len, num_targets)
        target_names = list(cfg.targets)[:num_targets]

        # Return flat per-target arrays: each value is (N, output_len)
        # Downstream metrics code can average/select steps as needed.
        return {
            "model_name":  "LSTM",
            "y_true_dict": {target_names[i]: y_true[:, :, i] for i in range(num_targets)},
            "y_pred_dict": {target_names[i]: y_pred[:, :, i] for i in range(num_targets)},
            "metadata": {
                "model_type":        "LSTM",
                "strategy":          "MIMO",
                "hidden_size":       hidden_size,
                "num_layers":        num_layers,
                "num_targets":       num_targets,
                "input_size":        input_size,
                "seq_len":           T,
                "output_len":        output_len,
                "epochs_trained":    len(train_losses),
                "best_val_loss":     best_val,
                "training_time_sec": train_time,
                "parameters":        sum(p.numel() for p in model.parameters()),
            },
        }