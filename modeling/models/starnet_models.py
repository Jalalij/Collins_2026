"""
models/starnet_models.py - Satellite GRU Seq2Seq, GRU+Attention, and FNN models.

Three registered model keys:
    satellite_gru       - SatelliteGRU_seq2seq  (GRU encoder-decoder)
    satellite_attention - SatelliteGRU_attention (GRU + Bahdanau attention)
    satellite_fnn       - Satellite_FNN          (feature-compress + linear)

Input convention
----------------
All three original models use:
    x_seq      : (B, T)      — historical values of the TARGET being predicted
    x_seq_attr : (B, T, F)   — all feature columns

Since every target column is also present as a feature column (verified across
all provider/experiment_type combinations), x_seq is extracted from the
features tensor by locating the target column's index in feature_cols.

For multi-target datasets the trainer loops over each target, trains a
separate model for that target, then stacks predictions — since all three
architectures have output_size=1 hardcoded.

Fixes applied to original code
-------------------------------
- Satellite_FNN.forward: squeeze(-1) instead of squeeze() — safe when batch=1
- pretrain_* / encode_forward / encoder_forward removed — not used here
- fc_attr_encoder removed from seq2seq — not used in forward()
- device passed into constructor so internal zero-tensors land on correct device
"""

from __future__ import annotations

import math
import time
from pathlib import Path
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from .base import BaseTrainer


# ---------------------------------------------------------------------------
# Model definitions  (kept as close to original as possible)
# ---------------------------------------------------------------------------

class BahdanauAttention(nn.Module):
    def __init__(self, hidden_size: int):
        super().__init__()
        self.Wa = nn.Linear(hidden_size, hidden_size)
        self.Ua = nn.Linear(hidden_size, hidden_size)
        self.Va = nn.Linear(hidden_size, 1)

    def forward(self, query, keys):
        # query: (B, num_layers, H)  keys: (B, T, H)
        scores  = self.Va(torch.tanh(
            self.Wa(query[:, -1, :].unsqueeze(1)) + self.Ua(keys)
        ))
        scores  = scores.squeeze(2).unsqueeze(1)       # (B, 1, T)
        weights = F.softmax(scores, dim=-1)
        context = torch.bmm(weights, keys)             # (B, 1, H)
        return context, weights


class SatelliteGRU_seq2seq(nn.Module):
    """
    GRU encoder-decoder.
    Encoder input : (B, T, feature_size+1)  — concat(x_seq_attr, x_seq)
    Decoder        : autoregressive single-step GRU, output_len steps
    """

    def __init__(self, seq_len: int, hidden_size: int, num_layers: int,
                 output_len: int, feature_size: int, device,
                 teacher_forcing_ratio: float = 0.3, dropout: float = 0.5):
        super().__init__()
        self.device        = device
        self.seq_len       = seq_len
        self.hidden_size   = hidden_size
        self.num_layers    = num_layers
        self.output_len    = output_len
        self.feature_size  = feature_size   # number of x_seq_attr columns

        self.gru_encoder = nn.GRU(
            feature_size + 1, hidden_size, num_layers, batch_first=True
        )
        self.gru_decoder = nn.GRU(1, hidden_size, num_layers, batch_first=True)
        self.fc_decoder  = nn.Sequential(
            nn.Linear(hidden_size, 128),
            nn.LeakyReLU(),
            nn.Linear(128, 1),
        )

    def _encode(self, x_input: torch.Tensor, B: int) -> torch.Tensor:
        h0 = torch.zeros(self.num_layers, B, self.hidden_size, device=self.device)
        _, hidden = self.gru_encoder(x_input, h0)
        return hidden

    def _decode(self, x_input: torch.Tensor, B: int,
                hidden: torch.Tensor) -> torch.Tensor:
        outputs       = torch.zeros(B, self.output_len, device=self.device)
        # seed decoder with last value of x_seq (index -1 of the last column)
        decoder_input = x_input[:, -1, -1].reshape(B, 1, 1)
        for t in range(self.output_len):
            out, hidden  = self.gru_decoder(decoder_input, hidden)
            out          = self.fc_decoder(out.squeeze(1))  # (B, 1)
            outputs[:, t] = out.squeeze(-1)
            decoder_input = out.unsqueeze(1)                # (B, 1, 1)
        return outputs

    def forward(self, x_seq: torch.Tensor,
                x_seq_attr: torch.Tensor, y_seq=None):
        """
        x_seq      : (B, T)
        x_seq_attr : (B, T, F)
        returns    : (B, output_len), None
        """
        B       = x_seq.size(0)
        x_input = torch.cat([x_seq_attr, x_seq.unsqueeze(-1)], dim=-1)  # (B,T,F+1)
        hidden  = self._encode(x_input, B)
        outputs = self._decode(x_input, B, hidden)
        return outputs, None


class SatelliteGRU_attention(nn.Module):
    """
    GRU encoder with Bahdanau attention decoder.
    """

    def __init__(self, seq_len: int, hidden_size: int, num_layers: int,
                 output_len: int, feature_size: int, device,
                 teacher_forcing_ratio: float = 0.3):
        super().__init__()
        self.device       = device
        self.hidden_size  = hidden_size
        self.num_layers   = num_layers
        self.output_len   = output_len
        self.feature_size = feature_size
        enc_input_dim     = feature_size + 1

        self.attention   = BahdanauAttention(hidden_size)
        self.gru_encoder = nn.GRU(enc_input_dim, hidden_size, num_layers,
                                  batch_first=True)
        # decoder input = context (H) + current step features (F+1)
        self.gru_decoder = nn.GRU(hidden_size + enc_input_dim, hidden_size,
                                  num_layers, batch_first=True)
        self.fc_decoder  = nn.Sequential(
            nn.Linear(hidden_size, 128),
            nn.LeakyReLU(),
            nn.Linear(128, enc_input_dim),
        )
        self.out = nn.Linear(enc_input_dim, 1)

    def _encode(self, x_input: torch.Tensor, B: int):
        h0 = torch.zeros(self.num_layers, B, self.hidden_size, device=self.device)
        enc_out, hidden = self.gru_encoder(x_input, h0)
        return enc_out, hidden

    def _decode_step(self, dec_in, hidden, enc_out):
        query   = hidden.permute(1, 0, 2)              # (B, layers, H)
        context, attn = self.attention(query, enc_out)  # (B,1,H), (B,1,T)
        gru_in  = torch.cat([dec_in, context], dim=2)  # (B,1,H+F+1)
        out, hidden = self.gru_decoder(gru_in, hidden)
        out     = self.fc_decoder(out)                 # (B,1,F+1)
        return out, hidden, attn

    def forward(self, x_seq: torch.Tensor,
                x_seq_attr: torch.Tensor, y_seq=None):
        """
        x_seq      : (B, T)
        x_seq_attr : (B, T, F)
        returns    : (B, output_len), attention weights
        """
        B       = x_seq.size(0)
        x_input = torch.cat([x_seq_attr, x_seq.unsqueeze(-1)], dim=-1)  # (B,T,F+1)
        enc_out, hidden = self._encode(x_input, B)

        dec_in  = x_input[:, -1:, :]   # (B, 1, F+1) — seed from last input step
        outputs    = []
        attentions = []
        for _ in range(self.output_len):
            out, hidden, attn = self._decode_step(dec_in, hidden, enc_out)
            scalar = self.out(out)               # (B, 1, 1)
            outputs.append(scalar.squeeze(-1))   # (B, 1)
            attentions.append(attn)
            dec_in = out.detach()                # feed predicted features as next input

        outputs    = torch.cat(outputs,    dim=1)  # (B, output_len)
        attentions = torch.cat(attentions, dim=1)  # (B, output_len, T)
        return outputs, attentions


class Satellite_FNN(nn.Module):
    """
    Feature-compress + linear map across time dimension.
    Fastest of the three — no recurrence.
    """

    def __init__(self, seq_len: int, hidden_size: int, num_layers: int,
                 output_len: int, feature_size: int, device, dropout: float = 0.5,
                 **kwargs):
        super().__init__()
        self.device      = device
        self.seq_len     = seq_len
        self.output_len  = output_len
        self.feature_size = feature_size

        self.fc_compress = nn.Sequential(
            nn.Linear(feature_size + 1, 1),
            nn.ReLU(),
        )
        self.fc_predict = nn.Linear(seq_len, output_len)

    def forward(self, x_seq: torch.Tensor,
                x_seq_attr: torch.Tensor, y_seq=None):
        """
        x_seq      : (B, T)
        x_seq_attr : (B, T, F)
        returns    : (B, output_len), None
        """
        x_input = torch.cat([x_seq_attr, x_seq.unsqueeze(-1)], dim=-1)  # (B,T,F+1)
        # squeeze(-1) — safe when batch=1 (original squeeze() was not)
        x       = self.fc_compress(x_input).squeeze(-1)   # (B, T)
        outputs = self.fc_predict(x)                       # (B, output_len)
        return outputs, None


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _unwrap(ds):
    if hasattr(ds, "datasets"):
        return ds.datasets[0]
    return ds


def _build_model(arch: str, seq_len: int, output_len: int,
                 feature_size: int, hidden_size: int,
                 num_layers: int, device) -> nn.Module:
    kw = dict(seq_len=seq_len, hidden_size=hidden_size, num_layers=num_layers,
              output_len=output_len, feature_size=feature_size, device=device)
    if arch == "gru":
        return SatelliteGRU_seq2seq(**kw).to(device)
    if arch == "attention":
        return SatelliteGRU_attention(**kw).to(device)
    if arch == "fnn":
        return Satellite_FNN(**kw).to(device)
    raise ValueError(f"Unknown satellite arch '{arch}'")


def _get_xseq(feat: torch.Tensor, target_col: str,
              feature_cols: list[str]) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Split a features tensor (B, T, F) into:
        x_seq      : (B, T)      — historical values of target_col
        x_seq_attr : (B, T, F-1) — all other feature columns

    Since every target is also a feature, we find its index in feature_cols
    and pull it out.  If somehow absent, fall back to zeros.
    """
    if target_col in feature_cols:
        idx        = feature_cols.index(target_col)
        x_seq      = feat[:, :, idx]                                    # (B, T)
        other_idx  = [i for i in range(feat.shape[2]) if i != idx]
        x_seq_attr = feat[:, :, other_idx]                              # (B, T, F-1)
    else:
        # Fallback: zeros for x_seq, all columns for x_seq_attr
        x_seq      = torch.zeros(feat.shape[0], feat.shape[1],
                                 device=feat.device)
        x_seq_attr = feat
    return x_seq, x_seq_attr


def _train_one_target(
    arch: str,
    target_col: str,
    feature_cols: list[str],
    train_loader: DataLoader,
    test_loader:  DataLoader,
    target_idx:   int,
    num_targets:  int,
    seq_len:      int,
    output_len:   int,
    hidden_size:  int,
    num_layers:   int,
    epochs:       int,
    lr:           float,
    patience:     int,
    device,
) -> nn.Module:
    """Train a single satellite model for one target column."""
    # feature_size = F-1 (x_seq takes one slot, x_seq_attr gets the rest)
    feature_size = len(feature_cols) - 1 if target_col in feature_cols \
                   else len(feature_cols)

    model     = _build_model(arch, seq_len, output_len, feature_size,
                             hidden_size, num_layers, device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, patience=5, factor=0.5, verbose=False,
    )
    criterion   = nn.MSELoss()
    best_val    = float("inf")
    no_improve  = 0
    best_state  = None

    for epoch in range(epochs):
        # ---- train ----
        model.train()
        ep_loss = 0.0
        for feat, hour, dow, tgt in train_loader:
            feat = feat.to(device)
            # Extract target for this target_idx
            if num_targets == 1:
                tgt_col = tgt.to(device)              # (B, H)
            else:
                tgt_col = tgt[:, :, target_idx].to(device)  # (B, H)

            x_seq, x_seq_attr = _get_xseq(feat, target_col, feature_cols)
            optimizer.zero_grad()
            preds, _ = model(x_seq, x_seq_attr)       # (B, H)
            loss = criterion(preds, tgt_col.float())
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            ep_loss += loss.item()

        # ---- validate ----
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for feat, hour, dow, tgt in test_loader:
                feat = feat.to(device)
                if num_targets == 1:
                    tgt_col = tgt.to(device)
                else:
                    tgt_col = tgt[:, :, target_idx].to(device)
                x_seq, x_seq_attr = _get_xseq(feat, target_col, feature_cols)
                preds, _ = model(x_seq, x_seq_attr)
                val_loss += criterion(preds, tgt_col.float()).item()

        avg_val = val_loss / max(1, len(test_loader))
        scheduler.step(avg_val)

        if avg_val < best_val:
            best_val   = avg_val
            no_improve = 0
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            no_improve += 1
            if no_improve >= patience:
                break

    if best_state is not None:
        model.load_state_dict({k: v.to(device) for k, v in best_state.items()})
    return model


# ---------------------------------------------------------------------------
# Shared trainer base
# ---------------------------------------------------------------------------

class _SatelliteTrainerBase(BaseTrainer):
    _arch: str = "gru"   # overridden in subclasses

    @classmethod
    def train_and_evaluate(
        cls,
        train_set,
        test_set,
        results_dir: str,
        config: dict,
        val_set=None,
    ) -> dict:
        arch        = cls._arch
        hidden_size = config.get("hidden_size", 128)
        num_layers  = config.get("num_layers",  2)
        epochs      = config.get("epochs",      100)
        lr          = config.get("lr",          1e-3)
        batch_size  = config.get("batch_size",  256)
        patience    = config.get("patience",    15)

        name = f"Satellite-{arch.upper()}"
        print(f"\n{'='*20} {name} {'='*20}")

        base_train   = _unwrap(train_set)
        num_targets  = getattr(base_train, "num_targets",  1)
        target_cols  = getattr(base_train, "target_cols",  ["var_0"])
        feature_cols = getattr(base_train, "feature_cols", [])
        target_stats = getattr(base_train, "target_stats", {})
        normalise    = getattr(base_train, "normalise",    "zscore")

        train_loader = DataLoader(train_set, batch_size=batch_size,
                                  shuffle=True,
                                  pin_memory=torch.cuda.is_available())
        test_loader  = DataLoader(test_set,  batch_size=batch_size,
                                  shuffle=False,
                                  pin_memory=torch.cuda.is_available())

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Infer seq_len and output_len from first batch
        feat_b, _, _, tgt_b = next(iter(train_loader))
        seq_len    = feat_b.shape[1]
        output_len = tgt_b.shape[1]

        def denorm(arr_t: torch.Tensor, col: str) -> np.ndarray:
            stats = target_stats.get(col, {})
            if not stats:
                return arr_t.cpu().numpy()
            if normalise == "zscore":
                return (arr_t.float() * stats["std"] + stats["mean"]).cpu().numpy()
            return (arr_t.float() * (stats["max"] - stats["min"]) + stats["min"]).cpu().numpy()

        def _short(col: str) -> str:
            return (col.replace("dish_", "")
                       .replace("_throughput_mbps", "")
                       .replace("_ms_mean", "_ms")
                       .replace("_mbps", ""))

        y_true_dict: dict[str, np.ndarray] = {}
        y_pred_dict: dict[str, np.ndarray] = {}
        t0 = time.time()

        for ti, target_col in enumerate(target_cols):
            print(f"  Training target {ti+1}/{num_targets}: {target_col}")

            model = _train_one_target(
                arch        = arch,
                target_col  = target_col,
                feature_cols = feature_cols,
                train_loader = train_loader,
                test_loader  = test_loader,
                target_idx   = ti,
                num_targets  = num_targets,
                seq_len      = seq_len,
                output_len   = output_len,
                hidden_size  = hidden_size,
                num_layers   = num_layers,
                epochs       = epochs,
                lr           = lr,
                patience     = patience,
                device       = device,
            )

            # Collect predictions for this target
            model.eval()
            preds_list, true_list = [], []
            with torch.no_grad():
                for feat, hour, dow, tgt in test_loader:
                    feat = feat.to(device)
                    if num_targets == 1:
                        tgt_col_t = tgt
                    else:
                        tgt_col_t = tgt[:, :, ti]
                    x_seq, x_seq_attr = _get_xseq(feat, target_col, feature_cols)
                    preds, _ = model(x_seq, x_seq_attr)
                    preds_list.append(denorm(preds.cpu(), target_col))
                    true_list.append(denorm(tgt_col_t.cpu().float(), target_col))

            preds_np   = np.vstack(preds_list)   # (N, H)
            targets_np = np.vstack(true_list)

            short = _short(target_col)
            for h in range(output_len):
                key = f"t+{h+1}s" if num_targets == 1 else f"{short}_t+{h+1}s"
                y_true_dict[key] = targets_np[:, h]
                y_pred_dict[key] = preds_np[:, h]

        train_time = time.time() - t0
        print(f"{name} complete in {train_time:.1f}s")

        return {
            "model_name":  name,
            "y_true_dict": y_true_dict,
            "y_pred_dict": y_pred_dict,
            "metadata": {
                "model_type":        name,
                "arch":              arch,
                "hidden_size":       hidden_size,
                "num_layers":        num_layers,
                "num_targets":       num_targets,
                "target_cols":       target_cols,
                "output_len":        output_len,
                "training_time_sec": train_time,
            },
        }


# ---------------------------------------------------------------------------
# Concrete trainer classes — one per architecture
# ---------------------------------------------------------------------------

class SatelliteGRUTrainer(_SatelliteTrainerBase):
    _arch = "gru"


class SatelliteAttentionTrainer(_SatelliteTrainerBase):
    _arch = "attention"


class SatelliteFNNTrainer(_SatelliteTrainerBase):
    _arch = "fnn"