"""
models/starnet_models.py - Satellite GRU Seq2Seq, GRU+Attention, and FNN models.

Three registered model keys:
    satellite_gru       - SatelliteGRU_seq2seq   (GRU encoder-decoder)
    satellite_attention - SatelliteGRU_attention  (GRU + Bahdanau attention)
    satellite_fnn       - Satellite_FNN           (feature-compress + linear)

Prediction mode: MIMO (Multiple Input, Multiple Output)
-------------------------------------------------
All three architectures receive the full feature tensor (B, T, F) and jointly
predict ALL target columns across ALL forecast horizons in a single forward pass.

    Input:  (B, T, F)                  — full normalised feature tensor
    Output: (B, output_len, num_targets)

Trainer contract
----------------
train_and_evaluate(cfg: DataConfig, results_dir: str, config: dict) -> dict
"""

from __future__ import annotations

import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from .base import BaseTrainer

try:
    from ..dataset import DataConfig, build_dataloaders, get_device
except ImportError:
    from dataset import DataConfig, build_dataloaders, get_device


# ---------------------------------------------------------------------------
# Attention module
# ---------------------------------------------------------------------------

class BahdanauAttention(nn.Module):
    """
    Additive (Bahdanau) attention.

    Forward:
        query : (B, num_layers, H) — current decoder hidden state
        keys  : (B, T, H)          — all encoder output steps

    Returns:
        context : (B, 1, H)
        weights : (B, 1, T)
    """
    def __init__(self, hidden_size: int):
        super().__init__()
        self.Wa = nn.Linear(hidden_size, hidden_size)
        self.Ua = nn.Linear(hidden_size, hidden_size)
        self.Va = nn.Linear(hidden_size, 1)

    def forward(self, query, keys):
        scores  = self.Va(
            torch.tanh(self.Wa(query[:, -1, :].unsqueeze(1)) + self.Ua(keys))
        )                                           # (B, T, 1)
        scores  = scores.squeeze(2).unsqueeze(1)   # (B, 1, T)
        weights = F.softmax(scores, dim=-1)         # (B, 1, T)
        context = torch.bmm(weights, keys)          # (B, 1, H)
        return context, weights


# ---------------------------------------------------------------------------
# Model definitions
# ---------------------------------------------------------------------------

class SatelliteGRU_seq2seq(nn.Module):
    """GRU encoder-decoder (MIMO). Output: (B, output_len, num_targets)."""

    def __init__(self, seq_len, hidden_size, num_layers, output_len,
                 feature_size, num_targets, device, dropout=0.5):
        super().__init__()
        self.device      = device
        self.hidden_size = hidden_size
        self.num_layers  = num_layers
        self.output_len  = output_len
        self.num_targets = num_targets

        self.gru_encoder = nn.GRU(feature_size, hidden_size, num_layers, batch_first=True)
        self.gru_decoder = nn.GRU(num_targets,  hidden_size, num_layers, batch_first=True)
        self.fc_decoder  = nn.Sequential(
            nn.Linear(hidden_size, 128), nn.LeakyReLU(),
            nn.Linear(128, num_targets),
        )

    def _encode(self, x_feat):
        B  = x_feat.size(0)
        h0 = torch.zeros(self.num_layers, B, self.hidden_size, device=self.device)
        _, hidden = self.gru_encoder(x_feat, h0)
        return hidden

    def _decode(self, x_feat, hidden):
        seed       = self.fc_decoder(hidden[-1])        # (B, num_targets)
        decoder_in = seed.unsqueeze(1)                  # (B, 1, num_targets)
        outputs    = []
        for _ in range(self.output_len):
            out, hidden  = self.gru_decoder(decoder_in, hidden)
            pred         = self.fc_decoder(out.squeeze(1))     # (B, num_targets)
            outputs.append(pred.unsqueeze(1))
            decoder_in   = pred.unsqueeze(1).detach()
        return torch.cat(outputs, dim=1)                # (B, output_len, num_targets)

    def forward(self, x_feat, y_seq=None):
        return self._decode(x_feat, self._encode(x_feat)), None


class SatelliteGRU_attention(nn.Module):
    """GRU encoder with Bahdanau attention decoder (MIMO). Output: (B, output_len, num_targets)."""

    def __init__(self, seq_len, hidden_size, num_layers, output_len,
                 feature_size, num_targets, device, teacher_forcing_ratio=0.3):
        super().__init__()
        self.device      = device
        self.hidden_size = hidden_size
        self.num_layers  = num_layers
        self.output_len  = output_len
        self.num_targets = num_targets

        self.attention   = BahdanauAttention(hidden_size)
        self.gru_encoder = nn.GRU(feature_size, hidden_size, num_layers, batch_first=True)
        self.gru_decoder = nn.GRU(hidden_size + num_targets, hidden_size, num_layers, batch_first=True)
        self.fc_decoder  = nn.Sequential(
            nn.Linear(hidden_size, 128), nn.LeakyReLU(),
            nn.Linear(128, feature_size),
        )
        self.out = nn.Linear(feature_size, num_targets)

    def _encode(self, x_feat):
        B  = x_feat.size(0)
        h0 = torch.zeros(self.num_layers, B, self.hidden_size, device=self.device)
        return self.gru_encoder(x_feat, h0)             # enc_out, hidden

    def _decode_step(self, dec_in, hidden, enc_out):
        query          = hidden.permute(1, 0, 2)
        context, attn  = self.attention(query, enc_out)
        gru_in         = torch.cat([dec_in, context], dim=2)
        gru_out, hidden = self.gru_decoder(gru_in, hidden)
        pred           = self.out(self.fc_decoder(gru_out)).squeeze(1)  # (B, num_targets)
        return pred, hidden, attn

    def forward(self, x_feat, y_seq=None):
        enc_out, hidden = self._encode(x_feat)
        seed    = self.out(self.fc_decoder(hidden[-1].unsqueeze(1)))    # (B, 1, num_targets)
        dec_in  = seed.detach()
        outputs, attentions = [], []
        for _ in range(self.output_len):
            pred, hidden, attn = self._decode_step(dec_in, hidden, enc_out)
            outputs.append(pred.unsqueeze(1))
            attentions.append(attn)
            dec_in = pred.unsqueeze(1).detach()
        return torch.cat(outputs, dim=1), torch.cat(attentions, dim=1)


class Satellite_FNN(nn.Module):
    """Feature-compress FNN (MIMO). Output: (B, output_len, num_targets)."""

    def __init__(self, seq_len, hidden_size, num_layers, output_len,
                 feature_size, num_targets, device, dropout=0.5, **kwargs):
        super().__init__()
        self.output_len  = output_len
        self.num_targets = num_targets

        self.fc_compress = nn.Sequential(nn.Linear(feature_size, 1), nn.ReLU())
        self.fc_predict  = nn.Linear(seq_len, output_len * num_targets)

    def forward(self, x_feat, y_seq=None):
        B = x_feat.size(0)
        x = self.fc_compress(x_feat).squeeze(-1)        # (B, T)
        x = self.fc_predict(x)                          # (B, output_len * num_targets)
        return x.view(B, self.output_len, self.num_targets), None


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _unwrap(ds):
    if hasattr(ds, "datasets"):
        return _unwrap(ds.datasets[0])
    return ds


def _build_model(arch, seq_len, output_len, feature_size, num_targets,
                 hidden_size, num_layers, device):
    kw = dict(seq_len=seq_len, hidden_size=hidden_size, num_layers=num_layers,
              output_len=output_len, feature_size=feature_size,
              num_targets=num_targets, device=device)
    if arch == "gru":
        return SatelliteGRU_seq2seq(**kw).to(device)
    if arch == "attention":
        return SatelliteGRU_attention(**kw).to(device)
    if arch == "fnn":
        return Satellite_FNN(**kw).to(device)
    raise ValueError(f"Unknown satellite arch '{arch}'")


# ---------------------------------------------------------------------------
# Shared trainer base
# ---------------------------------------------------------------------------

class _SatelliteTrainerBase(BaseTrainer):
    _arch: str = "gru"

    @classmethod
    def train_and_evaluate(
        cls,
        cfg: DataConfig,
        results_dir: str,
        config: dict,
    ) -> dict:
        arch        = cls._arch
        hidden_size = config.get("hidden_size", 128)
        num_layers  = config.get("num_layers",  2)
        epochs      = config.get("epochs",      100)
        lr          = config.get("lr",          1e-3)
        batch_size  = config.get("batch_size",  256)
        patience    = config.get("patience",    40)

        name = f"Satellite-{arch.upper()}"
        print(f"\n{'='*20} {name} (MIMO) {'='*20}")

        # ------------------------------------------------------------------ data
        train_loader, val_loader, test_loader, _, _, _ = build_dataloaders(cfg)

        eval_loader = val_loader if val_loader is not None else test_loader
        if eval_loader is None:
            eval_loader = train_loader

        base_ds     = _unwrap(train_loader.dataset)
        num_targets = cfg.num_targets
        output_len  = cfg.output_len
        target_cols = cfg.targets

        # Infer seq_len and feature_size from first batch
        feat_b, _, _, tgt_b = next(iter(train_loader))
        seq_len      = feat_b.shape[1]
        feature_size = feat_b.shape[2]

        # Verify MIMO target shape: (B, output_len, num_targets)
        assert tgt_b.shape[1:] == (output_len, num_targets), (
            f"Expected target (B, {output_len}, {num_targets}), got {tuple(tgt_b.shape)}"
        )

        print(f"  feature_size : {feature_size}")
        print(f"  num_targets  : {num_targets}  -> {target_cols}")
        print(f"  output_len   : {output_len}")
        print(f"  seq_len      : {seq_len}")

        # ------------------------------------------------------------------ model
        device = get_device()
        model  = _build_model(arch, seq_len, output_len, feature_size,
                              num_targets, hidden_size, num_layers, device)

        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, patience=5, factor=0.5,
        )
        criterion  = nn.MSELoss()
        best_val   = float("inf")
        no_improve = 0
        best_state = None
        t_start    = time.time()

        # ------------------------------------------------------------------ training
        for epoch in range(epochs):
            model.train()
            ep_loss = 0.0
            for feat, hour, dow, tgt in train_loader:
                feat = feat.to(device)                  # (B, T, F)
                tgt  = tgt.to(device).float()           # (B, output_len, num_targets)
                optimizer.zero_grad()
                preds, _ = model(feat)                  # (B, output_len, num_targets)
                loss = criterion(preds, tgt)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                ep_loss += loss.item()

            model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for feat, hour, dow, tgt in eval_loader:
                    feat = feat.to(device)
                    tgt  = tgt.to(device).float()
                    preds, _ = model(feat)
                    val_loss += criterion(preds, tgt).item()

            avg_val = val_loss / max(1, len(eval_loader))
            scheduler.step(avg_val)

            if avg_val < best_val:
                best_val   = avg_val
                no_improve = 0
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            else:
                no_improve += 1
                if no_improve >= patience:
                    print(f"  Early stopping at epoch {epoch + 1}")
                    break

        train_time = time.time() - t_start
        print(f"  {name} complete in {train_time:.1f}s")

        if best_state is not None:
            model.load_state_dict({k: v.to(device) for k, v in best_state.items()})

        # ------------------------------------------------------------------ inference
        eval_for_pred = test_loader if test_loader is not None else eval_loader

        model.eval()
        preds_list, true_list = [], []
        with torch.no_grad():
            for feat, hour, dow, tgt in eval_for_pred:
                preds, _ = model(feat.to(device))       # (B, output_len, num_targets)
                preds_list.append(preds.cpu())
                true_list.append(tgt)                   # already (B, output_len, num_targets)

        preds_arr   = torch.cat(preds_list, dim=0)      # (N, output_len, num_targets)
        targets_arr = torch.cat(true_list,  dim=0)

        # Denormalise using dataset helper — consistent with all other MIMO trainers
        preds_denorm = np.stack(
            [base_ds.inverse_transform_all(preds_arr[:, t, :]).numpy()
             for t in range(output_len)],
            axis=1,
        )   # (N, output_len, num_targets)
        true_denorm  = np.stack(
            [base_ds.inverse_transform_all(targets_arr[:, t, :].float()).numpy()
             for t in range(output_len)],
            axis=1,
        )

        # ------------------------------------------------------------------ output dicts
        def _short(col):
            return (col.replace("dish_", "")
                       .replace("_throughput_mbps", "")
                       .replace("_ms_mean", "_ms")
                       .replace("_mbps", ""))

        y_true_dict: dict[str, np.ndarray] = {}
        y_pred_dict: dict[str, np.ndarray] = {}

        if num_targets == 1:
            for h in range(output_len):
                key = f"t+{h+1}s"
                y_true_dict[key] = true_denorm[:,  h, 0]
                y_pred_dict[key] = preds_denorm[:, h, 0]
        else:
            for ti, col in enumerate(target_cols):
                short = _short(col)
                for h in range(output_len):
                    key = f"{short}_t+{h+1}s"
                    y_true_dict[key] = true_denorm[:,  h, ti]
                    y_pred_dict[key] = preds_denorm[:, h, ti]

        return {
            "model_name":  name,
            "y_true_dict": y_true_dict,
            "y_pred_dict": y_pred_dict,
            "metadata": {
                "model_type":        name,
                "strategy":          "MIMO",
                "arch":              arch,
                "hidden_size":       hidden_size,
                "num_layers":        num_layers,
                "num_targets":       num_targets,
                "target_cols":       target_cols,
                "output_len":        output_len,
                "feature_size":      feature_size,
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