"""
models/starnet_models.py - Satellite GRU Seq2Seq, GRU+Attention, and FNN models.

Three registered model keys:
    satellite_gru       - SatelliteGRU_seq2seq  (GRU encoder-decoder)
    satellite_attention - SatelliteGRU_attention (GRU + Bahdanau attention)
    satellite_fnn       - Satellite_FNN          (feature-compress + linear)

Prediction mode: MIMO (Multi-Input Multi-Output)
-------------------------------------------------
All three architectures receive the FULL feature tensor (B, T, F) as input
and jointly predict ALL target columns across ALL forecast horizons in a
single forward pass.

    Input:  (B, T, F)  — full feature tensor, targets included as features
    Output: (B, output_len, num_targets)

This replaces the original per-target training loop (which was effectively
MISO applied N times). Joint training allows the model to learn cross-target
correlations and is more parameter-efficient for multi-target settings.

Input convention
----------------
All three models receive:
    x_feat : (B, T, F) — full feature tensor (includes historical target values)

Output dimensions
-----------------
    GRU seq2seq  : (B, output_len, num_targets)
    GRU attention: (B, output_len, num_targets)
    FNN          : (B, output_len, num_targets)

Architecture changes vs original MISO code
-------------------------------------------
- GRU seq2seq  : fc_decoder terminal layer  → Linear(128, num_targets)
                 decoder output shape       → (B, output_len, num_targets)
                 decoder seed               → last timestep of x_feat[:, -1, :]
                                              (was incorrectly pinned to last col)

- GRU attention: self.out                   → Linear(feature_size, num_targets)
                 return shape               → (B, output_len, num_targets)
                 dec_in seed               → x_feat[:, -1:, :]  (full features)

- FNN           : fc_predict               → Linear(seq_len, output_len * num_targets)
                  forward reshape          → (B, output_len, num_targets)

- feature_size  : now len(feature_cols) — no column is stripped out
                  (all target cols remain as historical context features)

- _get_xseq     : removed; no longer needed (full feature tensor used directly)

- Trainer       : single call to _train_all_targets; no per-target loop

Fixes preserved from previous revision
---------------------------------------
- Satellite_FNN.forward: squeeze(-1) instead of squeeze() — safe when batch=1
- device passed into constructor for correct zero-tensor placement
- pretrain_* / encode_forward / encoder_forward not included
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
# Attention module
# ---------------------------------------------------------------------------

class BahdanauAttention(nn.Module):
    """
    Additive (Bahdanau) attention.

    Args:
        hidden_size: encoder/decoder hidden dimension H

    Forward:
        query : (B, num_layers, H) — current decoder hidden state
        keys  : (B, T, H)          — all encoder output steps

    Returns:
        context : (B, 1, H)   — weighted sum of encoder outputs
        weights : (B, 1, T)   — normalised attention scores
    """

    def __init__(self, hidden_size: int):
        super().__init__()
        self.Wa = nn.Linear(hidden_size, hidden_size)
        self.Ua = nn.Linear(hidden_size, hidden_size)
        self.Va = nn.Linear(hidden_size, 1)

    def forward(
        self,
        query: torch.Tensor,   # (B, num_layers, H)
        keys:  torch.Tensor,   # (B, T, H)
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # Use only the top layer of the decoder hidden state as the query.
        # query[:, -1, :] : (B, H) → unsqueeze → (B, 1, H) for broadcasting over T
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
    """
    GRU encoder-decoder (MIMO).

    Encoder processes the full feature tensor (B, T, F) and produces a
    hidden state that seeds the autoregressive decoder.  The decoder
    emits num_targets values at each step for output_len steps.

    Args:
        seq_len      : historical window length T
        hidden_size  : GRU hidden dimension H
        num_layers   : number of stacked GRU layers
        output_len   : forecast horizon
        feature_size : number of input feature columns F
        num_targets  : number of target variables to predict jointly
        device       : torch device (for zero-tensor initialisation)
        dropout      : (unused, kept for API compatibility)
    """

    def __init__(
        self,
        seq_len:      int,
        hidden_size:  int,
        num_layers:   int,
        output_len:   int,
        feature_size: int,
        num_targets:  int,
        device,
        dropout: float = 0.5,
    ):
        super().__init__()
        self.device       = device
        self.hidden_size  = hidden_size
        self.num_layers   = num_layers
        self.output_len   = output_len
        self.num_targets  = num_targets

        # Encoder: reads full feature sequence
        self.gru_encoder = nn.GRU(
            feature_size, hidden_size, num_layers, batch_first=True
        )

        # Decoder: one step at a time; input is previous prediction (num_targets,)
        self.gru_decoder = nn.GRU(
            num_targets, hidden_size, num_layers, batch_first=True
        )

        # Maps decoder hidden state → num_targets predictions
        self.fc_decoder = nn.Sequential(
            nn.Linear(hidden_size, 128),
            nn.LeakyReLU(),
            nn.Linear(128, num_targets),    # MIMO: num_targets outputs per step
        )

    def _encode(self, x_feat: torch.Tensor) -> torch.Tensor:
        """
        Run encoder GRU over full feature sequence.

        Args:
            x_feat : (B, T, F)

        Returns:
            hidden : (num_layers, B, H)
        """
        B  = x_feat.size(0)
        h0 = torch.zeros(self.num_layers, B, self.hidden_size, device=self.device)
        _, hidden = self.gru_encoder(x_feat, h0)
        return hidden

    def _decode(
        self,
        x_feat: torch.Tensor,   # (B, T, F) — used only to seed the first step
        hidden: torch.Tensor,   # (num_layers, B, H)
    ) -> torch.Tensor:
        """
        Autoregressive decoder.  Produces output_len × num_targets predictions.

        The decoder is seeded with the last observed values of the TARGET columns.
        Because all feature columns are passed in (including targets), we use the
        caller-supplied target_indices to slice x_feat[:, -1, :] correctly.  For
        the general case where the caller does not supply indices we simply use the
        full last-step feature vector projected to num_targets via fc_decoder once.

        Seed strategy: use the model's own output projection applied to the last
        encoder hidden state to produce an initial num_targets-dim seed.  This is
        cleaner and architecture-consistent.

        Returns:
            outputs : (B, output_len, num_targets)
        """
        B = x_feat.size(0)

        # Seed: project top encoder hidden layer to num_targets space
        # hidden[-1] : (B, H)
        seed         = self.fc_decoder(hidden[-1])              # (B, num_targets)
        decoder_in   = seed.unsqueeze(1)                        # (B, 1, num_targets)

        outputs = []
        for _ in range(self.output_len):
            out, hidden  = self.gru_decoder(decoder_in, hidden) # out: (B, 1, H)
            pred         = self.fc_decoder(out.squeeze(1))      # (B, num_targets)
            outputs.append(pred.unsqueeze(1))                   # (B, 1, num_targets)
            decoder_in   = pred.unsqueeze(1).detach()           # feed back prediction

        return torch.cat(outputs, dim=1)                        # (B, output_len, num_targets)

    def forward(
        self,
        x_feat: torch.Tensor,   # (B, T, F)
        y_seq=None,             # unused, kept for API compatibility
    ) -> tuple[torch.Tensor, None]:
        """
        Args:
            x_feat : (B, T, F) — full feature tensor

        Returns:
            outputs : (B, output_len, num_targets)
            None    : no auxiliary output for this architecture
        """
        hidden  = self._encode(x_feat)
        outputs = self._decode(x_feat, hidden)
        return outputs, None


class SatelliteGRU_attention(nn.Module):
    """
    GRU encoder with Bahdanau attention decoder (MIMO).

    Each decoder step attends over the full encoder output sequence to
    build a context vector, then predicts all num_targets values jointly.

    Args:
        seq_len      : historical window length T
        hidden_size  : GRU hidden dimension H
        num_layers   : number of stacked GRU layers
        output_len   : forecast horizon
        feature_size : number of input feature columns F
        num_targets  : number of target variables to predict jointly
        device       : torch device
    """

    def __init__(
        self,
        seq_len:      int,
        hidden_size:  int,
        num_layers:   int,
        output_len:   int,
        feature_size: int,
        num_targets:  int,
        device,
        teacher_forcing_ratio: float = 0.3,  # kept for API compatibility
    ):
        super().__init__()
        self.device      = device
        self.hidden_size = hidden_size
        self.num_layers  = num_layers
        self.output_len  = output_len
        self.num_targets = num_targets

        self.attention   = BahdanauAttention(hidden_size)

        # Encoder: full feature sequence
        self.gru_encoder = nn.GRU(
            feature_size, hidden_size, num_layers, batch_first=True
        )

        # Decoder input at each step: context (H) + previous prediction (num_targets)
        self.gru_decoder = nn.GRU(
            hidden_size + num_targets, hidden_size, num_layers, batch_first=True
        )

        # Intermediate projection (kept for representational depth)
        self.fc_decoder = nn.Sequential(
            nn.Linear(hidden_size, 128),
            nn.LeakyReLU(),
            nn.Linear(128, feature_size),   # project to feature space (intermediate)
        )

        # Final output head: feature_size → num_targets
        self.out = nn.Linear(feature_size, num_targets)

    def _encode(
        self,
        x_feat: torch.Tensor,   # (B, T, F)
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            enc_out : (B, T, H) — all encoder hidden states
            hidden  : (num_layers, B, H) — final encoder hidden state
        """
        B  = x_feat.size(0)
        h0 = torch.zeros(self.num_layers, B, self.hidden_size, device=self.device)
        enc_out, hidden = self.gru_encoder(x_feat, h0)
        return enc_out, hidden

    def _decode_step(
        self,
        dec_in:  torch.Tensor,  # (B, 1, num_targets) — previous prediction
        hidden:  torch.Tensor,  # (num_layers, B, H)
        enc_out: torch.Tensor,  # (B, T, H)
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        One decoder step: attend → GRU → project → output.

        Returns:
            pred    : (B, num_targets)  — prediction for this step
            hidden  : (num_layers, B, H)
            attn    : (B, 1, T)         — attention weights
        """
        query          = hidden.permute(1, 0, 2)           # (B, num_layers, H)
        context, attn  = self.attention(query, enc_out)    # (B, 1, H), (B, 1, T)
        gru_in         = torch.cat([dec_in, context], dim=2)   # (B, 1, H+num_targets)
        gru_out, hidden = self.gru_decoder(gru_in, hidden)     # (B, 1, H)
        feat_proj      = self.fc_decoder(gru_out)          # (B, 1, feature_size)
        pred           = self.out(feat_proj).squeeze(1)    # (B, num_targets)
        return pred, hidden, attn

    def forward(
        self,
        x_feat: torch.Tensor,   # (B, T, F)
        y_seq=None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x_feat : (B, T, F)

        Returns:
            outputs    : (B, output_len, num_targets)
            attentions : (B, output_len, T)
        """
        enc_out, hidden = self._encode(x_feat)

        # Seed: project top encoder hidden state to num_targets space
        # hidden[-1]: (B, H) → fc path → (B, num_targets) → (B, 1, num_targets)
        seed    = self.out(self.fc_decoder(hidden[-1].unsqueeze(1)))  # (B, 1, num_targets)
        dec_in  = seed.detach()

        outputs, attentions = [], []
        for _ in range(self.output_len):
            pred, hidden, attn = self._decode_step(dec_in, hidden, enc_out)
            outputs.append(pred.unsqueeze(1))           # (B, 1, num_targets)
            attentions.append(attn)                     # (B, 1, T)
            dec_in = pred.unsqueeze(1).detach()         # feed prediction back

        outputs    = torch.cat(outputs,    dim=1)       # (B, output_len, num_targets)
        attentions = torch.cat(attentions, dim=1)       # (B, output_len, T)
        return outputs, attentions


class Satellite_FNN(nn.Module):
    """
    Feature-compress + linear map across time dimension (MIMO).

    Fastest of the three — no recurrence, fully parallelisable.
    Compresses the feature dimension to a scalar per timestep, then
    maps the resulting (B, T) sequence to (B, output_len * num_targets)
    and reshapes to (B, output_len, num_targets).

    Args:
        seq_len      : historical window length T
        hidden_size  : unused (kept for API compatibility)
        num_layers   : unused (kept for API compatibility)
        output_len   : forecast horizon
        feature_size : number of input feature columns F
        num_targets  : number of target variables to predict jointly
        device       : torch device
        dropout      : (unused, kept for API compatibility)
    """

    def __init__(
        self,
        seq_len:      int,
        hidden_size:  int,
        num_layers:   int,
        output_len:   int,
        feature_size: int,
        num_targets:  int,
        device,
        dropout: float = 0.5,
        **kwargs,
    ):
        super().__init__()
        self.device      = device
        self.seq_len     = seq_len
        self.output_len  = output_len
        self.num_targets = num_targets

        # Compress F feature channels → 1 scalar per timestep
        self.fc_compress = nn.Sequential(
            nn.Linear(feature_size, 1),
            nn.ReLU(),
        )

        # Map compressed time sequence → all targets × all horizons
        # Output is (B, output_len * num_targets), then reshaped
        self.fc_predict = nn.Linear(seq_len, output_len * num_targets)

    def forward(
        self,
        x_feat: torch.Tensor,   # (B, T, F)
        y_seq=None,
    ) -> tuple[torch.Tensor, None]:
        """
        Args:
            x_feat : (B, T, F)

        Returns:
            outputs : (B, output_len, num_targets)
            None
        """
        B = x_feat.size(0)
        x = self.fc_compress(x_feat).squeeze(-1)    # (B, T) — squeeze(-1) safe for B=1
        x = self.fc_predict(x)                      # (B, output_len * num_targets)
        return x.view(B, self.output_len, self.num_targets), None


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _unwrap(ds):
    """Unwrap ConcatDataset to reach the underlying dataset for attribute access."""
    if hasattr(ds, "datasets"):
        return ds.datasets[0]
    return ds


def _build_model(
    arch:         str,
    seq_len:      int,
    output_len:   int,
    feature_size: int,
    num_targets:  int,
    hidden_size:  int,
    num_layers:   int,
    device,
) -> nn.Module:
    kw = dict(
        seq_len=seq_len,
        hidden_size=hidden_size,
        num_layers=num_layers,
        output_len=output_len,
        feature_size=feature_size,
        num_targets=num_targets,
        device=device,
    )
    if arch == "gru":
        return SatelliteGRU_seq2seq(**kw).to(device)
    if arch == "attention":
        return SatelliteGRU_attention(**kw).to(device)
    if arch == "fnn":
        return Satellite_FNN(**kw).to(device)
    raise ValueError(f"Unknown satellite arch '{arch}'")


def _train_all_targets(
    arch:         str,
    feature_cols: list[str],
    target_cols:  list[str],
    train_loader: DataLoader,
    test_loader:  DataLoader,
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
    """
    Train a single MIMO satellite model that jointly predicts all target columns.

    The full feature tensor (B, T, F) is fed to the model, which predicts
    (B, output_len, num_targets) in one forward pass.  Loss is MSE over all
    targets and horizons simultaneously.
    """
    feature_size = len(feature_cols)    # all columns used — no stripping

    model = _build_model(
        arch, seq_len, output_len, feature_size, num_targets,
        hidden_size, num_layers, device,
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, patience=5, factor=0.5,
    )
    criterion  = nn.MSELoss()
    best_val   = float("inf")
    no_improve = 0
    best_state = None

    for epoch in range(epochs):
        # ---- train ----
        model.train()
        ep_loss = 0.0
        for feat, hour, dow, tgt in train_loader:
            feat = feat.to(device)                  # (B, T, F)

            # tgt shape: (B, H) for single-target, (B, H, num_targets) for multi
            if num_targets == 1:
                tgt_all = tgt.to(device).unsqueeze(-1)          # (B, H, 1)
            else:
                tgt_all = tgt.to(device).float()                # (B, H, num_targets)

            optimizer.zero_grad()
            preds, _ = model(feat)                              # (B, H, num_targets)
            loss = criterion(preds, tgt_all)
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
                    tgt_all = tgt.to(device).unsqueeze(-1)
                else:
                    tgt_all = tgt.to(device).float()
                preds, _ = model(feat)
                val_loss += criterion(preds, tgt_all).item()

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

        train_loader = DataLoader(
            train_set, batch_size=batch_size, shuffle=True,
            pin_memory=torch.cuda.is_available(),
        )
        test_loader  = DataLoader(
            test_set, batch_size=batch_size, shuffle=False,
            pin_memory=torch.cuda.is_available(),
        )

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Infer seq_len and output_len from first batch
        feat_b, _, _, tgt_b = next(iter(train_loader))
        seq_len    = feat_b.shape[1]
        output_len = tgt_b.shape[1]

        # --- Train single MIMO model for all targets ---
        print(f"  Training MIMO model: {num_targets} target(s), "
              f"horizon={output_len}, features={len(feature_cols)}")

        t0 = time.time()
        model = _train_all_targets(
            arch         = arch,
            feature_cols = feature_cols,
            target_cols  = target_cols,
            train_loader = train_loader,
            test_loader  = test_loader,
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
        train_time = time.time() - t0
        print(f"{name} complete in {train_time:.1f}s")

        # --- Collect predictions ---
        def denorm(arr: np.ndarray, col: str) -> np.ndarray:
            stats = target_stats.get(col, {})
            if not stats:
                return arr
            if normalise == "zscore":
                return arr * stats["std"] + stats["mean"]
            return arr * (stats["max"] - stats["min"]) + stats["min"]

        def _short(col: str) -> str:
            return (col.replace("dish_", "")
                       .replace("_throughput_mbps", "")
                       .replace("_ms_mean", "_ms")
                       .replace("_mbps", ""))

        model.eval()
        preds_list, true_list = [], []
        with torch.no_grad():
            for feat, hour, dow, tgt in test_loader:
                feat = feat.to(device)                          # (B, T, F)
                preds, _ = model(feat)                          # (B, H, num_targets)
                preds_list.append(preds.cpu().float().numpy())

                if num_targets == 1:
                    true_list.append(tgt.unsqueeze(-1).float().numpy())
                else:
                    true_list.append(tgt.float().numpy())       # (B, H, num_targets)

        # preds_np : (N, output_len, num_targets)
        preds_np   = np.concatenate(preds_list, axis=0)
        targets_np = np.concatenate(true_list,  axis=0)

        # Build result dicts keyed by "col_t+Hs" or "t+Hs" (single target)
        y_true_dict: dict[str, np.ndarray] = {}
        y_pred_dict: dict[str, np.ndarray] = {}

        for ti, target_col in enumerate(target_cols):
            short = _short(target_col)
            pred_col   = denorm(preds_np[:, :, ti],   target_col)  # (N, H)
            target_col_ = denorm(targets_np[:, :, ti], target_col)

            for h in range(output_len):
                key = f"t+{h+1}s" if num_targets == 1 else f"{short}_t+{h+1}s"
                y_true_dict[key] = target_col_[:, h]
                y_pred_dict[key] = pred_col[:, h]

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
                "prediction_mode":   "MIMO",
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