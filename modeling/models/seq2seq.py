"""
models/seq2seq.py - Seq2Seq GRU with Bahdanau attention (MIMO).

Strategy: MIMO (Multiple Input, Multiple Output)
  Input  : (B, input_len, num_features + embed_dim)   — features + periodic time embedding
  Output : (B, output_len, num_targets)                — all targets over the full horizon

The model uses an autoregressive decoder with teacher forcing during training.
Teacher-forcing ratio is annealed from 0.5 → 0 over the first half of training
by train.py; the decoder never sees the ratio directly — train.py controls it.

NOTE: model.py has been inlined here because ThroughputSeq2Seq is private to
this trainer and nothing else imports it.  model.py can be deleted.

Batch format from ThroughputFileDataset (io_mode="MIMO"):
    feat : (B, input_len, num_features)         float32  normalised
    hour : (B, input_len)                        float32  raw 0-23
    dow  : (B, input_len)                        float32  raw 0-6
    tgt  : (B, output_len, num_targets)          float32  normalised  ← always 3-D
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from .base import BaseTrainer
from train import train as run_training

try:
    from ..dataset import DataConfig, build_dataloaders
except ImportError:
    from dataset import DataConfig, build_dataloaders


# ---------------------------------------------------------------------------
# Model components
# ---------------------------------------------------------------------------

class PeriodicEmbedding(nn.Module):
    """
    Maps (hour, dow) cyclical features → learned embedding of size embed_dim.

    Accepts sequence inputs:
        hour : (B, T)  raw 0-23
        dow  : (B, T)  raw 0-6
    Returns:
        (B, T, embed_dim)
    """
    def __init__(self, embed_dim: int = 16):
        super().__init__()
        self.proj = nn.Linear(4, embed_dim)

    def forward(self, hour: torch.Tensor, dow: torch.Tensor) -> torch.Tensor:
        x = torch.stack([
            torch.sin(2 * torch.pi * hour / 24),
            torch.cos(2 * torch.pi * hour / 24),
            torch.sin(2 * torch.pi * dow  /  7),
            torch.cos(2 * torch.pi * dow  /  7),
        ], dim=-1)                              # (B, T, 4)
        return self.proj(x)                     # (B, T, embed_dim)


class BahdanauAttention(nn.Module):
    """
    Additive (Bahdanau) attention over encoder outputs.

        decoder_hidden : (B, hidden_size)   — top-layer hidden state
        encoder_outputs: (B, T, hidden_size)

    Returns:
        context : (B, hidden_size)
        weights : (B, T)
    """
    def __init__(self, hidden_size: int = 128):
        super().__init__()
        self.query_proj  = nn.Linear(hidden_size, hidden_size)
        self.key_proj    = nn.Linear(hidden_size, hidden_size)
        self.energy_proj = nn.Linear(hidden_size, 1)

    def forward(
        self,
        decoder_hidden: torch.Tensor,
        encoder_outputs: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        q       = self.query_proj(decoder_hidden).unsqueeze(1)          # (B, 1, H)
        k       = self.key_proj(encoder_outputs)                        # (B, T, H)
        energy  = self.energy_proj(torch.tanh(q + k)).squeeze(-1)      # (B, T)
        weights = F.softmax(energy, dim=-1)                             # (B, T)
        context = torch.bmm(weights.unsqueeze(1), encoder_outputs).squeeze(1)  # (B, H)
        return context, weights


class Encoder(nn.Module):
    """
    2-layer GRU encoder.

    Input : (B, T, enc_in)
    Returns:
        outputs : (B, T, hidden_size)   — all hidden states
        hidden  : (num_layers, B, hidden_size)
    """
    def __init__(self, enc_in: int, hidden_size: int = 128, dropout: float = 0.3):
        super().__init__()
        self.gru = nn.GRU(
            enc_in, hidden_size,
            num_layers=2, batch_first=True, dropout=dropout,
        )

    def forward(self, x: torch.Tensor):
        return self.gru(x)


class ProjectionHead(nn.Module):
    """Single-target output head: hidden_size → scalar."""
    def __init__(self, hidden_size: int = 128, dropout: float = 0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.LeakyReLU(0.1),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class Decoder(nn.Module):
    """
    Autoregressive GRU decoder with Bahdanau attention.

    At each step:
      1. Attend over encoder outputs using the current top-layer hidden state.
      2. Concatenate context with a learned embedding of the previous prediction.
      3. Run one GRU step.
      4. Project to num_targets via independent heads.

    Output shapes:
        predictions : (B, output_len, num_targets)   ← always 3-D (no squeeze)
        attentions  : (B, output_len, T)
    """
    def __init__(
        self,
        enc_in: int,
        hidden_size: int = 128,
        dropout: float = 0.3,
        num_targets: int = 1,
    ):
        super().__init__()
        self.num_targets  = num_targets
        self.gru = nn.GRU(
            hidden_size + enc_in,   # context + embedded previous prediction
            hidden_size,
            num_layers=2, batch_first=True, dropout=dropout,
        )
        self.attention    = BahdanauAttention(hidden_size)
        self.heads        = nn.ModuleList([
            ProjectionHead(hidden_size, dropout) for _ in range(num_targets)
        ])
        # Projects last prediction (num_targets scalars) back into enc_in space
        # so the decoder input dimension is consistent regardless of num_targets.
        self.output_embed = nn.Linear(num_targets, enc_in)

    def _step(
        self,
        last_out: torch.Tensor,     # (B, num_targets) — previous step prediction
        hidden: torch.Tensor,       # (num_layers, B, hidden_size)
        enc_out: torch.Tensor,      # (B, T, hidden_size)
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        context, attn = self.attention(hidden[-1], enc_out)  # (B, H), (B, T)
        gru_in = torch.cat(
            [context, self.output_embed(last_out)], dim=-1
        ).unsqueeze(1)                                       # (B, 1, H+enc_in)
        h, new_hidden = self.gru(gru_in, hidden)             # (B, 1, H)
        pred = torch.cat(
            [head(h.squeeze(1)) for head in self.heads], dim=-1
        )                                                    # (B, num_targets)
        return pred, new_hidden, attn

    def forward(
        self,
        enc_out: torch.Tensor,          # (B, T, hidden_size)
        enc_hidden: torch.Tensor,       # (num_layers, B, hidden_size)
        output_len: int,
        target: torch.Tensor | None = None,   # (B, output_len, num_targets) or None
        tf_ratio: float = 0.5,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        B        = enc_out.size(0)
        hidden   = enc_hidden
        last_out = torch.zeros(B, self.num_targets, device=enc_out.device)
        preds, attns = [], []

        for t in range(output_len):
            pred, hidden, attn = self._step(last_out, hidden, enc_out)
            preds.append(pred)
            attns.append(attn)

            if target is not None and torch.rand(1).item() < tf_ratio:
                # target is always (B, output_len, num_targets) in MIMO mode
                last_out = target[:, t, :]          # (B, num_targets)
            else:
                last_out = pred.detach()

        predictions = torch.stack(preds, dim=1)     # (B, output_len, num_targets)
        attentions  = torch.stack(attns, dim=1)     # (B, output_len, T)
        return predictions, attentions


class ThroughputSeq2Seq(nn.Module):
    """
    Full Seq2Seq model: PeriodicEmbedding + Encoder + Decoder.

    Always outputs (B, output_len, num_targets) — no squeeze for num_targets==1.
    This keeps the output contract consistent with train.py's MIMO loss, which
    asserts preds.shape == target.shape == (B, output_len, num_targets).

    Args:
        num_features : raw feature count F (before time embedding)
        embed_dim    : periodic time embedding size (0 to disable)
        hidden_size  : GRU hidden units
        dropout      : dropout probability
        input_len    : encoder sequence length
        output_len   : decoder steps (forecast horizon)
        num_targets  : number of output targets
    """
    def __init__(
        self,
        num_features: int = 11,
        embed_dim: int = 16,
        hidden_size: int = 128,
        dropout: float = 0.3,
        input_len: int = 30,
        output_len: int = 15,
        num_targets: int = 1,
    ):
        super().__init__()
        self.output_len  = output_len
        self.num_targets = num_targets

        self.time_embed = PeriodicEmbedding(embed_dim) if embed_dim > 0 else None
        enc_in = num_features + (embed_dim if embed_dim > 0 else 0)

        self.encoder = Encoder(enc_in, hidden_size, dropout)
        self.decoder = Decoder(enc_in, hidden_size, dropout, num_targets)

    def forward(
        self,
        features: torch.Tensor,             # (B, T, num_features)
        hour: torch.Tensor,                 # (B, T)
        dow: torch.Tensor,                  # (B, T)
        target: torch.Tensor | None = None, # (B, output_len, num_targets) — training only
        teacher_forcing_ratio: float = 0.5,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            predictions : (B, output_len, num_targets)
            attentions  : (B, output_len, T)
        """
        x = features
        if self.time_embed is not None:
            x = torch.cat(
                [features, self.time_embed(hour.float(), dow.float())], dim=-1
            )                                               # (B, T, enc_in)

        enc_out, enc_hidden = self.encoder(x)
        return self.decoder(
            enc_out, enc_hidden,
            self.output_len, target, teacher_forcing_ratio,
        )


# ---------------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------------

def _unwrap(ds):
    """Recurse into ConcatDataset to reach the first ThroughputFileDataset."""
    if hasattr(ds, "datasets"):
        return _unwrap(ds.datasets[0])
    return ds


class Seq2SeqTrainer(BaseTrainer):
    """
    MIMO trainer for ThroughputSeq2Seq.

    Follows the same DataConfig-based API as LSTMTrainer:
      train_and_evaluate(cfg, results_dir, config) -> dict

    Output dict keys in y_true_dict / y_pred_dict:
      Single target  : "t+1s", "t+2s", ...
      Multi-target   : "<target_short>_t+1s", "<target_short>_t+2s", ...
    """

    @staticmethod
    def _short(col: str) -> str:
        """Shorten a column name for use as a dict-key prefix."""
        return (
            col.replace("dish_", "")
               .replace("_throughput_mbps", "")
               .replace("_ms_mean", "_ms")
               .replace("_mbps", "")
        )

    @staticmethod
    def train_and_evaluate(
        cfg: DataConfig,
        results_dir: str,
        config: dict,
    ) -> dict:
        print("\n" + "=" * 20 + " SEQ2SEQ MODEL (MIMO) " + "=" * 20)

        # ------------------------------------------------------------------ hparams
        hidden_size  = config.get("hidden_size",         128)
        embed_dim    = config.get("embed_dim",            16)
        dropout      = config.get("dropout",             0.3)
        epochs       = config.get("epochs",              100)
        lr           = config.get("lr",                 1e-3)
        batch_size   = config.get("batch_size",          512)
        weight_decay = config.get("weight_decay",       1e-3)
        es_patience  = config.get("early_stop_patience",  20)
        ema_alpha    = config.get("ema_alpha",            0.3)

        # ------------------------------------------------------------------ data
        train_loader, val_loader, test_loader, _, target_stats, actual_feature_cols = (
            build_dataloaders(cfg)
        )

        num_features = len(actual_feature_cols)
        num_targets  = cfg.num_targets
        output_len   = cfg.output_len
        target_cols  = cfg.targets

        print(f"  num_features : {num_features}  (actual after column intersection)")
        print(f"  num_targets  : {num_targets}  -> {target_cols}")
        print(f"  output_len   : {output_len}")

        # Sanity-check target shape from first batch
        feat_b, _, _, tgt_b = next(iter(train_loader))
        assert tgt_b.shape == (tgt_b.shape[0], output_len, num_targets), (
            f"Expected target shape (B, {output_len}, {num_targets}), "
            f"got {tuple(tgt_b.shape)}. "
            f"Ensure DataConfig.io_mode='MIMO'."
        )

        # ------------------------------------------------------------------ model
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model  = ThroughputSeq2Seq(
            num_features = num_features,
            embed_dim    = embed_dim,
            hidden_size  = hidden_size,
            dropout      = dropout,
            input_len    = feat_b.shape[1],
            output_len   = output_len,
            num_targets  = num_targets,
        ).to(device)

        # ------------------------------------------------------------------ training
        # val_loader falls back to test_loader if no validation split exists.
        effective_val = val_loader if val_loader is not None else test_loader

        history = run_training(
            model,
            train_loader,
            effective_val,
            test_loader         = test_loader,
            mode                = "mimo",       # explicit — never rely on the default
            num_epochs          = epochs,
            lr                  = lr,
            weight_decay        = weight_decay,
            early_stop_patience = es_patience,
            ema_alpha           = ema_alpha,
            save_dir            = results_dir,
        )

        # ------------------------------------------------------------------ inference
        base_ds = _unwrap(train_loader.dataset)

        model.eval()
        preds_list, true_list = [], []
        with torch.no_grad():
            for feat, hour, dow, tgt in test_loader:
                # Forward — no teacher forcing at inference
                preds, _ = model(feat.to(device), hour.to(device), dow.to(device))
                preds_list.append(preds.cpu())
                true_list.append(tgt)           # already (B, output_len, num_targets)

        # (N, output_len, num_targets) for both
        preds_arr   = torch.cat(preds_list, dim=0)
        targets_arr = torch.cat(true_list,  dim=0)

        # Denormalise using dataset's inverse_transform_all which handles
        # (*, num_targets) tensors.  We iterate over horizon steps.
        preds_denorm = np.stack(
            [base_ds.inverse_transform_all(preds_arr[:, t, :]).numpy()
             for t in range(output_len)],
            axis=1,
        )   # (N, output_len, num_targets)
        true_denorm  = np.stack(
            [base_ds.inverse_transform_all(targets_arr[:, t, :].float()).numpy()
             for t in range(output_len)],
            axis=1,
        )   # (N, output_len, num_targets)

        # ------------------------------------------------------------------ output dict
        y_true_dict: dict[str, np.ndarray] = {}
        y_pred_dict: dict[str, np.ndarray] = {}

        if num_targets == 1:
            for h in range(output_len):
                key = f"t+{h + 1}s"
                y_true_dict[key] = true_denorm[:,  h, 0]
                y_pred_dict[key] = preds_denorm[:, h, 0]
        else:
            for ti, col in enumerate(target_cols):
                short = Seq2SeqTrainer._short(col)
                for h in range(output_len):
                    key = f"{short}_t+{h + 1}s"
                    y_true_dict[key] = true_denorm[:,  h, ti]
                    y_pred_dict[key] = preds_denorm[:, h, ti]

        return {
            "model_name":  "Seq2Seq-GRU",
            "y_true_dict": y_true_dict,
            "y_pred_dict": y_pred_dict,
            "metadata": {
                "model_type":     "Seq2Seq-GRU",
                "strategy":       "MIMO",
                "hidden_size":    hidden_size,
                "embed_dim":      embed_dim,
                "num_features":   num_features,
                "num_targets":    num_targets,
                "target_cols":    target_cols,
                "output_len":     output_len,
                "epochs_trained": len(history["train_loss"]),
                "checkpoint":     history.get("checkpoint", ""),
                "parameters":     sum(p.numel() for p in model.parameters()),
            },
        }