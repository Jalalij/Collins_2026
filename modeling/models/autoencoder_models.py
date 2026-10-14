"""
models/autoencoder_models.py - Predictive Autoencoder variants (MISO).

MISO = Multiple Input, Single Output
  - Input  : full sequence (B, seq_len, F) → flattened to (B, seq_len * F)
  - Output : (B, output_len) — primary target across all horizon steps
  - No autoregressive loops; no decoder sequence generation.

Three variants selectable via config["variant"]:
    "standard"    - PredictiveAutoencoder  (encode → decode → predict)
    "sparse"      - SparseAutoencoder      (L1 sparsity penalty on latent)
    "variational" - VariationalAutoencoder (VAE with KL divergence)

Trainer contract
----------------
train_and_evaluate(cfg: DataConfig, results_dir: str, config: dict) -> dict
"""

from __future__ import annotations

import time
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from .base import BaseTrainer

try:
    from ..dataset import DataConfig, build_dataloaders, get_device
except ImportError:
    from dataset import DataConfig, build_dataloaders, get_device


# ---------------------------------------------------------------------------
# Model definitions
# ---------------------------------------------------------------------------

class PredictiveAutoencoder(nn.Module):
    """Standard autoencoder with a separate predictor head.

    Args:
        input_dim  : flattened sequence length (seq_len * n_features)
        output_dim : output_len (MISO: single target, all horizon steps)
        latent_dim : bottleneck size
        hidden_dim : hidden layer width
    """
    def __init__(self, input_dim, output_dim, latent_dim=32, hidden_dim=128):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim,  hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, latent_dim), nn.ReLU(),
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, input_dim),
        )
        self.predictor = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x):
        """x: (B, input_dim) → pred (B, output_dim), recon (B, input_dim), z"""
        z     = self.encoder(x)
        return self.predictor(z), self.decoder(z), z


class SparseAutoencoder(nn.Module):
    """Autoencoder with L1 sparsity penalty on the latent representation."""
    def __init__(self, input_dim, output_dim, latent_dim=32, hidden_dim=128,
                 sparsity_weight=1e-3):
        super().__init__()
        self.sparsity_weight = sparsity_weight
        self.encoder = nn.Sequential(
            nn.Linear(input_dim,  hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, latent_dim), nn.ReLU(),
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, input_dim),
        )
        self.predictor = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x):
        z             = self.encoder(x)
        sparsity_loss = self.sparsity_weight * z.abs().mean()
        return self.predictor(z), self.decoder(z), z, sparsity_loss


class VariationalAutoencoder(nn.Module):
    """VAE with reparameterisation trick."""
    def __init__(self, input_dim, output_dim, latent_dim=32, hidden_dim=128,
                 beta=1.0):
        super().__init__()
        self.beta = beta
        self.encoder      = nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.ReLU())
        self.mu_layer     = nn.Linear(hidden_dim, latent_dim)
        self.logvar_layer = nn.Linear(hidden_dim, latent_dim)
        self.decoder      = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, input_dim),
        )
        self.predictor    = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x):
        h      = self.encoder(x)
        mu     = self.mu_layer(h)
        logvar = self.logvar_layer(h)
        z      = mu + torch.exp(0.5 * logvar) * torch.randn_like(mu)
        kl     = self.beta * (
            -0.5 * (1 + logvar - mu.pow(2) - logvar.exp()).sum(dim=1).mean()
        )
        return self.predictor(z), self.decoder(z), z, kl


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _unwrap(ds):
    if hasattr(ds, "datasets"):
        return _unwrap(ds.datasets[0])
    return ds


# ---------------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------------

class PredictiveAutoencoderTrainer(BaseTrainer):

    @staticmethod
    def train_and_evaluate(
        cfg: DataConfig,
        results_dir: str,
        config: dict,
    ) -> dict:
        """
        MISO contract
        -------------
        Dataset (io_mode=MISO) yields:
            feat : (B, seq_len, F)
            hour : (B, seq_len)        ← positional, not target
            dow  : (B, seq_len)        ← positional, not target
            tgt  : (B, output_len)     ← primary target, all horizon steps

        The full feature sequence is flattened to (B, seq_len*F) as the
        autoencoder input.  The model predicts (B, output_len) directly.
        """
        variant    = config.get("variant",      "standard")
        batch_size = config.get("batch_size",   64)
        max_epochs = config.get("epochs",       100)
        patience   = config.get("patience",     40)
        lr         = config.get("lr",           1e-3)
        latent_dim = config.get("latent_dim",   32)
        hidden_dim = config.get("hidden_dim",   128)
        recon_w    = config.get("recon_weight", 0.5)

        print(f"\n{'='*20} AUTOENCODER ({variant.upper()}) (MISO) {'='*20}")

        # ------------------------------------------------------------------ data
        train_loader, val_loader, test_loader, _, _, _ = build_dataloaders(cfg)

        eval_loader = val_loader if val_loader is not None else test_loader
        if eval_loader is None:
            eval_loader = train_loader

        base_ds    = _unwrap(train_loader.dataset)
        output_len = cfg.output_len

        # Infer input dimensions from first batch.
        # Batch format: (feat, hour, dow, tgt)  — NOT (x, y)
        feat_b, _, _, tgt_b = next(iter(train_loader))
        seq_len   = feat_b.shape[1]
        feat_dim  = feat_b.shape[2]
        input_dim = seq_len * feat_dim   # flattened MISO input

        # MISO target: (B, output_len) — single target, full horizon
        assert tgt_b.dim() == 2 and tgt_b.shape[1] == output_len, (
            f"Expected MISO target (B, {output_len}), got {tuple(tgt_b.shape)}. "
            f"Ensure DataConfig.io_mode='MISO'."
        )

        print(f"  input_dim  : {input_dim}  (seq_len={seq_len}, feat_dim={feat_dim})")
        print(f"  output_len : {output_len}  (primary target: {cfg.primary_target})")

        # ------------------------------------------------------------------ model
        device = get_device()

        if variant == "sparse":
            model = SparseAutoencoder(
                input_dim, output_len, latent_dim, hidden_dim,
                config.get("sparsity_weight", 1e-3),
            ).to(device)
        elif variant == "variational":
            model = VariationalAutoencoder(
                input_dim, output_len, latent_dim, hidden_dim,
                config.get("beta", 1.0),
            ).to(device)
        else:
            model = PredictiveAutoencoder(
                input_dim, output_len, latent_dim, hidden_dim,
            ).to(device)

        optimizer = torch.optim.Adam(
            model.parameters(), lr=lr,
            weight_decay=config.get("weight_decay", 1e-5),
        )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, patience=5, factor=0.5,
        )
        mse = nn.MSELoss()

        # ------------------------------------------------------------------ training
        best_val, no_improve = float("inf"), 0
        train_losses = []
        train_time   = 0.0

        for epoch in range(max_epochs):
            model.train()
            ep_loss = 0.0
            t0 = time.time()

            for feat, hour, dow, tgt in train_loader:
                # Flatten full sequence to (B, seq_len * F)
                xf  = feat.view(feat.size(0), -1).to(device)   # (B, seq_len*F)
                tgt = tgt.to(device)                            # (B, output_len)

                optimizer.zero_grad()
                out        = model(xf)
                pred, recon = out[0], out[1]
                extra_loss  = out[3] if len(out) == 4 else 0.0

                loss = (mse(pred, tgt)
                        + recon_w * mse(recon, xf)
                        + extra_loss)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                ep_loss += loss.item()

            train_time += time.time() - t0

            model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for feat, hour, dow, tgt in eval_loader:
                    xf  = feat.view(feat.size(0), -1).to(device)
                    tgt = tgt.to(device)
                    val_loss += mse(model(xf)[0], tgt).item()

            avg_train = ep_loss  / max(1, len(train_loader))
            avg_val   = val_loss / max(1, len(eval_loader))
            train_losses.append(avg_train)
            scheduler.step(avg_val)

            if avg_val < best_val:
                best_val, no_improve = avg_val, 0
            else:
                no_improve += 1
                if no_improve >= patience:
                    print(f"  Early stopping at epoch {epoch + 1}")
                    break

        print(f"  Autoencoder ({variant}) complete in {train_time:.2f}s")

        # ------------------------------------------------------------------ inference
        eval_for_pred = test_loader if test_loader is not None else eval_loader

        model.eval()
        preds_list, true_list = [], []
        with torch.no_grad():
            for feat, hour, dow, tgt in eval_for_pred:
                xf   = feat.view(feat.size(0), -1).to(device)
                pred = model(xf)[0].cpu()       # (B, output_len)
                preds_list.append(pred)
                true_list.append(tgt)           # (B, output_len)

        preds_t   = torch.cat(preds_list, dim=0)   # (N, output_len)
        targets_t = torch.cat(true_list,  dim=0)

        # Denormalise using dataset's single-target inverse_transform
        preds_denorm  = base_ds.inverse_transform(preds_t,  target_idx=0).numpy()
        targets_denorm = base_ds.inverse_transform(targets_t, target_idx=0).numpy()

        # Build output dicts: one key per horizon step
        y_true_dict: dict[str, np.ndarray] = {}
        y_pred_dict: dict[str, np.ndarray] = {}
        for h in range(output_len):
            key = f"t+{h+1}s"
            y_true_dict[key] = targets_denorm[:, h]
            y_pred_dict[key] = preds_denorm[:,  h]

        return {
            "model_name":  f"Autoencoder-{variant}",
            "y_true_dict": y_true_dict,
            "y_pred_dict": y_pred_dict,
            "metadata": {
                "variant":           variant,
                "strategy":          "MISO",
                "input_dim":         input_dim,
                "latent_dim":        latent_dim,
                "output_len":        output_len,
                "seq_len":           seq_len,
                "feat_dim":          feat_dim,
                "primary_target":    cfg.primary_target,
                "epochs_trained":    len(train_losses),
                "training_time_sec": train_time,
                "parameters":        sum(p.numel() for p in model.parameters()),
            },
        }