"""
models/autoencoder_models.py - Predictive Autoencoder variants.

Three variants selectable via config["variant"]:
    "standard"    - PredictiveAutoencoder  (encode -> decode -> predict)
    "sparse"      - SparseAutoencoder      (L1 sparsity penalty on latent)
    "variational" - VariationalAutoencoder (VAE with KL divergence)
"""

from __future__ import annotations

import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from .base import BaseTrainer


# ---------------------------------------------------------------------------
# Model definitions
# ---------------------------------------------------------------------------

class PredictiveAutoencoder(nn.Module):
    def __init__(self, input_dim, latent_dim=32, hidden_dim=128):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, latent_dim), nn.ReLU(),
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, input_dim),
        )
        self.predictor = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, input_dim),
        )

    def forward(self, x):
        z    = self.encoder(x)
        recon = self.decoder(z)
        pred  = self.predictor(z)
        return pred, recon, z


class SparseAutoencoder(nn.Module):
    def __init__(self, input_dim, latent_dim=32, hidden_dim=128, sparsity_weight=1e-3):
        super().__init__()
        self.sparsity_weight = sparsity_weight
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, latent_dim), nn.ReLU(),
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, input_dim),
        )
        self.predictor = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, input_dim),
        )

    def forward(self, x):
        z     = self.encoder(x)
        recon = self.decoder(z)
        pred  = self.predictor(z)
        sparsity_loss = self.sparsity_weight * z.abs().mean()
        return pred, recon, z, sparsity_loss


class VariationalAutoencoder(nn.Module):
    def __init__(self, input_dim, latent_dim=32, hidden_dim=128, beta=1.0):
        super().__init__()
        self.beta = beta
        self.encoder = nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.ReLU())
        self.mu_layer      = nn.Linear(hidden_dim, latent_dim)
        self.logvar_layer  = nn.Linear(hidden_dim, latent_dim)
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, input_dim),
        )
        self.predictor = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, input_dim),
        )

    def reparameterise(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        return mu + std * torch.randn_like(std)

    def forward(self, x):
        h      = self.encoder(x)
        mu     = self.mu_layer(h)
        logvar = self.logvar_layer(h)
        z      = self.reparameterise(mu, logvar)
        recon  = self.decoder(z)
        pred   = self.predictor(z)
        kl     = self.beta * (-0.5 * (1 + logvar - mu.pow(2) - logvar.exp()).sum(dim=1).mean())
        return pred, recon, z, kl


# ---------------------------------------------------------------------------
# Shared trainer
# ---------------------------------------------------------------------------

class PredictiveAutoencoderTrainer(BaseTrainer):

    @staticmethod
    def train_and_evaluate(
        train_set,
        test_set,
        results_dir: str,
        config: dict,
    ) -> dict:
        variant    = config.get("variant",    "standard")
        batch_size = config.get("batch_size", 64)
        max_epochs = config.get("epochs",     100)
        patience   = config.get("patience",   10)
        lr         = config.get("lr",         1e-3)
        latent_dim = config.get("latent_dim", 32)
        hidden_dim = config.get("hidden_dim", 128)
        recon_w    = config.get("recon_weight", 0.5)

        print(f"\n{'='*20} AUTOENCODER ({variant.upper()}) {'='*20}")

        train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True,  drop_last=True)
        test_loader  = DataLoader(test_set,  batch_size=batch_size, shuffle=False, drop_last=False)

        xb, yb = next(iter(train_loader))
        # Use last timestep of input as the flat feature vector
        x_flat = xb[:, -1, :] if xb.dim() == 3 else xb
        input_dim = x_flat.shape[1]
        F = yb.shape[1]

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        if variant == "sparse":
            model = SparseAutoencoder(input_dim, latent_dim, hidden_dim,
                                      config.get("sparsity_weight", 1e-3)).to(device)
        elif variant == "variational":
            model = VariationalAutoencoder(input_dim, latent_dim, hidden_dim,
                                           config.get("beta", 1.0)).to(device)
        else:
            model = PredictiveAutoencoder(input_dim, latent_dim, hidden_dim).to(device)

        optimizer = torch.optim.Adam(model.parameters(), lr=lr,
                                     weight_decay=config.get("weight_decay", 1e-5))
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)
        mse = nn.MSELoss()

        best_val, no_improve = float("inf"), 0
        train_losses = []
        train_time = 0.0

        def get_xflat(xb):
            return (xb[:, -1, :] if xb.dim() == 3 else xb).to(device)

        for epoch in range(max_epochs):
            model.train()
            ep_loss = 0.0
            t0 = time.time()
            for xb, yb in train_loader:
                xf, yb = get_xflat(xb), yb.to(device)
                optimizer.zero_grad()
                out = model(xf)
                pred, recon = out[0], out[1]
                extra_loss  = out[3] if len(out) == 4 else 0.0
                loss = mse(pred[:, :F], yb) + recon_w * mse(recon, xf) + extra_loss
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                ep_loss += loss.item()
            train_time += time.time() - t0

            model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for xb, yb in test_loader:
                    xf, yb = get_xflat(xb), yb.to(device)
                    out = model(xf)
                    val_loss += mse(out[0][:, :F], yb).item()

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

        print(f"Autoencoder ({variant}) complete in {train_time:.2f}s")

        model.eval()
        preds_list, true_list = [], []
        with torch.no_grad():
            for xb, yb in test_loader:
                xf = get_xflat(xb)
                pred = model(xf)[0][:, :F]
                preds_list.append(train_set.inverse_transform(pred).cpu().numpy())
                true_list.append(train_set.inverse_transform(yb).numpy())

        y_pred  = np.vstack(preds_list)
        y_true  = np.vstack(true_list)
        targets = [f"var_{i}" for i in range(F)]
        name    = f"Autoencoder-{variant}"

        return {
            "model_name":  name,
            "y_true_dict": {targets[i]: y_true[:, i] for i in range(F)},
            "y_pred_dict": {targets[i]: y_pred[:, i] for i in range(F)},
            "metadata": {
                "model_type":        name,
                "variant":           variant,
                "latent_dim":        latent_dim,
                "epochs_trained":    len(train_losses),
                "training_time_sec": train_time,
                "parameters":        sum(p.numel() for p in model.parameters()),
            },
        }
