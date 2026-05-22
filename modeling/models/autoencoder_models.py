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
        z     = self.encoder(x)
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
        z            = self.encoder(x)
        recon        = self.decoder(z)
        pred         = self.predictor(z)
        sparsity_loss = self.sparsity_weight * z.abs().mean()
        return pred, recon, z, sparsity_loss


class VariationalAutoencoder(nn.Module):
    def __init__(self, input_dim, latent_dim=32, hidden_dim=128, beta=1.0):
        super().__init__()
        self.beta         = beta
        self.encoder      = nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.ReLU())
        self.mu_layer     = nn.Linear(hidden_dim, latent_dim)
        self.logvar_layer = nn.Linear(hidden_dim, latent_dim)
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
        kl     = self.beta * (
            -0.5 * (1 + logvar - mu.pow(2) - logvar.exp()).sum(dim=1).mean()
        )
        return pred, recon, z, kl


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _unwrap(ds):
    """Unwrap ConcatDataset to first constituent for metadata/inverse_transform."""
    if hasattr(ds, "datasets"):
        return ds.datasets[0]
    return ds


# ---------------------------------------------------------------------------
# Shared trainer
# ---------------------------------------------------------------------------

class PredictiveAutoencoderTrainer:
    """Framework-compatible autoencoder trainer."""

    @staticmethod
    def train_and_evaluate(
        train_set,
        test_set,
        results_dir,
        config,
        val_set=None,
    ) -> dict:

        print("\n" + "=" * 20 + " AUTOENCODER " + "=" * 20)

        try:
            import numpy as np
            import torch
            import torch.nn as nn
            from torch.utils.data import DataLoader

            # -----------------------------
            # CONFIG
            # -----------------------------
            batch_size = config.get("batch_size", 64)
            epochs = config.get("epochs", 20)
            lr = config.get("lr", 1e-3)

            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

            # -----------------------------
            # MODEL (KEEP YOUR ORIGINAL)
            # -----------------------------
            model = PredictiveAutoencoder(
                input_dim=train_set[0][0].shape[-1],
                hidden_dims=[128, 64, 32],
                latent_dim=16,
            ).to(device)

            optimizer = torch.optim.Adam(model.parameters(), lr=lr)
            criterion = nn.MSELoss()

            # -----------------------------
            # DATA LOADERS (FIXED SAFELY)
            # -----------------------------
            def collate(batch):
                x = torch.tensor(np.array([b[0] for b in batch]), dtype=torch.float32)
                y = torch.tensor(np.array([b[1] for b in batch]), dtype=torch.float32)
                return x, y

            train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, collate_fn=collate)
            test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, collate_fn=collate)

            # -----------------------------
            # TRAIN
            # -----------------------------
            model.train()
            for epoch in range(epochs):
                for xb, yb in train_loader:
                    xb, yb = xb.to(device), yb.to(device)

                    optimizer.zero_grad()

                    x_recon, x_pred, z = model(xb)

                    loss = criterion(x_pred, yb)

                    loss.backward()
                    optimizer.step()

            # -----------------------------
            # PREDICT
            # -----------------------------
            model.eval()
            y_true_list, y_pred_list = [], []

            with torch.no_grad():
                for xb, yb in test_loader:
                    xb = xb.to(device)

                    _, x_pred, _ = model(xb)

                    y_true_list.append(yb.numpy())
                    y_pred_list.append(x_pred.cpu().numpy())

            y_true = np.vstack(y_true_list)
            y_pred = np.vstack(y_pred_list)

            # -----------------------------
            # FORMAT OUTPUT
            # -----------------------------
            y_true_dict = {}
            y_pred_dict = {}

            for i in range(y_true.shape[1]):
                key = f"target_{i}"
                y_true_dict[key] = y_true[:, i]
                y_pred_dict[key] = y_pred[:, i]

            metadata = {
                "model": "PredictiveAutoencoder",
                "epochs": epochs,
                "parameters": sum(p.numel() for p in model.parameters()),
            }

            return {
                "model_name": "autoencoder",
                "y_true_dict": y_true_dict,
                "y_pred_dict": y_pred_dict,
                "metadata": metadata,
            }

        except Exception as e:
            import traceback
            print(f"[ERROR] Autoencoder failed: {e}")
            traceback.print_exc()
            return None