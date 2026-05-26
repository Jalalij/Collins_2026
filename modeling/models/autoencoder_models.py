"""
models/autoencoder_models.py - Predictive Autoencoder variants (MISO, single-step).

MISO = Multiple Input, Single Output
  - Input  : full sequence (batch, seq_len, input_dim) → flattened to (batch, seq_len * input_dim)
  - Output : single-step prediction (batch, n_out)
  - No autoregressive loops; no decoder sequence generation.

Three variants selectable via config["variant"]:
    "standard"    - PredictiveAutoencoder  (encode → decode → predict)
    "sparse"      - SparseAutoencoder      (L1 sparsity penalty on latent)
    "variational" - VariationalAutoencoder (VAE with KL divergence)
"""

from __future__ import annotations

import time
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from .base import BaseTrainer


# ---------------------------------------------------------------------------
# Model definitions
# ---------------------------------------------------------------------------

class PredictiveAutoencoder(nn.Module):
    """
    Standard autoencoder with a separate predictor head.

    Args:
        input_dim  : flattened sequence length (seq_len * n_features)
        output_dim : number of prediction targets (n_out)
        latent_dim : bottleneck size
        hidden_dim : hidden layer width
    """

    def __init__(self, input_dim: int, output_dim: int,
                 latent_dim: int = 32, hidden_dim: int = 128):
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
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x: torch.Tensor):
        """
        Args:
            x: (batch, input_dim) — pre-flattened sequence
        Returns:
            pred  : (batch, output_dim)
            recon : (batch, input_dim)
            z     : (batch, latent_dim)
        """
        z     = self.encoder(x)
        recon = self.decoder(z)
        pred  = self.predictor(z)
        return pred, recon, z


class SparseAutoencoder(nn.Module):
    """
    Autoencoder with an L1 sparsity penalty on the latent representation.

    The sparsity loss is pre-scaled by ``sparsity_weight`` and returned as a
    fourth element so the trainer can add it directly without double-weighting.
    """

    def __init__(self, input_dim: int, output_dim: int,
                 latent_dim: int = 32, hidden_dim: int = 128,
                 sparsity_weight: float = 1e-3):
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
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x: torch.Tensor):
        """
        Returns:
            pred          : (batch, output_dim)
            recon         : (batch, input_dim)
            z             : (batch, latent_dim)
            sparsity_loss : scalar tensor (already weighted)
        """
        z             = self.encoder(x)
        recon         = self.decoder(z)
        pred          = self.predictor(z)
        sparsity_loss = self.sparsity_weight * z.abs().mean()
        return pred, recon, z, sparsity_loss


class VariationalAutoencoder(nn.Module):
    """
    VAE with a reparameterisation trick.

    The KL divergence is pre-scaled by ``beta`` and returned as a fourth
    element so the trainer can add it directly without double-weighting.
    """

    def __init__(self, input_dim: int, output_dim: int,
                 latent_dim: int = 32, hidden_dim: int = 128,
                 beta: float = 1.0):
        super().__init__()
        self.beta = beta

        self.encoder    = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.ReLU(),
        )
        self.mu_layer     = nn.Linear(hidden_dim, latent_dim)
        self.logvar_layer = nn.Linear(hidden_dim, latent_dim)

        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, input_dim),
        )
        self.predictor = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def reparameterise(self, mu: torch.Tensor,
                       logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        return mu + std * torch.randn_like(std)

    def forward(self, x: torch.Tensor):
        """
        Returns:
            pred  : (batch, output_dim)
            recon : (batch, input_dim)
            z     : (batch, latent_dim)
            kl    : scalar tensor (already beta-weighted)
        """
        h      = self.encoder(x)
        mu     = self.mu_layer(h)
        logvar = self.logvar_layer(h)
        z      = self.reparameterise(mu, logvar)
        recon  = self.decoder(z)
        pred   = self.predictor(z)

        # KL divergence: beta * -0.5 * sum(1 + log σ² - μ² - σ²)
        kl = self.beta * (
            -0.5 * (1 + logvar - mu.pow(2) - logvar.exp()).sum(dim=1).mean()
        )
        return pred, recon, z, kl


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _unpack_batch(batch: tuple) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Return (x, y) from a batch tuple, ignoring any extra elements.
    Datasets may return (x, y) or (x, y, mask, ...) — we only need the first two.
    """
    return batch[0], batch[1]


def _flatten_sequence(xb: torch.Tensor) -> torch.Tensor:
    """
    Flatten a 3-D sequence tensor to 2-D for MISO models.

    Args:
        xb: (batch, seq_len, n_features)
    Returns:
        xf: (batch, seq_len * n_features)
    """
    batch_size = xb.shape[0]
    return xb.reshape(batch_size, -1)


def _normalise_targets(yb: torch.Tensor) -> torch.Tensor:
    """
    Ensure targets are always 2-D: (batch, n_out).
    Handles datasets that return 1-D (batch,) for single-target problems.
    """
    if yb.ndim == 1:
        yb = yb.unsqueeze(-1)   # (batch,) → (batch, 1)
    return yb


# ---------------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------------

class PredictiveAutoencoderTrainer(BaseTrainer):

    @staticmethod
    def train_and_evaluate(
        train_set,
        test_set,
        results_dir: str,
        config: dict,
        val_set=None,
    ) -> dict:
        """
        Train one of the three autoencoder variants and return predictions.

        MISO contract
        -------------
        * Input  : (batch, seq_len, n_features) — full sequence is flattened.
        * Output : (batch, n_out)               — single-step, multi-target.

        Args:
            train_set  : PyTorch Dataset returning (x, y[, ...])
            test_set   : held-out test set — used ONLY for final evaluation
            results_dir: directory for any saved artefacts
            config     : hyperparameter dict (see keys below)
            val_set    : optional validation set for early stopping;
                         falls back to test_set when None (not recommended
                         for final benchmarking — always supply val_set)

        Config keys
        -----------
            variant         : "standard" | "sparse" | "variational"  (default: "standard")
            batch_size      : int   (default: 64)
            epochs          : int   (default: 100)
            patience        : int   (default: 10)
            lr              : float (default: 1e-3)
            weight_decay    : float (default: 1e-5)
            latent_dim      : int   (default: 32)
            hidden_dim      : int   (default: 128)
            recon_weight    : float (default: 0.5)  — weight on reconstruction loss
            sparsity_weight : float (default: 1e-3) — sparse variant only
            beta            : float (default: 1.0)  — variational variant only
        """

        variant    = config.get("variant",      "standard")
        batch_size = config.get("batch_size",   64)
        max_epochs = config.get("epochs",       100)
        patience   = config.get("patience",     10)
        lr         = config.get("lr",           1e-3)
        latent_dim = config.get("latent_dim",   32)
        hidden_dim = config.get("hidden_dim",   128)
        recon_w    = config.get("recon_weight", 0.5)

        print(f"\n{'='*20} AUTOENCODER ({variant.upper()}) {'='*20}")

        # ------------------------------------------------------------------ #
        # Data loaders                                                         #
        # ------------------------------------------------------------------ #
        train_loader = DataLoader(
            train_set, batch_size=batch_size, shuffle=True, drop_last=True
        )
        test_loader = DataLoader(
            test_set, batch_size=batch_size, shuffle=False, drop_last=False
        )

        # Early stopping runs on val_set when provided; falls back to test_set.
        # Supplying a dedicated val_set avoids leaking test data into model selection.
        eval_set    = val_set if val_set is not None else test_set
        eval_loader = DataLoader(
            eval_set, batch_size=batch_size, shuffle=False, drop_last=False
        )

        # ------------------------------------------------------------------ #
        # Infer dimensions from one batch                                      #
        # ------------------------------------------------------------------ #
        xb_sample, yb_sample = _unpack_batch(next(iter(train_loader)))

        # MISO: flatten full sequence → (batch, seq_len * n_features)
        seq_len   = xb_sample.shape[1]
        feat_dim  = xb_sample.shape[2]
        input_dim = seq_len * feat_dim

        yb_sample = _normalise_targets(yb_sample)
        n_out     = yb_sample.shape[-1]

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # ------------------------------------------------------------------ #
        # Model                                                                #
        # ------------------------------------------------------------------ #
        if variant == "sparse":
            model = SparseAutoencoder(
                input_dim, n_out, latent_dim, hidden_dim,
                config.get("sparsity_weight", 1e-3),
            ).to(device)

        elif variant == "variational":
            model = VariationalAutoencoder(
                input_dim, n_out, latent_dim, hidden_dim,
                config.get("beta", 1.0),
            ).to(device)

        else:
            model = PredictiveAutoencoder(
                input_dim, n_out, latent_dim, hidden_dim,
            ).to(device)

        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=lr,
            weight_decay=config.get("weight_decay", 1e-5),
        )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, patience=5, factor=0.5
        )
        mse = nn.MSELoss()

        # ------------------------------------------------------------------ #
        # Training loop                                                        #
        # ------------------------------------------------------------------ #
        best_val    = float("inf")
        no_improve  = 0
        train_losses = []
        train_time  = 0.0

        for epoch in range(max_epochs):
            model.train()
            ep_loss = 0.0
            t0      = time.time()

            for batch in train_loader:
                xb, yb = _unpack_batch(batch)

                # MISO: flatten full sequence
                xf = _flatten_sequence(xb).to(device)       # (B, S*F)
                yb = _normalise_targets(yb).to(device)       # (B, n_out)

                optimizer.zero_grad()
                out = model(xf)

                pred, recon = out[0], out[1]
                # out[3] is the pre-weighted auxiliary loss (sparsity or KL);
                # for the standard variant there is no extra term (0.0 float).
                extra_loss = out[3] if len(out) == 4 else 0.0

                loss = (
                    mse(pred, yb)           # prediction loss
                    + recon_w * mse(recon, xf)  # reconstruction over full input
                    + extra_loss            # sparsity / KL (already weighted)
                )

                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

                ep_loss += loss.item()

            train_time += time.time() - t0

            # -------------------------------------------------------------- #
            # Validation (early stopping)                                      #
            # -------------------------------------------------------------- #
            model.eval()
            val_loss = 0.0

            with torch.no_grad():
                for batch in eval_loader:
                    xb, yb = _unpack_batch(batch)
                    xf = _flatten_sequence(xb).to(device)
                    yb = _normalise_targets(yb).to(device)
                    val_loss += mse(model(xf)[0], yb).item()

            avg_train = ep_loss  / max(1, len(train_loader))
            avg_val   = val_loss / max(1, len(eval_loader))

            train_losses.append(avg_train)
            scheduler.step(avg_val)

            if avg_val < best_val:
                best_val   = avg_val
                no_improve = 0
            else:
                no_improve += 1
                if no_improve >= patience:
                    print(f"Early stopping at epoch {epoch + 1}")
                    break

        print(f"Autoencoder ({variant}) complete in {train_time:.2f}s")

        # ------------------------------------------------------------------ #
        # Final evaluation on the held-out test set                           #
        # ------------------------------------------------------------------ #
        model.eval()
        preds_list, true_list = [], []

        with torch.no_grad():
            for batch in test_loader:
                xb, yb = _unpack_batch(batch)
                xf   = _flatten_sequence(xb).to(device)
                pred = model(xf)[0].cpu()

                preds_list.append(pred.numpy())
                true_list.append(_normalise_targets(yb).numpy())

        # Safe concatenation that works for both 1-D and 2-D target arrays
        y_pred = np.concatenate(preds_list, axis=0)   # (N, n_out)
        y_true = np.concatenate(true_list,  axis=0)   # (N, n_out)

        if y_pred.ndim == 1:
            y_pred = y_pred[:, None]
        if y_true.ndim == 1:
            y_true = y_true[:, None]

        targets = [f"var_{i}" for i in range(n_out)]

        return {
            "model_name": f"Autoencoder-{variant}",
            "y_true_dict": {targets[i]: y_true[:, i] for i in range(n_out)},
            "y_pred_dict": {targets[i]: y_pred[:, i] for i in range(n_out)},
            "metadata": {
                "variant":           variant,
                "input_dim":         input_dim,
                "latent_dim":        latent_dim,
                "seq_len":           seq_len,
                "feat_dim":          feat_dim,
                "epochs_trained":    len(train_losses),
                "training_time_sec": train_time,
                "parameters":        sum(p.numel() for p in model.parameters()),
            },
        }