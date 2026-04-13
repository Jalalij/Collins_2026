"""
Predictive Autoencoder Models Implementation
Includes Deterministic, Sparse, and Variational Autoencoders
"""

import os
import sys
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from datetime import datetime
from typing import Dict, List, Tuple, Any
import warnings

warnings.filterwarnings("ignore")

from ..evaluation.standardized_evaluation import STANDARD_TARGET_COLUMNS


class PredictiveAutoencoder(nn.Module):
    """Base Predictive Autoencoder for time series prediction."""

    def __init__(
        self, input_dim=7, hidden_dims=[64, 32, 16], latent_dim=8, dropout_rate=0.1
    ):
        super(PredictiveAutoencoder, self).__init__()

        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.latent_dim = latent_dim
        self.dropout_rate = dropout_rate

        # Encoder layers
        encoder_layers = []
        prev_dim = input_dim
        for hidden_dim in hidden_dims:
            encoder_layers.extend(
                [
                    nn.Linear(prev_dim, hidden_dim),
                    nn.ReLU(),
                    nn.Dropout(dropout_rate),
                    nn.BatchNorm1d(hidden_dim),
                ]
            )
            prev_dim = hidden_dim

        encoder_layers.append(nn.Linear(prev_dim, latent_dim))
        self.encoder = nn.Sequential(*encoder_layers)

        # Decoder layers (mirror of encoder)
        decoder_layers = []
        prev_dim = latent_dim
        for hidden_dim in reversed(hidden_dims):
            decoder_layers.extend(
                [
                    nn.Linear(prev_dim, hidden_dim),
                    nn.ReLU(),
                    nn.Dropout(dropout_rate),
                    nn.BatchNorm1d(hidden_dim),
                ]
            )
            prev_dim = hidden_dim

        decoder_layers.append(nn.Linear(prev_dim, input_dim))
        self.decoder = nn.Sequential(*decoder_layers)

        # Predictor head (from latent space)
        self.predictor = nn.Sequential(
            nn.Linear(latent_dim, latent_dim * 2),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(latent_dim * 2, input_dim),
        )

    def encode(self, x):
        return self.encoder(x)

    def decode(self, z):
        return self.decoder(z)

    def predict(self, z):
        return self.predictor(z)

    def forward(self, x):
        z = self.encode(x)
        x_recon = self.decode(z)
        x_pred = self.predict(z)
        return x_recon, x_pred, z


class SparseAutoencoder(PredictiveAutoencoder):
    """Sparse Autoencoder variant with L1 regularization on latent space."""

    def __init__(
        self,
        input_dim=7,
        hidden_dims=[64, 32, 16],
        latent_dim=8,
        dropout_rate=0.1,
        sparsity_weight=0.001,
    ):
        super().__init__(input_dim, hidden_dims, latent_dim, dropout_rate)
        self.sparsity_weight = sparsity_weight

    def forward(self, x):
        z = self.encode(x)
        x_recon = self.decode(z)
        x_pred = self.predict(z)

        # Add sparsity penalty
        sparsity_loss = torch.mean(torch.abs(z))

        return x_recon, x_pred, z, sparsity_loss


class VariationalAutoencoder(nn.Module):
    """Variational Autoencoder for probabilistic predictions."""

    def __init__(
        self, input_dim=7, hidden_dims=[64, 32, 16], latent_dim=8, dropout_rate=0.1
    ):
        super(VariationalAutoencoder, self).__init__()

        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.latent_dim = latent_dim
        self.dropout_rate = dropout_rate

        # Encoder to latent distribution parameters
        encoder_layers = []
        prev_dim = input_dim
        for hidden_dim in hidden_dims:
            encoder_layers.extend(
                [
                    nn.Linear(prev_dim, hidden_dim),
                    nn.ReLU(),
                    nn.Dropout(dropout_rate),
                    nn.BatchNorm1d(hidden_dim),
                ]
            )
            prev_dim = hidden_dim

        self.encoder_base = nn.Sequential(*encoder_layers)
        self.fc_mu = nn.Linear(prev_dim, latent_dim)
        self.fc_logvar = nn.Linear(prev_dim, latent_dim)

        # Decoder
        decoder_layers = []
        prev_dim = latent_dim
        for hidden_dim in reversed(hidden_dims):
            decoder_layers.extend(
                [
                    nn.Linear(prev_dim, hidden_dim),
                    nn.ReLU(),
                    nn.Dropout(dropout_rate),
                    nn.BatchNorm1d(hidden_dim),
                ]
            )
            prev_dim = hidden_dim

        decoder_layers.append(nn.Linear(prev_dim, input_dim))
        self.decoder = nn.Sequential(*decoder_layers)

        # Predictor
        self.predictor = nn.Sequential(
            nn.Linear(latent_dim, latent_dim * 2),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(latent_dim * 2, input_dim),
        )

    def encode(self, x):
        h = self.encoder_base(x)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        return self.decoder(z)

    def predict(self, z):
        return self.predictor(z)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        x_recon = self.decode(z)
        x_pred = self.predict(z)
        return x_recon, x_pred, mu, logvar


class PredictiveAutoencoderTrainer:
    """Trainer for all Predictive Autoencoder variants."""

    @staticmethod
    def train_and_evaluate(
        train_data,
        test_data,
        evaluator,
        results_dir,
        model_type="deterministic",
        direction="UL",
    ) -> Tuple[Dict[str, Any], Dict, Dict]:
        """Train and evaluate Predictive Autoencoder variants."""
        print(
            f"\n"
            + "=" * 20
            + f" PREDICTIVE AUTOENCODER ({model_type.upper()}) "
            + "=" * 20
        )

        try:
            from ..data.process_data import ExperimentDataset
            from src.data.process_data import ExperimentDataset

            csv_dir = (
                "/home/jonathangornet/Documents/SatCom_TimeSeries/data/processed_data"
            )

            files = [os.path.join(csv_dir, f) for f in os.listdir(csv_dir)]

            train_set = ExperimentDataset(files, mode="train", direction=direction)
            test_set = ExperimentDataset(files, mode="test", direction=direction)

            train_loader = DataLoader(train_set, batch_size=64, shuffle=True)
            test_loader = DataLoader(test_set, batch_size=64, shuffle=True)

            # Initialize model based on type
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

            if model_type == "deterministic":
                model = PredictiveAutoencoder(
                    input_dim=train_set.num_features,
                    hidden_dims=[128, 64, 32],
                    latent_dim=16,
                    dropout_rate=0.1,
                ).to(device)
            elif model_type == "sparse":
                model = SparseAutoencoder(
                    input_dim=train_set.num_features,
                    hidden_dims=[128, 64, 32],
                    latent_dim=16,
                    dropout_rate=0.1,
                    sparsity_weight=0.001,
                ).to(device)
            elif model_type == "variational":
                model = VariationalAutoencoder(
                    input_dim=train_set.num_features,
                    hidden_dims=[128, 64, 32],
                    latent_dim=16,
                    dropout_rate=0.1,
                ).to(device)
            else:
                raise ValueError(f"Unknown model type: {model_type}")

            # Loss functions and optimizer
            recon_criterion = nn.MSELoss()
            pred_criterion = nn.MSELoss()
            optimizer = torch.optim.Adam(
                model.parameters(), lr=0.001, weight_decay=1e-5
            )
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, patience=10, factor=0.5
            )

            # Training
            print(f"Training {model_type} Predictive Autoencoder...")
            start_time = datetime.now()
            model.train()

            train_losses = []
            val_losses = []
            epochs = 100
            best_val_loss = float("inf")
            patience = 20
            no_improve = 0

            for epoch in range(epochs):
                epoch_train_loss = 0
                for batch_X, batch_y in train_loader:
                    batch_X, batch_y = batch_X.to(device), batch_y.to(device)

                    optimizer.zero_grad()

                    if model_type == "variational":
                        x_recon, x_pred, mu, logvar = model(batch_X)

                        # Reconstruction loss
                        recon_loss = recon_criterion(x_recon, batch_X)
                        # Prediction loss
                        pred_loss = pred_criterion(x_pred, batch_y)
                        # KL divergence loss
                        kl_loss = (
                            -0.5
                            * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
                            / batch_X.size(0)
                        )

                        total_loss = recon_loss + pred_loss + 0.001 * kl_loss

                    elif model_type == "sparse":
                        x_recon, x_pred, z, sparsity_loss = model(batch_X)

                        # Reconstruction loss
                        recon_loss = recon_criterion(x_recon, batch_X)
                        # Prediction loss
                        pred_loss = pred_criterion(x_pred, batch_y)
                        # Sparsity loss
                        total_loss = (
                            recon_loss
                            + pred_loss
                            + model.sparsity_weight * sparsity_loss
                        )

                    else:  # deterministic
                        x_recon, x_pred, z = model(batch_X)

                        # Reconstruction loss
                        recon_loss = recon_criterion(x_recon, batch_X)
                        # Prediction loss
                        pred_loss = pred_criterion(x_pred, batch_y)

                        total_loss = recon_loss + pred_loss

                    total_loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    optimizer.step()

                    epoch_train_loss += total_loss.item()

                # Validation
                model.eval()
                epoch_val_loss = 0
                with torch.no_grad():
                    for batch_X, batch_y in test_loader:
                        batch_X, batch_y = batch_X.to(device), batch_y.to(device)

                        if model_type == "variational":
                            x_recon, x_pred, mu, logvar = model(batch_X)
                            recon_loss = recon_criterion(x_recon, batch_X)
                            pred_loss = pred_criterion(x_pred, batch_y)
                            kl_loss = (
                                -0.5
                                * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
                                / batch_X.size(0)
                            )
                            total_loss = recon_loss + pred_loss + 0.001 * kl_loss
                        elif model_type == "sparse":
                            x_recon, x_pred, z, sparsity_loss = model(batch_X)
                            recon_loss = recon_criterion(x_recon, batch_X)
                            pred_loss = pred_criterion(x_pred, batch_y)
                            total_loss = (
                                recon_loss
                                + pred_loss
                                + model.sparsity_weight * sparsity_loss
                            )
                        else:
                            x_recon, x_pred, z = model(batch_X)
                            recon_loss = recon_criterion(x_recon, batch_X)
                            pred_loss = pred_criterion(x_pred, batch_y)
                            total_loss = recon_loss + pred_loss

                        epoch_val_loss += total_loss.item()

                avg_train_loss = epoch_train_loss / len(train_loader)
                avg_val_loss = epoch_val_loss / len(test_loader)

                train_losses.append(avg_train_loss)
                val_losses.append(avg_val_loss)

                scheduler.step(avg_val_loss)

                if avg_val_loss < best_val_loss:
                    best_val_loss = avg_val_loss
                    no_improve = 0
                else:
                    no_improve += 1

                if (epoch + 1) % 20 == 0:
                    print(
                        f"Epoch {epoch+1:3d}: Train Loss = {avg_train_loss:.6f}, Val Loss = {avg_val_loss:.6f}"
                    )

                if no_improve >= patience:
                    print(f"Early stopping at epoch {epoch+1}")
                    break

                model.train()

            training_time = (datetime.now() - start_time).total_seconds()
            print(
                f"{model_type} Autoencoder training completed in {training_time:.2f} seconds"
            )

            # Generate predictions
            model.eval()
            y_pred_list = []
            y_true_list = []

            with torch.no_grad():
                for batch_X, batch_y in test_loader:
                    batch_X, batch_y = batch_X.to(device), batch_y.to(device)

                    if model_type == "variational":
                        _, x_pred, _, _ = model(batch_X)
                    elif model_type == "sparse":
                        _, x_pred, _, _ = model(batch_X)
                    else:
                        _, x_pred, _ = model(batch_X)

                    y_pred_list.append(x_pred.cpu().numpy())
                    y_true_list.append(batch_y.cpu().numpy())

            y_pred_array = np.vstack(y_pred_list)
            y_true_array = np.vstack(y_true_list)

            # Inverse transform predictions
            y_pred_original = (
                y_pred_array  # data_prep["scaler"].inverse_transform(y_pred_array)
            )
            y_true_original = (
                y_true_array  # data_prep["scaler"].inverse_transform(y_true_array)
            )

            # Create prediction dictionaries for evaluator
            y_true_dict = {}
            y_pred_dict = {}

            for i, target in enumerate(STANDARD_TARGET_COLUMNS):
                y_true_dict[target] = y_true_original[:, i]
                y_pred_dict[target] = y_pred_original[:, i]

            # Training metadata
            training_metadata = {
                "model_type": f"PredictiveAutoencoder_{model_type}",
                "training_time": training_time,
                "epochs_trained": len(train_losses),
                "final_train_loss": train_losses[-1] if train_losses else 0,
                "final_val_loss": val_losses[-1] if val_losses else 0,
                "training_samples": len(X_train),
                "test_samples": len(X_test),
                "latent_dim": model.latent_dim if hasattr(model, "latent_dim") else 16,
                "parameters": sum(p.numel() for p in model.parameters()),
                "loss_curves": {"train_losses": train_losses, "val_losses": val_losses},
            }

            if model_type == "sparse":
                training_metadata["sparsity_weight"] = model.sparsity_weight

            # Evaluate using standardized framework
            results = evaluator.evaluate_model_predictions(
                f"PredictiveAutoencoder_{model_type}",
                y_true_dict,
                y_pred_dict,
                training_metadata,
            )

            return results, y_true_dict, y_pred_dict

        except Exception as e:
            print(f"{model_type} Predictive Autoencoder training failed: {str(e)}")
            import traceback

            traceback.print_exc()
            return None, {}, {}
