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