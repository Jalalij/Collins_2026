#!/usr/bin/env python3
"""
Transformer Model Implementation for Time Series
- Works with flat windows [B, D] or sequence windows [B, T, F]
- If flat, D must be T*F; F inferred from STANDARD_TARGET_COLUMNS (or 7 fallback)
- Predicts next-step [B, F]
"""

import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from datetime import datetime
from typing import Dict, Tuple, Any
import warnings
import yaml 

warnings.filterwarnings("ignore")

from ..evaluation.standardized_evaluation import STANDARD_TARGET_COLUMNS


# ----------------------------- Model -----------------------------


class SimpleTransformer(nn.Module):
    """
    Encoder-only Transformer for next-step forecasting.
    Input:  x [B, T, F]
    Output: yhat [B, F]
    """

    def __init__(
        self,
        input_dim: int,  # = F (num variables)
        d_model: int = 64,
        nhead: int = 4,
        num_layers: int = 2,
        dim_feedforward: int = 256,
        dropout: float = 0.1,
        max_len: int = 2000,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.d_model = d_model

        self.input_projection = nn.Linear(input_dim, d_model)

        pe = torch.zeros(max_len, d_model, dtype=torch.float32)
        nn.init.normal_(pe, mean=0.0, std=0.02)
        self.register_buffer("positional_encoding", pe)  # [max_len, d_model]

        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer_encoder = nn.TransformerEncoder(
            enc_layer, num_layers=num_layers
        )

        self.output_projection = nn.Linear(d_model, input_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Expect [B, T, F]
        if x.dim() != 3:
            raise ValueError(f"Expected x of shape [B,T,F], got {tuple(x.shape)}")

        B, T, F = x.shape
        x = self.input_projection(x)  # [B, T, d_model]

        if T > self.positional_encoding.size(0):
            raise ValueError(
                f"seq_len {T} exceeds positional_encoding max_len {self.positional_encoding.size(0)}"
            )
        x = x + self.positional_encoding[:T, :].unsqueeze(0)  # [B, T, d_model]
        x = self.dropout(x)

        h = self.transformer_encoder(x)  # [B, T, d_model]
        last_h = h[:, -1, :]  # [B, d_model]
        yhat = self.output_projection(last_h)  # [B, F]
        return yhat


# ----------------------------- Trainer -----------------------------


class TransformerTrainer:
    """Transformer model trainer."""

    @staticmethod
    def _collect_files(csv_dir: str, direction: str) -> list[str]:
        direction = direction.upper()
        files = [
            os.path.join(csv_dir, f)
            for f in os.listdir(csv_dir)
            if f.endswith(f"_{direction}.csv")
        ]
        if not files:
            raise FileNotFoundError(
                f"No CSVs found in {csv_dir} for direction '{direction}'."
            )
        return sorted(files)

    @staticmethod
    def _infer_dims_from_loader(loader: DataLoader) -> Tuple[int, int, bool]:
        """
        Peek one batch to get (seq_len, num_vars, is_flat).
        Accepts xb as [B,T,F] or [B,D] (flat with D=T*F).
        """
        xb, yb = next(iter(loader))

        if xb.dim() == 3:
            # [B, T, F]
            _, T, F = xb.shape
            if yb.dim() != 2 or yb.shape[1] != F:
                raise ValueError(
                    f"Expected yb to be [B,F] with F={F}, got {tuple(yb.shape)}"
                )
            return T, F, False

        elif xb.dim() == 2:
            # [B, D] -> infer F from STANDARD_TARGET_COLUMNS (fallback 7), then T = D//F
            _, D = xb.shape
            F = len(STANDARD_TARGET_COLUMNS) if len(STANDARD_TARGET_COLUMNS) > 0 else 7
            if D % F != 0:
                raise ValueError(
                    f"Flat input width D={D} is not divisible by inferred F={F}; "
                    f"cannot reshape to [B,T,F]. Adjust window_size or target count."
                )
            T = D // F
            if yb.dim() != 2 or yb.shape[1] != F:
                raise ValueError(
                    f"Expected yb to be [B,F] with F={F}, got {tuple(yb.shape)}"
                )
            return T, F, True

        else:
            raise ValueError(f"Unsupported xb shape {tuple(xb.shape)}")

    @staticmethod
    def _reshape_if_flat(xb: torch.Tensor, seq_len: int, num_vars: int) -> torch.Tensor:
        # If flat [B, D], reshape to [B, T, F]; else return as-is
        if xb.dim() == 2:
            B, D = xb.shape
            if D != seq_len * num_vars:
                raise ValueError(f"Cannot reshape [B,{D}] to [B,{seq_len},{num_vars}]")
            return xb.view(B, seq_len, num_vars)
        return xb

    @staticmethod
    def train_and_evaluate(
        train_set, test_set, evaluator, results_dir,
        direction="UL", config: dict | None = None
    ) -> Tuple[Dict[str, Any], Dict, Dict]:

        print("\n" + "=" * 20 + " TRANSFORMER MODEL " + "=" * 20)

        try:
            # ---------------- Config ----------------
            cfg = config or {}
            batch_size   = cfg.get("batch_size", 64)
            max_epochs   = cfg.get("max_epochs", 100)
            patience     = cfg.get("patience", 10)
            timeout_sec  = cfg.get("timeout_sec", 600)

            # ---------------- Data ----------------
            train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, drop_last=True)
            test_loader  = DataLoader(test_set,  batch_size=batch_size, shuffle=False, drop_last=False)

            seq_len, num_vars, is_flat = TransformerTrainer._infer_dims_from_loader(train_loader)

            # ---------------- Model ----------------
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            model = SimpleTransformer(
                input_dim=num_vars,
                d_model=cfg.get("d_model", 64),
                nhead=cfg.get("nhead", 4),
                num_layers=cfg.get("num_layers", 2),
                dim_feedforward=cfg.get("dim_feedforward", 256),
                dropout=cfg.get("dropout", 0.1),
                max_len=max(2048, seq_len + 1),
            ).to(device)

            # ---------------- Optim ----------------
            criterion = nn.MSELoss()
            optimizer = torch.optim.Adam(
                model.parameters(),
                lr=cfg.get("lr", 1e-3),
                weight_decay=cfg.get("weight_decay", 1e-5),
            )
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, patience=5, factor=0.5
            )

            # ---------------- Train loop ----------------
            print("Training Transformer...")
            import time
            start_wall = time.time()

            best_val = float("inf")
            no_improve = 0
            train_losses, val_losses = [], []

            training_time = 0
            testing_time = 0
            
            for epoch in range(max_epochs):
                if time.time() - start_wall > timeout_sec:
                    print(f"⏰ Training timeout after {epoch} epochs")
                    break
                
                t0 = time.time()
                model.train()
                epoch_train_loss = 0.0

                for batch_X, batch_y in train_loader:
                    # Reshape flat input if needed
                    batch_X = TransformerTrainer._reshape_if_flat(
                        batch_X, seq_len, num_vars
                    )

                    batch_X = batch_X.to(device)  # [B,T,F]
                    batch_y = batch_y.to(device)  # [B,F]

                    optimizer.zero_grad()
                    outputs = model(batch_X)  # [B,F]
                    loss = criterion(outputs, batch_y)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    optimizer.step()

                    epoch_train_loss += loss.item()

                training_time += time.time() - t0

                t0 = time.time()
                # Validation
                model.eval()
                epoch_val_loss = 0.0
                with torch.no_grad():
                    for batch_X, batch_y in test_loader:
                        batch_X = TransformerTrainer._reshape_if_flat(
                            batch_X, seq_len, num_vars
                        )
                        batch_X = batch_X.to(device)
                        batch_y = batch_y.to(device)
                        outputs = model(batch_X)
                        loss = criterion(outputs, batch_y)
                        epoch_val_loss += loss.item()

                testing_time += time.time() - t0

                avg_train = epoch_train_loss / max(1, len(train_loader))
                avg_val = epoch_val_loss / max(1, len(test_loader))
                train_losses.append(avg_train)
                val_losses.append(avg_val)
                scheduler.step(avg_val)

                if (epoch + 1) % 5 == 0:
                    print(
                        f"Epoch {epoch+1:3d} | train {avg_train:.6f} | val {avg_val:.6f}"
                    )

                if avg_val < best_val:
                    best_val = avg_val
                    no_improve = 0
                else:
                    no_improve += 1
                    # if no_improve >= patience:
                    #     print(f"Early stopping at epoch {epoch+1}")
                    #     break

            print(f"Transformer training completed in {training_time:.2f}s")

            # ---------------- Predictions ----------------
            model.eval()
            y_pred_list, y_true_list = [], []
            with torch.no_grad():
                for batch_X, batch_y in test_loader:
                    batch_X = TransformerTrainer._reshape_if_flat(
                        batch_X, seq_len, num_vars
                    )
                    batch_X = batch_X.to(device)
                    outputs = model(batch_X)  # [B,F]
                    y_pred_list.append(
                        test_set.inverse_transform(outputs).cpu().numpy()
                    )
                    y_true_list.append(test_set.inverse_transform(batch_y).numpy())

            y_pred_array = np.vstack(y_pred_list)  # [N,F]
            y_true_array = np.vstack(y_true_list)  # [N,F]

            # Build dicts for evaluator (match STANDARD_TARGET_COLUMNS length if possible)
            F = y_true_array.shape[1]
            targets = (
                STANDARD_TARGET_COLUMNS[:F]
                if len(STANDARD_TARGET_COLUMNS) >= F
                else [f"var_{i}" for i in range(F)]
            )
            y_true_dict = {targets[i]: y_true_array[:, i] for i in range(F)}
            y_pred_dict = {targets[i]: y_pred_array[:, i] for i in range(F)}

            # ---------------- Metadata & evaluate ----------------
            param_count = sum(p.numel() for p in model.parameters())
            training_metadata = {
                "model_type": "Transformer",
                "direction": direction,
                "training_time_sec": training_time,
                "testing_time_sec": testing_time,
                "epochs_trained": len(train_losses),
                "final_train_loss": train_losses[-1] if train_losses else None,
                "final_val_loss": val_losses[-1] if val_losses else None,
                "train_batches": len(train_loader),
                "test_batches": len(test_loader),
                "sequence_length": seq_len,
                "num_features": num_vars,
                "d_model": model.d_model,
                "nhead": 4,
                "num_layers": 2,
                "dim_feedforward": 256,
                "parameters": param_count,
            }

            results = evaluator.evaluate_model_predictions(
                "Transformer", y_true_dict, y_pred_dict, training_metadata
            )

            return results, y_true_dict, y_pred_dict, training_metadata

        except Exception as e:
            print(f"Transformer training failed: {str(e)}")
            import traceback

            traceback.print_exc()
            return None, {}, {}, None
